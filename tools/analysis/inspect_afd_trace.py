#!/usr/bin/env python3
"""Summarize torch_npu profiler traces for the AFD A2E transport comparison.

Usage:
    inspect_afd_trace.py DIR [DIR ...] [--top N] [--json]

DIR is a torch_profiler_dir produced by vLLM's ``--profiler-config``. Each DP rank
writes its own trace file, so a directory usually holds several; every file is
summarized and then aggregated.

The A2E send op runs on the attention side, which is the only side with a
controllable profile window today, so this tool is aimed at answering one
question: does the transport time actually halve when the payload is quantized
to FP8, and how much of a step does it account for.

Output is deliberately split into three numbers per directory:

  kernel_total   sum of every device kernel duration
  span           wall time from the first to the last event in the trace
  span - total   the gap, i.e. time the stream spent waiting rather than running
                 a kernel. An asynchronous collective shows up here, not in
                 kernel_total, so both numbers matter.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ACLNN kernel names carry an opaque hash suffix, e.g.
#   MoeGatingTopKHash_84810d9a12f03db59eeda174f506e44c_10004
# Grouping on the raw name produces thousands of one-off rows.
_HASH_SUFFIX = re.compile(r"_[0-9a-f]{16,}(?:_[0-9]+)?$")

# Names that indicate the A2E/E2A transport rather than local compute.
_TRANSPORT_HINTS = ("a2e", "e2a", "cam", "hccl", "alltoall", "allgather", "dispatch")

# Event categories torch_npu emits. "Kernel" carries on-device execution; the
# others are host-side bookkeeping and would double count.
_DEVICE_CATEGORIES = ("Kernel", "kernel", "aclnn_kernel")
_Number = int | float


def _normalize(name: str) -> str:
    return _HASH_SUFFIX.sub("", name)


def _load_events(path: Path) -> list[dict]:
    # utf-8-sig also strips a BOM, which traces copied through editors pick up.
    payload = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    return [event for event in events if isinstance(event, dict)]


def _is_device_kernel(event: dict) -> bool:
    if event.get("ph") != "X":
        return False
    if not isinstance(event.get("dur"), _Number) or event["dur"] <= 0:
        return False
    category = str(event.get("cat", ""))
    return category in _DEVICE_CATEGORIES


def summarize(path: Path, top: int) -> dict:
    events = _load_events(path)

    by_name: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if _is_device_kernel(event):
            by_name[_normalize(str(event.get("name", "<unnamed>")))].append(
                float(event["dur"])
            )

    kernel_total = sum(sum(durations) for durations in by_name.values())

    starts = [event["ts"] for event in events if isinstance(event.get("ts"), _Number)]
    ends = [
        event["ts"] + event["dur"]
        for event in events
        if isinstance(event.get("ts"), _Number)
        and isinstance(event.get("dur"), _Number)
    ]
    span = (max(ends) - min(starts)) if starts and ends else 0.0

    ranked = sorted(
        ((name, len(durations), sum(durations)) for name, durations in by_name.items()),
        key=lambda row: row[2],
        reverse=True,
    )
    transport = [
        row
        for row in ranked
        if any(hint in row[0].lower() for hint in _TRANSPORT_HINTS)
    ]

    return {
        "file": path.name,
        "event_count": len(events),
        "distinct_kernels": len(by_name),
        "kernel_total_us": kernel_total,
        "span_us": span,
        "gap_us": max(span - kernel_total, 0.0),
        "top": ranked[:top],
        "transport": transport,
    }


def _merge(summaries: list[dict], top: int) -> dict:
    """Sum per-rank results so a 4-rank directory yields one row."""
    merged: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for summary in summaries:
        for name, count, total in summary["top"]:
            merged[name][0] += count
            merged[name][1] += total
    ranked = sorted(
        ((name, int(values[0]), values[1]) for name, values in merged.items()),
        key=lambda row: row[2],
        reverse=True,
    )
    kernel_total = sum(summary["kernel_total_us"] for summary in summaries)
    span = max((summary["span_us"] for summary in summaries), default=0.0)
    return {
        "ranks": len(summaries),
        "kernel_total_us": kernel_total,
        "span_us": span,
        "gap_us": max(span - kernel_total, 0.0),
        "top": ranked[:top],
    }


def _ms(value: float) -> str:
    return f"{value / 1000.0:9.1f}"


def _print_directory(label: str, summaries: list[dict], top: int) -> dict:
    merged = _merge(summaries, top)
    print(f"\n=== {label} ===")
    print(f"  traces              : {len(summaries)} file(s)")
    print(f"  device kernel total : {_ms(merged['kernel_total_us'])} ms")
    print(f"  trace span          : {_ms(merged['span_us'])} ms")
    print(
        f"  idle (span - kernel): {_ms(merged['gap_us'])} ms  <- waiting, not running"
    )
    if merged["span_us"] > 0:
        share = merged["kernel_total_us"] / merged["span_us"] * 100.0
        print(f"  kernel share of span: {share:9.1f} %")

    print(f"\n  top {top} device kernels by total time:")
    print(f"    {'total(ms)':>10} {'count':>8}  name")
    for name, count, total in merged["top"]:
        print(f"    {_ms(total)} {count:>8}  {name}")

    transport = [
        row
        for row in merged["top"]
        if any(h in row[0].lower() for h in _TRANSPORT_HINTS)
    ]
    transport_total = sum(row[2] for row in transport)
    print(f"\n  transport-like kernels: {_ms(transport_total)} ms")
    if merged["span_us"] > 0:
        transport_share = transport_total / merged["span_us"] * 100.0
        print(f"  transport share      : {transport_share:9.1f} %")
    for name, count, total in transport[:10]:
        print(f"    {_ms(total)} {count:>8}  {name}")

    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--json", action="store_true", help="emit raw numbers as JSON")
    args = parser.parse_args()

    results = {}
    for directory in args.directories:
        if not directory.is_dir():
            print(f"not a directory: {directory}", file=sys.stderr)
            return 2
        traces = sorted(directory.glob("*.json"))
        if not traces:
            print(f"no trace json under {directory}", file=sys.stderr)
            return 2
        summaries = [summarize(trace, args.top) for trace in traces]
        results[str(directory)] = _print_directory(str(directory), summaries, args.top)

    if len(results) == 2:
        labels = list(results)
        left, right = (results[labels[0]], results[labels[1]])
        print("\n=== comparison (ms) ===")
        print(
            f"  {'metric':>22} {_ms(left['kernel_total_us']):>12} "
            f"{_ms(right['kernel_total_us']):>12}"
        )
        for title, key in (
            ("kernel total", "kernel_total_us"),
            ("span", "span_us"),
            ("idle", "gap_us"),
        ):
            row = f"  {title:>22} {_ms(left[key]):>12} {_ms(right[key]):>12}"
            print(row)
        print(f"\n  left  = {labels[0]}")
        print(f"  right = {labels[1]}")

    if args.json:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
