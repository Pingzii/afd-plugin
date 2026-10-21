#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Classify and review three-way updates of the two A3 layered GMM operators.

This tool deliberately does not edit source files. It identifies the files that
can be replayed mechanically and the files that need a semantic review.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

OPERATORS = {
    "gmm": ("grouped_matmul", "grouped_matmul_layered"),
    "swiglu": (
        "grouped_matmul_swiglu_quant_v2",
        "grouped_matmul_swiglu_quant_v2_layered",
    ),
}

# Upstream op_host/op_api/ file stem -> AFD op_api/ filename. The op_api layer
# was consolidated and renamed per file, so these are explicit.
OP_API_FILE_MAP = {
    "gmm": {
        "aclnn_grouped_matmul.cpp": "aclnn_grouped_matmul_layered.cpp",
        "aclnn_grouped_matmul.h": "aclnn_grouped_matmul_layered.h",
        "grouped_matmul.cpp": "aclnn_grouped_matmul_l0op.cpp",
        "grouped_matmul.h": "grouped_matmul.h",
        "grouped_matmul_util.cpp": "aclnn_grouped_matmul_util.cpp",
        "grouped_matmul_util.h": "grouped_matmul_util.h",
    },
    "swiglu": {
        "grouped_matmul_swiglu_quant_v2.cpp": (
            "aclnn_grouped_matmul_swiglu_quant_v2_layered_l0op.cpp"
        ),
        "grouped_matmul_swiglu_quant_v2.h": "grouped_matmul_swiglu_quant_v2_layered.h",
        "grouped_matmul_swiglu_quant_v2_utils.h": (
            "grouped_matmul_swiglu_quant_v2_layered_utils.h"
        ),
        "aclnn_grouped_matmul_swiglu_quant_v2.cpp": (
            "aclnn_grouped_matmul_swiglu_quant_v2_layered.cpp"
        ),
        "aclnn_grouped_matmul_swiglu_quant_v2.h": (
            "aclnn_grouped_matmul_swiglu_quant_v2_layered.h"
        ),
    },
}

# Full-path renames outside op_api. gmm keeps every upstream filename except
# the kernel entry; swiglu renames the operator token in every remaining path,
# so it needs no entry here.
RENAMED_PATHS = {
    "gmm": {
        "op_kernel/grouped_matmul.cpp": "op_kernel/grouped_matmul_layered.cpp",
    },
    "swiglu": {},
}

EXCLUDED_PARTS = ("arch35", "tests", "examples", "docs")
RENAME_TOKENS = (
    ("grouped_matmul_swiglu_quant_v2_layered", "grouped_matmul_swiglu_quant_v2"),
    ("GroupedMatmulSwigluQuantV2Layered", "GroupedMatmulSwigluQuantV2"),
    ("grouped_matmul_layered", "grouped_matmul"),
    ("GroupedMatmulLayered", "GroupedMatmul"),
    ("ALL_WEIGHT", "WEIGHT"),
    ("ALL_SCALE", "SCALE"),
    ("ALL_BIAS", "BIAS"),
    ("all_weight", "weight"),
    ("all_weight_scale", "weight_scale"),
    ("all_weight_assist_matrix", "weight_assist_matrix"),
)


ANCHOR_MARKERS = ("layered:", "PATCH START")

# Default path rule per operator when no explicit map entry applies: gmm kept
# the upstream filenames verbatim (only the kernel entry was renamed), while
# swiglu renamed the operator token in every remaining path.
TOKEN_SWAP_KINDS = ("swiglu",)


def has_layered_anchor(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return any(marker in text for marker in ANCHOR_MARKERS)


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


def counterpart(upstream_root: Path, afd_root: Path, relative: Path, kind: str) -> Path:
    """Map an upstream file to its AFD layered counterpart.

    op_api files follow the explicit consolidated map; outside op_api, gmm
    falls back to the upstream path verbatim and swiglu falls back to
    replacing the operator token.
    """
    source_name, layered_name = OPERATORS[kind]
    text = str(relative)
    renamed = RENAMED_PATHS[kind].get(text)
    if renamed is not None:
        return afd_root / renamed
    if text.startswith("op_host/op_api/"):
        stem = text[len("op_host/op_api/") :]
        mapped = OP_API_FILE_MAP[kind].get(stem)
        if mapped is None:
            if kind in TOKEN_SWAP_KINDS:
                stem = stem.replace(source_name, layered_name)
            mapped = stem
        return afd_root / "op_api" / mapped
    if kind in TOKEN_SWAP_KINDS:
        text = text.replace(source_name, layered_name)
    return afd_root / text


def normalized(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    for old, new in RENAME_TOKENS:
        text = text.replace(old, new)
    # Formatting changes are noise for migration triage. Keep comments because
    # upstream comments can expose a changed contract.
    return "\n".join(line.strip() for line in text.splitlines())


def classify(upstream: Path, afd: Path, kind: str) -> list[tuple[str, str, bool]]:
    rows = []
    for source in sorted(upstream.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(upstream)
        if is_excluded(relative):
            continue
        target = counterpart(upstream, afd, relative, kind)
        if not target.exists():
            status = "ABSENT_IN_AFD"
        elif source.read_bytes() == target.read_bytes():
            status = "IDENTICAL"
        elif normalized(source) == normalized(target):
            status = "RENAME_OR_FORMAT_ONLY"
        else:
            status = "SUBSTANTIVE"
        anchored = target.exists() and has_layered_anchor(target)
        rows.append((status, str(relative), anchored))
    return rows


def print_classification(upstream: Path, afd: Path, kind: str) -> int:
    rows = classify(upstream, afd, kind)
    print(f"## {kind}")
    print("| status | upstream file | layered anchor |")
    print("|---|---|---|")
    for status, relative, anchor in rows:
        print(f"| {status} | `{relative}` | {'yes' if anchor else 'no'} |")
    counts = {}
    for status, _, _ in rows:
        counts[status] = counts.get(status, 0) + 1
    summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    print("\nCounts: " + summary)
    return 0


def print_three_way(old: Path, new: Path, afd: Path, kind: str) -> int:
    source_old = old
    source_new = new
    source_name, _ = OPERATORS[kind]
    print(f"## three-way replay: {kind}")
    print("| action | upstream file | reason |")
    print("|---|---|---|")
    old_files = {
        p.relative_to(source_old) for p in source_old.rglob("*") if p.is_file()
    }
    new_files = {
        p.relative_to(source_new) for p in source_new.rglob("*") if p.is_file()
    }
    for relative in sorted(new_files | old_files):
        if is_excluded(relative):
            continue
        old_file = source_old / relative
        new_file = source_new / relative
        target = counterpart(source_new, afd, relative, kind)
        if not old_file.exists():
            action, reason = (
                "REVIEW",
                "new upstream file; decide whether A3 scope includes it",
            )
        elif not new_file.exists():
            action, reason = "REVIEW", "upstream file removed"
        elif old_file.read_bytes() == new_file.read_bytes():
            continue
        elif not target.exists():
            action, reason = (
                "REVIEW",
                "upstream file changed but AFD has no counterpart",
            )
        elif has_layered_anchor(target):
            action, reason = "HAND_MERGE", "upstream delta touches a layered anchor"
        else:
            action, reason = "REPLAY_AND_REVIEW", "AFD file has no layered anchor"
        print(f"| {action} | `{relative}` | {reason} |")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("classify", "threeway"))
    parser.add_argument("--kind", choices=sorted(OPERATORS), required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--afd", type=Path, required=True)
    parser.add_argument("--old-upstream", type=Path)
    args = parser.parse_args()
    if args.command == "classify":
        return print_classification(args.upstream, args.afd, args.kind)
    if args.old_upstream is None:
        parser.error("threeway requires --old-upstream")
    return print_three_way(args.old_upstream, args.upstream, args.afd, args.kind)


if __name__ == "__main__":
    sys.exit(main())
