# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Launcher-side pieces of the external-routing verification.

The device bootstrap lives in ``scripts/verify/run_routing_verify.py``; what is here is
the part that can be decided without a device -- the run id every rank shares, the layer
selection, the exit-code mapping and the aggregation of the per-rank files.

Keeping these in the package rather than in the script is what lets them be tested on a
CPU host at all.

Exit codes: 0 pass, 2 not executed, 3 unsupported topology, 4 numerical mismatch,
5 organization failure. A caller-side ``timeout`` uses 124, which is never a pass.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

EXIT_PASS = 0
EXIT_NOT_EXECUTED = 2
EXIT_UNSUPPORTED = 3
EXIT_MISMATCH = 4
EXIT_ORGANIZATION = 5

DEFAULT_RESULTS_ROOT = "verify-runs"


def resolve_run_id(value: str | None) -> str:
    """Return the id every rank shares.

    Generated once by the launcher and passed to the workers as an argument. A rank that
    generated its own would write into a different directory from its peers, and the
    summary would be built from whichever file happened to exist.
    """

    if value:
        return value
    return time.strftime("%Y%m%d-%H%M%S")


def resolve_output_dir(value: str | None, run_id: str, *, root: Path | None = None) -> Path:
    """Return the directory for this run, shared by every rank."""

    base = Path(value) if value else (root or Path.cwd()) / DEFAULT_RESULTS_ROOT
    return base / run_id


def select_layer(model, requested: int | None) -> int:
    """Return the model layer index to verify.

    The value recorded has to be a model layer index, not a position in the experts
    list: the two differ on a model whose MoE layers are not the first ones.
    """

    layer_indices = tuple(int(index) for index in model.get_experts_layer_indices())
    if not layer_indices:
        raise ValueError("the model has no MoE layers to verify")
    if requested is None:
        return layer_indices[0]
    if requested not in layer_indices:
        raise ValueError(
            f"layer {requested} is not an MoE layer; the model has {list(layer_indices)}",
        )
    return requested


def plan_signature(plan) -> str:
    """Return a canonical, hashable form of the plan for persistence.

    ``VerificationPlan.signature()`` returns nested tuples, which JSON turns into nested
    lists. Comparing those as set elements raises an unhashable-type error the moment a
    real run writes them, so the persisted form is a JSON string with sorted keys: stable
    across ranks, hashable, and readable in the file.
    """

    return json.dumps(plan.signature(), sort_keys=True, default=list)


def exit_code_for(record) -> int:
    """Map a verdict onto this entry point's exit codes."""

    if record is None:
        return EXIT_ORGANIZATION
    if record.topology_refusal:
        return EXIT_UNSUPPORTED
    if record.summary is None:
        return EXIT_NOT_EXECUTED
    return EXIT_MISMATCH if record.summary.failed else EXIT_PASS


REQUIRED_VERSIONS = ("afd-plugin", "vllm", "vllm-ascend")

# Every result file has to say which run it belongs to. Without this, a file left over
# from an earlier run in the same directory would count toward this run's verdict.
REQUIRED_FIELDS = ("run_id", "rank", "exit_code", "plan_signature", "versions")


def write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def aggregate(
    output_dir: Path,
    dp_size: int,
    *,
    expected_run_id: str | None = None,
) -> tuple[int, list[str]]:
    """Read every rank's file and decide the run's verdict.

    Reading files needs no collective, so the summary can also be produced offline. A
    missing file, a file from another run, a missing field, disagreeing plans,
    disagreeing versions or disagreeing per-rank outcomes all fail the run: a subset of
    ranks agreeing on anything is not a result.
    """

    reasons: list[str] = []
    payloads = []
    for rank in range(dp_size):
        path = Path(output_dir) / f"rank{rank}.json"
        if not path.exists():
            reasons.append(f"rank {rank} produced no result at {path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        missing = [field for field in REQUIRED_FIELDS if field not in payload]
        if missing:
            reasons.append(f"rank {rank} result is missing {missing}")
            continue
        if int(payload["rank"]) != rank:
            reasons.append(
                f"rank {rank} result claims rank {payload['rank']}; the file name and "
                "its contents have to agree",
            )
            continue
        if expected_run_id is not None and payload["run_id"] != expected_run_id:
            reasons.append(
                f"rank {rank} result belongs to run {payload['run_id']!r}, "
                f"not {expected_run_id!r}",
            )
            continue
        absent = [name for name in REQUIRED_VERSIONS if name not in payload["versions"]]
        if absent:
            reasons.append(f"rank {rank} result is missing versions {absent}")
            continue
        payloads.append(payload)
    if reasons:
        return EXIT_ORGANIZATION, reasons

    signatures = {payload["plan_signature"] for payload in payloads}
    if len(signatures) != 1:
        reasons.append(f"the ranks reported different plans: {sorted(signatures)}")
    versions = {json.dumps(payload["versions"], sort_keys=True) for payload in payloads}
    if len(versions) != 1:
        reasons.append("the ranks reported different versions")

    codes = {int(payload["exit_code"]) for payload in payloads}
    for payload in payloads:
        reasons.extend(payload.get("reasons", []))
    if len(codes) != 1:
        reasons.append(f"the ranks disagreed on the outcome: {sorted(codes)}")
    if reasons:
        # A consistent pass cannot carry reasons. Plans or versions that disagree, or a
        # rank that filed an objection, make this an aggregation failure even when every
        # per-rank code happens to be success.
        worst = max(codes) if codes else EXIT_ORGANIZATION
        return (worst if worst != EXIT_PASS else EXIT_ORGANIZATION), reasons
    return codes.pop(), []


__all__ = [
    "DEFAULT_RESULTS_ROOT",
    "EXIT_MISMATCH",
    "EXIT_NOT_EXECUTED",
    "EXIT_ORGANIZATION",
    "EXIT_PASS",
    "EXIT_UNSUPPORTED",
    "REQUIRED_FIELDS",
    "REQUIRED_VERSIONS",
    "aggregate",
    "exit_code_for",
    "plan_signature",
    "resolve_output_dir",
    "resolve_run_id",
    "select_layer",
    "write_result",
]
