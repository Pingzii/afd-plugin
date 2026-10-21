# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Compare the native routing path with the external-routing prepare patches.

The prepare patches only matter if Attention-side routing through them yields the
same routed output the native path produces for the same routing. That comparison
runs on a device, in two levels:

- **Communication level** — a real multi-rank HCCL group, ``prepare``, a
  hand-built routed contribution, ``finalize``. Proves row identity, padding,
  ``mc2_mask``, scale transport and the reduce/trim, without needing expert
  weights. A stubbed group does not count: it only re-tests CPU framework logic.
- **Expert level** — the production-loaded W4A8MXFP experts, ``prepare`` then
  ``apply`` then ``finalize``. Proves the full routed values agree. Both sides must
  use the same weights and the same routing, the reference ``prepare`` must be the
  unpatched one, and the comparison must exclude the shared-expert output and
  apply routed scaling exactly once.

Neither level can run from this entry point: both need a normally-initialised FFN
worker, because that worker owns the HCCL groups and the production weight loading
(``AFDNPUFFNModelRunner.load_model`` → native ``loader.load_model``). Constructing a
runner without initialisation, or fabricating packed FP4 weights, would not
exercise the production path.

This module holds the part that does not need a device: the case matrix, the
result formatting, and the comparison itself once two routed tensors exist.
``--dry-run`` prints the matrix without touching a device.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Final

# Checkpoint the FFN serve script records; DSV4_MODEL_PATH or --model overrides it.
DEFAULT_MODEL_PATH: Final[str] = "/mnt/share/weight/DeepSeek-V4-Flash"

# Absolute tolerance for BF16 activations through a W4A8MXFP expert stack. The
# comparison reports the measured error as well, so a near miss is visible rather
# than hidden behind a pass.
DEFAULT_ATOL: Final[float] = 2e-2
DEFAULT_RTOL: Final[float] = 2e-2


@dataclass(frozen=True)
class RoutingCase:
    """One shape of work the two prepare paths must agree on.

    Attributes:
        name: Short label used in the report.
        tokens_per_rank: Local row count for each rank. Uneven entries exercise
            the padding rule; a zero exercises an empty shard.
        top_k: Experts per token.
        with_scale: Whether the case carries an E8M0 scale (FP8) or not (BF16).
        note: Why the case exists, printed in the dry run.
    """

    name: str
    tokens_per_rank: tuple[int, ...]
    top_k: int
    with_scale: bool
    note: str

    @property
    def dp_size(self) -> int:
        return len(self.tokens_per_rank)

    @property
    def max_rows(self) -> int:
        return max(self.tokens_per_rank)

    @property
    def gathered_rows(self) -> int:
        """Rows the ALLGATHER path produces: ``dp_size * max(local_rows)``."""

        return self.dp_size * self.max_rows


DEFAULT_CASES: Final[tuple[RoutingCase, ...]] = (
    RoutingCase(
        name="uniform_bf16",
        tokens_per_rank=(2, 2, 2, 2),
        top_k=8,
        with_scale=False,
        note="every rank the same size; the simplest agreement check",
    ),
    RoutingCase(
        name="uneven_bf16",
        tokens_per_rank=(3, 1, 2, 0),
        top_k=8,
        with_scale=False,
        note="uneven rows plus an empty rank; catches padding and shift bugs",
    ),
    RoutingCase(
        name="uneven_fp8",
        tokens_per_rank=(3, 1, 2, 0),
        top_k=8,
        with_scale=True,
        note="same shape with a scale, so the scale must follow the rows",
    ),
    RoutingCase(
        name="single_token_bf16",
        tokens_per_rank=(1, 1, 1, 1),
        top_k=8,
        with_scale=False,
        note="boundary case at the smallest non-empty batch",
    ),
)


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of comparing one case's candidate routing against the reference."""

    case: str
    passed: bool
    max_abs_error: float
    max_rel_error: float
    shape_ok: bool
    finite_ok: bool
    detail: str = ""

    def render(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{status} {self.case:<20} "
            f"shape_ok={self.shape_ok} finite_ok={self.finite_ok} "
            f"max_abs={self.max_abs_error:.3e} max_rel={self.max_rel_error:.3e}"
            + (f"  {self.detail}" if self.detail else "")
        )


@dataclass
class CaseReport:
    """Collected results, so the device half can print one summary."""

    results: list[ComparisonResult] = field(default_factory=list)

    def add(self, result: ComparisonResult) -> None:
        self.results.append(result)

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(item.passed for item in self.results)

    def render(self) -> str:
        lines = [result.render() for result in self.results]
        lines.append(
            "all cases passed" if self.passed else "at least one case failed",
        )
        return "\n".join(lines)


def compare_routed(reference, candidate, *, case: str, atol: float = DEFAULT_ATOL,
                   rtol: float = DEFAULT_RTOL) -> ComparisonResult:
    """Compare the candidate's routed output against the reference.

    Shapes are checked first because a silent row-count difference is the failure
    mode the padding rule is most likely to produce. Non-finite values fail before
    the error numbers mean anything.

    ``reference`` and ``candidate`` are the per-rank local routed tensors: the
    result after ``finalize`` has trimmed back to this rank's rows.
    """

    import torch

    if reference.shape != candidate.shape:
        return ComparisonResult(
            case=case,
            passed=False,
            max_abs_error=float("inf"),
            max_rel_error=float("inf"),
            shape_ok=False,
            finite_ok=False,
            detail=f"shape {tuple(candidate.shape)} != {tuple(reference.shape)}",
        )

    finite_ok = bool(torch.isfinite(candidate).all())
    if not finite_ok:
        return ComparisonResult(
            case=case,
            passed=False,
            max_abs_error=float("inf"),
            max_rel_error=float("inf"),
            shape_ok=True,
            finite_ok=False,
            detail="candidate contains non-finite values",
        )

    delta = (candidate.float() - reference.float()).abs()
    max_abs = float(delta.max()) if delta.numel() else 0.0
    scale = reference.float().abs().clamp_min(torch.finfo(torch.float32).tiny)
    max_rel = float((delta / scale).max()) if delta.numel() else 0.0

    passed = bool(torch.allclose(
        candidate.float(),
        reference.float(),
        atol=atol,
        rtol=rtol,
        equal_nan=False,
    ))
    return ComparisonResult(
        case=case,
        passed=passed,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        shape_ok=True,
        finite_ok=True,
    )


def render_dry_run(cases: tuple[RoutingCase, ...] = DEFAULT_CASES) -> str:
    """Describe the matrix without touching a device."""

    lines = ["external routing FFN comparison (dry run)", ""]
    for case in cases:
        lines.append(f"{case.name:<20} {case.note}")
        lines.append(
            f"{'':<20} tokens_per_rank={case.tokens_per_rank} "
            f"dp_size={case.dp_size} max_rows={case.max_rows} "
            f"gathered_rows={case.gathered_rows} top_k={case.top_k} "
            f"scale={'E8M0' if case.with_scale else 'none'}",
        )
    lines.append("")
    lines.append(
        "device run must: save the unpatched prepare, compute the reference with "
        "it, install the patch, then compare per-rank local routed output",
    )
    lines.append(
        "switching coverage: ALLGATHER -> MC2 -> ALLGATHER, with stash / scale / "
        "mask residue checked between steps",
    )
    return "\n".join(lines)


def resolve_model_path(explicit: str | None = None) -> str:
    """Resolve the checkpoint directory: explicit, then env, then the recorded one."""

    if explicit:
        return explicit
    return os.environ.get("DSV4_MODEL_PATH", DEFAULT_MODEL_PATH)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the case matrix and exit without touching a device",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "checkpoint directory for the expert-path level; defaults to "
            "DSV4_MODEL_PATH or the path the FFN serve script records"
        ),
    )
    args = parser.parse_args(argv)
    model_path = resolve_model_path(args.model)

    if args.dry_run:
        print(render_dry_run())
        return 0

    # A real run needs the device stack. Failing here is clearer than an ImportError
    # halfway through a comparison.
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:
        print(
            "external routing FFN comparison needs an NPU host "
            f"(import torch_npu failed: {type(exc).__name__}). "
            "Use --dry-run to inspect the matrix without a device.",
            file=sys.stderr,
        )
        return 2

    # The device levels run inside a normally-initialised FFN worker: they need its
    # real HCCL groups and its production-loaded W4A8MXFP experts, which cannot be
    # recreated here. Report that instead of exiting 0, so a wrapper never mistakes
    # "nothing ran" for "the comparison passed".
    print(
        "device comparison is not available from this entry point: it must run "
        "inside an initialised FFN worker, which owns the HCCL groups and the "
        f"loaded experts (checkpoint: {model_path}). "
        "Use --dry-run to inspect the matrix, or run the FFN worker verification.",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "CaseReport",
    "ComparisonResult",
    "DEFAULT_CASES",
    "DEFAULT_ATOL",
    "DEFAULT_RTOL",
    "RoutingCase",
    "compare_routed",
    "resolve_model_path",
    "render_dry_run",
]
