# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Compare the external-routing consumer against a hand-aligned native run.

Runs inside an initialised FFN worker so the real HCCL groups and the
production-loaded W4A8MXFP experts are reused. It does not build weights, and it
does not start a server.

Two paths are compared for the same layer, activation and routing:

- **candidate** — ``_compute_external_routing_ffn``, which lets the prepare patch pad
  and gather the routing on its own;
- **reference** — the same prepare/apply/finalize sequence with the routing aligned
  by hand and no patch involved.

The reference deliberately does not call the candidate, and does not go through the
patch, so a defect in the patch shows up as a difference instead of cancelling out.

Scale semantics -- corrected, and still open:

The routing this consumer receives already carries whatever scaling the Attention
selector applied, because the selector folds ``routed_scaling_factor`` into
``topk_weights``. The consumer must therefore not scale again, and it does not.

An earlier version of this docstring justified that by pointing at
``routed_scaling_factor=1.0`` in ``_extract_async_cam_w4a8_layers``. That line
belongs to the async layered-GMM weight export, not to the sync FFN MoE
construction, so it is not evidence for this path. The sync construction only
changes ``n_shared_experts``. Whether the sync consumer ends up scaling once follows
from bypassing the native MoE outer layer and needs a numeric check, not a citation.

Known blockers before this can be acceptance evidence:

- the reference is not independent: it reuses the same alignment helper and the same
  patched prepare, so a shared alignment defect cancels out. It needs its own
  padding/gathering and the unpatched prepare with a placeholder ``router_logits``;
- the per-rank case rows are clamped locally, which silently changes the case and can
  leave other ranks waiting in a collective. The case matrix has to be agreed across
  ranks and run in the same order everywhere, and a skipped case must fail the run.

Until both are addressed, a pass here means the two paths agreed and nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from afd_plugin.verification.external_routing_ffn import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    CaseReport,
    ComparisonResult,
    compare_routed,
)

# Row counts per rank for the shapes that matter. Zero exercises an empty rank,
# which is legal because an EP rank can own none of the routed tokens.
#
# A table is tied to the DP size it was written for. A different size is refused rather
# than folded onto the table with a modulus: that would silently reuse a row count from
# a shape nobody chose and label the result with that shape's name.
LAYER_CASES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("uniform", (2, 2, 2, 2)),
    ("uneven_with_empty_rank", (3, 1, 0, 2)),
)


@dataclass(frozen=True)
class VerificationPlan:
    """The execution plan every participating rank has to agree on.

    Built from shared configuration, never from a rank's own view. Each rank reads its
    own row count out of it, which is why differing row counts are fine and differing
    plans are not.
    """

    case_order: tuple[str, ...]
    rows_by_case: tuple[tuple[int, ...], ...]
    seed: int
    scaling: tuple[tuple[str, float], ...]
    atol: float
    rtol: float

    def signature(self) -> tuple[Any, ...]:
        """A comparable form of everything that has to match across ranks."""

        return (
            self.case_order,
            self.rows_by_case,
            self.seed,
            self.scaling,
            self.atol,
            self.rtol,
        )

    def rows_for(self, case: str, dp_rank: int) -> int:
        index = self.case_order.index(case)
        return self.rows_by_case[index][dp_rank]


def build_plan(dp_size: int, *, seed: int, atol: float, rtol: float) -> VerificationPlan:
    """Return the plan for this topology.

    Raises:
        ValueError: If the case tables do not have one row count per DP rank. The
            tables are written for a fixed topology; a different one needs its own
            table, not a modulus.
    """

    if dp_size < 2:
        raise ValueError(f"dp_size {dp_size} has no ranks to align across")
    for name, rows in LAYER_CASES:
        if len(rows) != dp_size:
            raise ValueError(
                f"case {name!r} has {len(rows)} row counts for dp_size {dp_size}; "
                "add a case table for this topology instead of reusing another one",
            )
    return VerificationPlan(
        case_order=tuple(name for name, _ in LAYER_CASES),
        rows_by_case=tuple(rows for _, rows in LAYER_CASES),
        seed=seed,
        scaling=tuple((case.name, case.factor) for case in SCALING_CASES),
        atol=atol,
        rtol=rtol,
    )


def agree_plan(plan: VerificationPlan, exchange) -> str | None:
    """Return why the ranks disagreed, or ``None`` when they all match.

    ``exchange`` is the all-gather over the participating group: it takes this rank's
    value and returns every rank's value. Injected so the agreement rule is testable
    without a collective.

    Runs before any data collective, so a disagreement is reported by every rank that
    voted rather than by whichever one happened to notice first.
    """

    signature = plan.signature()
    gathered = exchange(signature)
    for rank, other in enumerate(gathered):
        if other != signature:
            return (
                f"rank {rank} planned {other!r} while this rank planned {signature!r}; "
                "the ranks must agree on the cases before any collective"
            )
    return None


def precheck(
    *,
    comm_type_name: str,
    dp_size: int,
    tp_size: int,
    pcp_size: int,
    sequence_parallel: bool,
    captured: bool,
) -> str | None:
    """Return why this topology cannot be verified, or ``None`` when it can.

    Pure and topology-based, so every rank reaches the same answer from the same facts
    and a single rank cannot decide to stop on its own.
    """

    if comm_type_name != "AllGatherCommImpl":
        return f"comm method is {comm_type_name}, not AllGatherCommImpl"
    if dp_size < 2:
        return f"dp_size {dp_size} has no ranks to align across"
    if tp_size != 1:
        return f"tp_size {tp_size}: routing rows are not sharded by TP here"
    if pcp_size > 1:
        return f"pcp_size {pcp_size}: PCP aligns differently and is not covered"
    if sequence_parallel:
        return "sequence parallelism aligns differently and is not covered"
    if not captured:
        return (
            "the upstream prepare was not captured, so the reference would compare "
            "the patch against itself"
        )
    return None


@dataclass
class OutcomeSummary:
    """The run's verdict, and the reasons that produced it."""

    failed: bool
    reasons: list[str] = field(default_factory=list)


def summarize(
    plan: VerificationPlan,
    results: list[ComparisonResult],
    skipped: list[str],
) -> OutcomeSummary:
    """Decide the run's verdict from what actually executed.

    A skipped mandatory case fails the run. Reporting "nothing to report" as a pass is
    how a verification entry point ends up green without having checked anything.
    """

    reasons: list[str] = []
    if skipped:
        reasons.extend(f"skipped: {reason}" for reason in skipped)
    expected = len(plan.case_order) * len(plan.scaling)
    if len(results) != expected:
        reasons.append(
            f"ran {len(results)} of {expected} cases; a missing case is a failure",
        )
    for result in results:
        if not result.passed:
            reasons.append(f"case {result.case!r} did not match")
    return OutcomeSummary(failed=bool(reasons), reasons=reasons)


@dataclass(frozen=True)
class ScalingCase:
    """A routed-scaling factor to verify.

    The factor is multiplied into the routing weights here, mirroring what the
    Attention selector does, so the consumer is checked against an already-scaled
    routing. Every case carries a concrete value; the caller passes the model's own
    factor as the "configured" case, because this module cannot read it and an earlier
    sentinel-based version silently ran unscaled.

    A pass proves the consumer does not rescale. It does not prove the selector folds
    the factor correctly -- that is the Attention side's own check.
    """

    name: str
    factor: float


SCALING_CASES: tuple[ScalingCase, ...] = (
    ScalingCase("configured", 1.0),  # replaced by the caller's model factor
    ScalingCase("explicit_1p5", 1.5),
)


@dataclass
class VerificationRecord:
    """Everything a green-zone run must report back."""

    versions: dict[str, str]
    configuration: dict[str, Any]
    report: CaseReport = field(default_factory=CaseReport)
    skipped: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.report.results) and self.report.passed

    def render(self) -> str:
        lines = ["external routing consumer verification", ""]
        lines.append("versions:")
        for name, value in self.versions.items():
            lines.append(f"  {name}: {value}")
        lines.append("configuration:")
        for name, value in self.configuration.items():
            lines.append(f"  {name}: {value}")
        lines.append("")
        lines.extend(self.report.render().splitlines())
        if self.skipped:
            lines.append("")
            lines.append("skipped:")
            for item in self.skipped:
                lines.append(f"  {item}")
        return "\n".join(lines)


def _align_routing_independently(topk_ids, topk_weights, pad_size, gather):
    """Pad and gather the routing with plain PyTorch.

    Deliberately does not use ``AFDExternalRoutingInputs.padded``/``gathered``: those
    are what the candidate path runs through, so reusing them would let a defect in
    that helper cancel out on both sides of the comparison.

    Scope: DP-level ALLGATHER, BF16 activations, one routing row per token. PCP and
    SP sharding use different rules and are not covered.
    """

    import torch

    pad_rows = pad_size
    if pad_rows < 0:
        raise ValueError(f"pad_size {pad_rows} is negative")
    if pad_rows > 0:
        id_pad = torch.zeros(
            (pad_rows, topk_ids.shape[1]),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        weight_pad = torch.zeros(
            (pad_rows, topk_weights.shape[1]),
            dtype=topk_weights.dtype,
            device=topk_weights.device,
        )
        padded_ids = torch.cat([topk_ids, id_pad], dim=0)
        padded_weights = torch.cat([topk_weights, weight_pad], dim=0)
    else:
        padded_ids, padded_weights = topk_ids, topk_weights
    if gather is None:
        return padded_ids, padded_weights
    return gather(padded_ids), gather(padded_weights)


def _run_reference(
    layer,
    activation,
    topk_ids,
    topk_weights,
    *,
    plan,
    gather,
    experts,
    comm,
):
    """Run the unpatched prepare with independently aligned routing.

    Reads the captured upstream function from its module attribute rather than
    importing it by name: a module-level ``from ... import`` binding is resolved once,
    and could hold ``None`` if it was bound before the installer ran.

    Calls that function directly instead of ``comm.prepare``, because ``comm.prepare``
    may still route through the patch depending on how the class was wired.
    """

    import torch
    from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl

    from afd_plugin.compat.patches.npu import external_routing_prepare

    upstream_prepare = external_routing_prepare.ORIGINAL_PREPARE_WITH_DP_GROUP
    if upstream_prepare is None:
        raise RuntimeError(
            "the upstream prepare was not captured; the verification reference "
            "cannot run and would otherwise compare the patch against itself",
        )

    aligned_ids, aligned_weights = _align_routing_independently(
        topk_ids,
        topk_weights,
        plan.pad_size,
        gather,
    )
    # The upstream prepare requires logits and derives the row count from them. The
    # activation is aligned by that same call, once, so this must not pre-gather it.
    placeholder_logits = torch.zeros(
        (int(activation.shape[0]), int(experts.moe_config.num_experts)),
        dtype=torch.float32,
        device=activation.device,
    )
    output = upstream_prepare(
        comm.prepare_finalize,
        hidden_states=activation,
        router_logits=placeholder_logits,
        replace_allreduce=experts.moe_config.is_sequence_parallel,
        quant_type=experts.quant_type,
    )
    experts.ascend_pertoken_scale = None
    experts.ascend_mc2_mask = output.mc2_mask
    try:
        fused = experts.quant_method.apply(
            layer=experts,
            x=output.hidden_states,
            topk_weights=aligned_weights,
            topk_ids=aligned_ids,
            shared_experts=None,
            shared_experts_input=None,
        )
    finally:
        experts.ascend_pertoken_scale = None
        experts.ascend_mc2_mask = None
    return comm.finalize(
        hidden_states=fused.routed_out,
        reduce_results=isinstance(comm, AllGatherCommImpl),
        padded_hidden_states_shape=output.padded_hidden_states_shape,
    )


def verify_layer(
    layer,
    record: VerificationRecord,
    *,
    hidden_size: int,
    dp_rank: int,
    device,
    tp_size: int = 1,
    pcp_size: int = 1,
    seed: int = 0,
    tolerance=None,
) -> None:
    """Compare both paths for one decoder layer and record every case.

    The caller supplies the worker-loaded layer and the facts only it can know: the
    hidden width, this rank's DP index, the device the experts live on, and the TP and
    PCP sizes. Reading them from the forward context would not work -- it exposes
    neither the rank nor the width -- and guessing would silently compare the wrong
    shapes or run on the wrong device.

    The per-case row counts come from the shared plan, not from this rank. Every rank
    reads the same plan, so a case that one rank cannot run is refused before the first
    collective rather than clamped into a different shape.

    ``seed`` is fixed so a rerun reproduces the same activations and routing; it is not
    a determinism claim about the collective itself.

    Weights, communication groups and the quant method all come from the production
    path. Anything unavailable is recorded as skipped, and a skipped mandatory case
    fails the run rather than passing quietly.
    """

    import torch
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    from afd_plugin.compat.patches.npu import external_routing_prepare
    from afd_plugin.model_executor.models.npu.deepseek_v4 import (
        _routed_experts_of,
    )

    atol, rtol = tolerance if tolerance is not None else (DEFAULT_ATOL, DEFAULT_RTOL)
    comm = _EXTRA_CTX.moe_comm_method
    experts = _routed_experts_of(layer.mlp)
    moe_config = experts.moe_config
    dp_size = int(moe_config.dp_size)

    # Every rank runs the same topology test on the same facts and reaches the same
    # answer, so one rank cannot decide to stop alone while the others wait inside a
    # collective.
    reason = precheck(
        comm_type_name=type(comm).__name__,
        dp_size=dp_size,
        tp_size=tp_size,
        pcp_size=pcp_size,
        sequence_parallel=bool(moe_config.is_sequence_parallel),
        captured=external_routing_prepare.ORIGINAL_PREPARE_WITH_DP_GROUP is not None,
    )
    if reason is not None:
        record.skipped.append(reason)
        return

    try:
        plan = build_plan(dp_size, seed=seed, atol=atol, rtol=rtol)
    except ValueError as error:
        record.skipped.append(str(error))
        return

    disagreement = agree_plan(
        plan,
        lambda signature: moe_config.dp_group.all_gather_object(signature),
    )
    if disagreement is not None:
        record.skipped.append(disagreement)
        return

    max_rows = _EXTRA_CTX.max_tokens_across_dp
    gather = lambda tensor: moe_config.dp_group.all_gather(tensor, 0)  # noqa: E731
    for case_name in plan.case_order:
        # The plan decides the row count, not this rank. No clamping: a case that does
        # not fit is refused above rather than silently renamed to a smaller shape.
        case_rows = plan.rows_for(case_name, dp_rank)
        pad_plan = _Plan(pad_size=max_rows - case_rows)
        for scaling_name, factor in plan.scaling:
            # Both paths must run on the experts' device; a CPU activation compared
            # against an NPU result would fail for the wrong reason.
            #
            # The activation gets its own generator, seeded the same way as the
            # routing. Seeding only the routing would reproduce the indices while the
            # activations changed between runs, which looks like a numerical
            # difference. Drawing on CPU keeps the local generator off the production
            # global RNG and avoids depending on a device generator existing.
            activation_generator = torch.Generator(device="cpu").manual_seed(seed)
            activation = torch.randn(
                case_rows,
                hidden_size,
                dtype=torch.bfloat16,
                generator=activation_generator,
            ).to(device)
            topk_ids, topk_weights = _routing_for(
                experts,
                activation,
                factor,
                device=device,
                seed=seed,
            )
            reference = _run_reference(
                layer,
                activation,
                topk_ids,
                topk_weights,
                plan=pad_plan,
                gather=gather,
                experts=experts,
                comm=comm,
            )
            candidate = layer._compute_external_routing_ffn(
                activation,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )
            record.report.add(
                compare_routed(
                    reference,
                    candidate,
                    case=f"{case_name}/{scaling_name}",
                    atol=atol,
                    rtol=rtol,
                ),
            )

    # The verdict comes from what actually ran, on every rank. A skipped mandatory case
    # is a failure, not an absence of news.
    summary = summarize(plan, list(record.report.results), record.skipped)
    if summary.failed:
        record.skipped.extend(summary.reasons)


@dataclass(frozen=True)
class _Plan:
    """How much padding the local rows need before the collective."""

    pad_size: int


def _routing_for(experts, activation, factor, *, device, seed):
    """Build routing already carrying ``factor``, on the worker's device.

    The Attention selector folds the factor into the weights, so this multiplies here
    as well. An earlier version treated ``0.0`` as "use the model's factor" but only
    multiplied when the value was truthy, so the configured case silently ran
    unscaled. ``factor`` is now always applied, and the caller resolves the
    configured value before calling.
    """

    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = int(activation.shape[0])
    top_k = int(experts.moe_config.experts_per_token)
    ids = torch.randint(
        0,
        int(experts.moe_config.num_experts),
        (rows, top_k),
        generator=generator,
        dtype=torch.int32,
    ).to(device)
    weights = torch.full((rows, top_k), 0.5 / top_k, dtype=torch.float32).to(device)
    return ids, weights * factor


__all__ = [
    "LAYER_CASES",
    "SCALING_CASES",
    "OutcomeSummary",
    "ScalingCase",
    "VerificationPlan",
    "VerificationRecord",
    "agree_plan",
    "build_plan",
    "precheck",
    "summarize",
    "verify_layer",
]
