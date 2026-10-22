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

from contextlib import contextmanager
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
    """Return why the ranks disagreed on the plan alone, or ``None`` when they match.

    Kept separate from :func:`decide_together` because the precheck and the plan are
    answered by different votes; a refusal is not a planning disagreement and should not
    be reported as one. ``decide_together`` is what a run uses.

    ``exchange`` is the object all-gather over the participating group.
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


@dataclass(frozen=True)
class _Vote:
    """One rank's participation answer, exchanged before anything collective runs."""

    rank: int
    reason: str | None
    plan: tuple[Any, ...] | None


def decide_together(*, dp_rank: int, precheck_reason: str | None, plan, exchange) -> str | None:
    """Return the run's shared refusal reason, or ``None`` when every rank agrees to run.

    Local refusals do not exit on their own. A rank whose precheck failed, or whose plan
    could not be built, still takes part in this exchange, because the others are going
    to wait for it in the next collective. The decision is taken from every vote, so all
    ranks stop together.

    ``exchange`` is the object all-gather over the participating group.
    """

    if precheck_reason is None and plan is None:
        # No plan and no stated reason is an inconsistent vote, not an approval. Every
        # rank that cannot build a plan has to say why, and one that says nothing must
        # not be counted as agreeing.
        precheck_reason = "no plan was built for this rank"

    vote = _Vote(
        rank=dp_rank,
        reason=precheck_reason,
        plan=None if plan is None else plan.signature(),
    )
    votes = exchange(vote)
    refusals = [other for other in votes if other.reason is not None]
    if refusals:
        lowest = min(refusals, key=lambda item: item.rank)
        return f"rank {lowest.rank} refused: {lowest.reason}"
    signatures = {other.plan for other in votes}
    if len(signatures) != 1:
        return (
            f"the ranks planned differently: {sorted(map(repr, signatures))}; "
            "they must agree before any collective"
        )
    return None


@contextmanager
def case_forward_context(max_rows: int):
    """Present ``max_rows`` to the native prepare for one case.

    The native prepare reads the padding target from the forward context, not from an
    argument, so a per-case value has to be installed there. Without this every case
    would pad to whatever the entry context happened to hold, and a case larger than
    that would ask for negative padding on one rank while the others had already
    entered the collective.

    This only overwrites the padding target of a context that already exists. It does
    not make a context, so it is a fallback for callers that cannot build one; a caller
    that can should pass a ``context_factory`` to :func:`verify_layer` instead, because
    a field being present is not the same as the rest of the context being valid.
    """

    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    previous = _EXTRA_CTX.max_tokens_across_dp
    _EXTRA_CTX.max_tokens_across_dp = max_rows
    try:
        yield max_rows
    finally:
        _EXTRA_CTX.max_tokens_across_dp = previous


@dataclass
class OutcomeSummary:
    """The run's verdict, and the reasons that produced it."""

    failed: bool
    reasons: list[str] = field(default_factory=list)


def summarize(
    plan: VerificationPlan,
    entries: list[tuple[int, str, bool]],
    skipped: list[str],
    participating_ranks: list[int],
) -> OutcomeSummary:
    """Decide the run's verdict from what every rank reported.

    ``entries`` are ``(rank, case, passed)`` gathered across the participating ranks. A
    local-only view cannot see a remote rank's failure, and a count cannot tell a
    duplicated case from a missing one, so the check is on the named set: every planned
    ``(case, scaling)`` must be reported by every participating rank, exactly once.

    A skipped mandatory case fails the run. Reporting "nothing to report" as a pass is
    how a verification entry point ends up green without having checked anything.
    """

    reasons: list[str] = [f"skipped: {reason}" for reason in skipped]
    expected = {
        f"{case}/{scaling}" for case in plan.case_order for scaling, _ in plan.scaling
    }
    by_case: dict[str, list[tuple[int, bool]]] = {}
    for rank, case, passed in entries:
        by_case.setdefault(case, []).append((rank, passed))

    for case in sorted(expected):
        reported = by_case.get(case, [])
        ranks = [rank for rank, _ in reported]
        if sorted(ranks) != sorted(participating_ranks):
            reasons.append(
                f"case {case!r} reported by ranks {sorted(ranks)}, "
                f"expected {sorted(participating_ranks)}",
            )
            continue
        if len(ranks) != len(set(ranks)):
            reasons.append(f"case {case!r} reported more than once by one rank")
        if not all(passed for _, passed in reported):
            failing = sorted(rank for rank, passed in reported if not passed)
            reasons.append(f"case {case!r} did not match on ranks {failing}")

    unplanned = sorted(set(by_case) - expected)
    if unplanned:
        reasons.append(f"cases outside the plan were reported: {unplanned}")

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
    summary: OutcomeSummary | None = None
    # Set when the refusal came from the topology check rather than from a run that
    # could not complete. The two map to different exit codes, and folding them together
    # would report "unsupported setup" as "never ran".
    topology_refusal: bool = False

    @property
    def passed(self) -> bool:
        """Whether the run passed. False until a verdict from all ranks exists.

        Reading this before ``summarize`` runs would report a pass from a local view
        that has not seen the other ranks, and a local case matching says nothing about
        a rank that failed or never ran.
        """

        if self.summary is None:
            return False
        return not self.summary.failed

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
    context_factory=None,
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

    # Neither a failed precheck nor an unbuildable plan may exit on its own: the other
    # ranks are about to wait for this one in the exchange. Both results travel into
    # the same vote, and every rank reads the same decision out of it.
    reason = precheck(
        comm_type_name=type(comm).__name__,
        dp_size=dp_size,
        tp_size=tp_size,
        pcp_size=pcp_size,
        sequence_parallel=bool(moe_config.is_sequence_parallel),
        captured=external_routing_prepare.ORIGINAL_PREPARE_WITH_DP_GROUP is not None,
    )
    plan = None
    try:
        plan = build_plan(dp_size, seed=seed, atol=atol, rtol=rtol)
    except ValueError as error:
        if reason is None:
            reason = str(error)

    # Recorded separately: a topology refusal is a different outcome from a run that
    # could not complete, and the two must not share an exit code.
    topology_refusal = reason is not None

    import torch.distributed as dist

    def exchange(vote):
        votes = [None] * dp_size
        # The plan is a plain object, so it travels over the CPU group. vLLM's
        # GroupCoordinator exposes all_gather for tensors only; the CPU group is the
        # supported route for object exchange.
        dist.all_gather_object(votes, vote, group=moe_config.dp_group.cpu_group)
        return votes

    refusal = decide_together(
        dp_rank=dp_rank,
        precheck_reason=reason,
        plan=plan,
        exchange=exchange,
    )
    if refusal is not None:
        record.skipped.append(refusal)
        record.topology_refusal = topology_refusal
        return
    assert plan is not None, "decide_together accepted a run without a plan"

    gather = lambda tensor: moe_config.dp_group.all_gather(tensor, 0)  # noqa: E731
    for case_name in plan.case_order:
        # The plan decides the row count, not this rank. No clamping: a case that does
        # not fit is refused above rather than silently renamed to a smaller shape.
        case_rows = plan.rows_for(case_name, dp_rank)
        # The padding target this case needs, derived from the plan so every rank pads
        # to the same total. The native prepare reads it from the forward context, so it
        # has to be installed there, not just computed here.
        max_rows = max(plan.rows_by_case[plan.case_order.index(case_name)])
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
            with (
                context_factory(case_rows, max_rows)
                if context_factory is not None
                else case_forward_context(max_rows)
            ):
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

    # The verdict needs every rank's result, not just this one's. A local pass says
    # nothing about a rank that failed or never ran, so the named results are exchanged
    # before anything is decided.
    entries = [
        (result.case, result.passed) for result in record.report.results
    ]
    gathered_entries = [None] * dp_size
    dist.all_gather_object(
        gathered_entries,
        (dp_rank, entries),
        group=moe_config.dp_group.cpu_group,
    )
    flattened: list[tuple[int, str, bool]] = []
    for vote in gathered_entries:
        rank, per_rank = vote
        for case, passed in per_rank:
            flattened.append((rank, case, passed))
    record.summary = summarize(
        plan,
        flattened,
        record.skipped,
        list(range(dp_size)),
    )


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
    "case_forward_context",
    "decide_together",
    "precheck",
    "summarize",
    "verify_layer",
]
