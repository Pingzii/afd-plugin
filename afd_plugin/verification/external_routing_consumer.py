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
LAYER_CASES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("uniform", (2, 2, 2, 2)),
    ("uneven_with_empty_rank", (3, 1, 0, 2)),
)


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


def _align_by_hand(activation, topk_ids, topk_weights, plan, gather):
    """Pad and gather the routing without the patch, using the documented rule."""

    from afd_plugin.connectors.npu.external_routing import (
        AFDExternalRoutingInputs,
    )

    routing = AFDExternalRoutingInputs(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    ).padded(plan.pad_size)
    if gather is not None:
        routing = routing.gathered(gather)
    return activation, routing


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
    """Run prepare/apply/finalize with hand-aligned routing and no patch."""

    from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl

    aligned_activation, routing = _align_by_hand(
        activation,
        topk_ids,
        topk_weights,
        plan,
        gather,
    )
    output = comm.prepare(
        hidden_states=aligned_activation,
        router_logits=None,
        replace_allreduce=experts.moe_config.is_sequence_parallel,
        quant_type=experts.quant_type,
    )
    experts.ascend_pertoken_scale = None
    experts.ascend_mc2_mask = output.mc2_mask
    try:
        fused = experts.quant_method.apply(
            layer=experts,
            x=output.hidden_states,
            topk_weights=routing.topk_weights,
            topk_ids=routing.topk_ids,
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
    local_rows: int,
    hidden_size: int,
    dp_rank: int,
    tolerance=None,
) -> None:
    """Compare both paths for one decoder layer and record every case.

    The caller supplies the worker-loaded layer and the three facts only it can know:
    this rank's local row count, the hidden width, and this rank's DP index. Reading
    them from the forward context would not work -- it exposes neither the rank nor
    the width -- and guessing would silently compare the wrong shapes.

    Weights, communication groups and the quant method all come from the production
    path. Anything unavailable is recorded as skipped rather than treated as a pass.
    """

    import torch
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl

    from afd_plugin.model_executor.models.npu.deepseek_v4 import (
        _routed_experts_of,
    )

    atol, rtol = tolerance if tolerance is not None else (DEFAULT_ATOL, DEFAULT_RTOL)
    comm = _EXTRA_CTX.moe_comm_method
    if not isinstance(comm, AllGatherCommImpl):
        record.skipped.append(
            f"comm method is {type(comm).__name__}, not AllGatherCommImpl",
        )
        return
    experts = _routed_experts_of(layer.mlp)
    moe_config = experts.moe_config
    if moe_config.dp_size <= 1:
        record.skipped.append("dp_size <= 1: nothing to align across ranks")
        return
    if not 0 <= dp_rank < moe_config.dp_size:
        record.skipped.append(f"dp_rank {dp_rank} outside dp_size {moe_config.dp_size}")
        return

    max_rows = _EXTRA_CTX.max_tokens_across_dp
    if local_rows > max_rows:
        record.skipped.append(
            f"local_rows {local_rows} exceeds max_tokens_across_dp {max_rows}",
        )
        return
    gather = lambda tensor: moe_config.dp_group.all_gather(tensor, 0)  # noqa: E731
    for case_name, rows_by_rank in LAYER_CASES:
        # Each rank uses its own row count from the case, so one run covers the
        # uniform and uneven shapes without coordinating across ranks.
        case_rows = rows_by_rank[dp_rank % len(rows_by_rank)]
        case_rows = min(case_rows, local_rows, max_rows)
        plan = _Plan(pad_size=max_rows - case_rows)
        for scaling in SCALING_CASES:
            activation = torch.randn(
                case_rows,
                hidden_size,
                dtype=torch.bfloat16,
            )
            topk_ids, topk_weights = _routing_for(
                experts,
                activation,
                scaling.factor,
            )
            reference = _run_reference(
                layer,
                activation,
                topk_ids,
                topk_weights,
                plan=plan,
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
                    case=f"{case_name}/{scaling.name}",
                    atol=atol,
                    rtol=rtol,
                ),
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
    "ScalingCase",
    "VerificationRecord",
    "verify_layer",
]
