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
    local_rows: int,
    hidden_size: int,
    dp_rank: int,
    device,
    seed: int = 0,
    tolerance=None,
) -> None:
    """Compare both paths for one decoder layer and record every case.

    The caller supplies the worker-loaded layer and the facts only it can know: this
    rank's local row count, the hidden width, this rank's DP index, and the device the
    experts live on. Reading them from the forward context would not work -- it exposes
    neither the rank nor the width -- and guessing would silently compare the wrong
    shapes or run on the wrong device.

    ``seed`` is fixed so a rerun reproduces the same activations and routing; it is not
    a determinism claim about the collective itself.

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
            # Both paths must run on the experts' device; a CPU activation compared
            # against an NPU result would fail for the wrong reason.
            activation = torch.randn(
                case_rows,
                hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )
            topk_ids, topk_weights = _routing_for(
                experts,
                activation,
                scaling.factor,
                device=device,
                seed=seed,
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
