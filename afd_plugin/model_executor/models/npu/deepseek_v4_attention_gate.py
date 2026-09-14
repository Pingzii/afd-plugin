# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V4 Attention-side routing helpers for Async CAM."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from afd_plugin.model_executor.models.npu.deepseek_v4 import (
        AFDDeepseekV4AttentionGateRemoteMoE,
    )


def local_hash_input_ids(
    *,
    input_ids: torch.Tensor | None,
    router_tokens: int,
    flash_comm_v1_enabled: bool,
    pad_size: int,
    tp_group: object | None = None,
) -> torch.Tensor:
    """Return the rank-local token ids that a Hash layer routes on.

    A DSV4 Hash layer routes by token identity rather than by router logits, so
    the FFN rank executing that layer needs the ids of exactly the tokens it
    computes on. On Attention the forward context carries the *global* ids while
    FlashComm v1 shards router logits across TP ranks, so the global vector must
    receive the same padding and contiguous TP split as the logits before it can
    be sent.

    This mirrors the slicing ``_compute_sqrtsoftplus_topk`` already applies for
    local Hash routing, so the ids that cross the AFD boundary describe the same
    tokens the Attention-side routing used.

    Args:
        input_ids: Global ids from the forward context, or ``None``.
        router_tokens: Token count of this rank's router logits.
        flash_comm_v1_enabled: Whether FlashComm v1 is active for this forward.
        pad_size: FlashComm v1 padding applied to the activation.
        tp_group: Tensor-parallel group used for the split. Required only when
            FlashComm v1 padding actually has to be applied.

    Returns:
        A one-dimensional ``int64`` tensor of ``router_tokens`` local ids.

    Raises:
        RuntimeError: If ids are unavailable, or if the ids that would be sent
            do not describe exactly ``router_tokens`` tokens. Both cases would
            otherwise let a token-keyed router select experts for the wrong
            tokens, so they fail on the Attention side before any transfer.
    """

    if input_ids is None:
        # Names the switch as a literal because importing deepseek_v4 here would
        # pull the whole model into this tensor-only helper. The literal mirrors
        # AFD_DSV4_SKIP_INPUT_IDS_ENV there.
        raise RuntimeError(
            "DSV4 Hash routing requires input_ids to send towards the FFN role, "
            "but the forward context carries none. Both roles decide the ids "
            "channel from the same switch, so set AFD_DSV4_SKIP_INPUT_IDS=1 on "
            "both to transfer activations only and accept that Hash layers "
            "cannot route correctly.",
        )
    ids = input_ids.reshape(-1).to(torch.int64)
    if flash_comm_v1_enabled and ids.numel() != router_tokens:
        from vllm.distributed import get_tp_group
        from vllm_ascend.distributed.utils import split_tensor_along_first_dim

        if pad_size > 0:
            ids = torch.nn.functional.pad(ids, (0, pad_size))
        group = tp_group if tp_group is not None else get_tp_group()
        ids = split_tensor_along_first_dim(
            ids,
            num_partitions=group.world_size,
            contiguous_split_chunks=True,
        )[group.rank_in_group]
    if ids.numel() != router_tokens:
        raise RuntimeError(
            "DSV4 Hash routing cannot align the ids sent to FFN with the local "
            f"tokens: ids={ids.numel()} router_tokens={router_tokens}",
        )
    return ids


def hash_input_ids_from_context(
    *,
    forward_context: object,
    router_tokens: int,
) -> torch.Tensor:
    """Return the rank-local token ids that a Hash layer routes on.

    The ids channel belongs to the transfer, not to one layer: the FFN role
    decides once per run that it expects ids, and it cannot tell a Hash layer
    from a non-Hash one by looking at the hidden states. A forward context that
    carries no ids therefore cannot be answered by sending activations alone --
    the FFN rank would read an ids slot the operator never wrote. The call is
    routed through ``local_hash_input_ids`` so that case raises here, on the
    Attention side, instead of corrupting routing on the FFN side.
    """

    return local_hash_input_ids(
        input_ids=getattr(forward_context, "input_ids", None),
        router_tokens=router_tokens,
        flash_comm_v1_enabled=bool(
            getattr(forward_context, "flash_comm_v1_enabled", False),
        ),
        pad_size=int(getattr(forward_context, "pad_size", 0) or 0),
    )


def compute_attention_gate_topk(
    moe: AFDDeepseekV4AttentionGateRemoteMoE,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run DSV4 routing without entering vLLM's native MoE communicator.

    AFD Async CAM owns the cross-role dispatch.  The vLLM-Ascend fused
    selector's hash path instead assumes native EP/SP communication and calls
    ``forward_context.moe_comm_method.pad_and_split_input_ids``.  That object
    is intentionally absent on Attention ranks, including the KV-cache profile
    forward.  Use the same CANN routing operators directly on local Attention
    tokens, then hand their IDs and weights to CAM dispatch.
    """

    router_logits, _ = moe.gate(hidden_states)
    if moe.scoring_func == "sqrtsoftplus":
        topk_weights, topk_ids = _compute_sqrtsoftplus_topk(moe, router_logits)
    else:
        topk_weights, topk_ids = _compute_standard_topk(moe, router_logits)
    return topk_weights.to(torch.float32), topk_ids


def _compute_sqrtsoftplus_topk(
    moe: AFDDeepseekV4AttentionGateRemoteMoE,
    router_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run DSV4's sqrtsoftplus CANN router without native MoE communication."""

    if moe.scoring_func != "sqrtsoftplus":
        raise RuntimeError(
            "DSV4 Hash routing requires scoring_func='sqrtsoftplus', got "
            f"{moe.scoring_func!r}",
        )

    tid2eid = moe.gate.tid2eid
    input_ids = None
    if tid2eid is not None:
        from vllm.forward_context import get_forward_context

        forward_context = get_forward_context()
        input_ids = local_hash_input_ids(
            input_ids=getattr(forward_context, "input_ids", None),
            router_tokens=router_logits.shape[0],
            flash_comm_v1_enabled=forward_context.flash_comm_v1_enabled,
            pad_size=forward_context.pad_size,
        )
        input_ids = torch.where(input_ids == -1, 0, input_ids)
        tid2eid = tid2eid.to(torch.int32)
    correction_bias = moe.gate.e_score_correction_bias
    if correction_bias is not None and correction_bias.dtype != router_logits.dtype:
        correction_bias = correction_bias.to(router_logits.dtype)
    topk_weights, topk_ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
        x=router_logits,
        k=moe.top_k,
        bias=correction_bias,
        input_ids=input_ids,
        tid2eid=tid2eid,
        k_group=moe.topk_group,
        group_count=moe.num_expert_group,
        routed_scaling_factor=moe.routed_scaling_factor,
        eps=1e-20,
        group_select_mode=1,
        renorm=0,
        norm_type=2,
        out_flag=False,
    )
    return topk_weights, topk_ids


def _compute_standard_topk(
    moe: AFDDeepseekV4AttentionGateRemoteMoE,
    router_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the non-Hash CANN selector without native MoE communication."""

    from vllm_ascend.device.device_op import DeviceOperator

    norm_type_by_scoring_func = {"softmax": 0, "sigmoid": 1}
    try:
        norm_type = norm_type_by_scoring_func[moe.scoring_func]
    except KeyError as exc:
        raise RuntimeError(
            f"Unsupported non-Hash DSV4 routing scoring function: {moe.scoring_func!r}",
        ) from exc
    correction_bias = moe.gate.e_score_correction_bias
    if correction_bias is not None and correction_bias.dtype != router_logits.dtype:
        correction_bias = correction_bias.to(router_logits.dtype)
    topk_weights, topk_ids, _ = DeviceOperator.moe_gating_top_k(
        router_logits,
        k=moe.top_k,
        k_group=moe.topk_group,
        group_count=moe.num_expert_group,
        group_select_mode=1,
        renorm=int(moe.renormalize),
        norm_type=norm_type,
        out_flag=False,
        routed_scaling_factor=moe.routed_scaling_factor,
        eps=1e-20,
        bias_opt=correction_bias,
    )
    return topk_weights, topk_ids
