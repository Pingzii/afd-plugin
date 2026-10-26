# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Let MC2 dispatch take an activation that is already quantized.

Upstream source: vllm_ascend/ops/fused_moe/token_dispatcher.py (8d4409d),
``TokenDispatcherWithMC2.get_dispatch_mc2_kwargs``.

Patch reason: the Attention-side gate hands the FFN an activation quantized on the
other role, together with a two-dimensional E8M0 scale, and the prepare patch
publishes that scale as the routing's ``pertoken_scale``. MC2's dispatch kwargs do
not read it: ``scales`` is ``None`` and ``quant_mode`` resolves to the MXFP
communication mode, so the operator would quantize a tensor that is already
quantized and the scale from the other role would be dropped.

Patch functionality: when the routing carries a scale, pass it through and set
``quant_mode`` to 0, which tells the operator the activation is already quantized.
Every other configuration keeps upstream's behaviour exactly: no scale means
``scales`` stays ``None`` and the quant mode is resolved as before.

The device smoke in ``tests/npu/mc2_prequant_smoke.py`` established that the
installed operator accepts FP8 activations with a two-dimensional E8M0 scale at
``quant_mode=0`` and returns a scale of the same shape next to the activation.
Where that output goes downstream is upstream's existing path: ``token_dispatch``
unpacks it as ``dynamic_scale`` and the MXFP quant method normalizes its layout
when it consumes it, so nothing here converts it.

Signature: matches upstream; no added parameters.
"""

from __future__ import annotations

from typing import Any

import torch

# Resolved on first use rather than at import time. The patch is installed inside a
# worker process, but keeping the import out of module scope lets a CPU-only host
# import this module and exercise the patch's branching against stand-ins.
TokenDispatcherWithMC2: Any = None
_get_expert_token_nums_type: Any = None

_ORIGINAL_GET_DISPATCH_MC2_KWARGS: Any = None


def _ensure_upstream_imports() -> None:
    """Resolve the vLLM-Ascend names once, leaving supplied stand-ins alone."""

    global TokenDispatcherWithMC2, _get_expert_token_nums_type
    global _ORIGINAL_GET_DISPATCH_MC2_KWARGS

    if TokenDispatcherWithMC2 is not None:
        return

    from vllm_ascend.ops.fused_moe.token_dispatcher import (
        TokenDispatcherWithMC2 as upstream_dispatcher,
    )
    from vllm_ascend.ops.fused_moe.token_dispatcher import (
        _get_expert_token_nums_type as upstream_nums_type,
    )

    TokenDispatcherWithMC2 = upstream_dispatcher
    _get_expert_token_nums_type = upstream_nums_type
    _ORIGINAL_GET_DISPATCH_MC2_KWARGS = upstream_dispatcher.get_dispatch_mc2_kwargs


def _for_testing_reset() -> None:
    """Drop resolved upstream objects. Only tests use this."""

    global TokenDispatcherWithMC2, _get_expert_token_nums_type
    global _ORIGINAL_GET_DISPATCH_MC2_KWARGS

    TokenDispatcherWithMC2 = None
    _get_expert_token_nums_type = None
    _ORIGINAL_GET_DISPATCH_MC2_KWARGS = None


# Upstream source: vllm_ascend/ops/fused_moe/token_dispatcher.py (8d4409d),
# TokenDispatcherWithMC2.get_dispatch_mc2_kwargs, copied in full.
# Patch reason: ``scales`` was hardcoded to None and the quant mode ignored the
# routing's pre-quantized activation, so a scale produced on the Attention side
# could not reach the operator.
# Patch functionality: forward the routing's pertoken scale and select
# quant_mode 0 when it is present; leave every other path as upstream has it.
# Signature: matches upstream; no added parameters.
def get_dispatch_mc2_kwargs(
    self,
    token_dispatch_input: Any,
):
    _ensure_upstream_imports()

    hidden_states = token_dispatch_input.hidden_states
    topk_weights = token_dispatch_input.topk_weights
    topk_ids = token_dispatch_input.topk_ids
    expert_map = token_dispatch_input.routing.expert_map
    global_redundant_expert_num = token_dispatch_input.routing.global_redundant_expert_num
    comm_quant_mode = token_dispatch_input.quant.comm_quant_mode

    # ### PATCH START: the activation may already be quantized
    # Set only when the Attention side quantized the activation and sent its scale;
    # absent for the FFN-side gate, which sends a model-dtype activation.
    routing_scale = token_dispatch_input.routing.pertoken_scale
    # ### PATCH END: the activation may already be quantized

    assert expert_map is not None, "expert_map is required for MC2 token dispatch."
    # NOTE: quant_mode differs by quant feature:
    # - Legacy int communication quantization uses quant_mode=2.
    # - A5 MXFP communication uses quant_mode=4.
    if comm_quant_mode is not None:
        quant_mode = comm_quant_mode
    elif token_dispatch_input.quant.dispatch_with_quant:
        quant_mode = 4 if self.need_shared_expert_args and token_dispatch_input.quant.is_mxfp else 2
    else:
        quant_mode = 0
    # ### PATCH START: a scale from the Attention side means no quantization here
    # quant_mode 0 states the activation is already quantized, so the operator reads
    # the scale instead of producing one.
    if routing_scale is not None:
        quant_mode = 0
    # ### PATCH END: a scale from the Attention side means no quantization here
    self.moe_expert_num = len(expert_map) + global_redundant_expert_num
    expert_token_nums_type = _get_expert_token_nums_type(token_dispatch_input)
    kwargs_mc2 = {
        "x": hidden_states,
        "expert_ids": topk_ids,
        "expert_shard_type": 0,
        "shared_expert_rank_num": 0,
        "moe_expert_num": self.moe_expert_num,
        "global_bs": self.global_bs,
        "expert_token_nums_type": expert_token_nums_type,
    }
    if self.global_bs == 0:
        kwargs_mc2["x_active_mask"] = token_dispatch_input.routing.mc2_mask

    stage1_kwargs = {
        # ### PATCH START: forward the scale that arrived with the activation
        # None keeps upstream's behaviour for an activation this side quantized.
        "scales": routing_scale,
        # ### PATCH END: forward the scale that arrived with the activation
        "quant_mode": quant_mode,
        "group_ep": self.moe_all_to_all_group_name,
        "ep_world_size": self.ep_world_size,
        "ep_rank_id": self.ep_rank_id,
        "comm_alg": self.mc2_comm_alg,
    }
    if self.need_extra_args:
        stage1_kwargs.update(
            {
                "group_tp": self.moe_all_to_all_group_name,
                "tp_world_size": 1,
                "tp_rank_id": 0,
            }
        )
    # Only dispatch-enabled MXFP paths pass y_dtype through MC2.
    if (
        self.need_shared_expert_args
        and (token_dispatch_input.quant.is_mxfp or token_dispatch_input.quant.is_fp8)
        and token_dispatch_input.quant.dispatch_with_quant
    ):
        y_dtype = torch.float8_e4m3fn
        if (
            token_dispatch_input.quant.mxfp is not None
            and token_dispatch_input.quant.mxfp.act_quant_type is not None
        ):
            y_dtype = token_dispatch_input.quant.mxfp.act_quant_type
        stage1_kwargs.update({"tp_world_size": 1, "tp_rank_id": 0, "y_dtype": y_dtype})
    if self.need_expert_scale:
        stage1_kwargs.update(
            {
                "expert_scales": topk_weights.to(torch.float32),
            }
        )

    kwargs_mc2.update(stage1_kwargs)
    return kwargs_mc2


def apply_afd_mc2_dispatch_prequant_patch() -> bool:
    """Install the patch once. Returns whether it is installed."""

    _ensure_upstream_imports()
    if TokenDispatcherWithMC2.get_dispatch_mc2_kwargs is get_dispatch_mc2_kwargs:
        return True
    # The multi-patch and all-to-all dispatchers share this method's shape but not
    # its routing contract, so only the MC2 dispatcher is replaced.
    if (
        TokenDispatcherWithMC2.get_dispatch_mc2_kwargs
        is not _ORIGINAL_GET_DISPATCH_MC2_KWARGS
    ):
        raise RuntimeError(
            "AFD MC2 dispatch patch: get_dispatch_mc2_kwargs is already replaced by "
            "something else, so the upstream body this patch copies may not match",
        )
    TokenDispatcherWithMC2.get_dispatch_mc2_kwargs = get_dispatch_mc2_kwargs
    return True


__all__ = ["apply_afd_mc2_dispatch_prequant_patch", "get_dispatch_mc2_kwargs"]
