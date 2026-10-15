# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Deliver an externally supplied MXFP8 activation scale to the Ascend MoE.

Upstream source: ``vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py``.

The A2E MXFP8 transport quantizes the Attention activation and ships it to the
FFN together with its E8M0 MX scale. The scale has to reach
``build_fused_experts_input`` so the MoE treats the activation as already
quantized instead of quantizing it a second time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts

AFD_MXFP8_SCALE_ATTR = "afd_mxfp8_scale"
_MXFP8_SCALE_PATCH_ATTR = "_afd_plugin_moe_mxfp8_scale_patch_state"


def apply_afd_moe_mxfp8_scale_patch() -> bool:
    """Install the MoE MXFP8 scale resolver, returning whether it applied."""

    try:
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        from vllm_ascend.ops.fused_moe.dataclass.fused_experts import (
            build_fused_experts_input,
        )
        from vllm_ascend.quantization.methods.w4a8 import w4a8_mxfp4
    except ImportError:
        return False

    if hasattr(w4a8_mxfp4, _MXFP8_SCALE_PATCH_ATTR):
        return True

    method = w4a8_mxfp4.AscendW4A8MXFPDynamicFusedMoEMethod
    original_apply = method.apply

    # Patch reason: ``AscendRoutedExperts`` assigns ``ascend_pertoken_scale``
    # from the MoE prepare output, which is ``None`` on the DP path. That
    # overwrites the scale the A2E MXFP8 transport pinned on the layer, so the
    # MoE would quantize an already-quantized activation a second time and drop
    # the delivered scale.
    # Patch functionality: resolve ``pertoken_scale`` from the AFD-owned
    # attribute when AFD set one, and otherwise keep the upstream argument
    # unchanged.
    # Signature: matches upstream; no added parameters.
    def apply(
        self,
        layer: AscendRoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        import torch_npu

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        # ### PATCH START: resolve the A2E MXFP8 transport scale
        transported_scale = getattr(layer, AFD_MXFP8_SCALE_ATTR, None)
        pertoken_scale = (
            layer.ascend_pertoken_scale
            if transported_scale is None
            else transported_scale
        )
        # ### PATCH END: resolve the A2E MXFP8 transport scale
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                layer=layer,
                quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=layer.ascend_expert_map,
                global_redundant_expert_num=layer.global_redundant_expert_num,
                mc2_mask=layer.ascend_mc2_mask,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                pertoken_scale=pertoken_scale,
                activation=layer.activation,
                mxfp_act_quant_type=torch.float8_e4m3fn,
                mxfp_weight_quant_type=torch_npu.float4_e2m1fn_x2,
                mxfp_scale_dtype=torch_npu.float8_e8m0fnu,
                mxfp_per_token_scale_dtype=torch_npu.float8_e8m0fnu,
                mxfp_use_bf16=(x.dtype in [torch.bfloat16, torch.float8_e4m3fn]),
            ),
            quant_method=self,
        )

    method.apply = apply
    setattr(w4a8_mxfp4, _MXFP8_SCALE_PATCH_ATTR, original_apply)
    return True


__all__ = [
    "AFD_MXFP8_SCALE_ATTR",
    "apply_afd_moe_mxfp8_scale_patch",
]
