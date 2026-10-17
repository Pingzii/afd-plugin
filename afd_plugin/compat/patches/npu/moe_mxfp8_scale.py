# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Deliver an externally supplied MXFP8 activation scale to the Ascend MoE.

Upstream source: ``vllm_ascend/ops/fused_moe/dataclass/fused_experts.py``.

The A2E MXFP8 transport quantizes the Attention activation and ships it to the
FFN together with its E8M0 MX scale. That scale has to reach
``build_fused_experts_input`` so the MoE treats the activation as already
quantized: the routing op selects ``quant_mode = -1`` (permute only) exactly when
``routing.pertoken_scale`` is present, and ``_quant_hidden_states`` skips its
``npu_dynamic_mx_quant`` on the same condition.

Every MXFP8 scheme forwards ``pertoken_scale=layer.ascend_pertoken_scale``, and
``AscendRoutedExperts`` overwrites that attribute from the MoE prepare output,
which is ``None`` on the DP path. Resolving the scale inside
``build_fused_experts_input`` covers all of them at once, including schemes this
plugin does not name.
"""

from __future__ import annotations

from typing import Any

AFD_MXFP8_SCALE_ATTR = "afd_mxfp8_scale"
AFD_MXFP8_SCALE_EXPANDED_ATTR = "afd_mxfp8_scale_expanded"
AFD_MXFP8_SHARED_INPUT_ATTR = "afd_mxfp8_shared_input"
_MXFP8_SCALE_PATCH_ATTR = "_afd_plugin_moe_mxfp8_scale_patch_state"

# Every module that binds ``build_fused_experts_input`` by name at import time.
# A module-level rebind there is what the scheme methods' global lookup finds;
# patching the defining module alone would leave those bindings untouched.
_USER_MODULES = (
    "vllm_ascend.ops.fused_moe.routed_experts",
    "vllm_ascend.quantization.methods.w4a4.w4a4_mxfp4",
    "vllm_ascend.quantization.methods.w4a8.w4a8",
    "vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4",
    "vllm_ascend.quantization.methods.w8a8.w8a8_dynamic",
    "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8",
    "vllm_ascend.quantization.methods.wna16.w4a16",
    "vllm_ascend.quantization.methods.wna16.w4a16_mxfp4",
)


def apply_afd_moe_mxfp8_scale_patch() -> bool:
    """Install the MoE MXFP8 scale resolver, returning whether it applied."""

    try:
        from vllm_ascend.ops.fused_moe.dataclass import fused_experts
    except ImportError:
        return False

    if hasattr(fused_experts, _MXFP8_SCALE_PATCH_ATTR):
        return True

    original = fused_experts.build_fused_experts_input

    def _expand_scale_for_layer(layer: Any) -> Any:
        """Gather the transported scale onto the DP row layout the routing op sees.

        ``_prepare_with_dp_group`` pads the activation to ``max_tokens_across_dp``
        and all-gathers it, so the routing op sees ``dp_size * max_tokens_across_dp``
        rows. A transport scale covers only the rows this rank received, so it has
        to follow the same pad-then-gather path or its first dimension will not
        match the activation's.

        This runs from ``forward_impl`` -- entered once per layer by every rank that
        reaches the MoE, which is what makes the collective symmetric -- and never
        from ``build_fused_experts_input``, whose caller set varies between steps.
        A collective launched there never lines up and HCCL rejects it with
        "execute kernel param invalid". Ranks holding no scale still join with a
        neutral dummy and discard the result.

        Returns ``None`` when the expansion cannot be done, so the caller keeps the
        upstream behavior instead of guessing.
        """

        from afd_plugin.envs import camp2p_mxfp8_direct_enabled

        if not camp2p_mxfp8_direct_enabled():
            return None

        import torch
        from vllm.distributed import get_dp_group
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX

        transported = getattr(layer, AFD_MXFP8_SCALE_ATTR, None)
        dp_group = get_dp_group()
        if dp_group.world_size <= 1:
            return transported
        target_rows = getattr(_EXTRA_CTX, "max_tokens_across_dp", None)
        if target_rows is None:
            return None
        target_rows = int(target_rows)

        # Take the group layout from the layer weights: every rank holds them,
        # so the contributed shape cannot depend on per-rank forward state.
        weight = getattr(layer, "w13_weight", None)
        if weight is None:
            return None
        columns = -(-int(weight.shape[-1]) // 64)

        if transported is None:
            # 127 is 2^0, matching the neutral padding the transport uses.
            dummy = torch.full(
                (target_rows, columns, 2),
                127,
                dtype=torch.uint8,
                device=weight.device,
            )
            dp_group.all_gather(dummy, dim=0)
            return None

        # Stay on the byte view for both steps: HCCL has no e8m0 collective, and
        # 127 is 2^0, which leaves the zero-padded activation rows at zero.
        raw = transported.view(torch.uint8)
        rows = int(raw.shape[0])
        if target_rows > rows:
            raw = torch.nn.functional.pad(
                raw,
                (0, 0, 0, 0, 0, target_rows - rows),
                value=127,
            )
        return dp_group.all_gather(raw, dim=0).view(torch.float8_e8m0fnu)

    # Patch reason: ``AscendRoutedExperts`` assigns ``ascend_pertoken_scale`` from
    # the MoE prepare output, which is ``None`` on the DP path, so it overwrites
    # the scale the A2E MXFP8 transport pinned on the layer. The MoE would then
    # quantize an already-quantized activation a second time and drop the
    # delivered scale.
    # Patch functionality: substitute the AFD-owned scale when the layer carries
    # one, expanded to the same DP row layout as the activation, and otherwise
    # keep the caller's argument unchanged. The expansion runs on every rank so
    # the collective stays symmetric.
    # Signature: matches upstream; no added parameters.
    def build_fused_experts_input(*args: Any, **kwargs: Any) -> Any:
        layer = kwargs.get("layer")
        # ### PATCH START: prefer the A2E MXFP8 transport scale
        if layer is not None:
            expanded = getattr(layer, AFD_MXFP8_SCALE_EXPANDED_ATTR, None)
            if expanded is not None:
                kwargs["pertoken_scale"] = expanded
        # ### PATCH END: prefer the A2E MXFP8 transport scale
        return original(*args, **kwargs)

    patched = 0
    for module_name in _USER_MODULES:
        import importlib

        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if getattr(module, "build_fused_experts_input", None) is original:
            module.build_fused_experts_input = build_fused_experts_input  # type: ignore[attr-defined]
            patched += 1

    if not patched:
        return False

    fused_experts.build_fused_experts_input = build_fused_experts_input
    setattr(fused_experts, _MXFP8_SCALE_PATCH_ATTR, original)

    try:
        from vllm_ascend.ops.fused_moe import fused_moe
    except ImportError:
        return True

    runner = getattr(fused_moe, "AscendMoERunner", None)
    if runner is None or hasattr(runner, _MXFP8_SCALE_PATCH_ATTR):
        return True

    original_forward = runner.forward

    # Patch reason: the shared experts quantize their input unconditionally, so
    # they reject an already-quantized activation. ``AscendMoERunner.forward``
    # already accepts ``shared_experts_input`` and routes it to them
    # (``shared_hidden_states = shared_experts_input or hidden_states``), but no
    # caller supplies it on the AFD path.
    # Patch functionality: fall back to the AFD-pinned model-dtype copy, which
    # the transport receiver already produces for the router. It also expands the
    # transported MX scale here, because this is the MoE layer entry that every
    # rank reaches once per layer: the expansion pads and all-gathers, so it can
    # only live where the participation set matches the activation's DP gather.
    # Delegation over copying is deliberate -- the upstream body is the MoE layer
    # pipeline and only the two marked blocks differ.
    # Signature: matches upstream; no added parameters.
    def forward(
        self: Any,
        hidden_states: Any,
        router_logits: Any,
        input_ids: Any = None,
        shared_experts_input: Any = None,
    ) -> Any:
        # ### PATCH START: give the shared experts the model-dtype copy
        if shared_experts_input is None:
            transported_input = getattr(self, AFD_MXFP8_SHARED_INPUT_ATTR, None)
            if transported_input is not None:
                shared_experts_input = transported_input
        # ### PATCH END: give the shared experts the model-dtype copy
        # ### PATCH START: expand the transported MX scale for the DP row layout
        routed_experts = getattr(self, "routed_experts", None)
        if routed_experts is not None:
            setattr(
                routed_experts,
                AFD_MXFP8_SCALE_EXPANDED_ATTR,
                _expand_scale_for_layer(routed_experts),
            )
        # ### PATCH END: expand the transported MX scale for the DP row layout
        return original_forward(
            self,
            hidden_states,
            router_logits,
            input_ids,
            shared_experts_input,
        )

    runner.forward = forward
    setattr(runner, _MXFP8_SCALE_PATCH_ATTR, original_forward)
    return True


__all__ = [
    "AFD_MXFP8_SCALE_ATTR",
    "AFD_MXFP8_SHARED_INPUT_ATTR",
    "apply_afd_moe_mxfp8_scale_patch",
]
