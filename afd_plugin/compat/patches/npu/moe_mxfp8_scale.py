# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Deliver an externally supplied MXFP8 activation scale to the Ascend MoE.

Upstream source: ``vllm_ascend/ops/fused_moe/prepare_finalize.py`` and
``vllm_ascend/ops/fused_moe/fused_moe.py``.

The A2E MXFP8 transport quantizes the Attention activation and ships it to the
FFN together with its E8M0 MX scale. For the MoE to treat the activation as
already quantized, that scale has to arrive as ``pertoken_scale``: the routing op
selects ``quant_mode = -1`` (permute only) exactly when ``routing.pertoken_scale``
is present, and ``_quant_hidden_states`` skips its ``npu_dynamic_mx_quant`` on the
same condition.

Two upstream gaps stand in the way.

1. ``PrepareAndFinalizeWithAllGather._prepare_with_dp_group`` pads and
   all-gathers the activation but always reports ``pertoken_scale=None``, so the
   scale never reaches the MoE on the DP path. Its EP counterpart,
   ``_prepare_with_ep_group``, already gathers the scale next to the activation;
   the DP path simply lacks the same step.

2. The shared experts quantize their input unconditionally, so they reject an
   already-quantized activation. ``AscendMoERunner.forward`` accepts
   ``shared_experts_input`` and routes it to them, but no AFD caller supplies it.

The scale gather must happen inside ``_prepare_with_dp_group``, not from an
outer hook. It is a collective: launched anywhere else it runs on a different set
of ranks than the framework's own gather, and HCCL rejects it with "execute
kernel param invalid".
"""

from __future__ import annotations

from typing import Any

AFD_MXFP8_SCALE_ATTR = "afd_mxfp8_scale"
AFD_MXFP8_SHARED_INPUT_ATTR = "afd_mxfp8_shared_input"
_MXFP8_SCALE_PATCH_ATTR = "_afd_plugin_moe_mxfp8_scale_patch_state"
_MXFP8_NEUTRAL_SCALE_ATTR = "_afd_plugin_moe_mxfp8_neutral_scale"

#: E8M0 byte for 2^0. Used as the neutral fill so the activation rows the
#: framework pads with zeros stay zero after scaling.
_E8M0_ONE = 127

#: MX block size: one E8M0 scale byte covers 32 elements.
_MX_GROUP_SIZE = 32


def _max_tokens_across_dp() -> int:
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    return int(_EXTRA_CTX.max_tokens_across_dp)


def _pad_transported_scale(
    owner: Any, transported: Any, hidden_dim: int, target_rows: int
) -> Any:
    """Pad a transported MX scale to ``target_rows`` on its E8M0 byte view.

    The scale arrives as ``[rows, hidden/32]`` E8M0 bytes. The routing op wants
    ``(rows, hidden/64, 2)``, which is the same bytes re-viewed, so both the pad
    and the later all-gather stay on the byte view: HCCL has no E8M0 collective.

    A rank with no transported scale contributes ``_E8M0_ONE`` so the collective
    keeps the same shape everywhere; the caller discards that result. That dummy is
    reused rather than rebuilt: one allocation per (rows, pairs) instead of one per
    layer per step, which at 43 layers was churning hundreds of megabytes and
    fragmenting the cache that the following routing op depends on.
    """

    import torch

    pairs = -(-int(hidden_dim) // (2 * _MX_GROUP_SIZE))
    if transported is None:
        return _neutral_scale(
            owner,
            target_rows,
            pairs,
        )
    raw = transported.view(torch.uint8)
    rows = int(raw.shape[0])
    if target_rows > rows:
        raw = torch.nn.functional.pad(
            raw,
            (0, 0, 0, 0, 0, target_rows - rows),
            value=_E8M0_ONE,
        )
    return raw


def _neutral_scale(owner: Any, rows: int, pairs: int) -> Any:
    """Return a cached all-ones E8M0 block sized ``(rows, pairs, 2)``.

    The block is only read by the collective, so sharing one instance across
    layers is safe. It is held on the long-lived prepare/finalize object, which
    keeps the lifetime bounded by the process rather than by a forward pass.
    """

    import torch

    cache = getattr(owner, _MXFP8_NEUTRAL_SCALE_ATTR, None)
    if cache is not None and cache.shape == (rows, pairs, 2):
        return cache
    block = torch.full(
        (rows, pairs, 2),
        _E8M0_ONE,
        dtype=torch.uint8,
        device=torch.npu.current_device(),
    )
    setattr(owner, _MXFP8_NEUTRAL_SCALE_ATTR, block)
    return block


def apply_afd_moe_mxfp8_scale_patch() -> bool:
    """Install the MoE MXFP8 scale plumbing, returning whether it applied."""

    try:
        from vllm_ascend.ops.fused_moe import fused_moe
    except ImportError:
        return False

    runner = getattr(fused_moe, "AscendMoERunner", None)
    if runner is None or hasattr(runner, _MXFP8_SCALE_PATCH_ATTR):
        return hasattr(runner, _MXFP8_SCALE_PATCH_ATTR) if runner else False

    original_forward = runner.forward

    # Patch reason: the shared experts quantize their input unconditionally, so
    # they reject an already-quantized activation. ``AscendMoERunner.forward``
    # already accepts ``shared_experts_input`` and routes it to them
    # (``shared_hidden_states = shared_experts_input or hidden_states``), but no
    # AFD caller supplies it. The transported scale also has to reach the prepare
    # stage, and ``MoEPrepareOutput`` is frozen, so it is staged on
    # ``moe_config`` -- a plain dataclass -- for that stage to pick up.
    # Patch functionality: fall back to the AFD-pinned model-dtype copy for the
    # shared experts, and stage the transported scale on ``moe_config``.
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
        # ### PATCH START: stage the transported MX scale for the prepare stage
        routed_experts = getattr(self, "routed_experts", None)
        moe_config = getattr(self, "moe_config", None)
        # The pin may land on the runner or on the routed-experts layer,
        # depending on which object the model holds; check both.
        staged = getattr(self, AFD_MXFP8_SCALE_ATTR, None)
        if staged is None and routed_experts is not None:
            staged = getattr(routed_experts, AFD_MXFP8_SCALE_ATTR, None)
        # ``get_moe_comm_method`` caches one method object per (comm type, config)
        # key, and that object -- not this runner -- owns the prepare/finalize
        # instance whose ``_prepare_with_dp_group`` runs. Ask for it the same way
        # ``forward_impl`` will, rather than reading ``moe_comm_method`` from the
        # forward context: that field is only set once ``forward_impl`` starts, so
        # it is still unset for this layer.
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

        staged_targets = []
        moe_comm_method = getattr(_EXTRA_CTX, "moe_comm_method", None)
        if moe_comm_method is None:
            # ``forward_impl`` is what sets that field, so the very first layer of
            # the very first step has to resolve the cached method itself. The
            # released signature differs from the checkout (config argument
            # optional), so accept both without guessing.
            from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

            moe_comm_type = getattr(_EXTRA_CTX, "moe_comm_type", None)
            if moe_comm_type is not None:
                try:
                    moe_comm_method = get_moe_comm_method(moe_comm_type)
                except TypeError:
                    moe_comm_method = None
        if moe_comm_method is not None:
            prepare_finalize = getattr(moe_comm_method, "prepare_finalize", None)
            if prepare_finalize is not None:
                staged_targets.append(prepare_finalize)
            cached_config = getattr(moe_comm_method, "moe_config", None)
            if cached_config is not None:
                staged_targets.append(cached_config)
        if moe_config is not None:
            staged_targets.append(moe_config)
        for target in staged_targets:
            setattr(target, AFD_MXFP8_SCALE_ATTR, staged)
        # ### PATCH END: stage the transported MX scale for the prepare stage
        return original_forward(
            self,
            hidden_states,
            router_logits,
            input_ids,
            shared_experts_input,
        )

    runner.forward = forward
    setattr(runner, _MXFP8_SCALE_PATCH_ATTR, original_forward)

    try:
        from vllm_ascend.ops.fused_moe import prepare_finalize
    except ImportError:
        return True

    all_gather_prepare = getattr(
        prepare_finalize, "PrepareAndFinalizeWithAllGather", None
    )
    if all_gather_prepare is None or hasattr(
        all_gather_prepare, _MXFP8_SCALE_PATCH_ATTR
    ):
        return True

    original_prepare_with_dp_group = all_gather_prepare._prepare_with_dp_group

    # Patch reason: this stage pads and all-gathers the activation to
    # ``dp_size * max_tokens_across_dp`` rows but reports ``pertoken_scale=None``,
    # so a scale covering only this rank's rows cannot reach the routing op. Its
    # EP counterpart already gathers the scale beside the activation.
    # Patch functionality: gather the transported scale with the same group
    # object, in the same sequence, and return it as ``pertoken_scale``. Ranks
    # without a transported scale join with a neutral dummy and discard the
    # result. Only a process-wide switch can decide whether to enter, because the
    # dummy is what keeps the collective symmetric.
    # Signature: matches upstream; no added parameters.
    def _prepare_with_dp_group(
        self: Any,
        hidden_states: Any,
        router_logits: Any,
        replace_allreduce: bool = False,
        quant_type: Any = None,
    ) -> Any:
        import dataclasses

        import torch

        from afd_plugin.envs import camp2p_mxfp8_direct_enabled

        # ### PATCH START: gather the transported MX scale with the activation
        dp_group = getattr(self.moe_config, "dp_group", None)
        if dp_group is None:
            # ``AscendMoERunner.__init__`` fills this in, but a config built
            # elsewhere leaves it unset; fall back to the process-wide group.
            from vllm.distributed import get_dp_group

            dp_group = get_dp_group()
        transported = getattr(self, AFD_MXFP8_SCALE_ATTR, None)
        if transported is None:
            transported = getattr(self.moe_config, AFD_MXFP8_SCALE_ATTR, None)
        padded_scale = None
        if (
            camp2p_mxfp8_direct_enabled()
            and dp_group is not None
            and dp_group.world_size > 1
        ):
            padded_scale = _pad_transported_scale(
                self,
                transported,
                int(self.moe_config.hidden_dim),
                _max_tokens_across_dp(),
            )
        # ### PATCH END: gather the transported MX scale with the activation
        output = original_prepare_with_dp_group(
            self,
            hidden_states,
            router_logits,
            replace_allreduce,
            quant_type,
        )
        # ### PATCH START: gather the transported MX scale with the activation
        if padded_scale is not None:
            gathered = dp_group.all_gather(padded_scale, 0).view(torch.float8_e8m0fnu)
            if transported is not None:
                output = dataclasses.replace(output, pertoken_scale=gathered)
        # ### PATCH END: gather the transported MX scale with the activation
        return output

    all_gather_prepare._prepare_with_dp_group = _prepare_with_dp_group
    setattr(all_gather_prepare, _MXFP8_SCALE_PATCH_ATTR, original_prepare_with_dp_group)
    return True


__all__ = [
    "AFD_MXFP8_SCALE_ATTR",
    "AFD_MXFP8_SHARED_INPUT_ATTR",
    "apply_afd_moe_mxfp8_scale_patch",
]
