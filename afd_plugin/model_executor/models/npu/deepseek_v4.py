# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend A5 AFD wrapper for the native vLLM-Ascend DeepSeek-V4 model.

The Attention worker retains DSA attention, normalization, and the complete
mHC residual stream. The FFN worker owns the native Ascend V4 MoE. Only the
normalized two-dimensional FFN activation and token-aligned input IDs cross
the synchronous A5 HCCL P2P boundary.

The connector API intentionally represents input IDs as an optional payload.
That same contract is the integration seam for the planned A5 A2E/E2A
operators; they can replace the P2P transport without changing this model.
"""

from collections.abc import Iterable
from importlib import import_module
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context

from afd_plugin.config import parse_afd_config
from afd_plugin.connectors.metadata import AFDTransferContext, AFDTransferMetadata
from afd_plugin.connectors.npu.camp2p_a5 import is_a5
from afd_plugin.model_executor.models import get_afd_metadata_from_forward_context
from afd_plugin.model_executor.models.deepseek_v4_common import _iter_role_weights
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield


def _import_native_deepseek_v4() -> ModuleType:
    """Load DeepSeek-V4 across the flat and package Ascend layouts."""
    module_name = "vllm_ascend.models.deepseek_v4"
    deepseek_v4 = import_module(module_name)
    if hasattr(deepseek_v4, "__path__"):
        return import_module(f"{module_name}.model")
    return deepseek_v4


native = _import_native_deepseek_v4()


class RemoteNPUDeepseekV4FFN(nn.Module):
    """Parameter-free FFN proxy carrying V4 hash-router token identifiers."""

    def __init__(self, *, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if input_ids is None:
            raise RuntimeError("DeepSeek-V4 remote FFN requires input_ids")
        if input_ids.ndim != 1 or input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "DeepSeek-V4 input_ids must be one-dimensional and token-aligned",
            )

        afd_metadata = get_afd_metadata_from_forward_context()
        if afd_metadata is None:
            raise RuntimeError("RemoteNPUDeepseekV4FFN requires AFD metadata")
        forward_context = get_forward_context()
        stage_idx = int(
            getattr(forward_context, "ubatch_idx", afd_metadata.stage_idx),
        )
        afd_metadata.stage_idx = stage_idx
        metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=self.layer_idx,
            stage_idx=stage_idx,
            seq_len=int(hidden_states.shape[0]),
        )
        context = AFDTransferContext(metadata=metadata)
        afd_metadata.connector.send_attn_output(
            hidden_states,
            context,
            input_ids=input_ids,
        )
        hidden_states = maybe_apply_dbo_yield(
            hidden_states,
            role="attention",
        )
        return afd_metadata.connector.recv_ffn_output(
            ref_tensor=hidden_states,
            ubatch_idx=stage_idx,
        )


class AFDNPUDeepseekV4DecoderLayer(native.DeepseekV2DecoderLayer):
    """Ascend DeepSeek-V4 decoder with an FFN-boundary AFD split."""

    # Patch reason: the native Ascend layer always allocates Attention and FFN.
    # Patch functionality: allocate only the stage owned by the active AFD role.
    # Signature: matches upstream; no added parameters.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v4/model.py
    # Commit: 4fe7ddbf94bc28bcd2b9f3d2d93f0fe1f0499cf5
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: torch.Tensor | None = None,
        is_draft_layer: bool = False,
    ) -> None:
        # ### PATCH START: require a role before allocating native stages.
        nn.Module.__init__(self)
        afd_config = parse_afd_config(vllm_config, validate=False)
        # ### PATCH END

        if config is None:
            config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = config.rope_parameters[
            "original_max_position_embeddings"
        ]
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        self.norm_eps = config.rms_norm_eps

        # ### PATCH START: replace the remote stage with a parameter-free proxy.
        if afd_config.role == "attention":
            self.self_attn = native.DeepseekV4Attention(
                vllm_config=vllm_config,
                config=config,
                max_position_embeddings=max_position_embeddings,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
            )
            self.mlp = RemoteNPUDeepseekV4FFN(layer_idx=layer_idx)
        elif afd_config.role == "ffn":
            self.self_attn = native.PPMissingLayer()
            self.mlp = native.DeepseekV4MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                is_draft_layer=is_draft_layer,
            )
        else:
            raise ValueError(f"unsupported AFD role {afd_config.role!r}")
        # ### PATCH END

        # ### PATCH START: mHC state and normalization are Attention-owned.
        if afd_config.role == "ffn":
            return
        # ### PATCH END
        self.input_layernorm = native.RMSNorm(
            config.hidden_size,
            eps=self.norm_eps,
        )
        self.post_attention_layernorm = native.RMSNorm(
            config.hidden_size,
            eps=self.norm_eps,
        )
        self.routed_scaling_factor = getattr(
            config,
            "routed_scaling_factor",
            1.0,
        )
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32),
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32),
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
        )
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    # Patch reason: native forward directly invokes its locally allocated FFN.
    # Patch functionality: retain native NPU mHC locally while the proxy sends
    # only normalized FFN activations and token IDs to the FFN worker.
    # Signature: matches upstream; no added parameters.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v4/model.py
    # Commit: 4fe7ddbf94bc28bcd2b9f3d2d93f0fe1f0499cf5
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ### PATCH START: prohibit execution on the FFN worker.
        if isinstance(self.self_attn, native.PPMissingLayer):
            raise RuntimeError("DeepSeek-V4 decoder forward is Attention-owned")
        # ### PATCH END
        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
        )
        hidden_states = self.input_layernorm(hidden_states)
        attn_kwargs = {
            "positions": positions,
            "hidden_states": hidden_states,
            "llama_4_scaling": llama_4_scaling,
        }
        hidden_states = self.self_attn(**attn_kwargs)
        hidden_states = self.hc_post(hidden_states, residual, post, comb)
        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        # ### PATCH START: this call enters the synchronous remote FFN proxy.
        hidden_states = self.mlp(hidden_states, input_ids)
        # ### PATCH END
        hidden_states = self.hc_post(hidden_states, residual, post, comb)
        return hidden_states, residual

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Execute the native Ascend V4 MoE, including hash routing."""
        if not isinstance(self.mlp, native.DeepseekV4MoE):
            raise RuntimeError("DeepSeek-V4 FFN compute is FFN-role only")
        if input_ids is None:
            raise RuntimeError("DeepSeek-V4 FFN compute requires input_ids")
        return self.mlp(hidden_states, input_ids)


class AFDNPUDeepseekV4Model(native.DeepseekV4Model):
    """Role-aware Ascend DeepSeek-V4 model for the initial A5 P2P route."""

    # Patch reason: native Ascend V4 allocates all decoder stages on every rank.
    # Patch functionality: construct AFD role-aware layers and resources.
    # Signature: matches upstream; no added parameters.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v4/model.py
    # Commit: 4fe7ddbf94bc28bcd2b9f3d2d93f0fe1f0499cf5
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: validate the deliberately narrow first release.
        nn.Module.__init__(self)
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        if native.current_platform.device_type != "npu":
            raise RuntimeError("AFD NPU DeepSeek-V4 requires an Ascend platform")
        if not is_a5():
            raise RuntimeError("AFD NPU DeepSeek-V4 currently supports A5 only")
        if self.afd_config.connector != "CAMP2pAFDConnector":
            raise RuntimeError(
                "AFD NPU DeepSeek-V4 requires CAMP2pAFDConnector",
            )
        if self.afd_config.compute_gate_on_attention:
            raise RuntimeError(
                "AFD NPU DeepSeek-V4 requires gate computation on FFN",
            )
        if not vllm_config.model_config.enforce_eager:
            raise RuntimeError(
                "AFD NPU DeepSeek-V4 A5 P2P requires --enforce-eager",
            )
        if vllm_config.speculative_config is not None:
            raise RuntimeError(
                "AFD NPU DeepSeek-V4 does not yet support speculative decoding",
            )
        if vllm_config.lora_config is not None:
            raise RuntimeError("AFD NPU DeepSeek-V4 does not yet support LoRA")
        parallel_config = vllm_config.parallel_config
        if parallel_config.pipeline_parallel_size != 1:
            raise RuntimeError("AFD NPU DeepSeek-V4 does not support PP")
        if (
            parallel_config.prefill_context_parallel_size != 1
            or parallel_config.decode_context_parallel_size != 1
        ):
            raise RuntimeError("AFD NPU DeepSeek-V4 does not support CP")
        if parallel_config.use_sequence_parallel_moe:
            raise RuntimeError("AFD NPU DeepSeek-V4 does not support SP MoE")
        if parallel_config.enable_eplb or parallel_config.enable_elastic_ep:
            raise RuntimeError(
                "AFD NPU DeepSeek-V4 does not support EPLB or elastic EP",
            )
        # ### PATCH END

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = native.current_platform.device_type
        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")

        # ### PATCH START: sparse-index storage is Attention-owned.
        if self.afd_config.role == "attention" and self.is_v32:
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None
        self.topk_indices_buffer = topk_indices_buffer
        # ### PATCH END

        if native.get_pp_group().is_first_rank:
            self.embed_tokens = native.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = native.PPMissingLayer()

        # ### PATCH START: use the role-aware Ascend decoder constructor.
        self.start_layer, self.end_layer, self.layers = native.make_layers(
            config.num_hidden_layers,
            lambda prefix: AFDNPUDeepseekV4DecoderLayer(
                vllm_config,
                prefix,
                topk_indices_buffer=topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )
        # ### PATCH END

        if native.get_pp_group().is_last_rank:
            self.norm = native.RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
        else:
            self.norm = native.PPMissingLayer()

        def make_empty_intermediate_tensors(
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> native.IntermediateTensors:
            return native.IntermediateTensors(
                {
                    "hidden_states": torch.zeros(
                        (batch_size, self.hc_mult, config.hidden_size),
                        dtype=dtype,
                        device=device,
                    ),
                },
            )

        # ### PATCH START: preserve the pre-module-split Ascend PP contract.
        make_pp_empty = getattr(
            native,
            "make_pp_empty_intermediate_tensors",
            None,
        )
        if make_pp_empty is None:
            # vLLM-Ascend before the DSV4 module split exposes the factory
            # directly instead of wrapping it for pipeline-parallel models.
            self.make_empty_intermediate_tensors = make_empty_intermediate_tensors
        else:
            self.make_empty_intermediate_tensors = make_pp_empty(
                self,
                make_empty_intermediate_tensors,
            )
        # ### PATCH END

        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        hc_dim = hc_mult * config.hidden_size

        # ### PATCH START: final mHC state is constructed only on Attention.
        if self.afd_config.role == "attention":
            self.hc_head_fn = nn.Parameter(
                torch.empty(hc_mult, hc_dim, dtype=torch.float32),
            )
            self.hc_head_base = nn.Parameter(
                torch.empty(hc_mult, dtype=torch.float32),
            )
            self.hc_head_scale = nn.Parameter(
                torch.empty(1, dtype=torch.float32),
            )
            self.hc_norm = native.RMSNorm(
                hc_dim,
                eps=config.rms_norm_eps,
                has_weight=False,
                dtype=torch.float32,
            )
        else:
            self.hc_head_fn = None
            self.hc_head_base = None
            self.hc_head_scale = None
            self.hc_norm = native.PPMissingLayer()
        self._mtp_hidden_buffer = None
        # ### PATCH END

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        *,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.layers[layer_idx].compute_ffn_output(
            hidden_states,
            input_ids=input_ids,
        )

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return tuple(range(int(self.config.num_hidden_layers)))


class AFDNPUDeepseekV4ForCausalLM(native.AscendDeepseekV4ForCausalLM):
    """Ascend V4 causal LM exposing the NPU FFN-runner model contract."""

    model_cls = AFDNPUDeepseekV4Model
    afd_requires_input_ids = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = self.afd_config.role
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        *,
        input_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.model.compute_ffn_output(
            hidden_states,
            layer_idx,
            input_ids=input_ids,
        )

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.model.get_experts_layer_indices()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Return native expert mappings only where real experts are owned."""
        if self.afd_role == "attention":
            return []
        return super().get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(
            _iter_role_weights(weights, role=self.afd_role),
        )


__all__ = ["AFDNPUDeepseekV4ForCausalLM"]
