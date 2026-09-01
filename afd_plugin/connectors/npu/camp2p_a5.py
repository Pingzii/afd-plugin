# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A5 (Ascend950) HCCL p2p data movement for the CAMP2p connector.

On Atlas A5 both the a2e/e2a custom ops and the torch_npu native MoE
dispatch/combine ops fail on the AFD mixed group with ``507035`` (MTE
out-of-range). This module provides the Route-B2 replacement: plain
``torch.distributed.send`` / ``recv`` over the plugin-owned ``afd`` HCCL
process group, moving hidden states between Attention and FFN ranks. The
FFN side then runs its MoE internally through the standard vLLM-Ascend EP
path, where the native ops are proven.

Attention<->FFN rank mapping matches ``camp2p._num_tokens_for_ffn_rank``:
with ``group_size = attention_size // ffn_size``, Attention local rank ``i``
maps to FFN rank ``i // group_size`` and FFN rank ``j`` receives from the
consecutive Attention local ranks ``[j*group_size, (j+1)*group_size)``.

Only ``hidden_states`` crosses the wire (CAMP2p enforces gate-on-FFN, so
``router_logits`` never enters the connector). The per-peer token counts
come from the DP metadata control plane, so the FFN side can size its
receive buffers and split results back exactly (no equal-ratio split).
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache

import torch
import torch.distributed as dist


@cache
def is_a5() -> bool:
    """Return whether this process runs on Atlas A5 (Ascend950)."""
    try:
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        return get_ascend_device_type() == AscendDeviceType.A5
    except Exception:
        # Not on an Ascend platform or vllm_ascend unavailable.
        return False


def attention_group_size(attention_size: int, ffn_size: int) -> int:
    """Number of Attention ranks mapped to each FFN rank."""
    return attention_size // ffn_size


def dst_ffn_for_attention(
    attn_local_rank: int,
    attention_size: int,
    ffn_size: int,
) -> int:
    """FFN rank (0-based) that an Attention rank sends its tokens to."""
    return attn_local_rank // attention_group_size(attention_size, ffn_size)


def attention_peers_for_ffn(
    ffn_rank: int,
    attention_size: int,
    ffn_size: int,
) -> list[int]:
    """Attention local ranks that an FFN rank receives from (ascending)."""
    group_size = attention_group_size(attention_size, ffn_size)
    start = ffn_rank * group_size
    return list(range(start, start + group_size))


def attention_token_counts(
    dp_metadata_list: Mapping[int, object],
    stage_idx: int,
    attention_size: int,
) -> list[int] | None:
    """Per-Attention-rank token counts for a stage (DP expanded to AFD ranks).

    Mirrors ``camp2p._num_tokens_for_ffn_rank``'s DP -> AFD expansion: when TP
    creates several Attention workers per DP rank, the DP token count is
    replicated ``tp_size`` times. Returns ``None`` when the metadata is
    missing or cannot be expanded (the caller falls back to an even split).
    """
    dp_metadata = dp_metadata_list.get(stage_idx)
    if dp_metadata is None:
        return None
    token_counts = dp_metadata.num_tokens_across_dp_cpu
    counts = token_counts.flatten().tolist()
    if len(counts) < attention_size and attention_size % len(counts) == 0:
        tp_size = attention_size // len(counts)
        counts = [counts[i // tp_size] for i in range(attention_size)]
    if len(counts) < attention_size:
        return None
    return counts


def p2p_send(pg, tensor: torch.Tensor, dst_rank: int) -> None:
    """Blocking HCCL send of ``tensor`` to ``dst_rank`` over the AFD group."""
    dist.send(tensor.contiguous(), dst=dst_rank, group=pg)


def p2p_recv(
    pg,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    src_rank: int,
) -> torch.Tensor:
    """Blocking HCCL recv of a tensor with the given shape from ``src_rank``."""
    buf = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(buf, src=src_rank, group=pg)
    return buf


__all__ = [
    "attention_group_size",
    "attention_peers_for_ffn",
    "attention_token_counts",
    "dst_ffn_for_attention",
    "is_a5",
    "p2p_recv",
    "p2p_send",
]
