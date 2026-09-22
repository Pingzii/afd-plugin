# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A2E tile layout: which Attention rank fills which FFN rank tile.

A2E and its E2A twin move one tile per Attention peer of an FFN rank and read the
same row count from every peer, so the peer set, the tile row count, and the rows
the FFN rank computes on all have to be derived identically by the connector, the
FFN runner, and the graph-key builder. The operators fix both properties
(``csrc/npu/ascend_kernels/a2e/op_kernel/a2e.h``,
``csrc/npu/ascend_kernels/e2a/op_kernel/e2a.h``):

* An FFN rank ``r`` reads one tile per Attention peer ``r, r + ffn_size,
  r + 2 * ffn_size, ...``. The receiver walks ``sendRank = rank + (index + 1) *
  expertRankSize`` for both the ids and the activations, and the sender indexes
  its tile as ``rank / expertRankSize - 1`` on the destination window ``rank %
  expertRankSize``. The peer group is therefore *strided*, unlike the contiguous
  blocks the GPU P2P mapping builds in
  :func:`afd_plugin.distributed.subgroup_attention_block`.
* The receiver derives its tile from the total it was given (``recvBatchSize =
  batchSize / attnToMoeRatio``) and then reads exactly that many rows from every
  peer, so the peers of one FFN rank have to write the same row count. DP ranks
  hold independent batches, so the transport pads each peer up to the largest
  count of its peer group rather than assuming the counts already match.

Counts are positional in Attention-rank order: ``num_tokens_across_dp_cpu`` holds
one count per DP rank and every count is replicated across that DP rank's TP
workers. The module deliberately depends on nothing else, because the NPU
connector, the NPU FFN runner, and the CPU-safe graph-key helper all import it.
"""

from __future__ import annotations

from collections.abc import Sequence


def _divides_evenly(attention_size: int, ffn_size: int) -> bool:
    """Report whether the ``attention_size // ffn_size`` tile topology applies."""

    if ffn_size <= 0 or attention_size < ffn_size:
        return False
    return attention_size % ffn_size == 0


def attention_rank_token_counts(
    dp_counts: Sequence[int],
    *,
    attention_size: int,
) -> list[int] | None:
    """Expand per-DP-rank counts to one count per Attention rank.

    Returns ``None`` when the counts cannot describe every Attention rank, which
    leaves the caller's fallback in place instead of inventing counts.
    """

    counts = [int(count) for count in dp_counts]
    if not counts:
        return None
    if len(counts) < attention_size and attention_size % len(counts) == 0:
        attention_ranks_per_count = attention_size // len(counts)
        counts = [
            counts[rank // attention_ranks_per_count] for rank in range(attention_size)
        ]
    if len(counts) < attention_size:
        return None
    return counts[:attention_size]


def attention_peer_ranks(
    ffn_rank: int,
    *,
    attention_size: int,
    ffn_size: int,
) -> range | None:
    """Return the Attention ranks that fill one FFN rank's tiles, in tile order.

    Returns ``None`` for a topology this layout does not describe, so callers keep
    their fallback instead of guessing a peer set.
    """

    if not _divides_evenly(attention_size, ffn_size) or not 0 <= ffn_rank < ffn_size:
        return None
    return range(ffn_rank, attention_size, ffn_size)


def ffn_rank_for_attention_rank(
    attention_rank: int,
    *,
    attention_size: int,
    ffn_size: int,
) -> int | None:
    """Return the FFN rank one Attention rank writes its tile to."""

    if not _divides_evenly(attention_size, ffn_size):
        return None
    if not 0 <= attention_rank < attention_size:
        return None
    return attention_rank % ffn_size


def padded_tile_rows(
    dp_counts: Sequence[int],
    *,
    ffn_rank: int,
    attention_size: int,
    ffn_size: int,
) -> int | None:
    """Return the row count every Attention peer of one FFN rank writes.

    The value is the largest count in that FFN rank's peer group, which is what
    each peer pads its own payload up to. Returns ``None`` when the metadata or
    the topology cannot describe the group.
    """

    peers = attention_peer_ranks(
        ffn_rank,
        attention_size=attention_size,
        ffn_size=ffn_size,
    )
    counts = attention_rank_token_counts(dp_counts, attention_size=attention_size)
    if peers is None or counts is None:
        return None
    return max(1, max(counts[peer] for peer in peers))


def padded_ffn_token_counts(
    dp_counts: Sequence[int],
    *,
    attention_size: int,
    ffn_size: int,
) -> tuple[int, ...] | None:
    """Return the rows each FFN rank receives, one entry per FFN role rank.

    Every FFN rank receives one padded tile per Attention peer, so its count is
    ``attention_size // ffn_size`` tiles of ``padded_tile_rows`` rows.
    """

    if not _divides_evenly(attention_size, ffn_size):
        return None
    tiles = attention_size // ffn_size
    counts: list[int] = []
    for ffn_rank in range(ffn_size):
        rows = padded_tile_rows(
            dp_counts,
            ffn_rank=ffn_rank,
            attention_size=attention_size,
            ffn_size=ffn_size,
        )
        if rows is None:
            return None
        counts.append(tiles * rows)
    return tuple(counts)


__all__ = [
    "attention_peer_ranks",
    "attention_rank_token_counts",
    "ffn_rank_for_attention_rank",
    "padded_ffn_token_counts",
    "padded_tile_rows",
]
