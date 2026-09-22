# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for the A2E tile layout.

The layout is plain integer arithmetic that mirrors the A2E/E2A kernels, so it is
testable without an Ascend device. The properties worth pinning are the peer set
(the kernel pairs an FFN rank with strided Attention ranks) and the padded tile
that every peer of a group derives identically.
"""

from __future__ import annotations

import pytest

from afd_plugin.a2e_layout import (
    attention_peer_ranks,
    attention_rank_token_counts,
    ffn_rank_for_attention_rank,
    padded_ffn_token_counts,
    padded_tile_rows,
)


class TestAttentionPeerRanks:
    def test_pairs_strided_attention_ranks_with_each_ffn_rank(self):
        # 4A2F: the kernel reads peers rank + (i + 1) * ffn_size.
        assert list(attention_peer_ranks(0, attention_size=4, ffn_size=2)) == [0, 2]
        assert list(attention_peer_ranks(1, attention_size=4, ffn_size=2)) == [1, 3]

    def test_keeps_one_peer_per_ffn_rank_when_the_sizes_match(self):
        assert list(attention_peer_ranks(0, attention_size=2, ffn_size=2)) == [0]
        assert list(attention_peer_ranks(1, attention_size=2, ffn_size=2)) == [1]

    @pytest.mark.parametrize(
        ("attention_size", "ffn_size"),
        [(2, 4), (3, 2), (0, 2), (4, 0)],
    )
    def test_rejects_a_topology_it_cannot_describe(self, attention_size, ffn_size):
        assert (
            attention_peer_ranks(0, attention_size=attention_size, ffn_size=ffn_size)
            is None
        )

    def test_rejects_a_rank_outside_the_role(self):
        assert attention_peer_ranks(2, attention_size=4, ffn_size=2) is None

    def test_reports_the_ffn_rank_each_attention_rank_feeds(self):
        assert ffn_rank_for_attention_rank(0, attention_size=4, ffn_size=2) == 0
        assert ffn_rank_for_attention_rank(3, attention_size=4, ffn_size=2) == 1
        assert ffn_rank_for_attention_rank(4, attention_size=4, ffn_size=2) is None


class TestAttentionRankTokenCounts:
    def test_expands_dp_counts_across_tp_workers(self):
        # DP=2 with TP=2 replicates each DP rank's count over its TP workers.
        assert attention_rank_token_counts([4, 8], attention_size=4) == [4, 4, 8, 8]

    def test_keeps_one_count_per_attention_rank(self):
        assert attention_rank_token_counts([4, 8, 16, 16], attention_size=4) == [
            4,
            8,
            16,
            16,
        ]

    def test_returns_none_when_the_counts_cannot_cover_the_role(self):
        assert attention_rank_token_counts([], attention_size=4) is None
        assert attention_rank_token_counts([4, 8, 16], attention_size=4) is None


class TestPaddedTile:
    def test_pads_to_the_largest_count_of_the_peer_group(self):
        counts = [4, 8, 16, 16]

        # FFN rank 0 owns Attention ranks 0 and 2, whose counts are 4 and 16.
        assert padded_tile_rows(counts, ffn_rank=0, attention_size=4, ffn_size=2) == 16
        # FFN rank 1 owns Attention ranks 1 and 3, both holding 16.
        assert padded_tile_rows(counts, ffn_rank=1, attention_size=4, ffn_size=2) == 16

    def test_every_peer_of_a_group_derives_the_same_tile(self):
        counts = [5, 7, 6, 6]

        # Attention ranks 1 and 3 both resolve to FFN rank 1 and have to agree.
        tiles = {
            padded_tile_rows(
                counts,
                ffn_rank=ffn_rank_for_attention_rank(
                    rank, attention_size=4, ffn_size=2
                ),
                attention_size=4,
                ffn_size=2,
            )
            for rank in (1, 3)
        }

        assert tiles == {7}

    def test_keeps_a_single_peer_tile_unchanged(self):
        assert padded_tile_rows([6, 6], ffn_rank=0, attention_size=2, ffn_size=2) == 6

    def test_returns_none_without_counts(self):
        assert padded_tile_rows([], ffn_rank=0, attention_size=4, ffn_size=2) is None


class TestPaddedFfnTokenCounts:
    def test_counts_one_padded_tile_per_attention_peer(self):
        # 4A2F: two tiles per FFN rank, each the size of its group maximum.
        assert padded_ffn_token_counts(
            [4, 8, 16, 16], attention_size=4, ffn_size=2
        ) == (
            32,
            32,
        )

    def test_keeps_even_groups_at_their_real_total(self):
        assert padded_ffn_token_counts([6, 6, 6, 6], attention_size=4, ffn_size=2) == (
            12,
            12,
        )

    def test_matches_the_single_tile_layout_when_the_sizes_match(self):
        # 2A2F: one tile per FFN rank, so the padded total is the peer's count.
        assert padded_ffn_token_counts([5, 7], attention_size=2, ffn_size=2) == (5, 7)

    def test_expands_dp_counts_before_aggregating(self):
        # DP=1 with TP=2 replicates the single count over both Attention ranks.
        assert padded_ffn_token_counts([8], attention_size=2, ffn_size=2) == (8, 8)

    @pytest.mark.parametrize(
        ("attention_size", "ffn_size"),
        [(2, 4), (3, 2), (4, 0)],
    )
    def test_returns_none_for_a_topology_it_cannot_describe(
        self,
        attention_size,
        ffn_size,
    ):
        assert (
            padded_ffn_token_counts(
                [4, 4], attention_size=attention_size, ffn_size=ffn_size
            )
            is None
        )
