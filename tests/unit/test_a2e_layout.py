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
    attention_tile_rows,
    fallback_tile_rows,
    ffn_rank_for_attention_rank,
    ffn_receive_rows,
    ffn_tile_count,
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


class TestFfnReceiveRows:
    def test_counts_one_tile_per_attention_peer(self):
        # 4A2F: F0 owns {A0, A2} and F1 owns {A1, A3}, each receiving two tiles
        # sized by its own group maximum.
        assert ffn_receive_rows([4, 8, 16, 16], 0, attention_size=4, ffn_size=2) == 32
        assert ffn_receive_rows([4, 8, 16, 16], 1, attention_size=4, ffn_size=2) == 32

    def test_keeps_even_groups_at_their_real_total(self):
        assert ffn_receive_rows([6, 6, 6, 6], 0, attention_size=4, ffn_size=2) == 12
        assert ffn_receive_rows([6, 6, 6, 6], 1, attention_size=4, ffn_size=2) == 12

    def test_matches_the_single_tile_layout_when_the_sizes_match(self):
        # 2A2F: one tile per FFN rank, so the receive is the peer's count.
        assert ffn_receive_rows([5, 7], 0, attention_size=2, ffn_size=2) == 5
        assert ffn_receive_rows([5, 7], 1, attention_size=2, ffn_size=2) == 7

    def test_expands_dp_counts_before_aggregating(self):
        # DP=1 with TP=2 replicates the single count over both Attention ranks.
        assert ffn_receive_rows([8], 0, attention_size=2, ffn_size=2) == 8
        assert ffn_receive_rows([8], 1, attention_size=2, ffn_size=2) == 8

    def test_sizes_a_missing_counts_step_by_the_shared_fallback(self):
        # No counts: the tile is the run-level fallback, and the receive is two of
        # those tiles, which is the number the Attention ranks pad up to.
        assert ffn_receive_rows([], 0, attention_size=4, ffn_size=2, fallback=64) == 128
        assert (
            attention_tile_rows(
                [], ffn_rank=0, attention_size=4, ffn_size=2, fallback=64
            )
            == 64
        )

    @pytest.mark.parametrize(
        ("attention_size", "ffn_size"),
        [(2, 4), (3, 2), (4, 0)],
    )
    def test_falls_back_to_one_tile_for_a_topology_it_cannot_describe(
        self,
        attention_size,
        ffn_size,
    ):
        # One tile of the all-rank fallback, which is the tile the Attention ranks
        # derive for the same topology.
        assert (
            ffn_receive_rows(
                [4, 4],
                0,
                attention_size=attention_size,
                ffn_size=ffn_size,
                fallback=16,
            )
            == 16
        )


class TestFallbackTileRows:
    def test_keeps_the_all_rank_count_both_roles_pass(self):
        assert fallback_tile_rows(fallback=64) == 64

    def test_never_returns_zero_rows(self):
        assert fallback_tile_rows(fallback=1) == 1
        assert fallback_tile_rows(fallback=0) == 1


class TestAttentionTileRows:
    def test_uses_the_peer_group_maximum(self):
        assert (
            attention_tile_rows([2, 2, 5, 7], ffn_rank=0, attention_size=4, ffn_size=2)
            == 5
        )
        assert (
            attention_tile_rows([2, 2, 5, 7], ffn_rank=1, attention_size=4, ffn_size=2)
            == 7
        )

    def test_falls_back_to_the_shared_all_rank_count(self):
        # A missing-counts step sizes both roles by the same all-rank fallback
        # instead of letting the receiver invent a larger tile than the sender
        # writes, which would make A2E read rows no peer wrote.
        assert (
            attention_tile_rows(
                [],
                ffn_rank=0,
                attention_size=4,
                ffn_size=2,
                fallback=64,
            )
            == 64
        )

    def test_keeps_even_groups_at_one_tile_per_peer(self):
        assert (
            attention_tile_rows(
                [12, 12, 12, 12], ffn_rank=0, attention_size=4, ffn_size=2
            )
            == 12
        )


class TestFfnTileCount:
    def test_counts_the_tiles_of_one_ffn_rank(self):
        assert ffn_tile_count(attention_size=4, ffn_size=2) == 2
        assert ffn_tile_count(attention_size=2, ffn_size=2) == 1

    def test_uses_one_tile_for_a_topology_it_cannot_describe(self):
        assert ffn_tile_count(attention_size=3, ffn_size=2) == 1

    def test_the_receive_total_is_the_senders_tiles(self):
        # The two roles have to agree on one number: the Attention rank writes
        # ``tile`` rows and the FFN rank sizes its receive by ``tiles * tile``,
        # which the operator divides back into one tile per peer.
        counts = [24, 24]
        tile = attention_tile_rows(counts, ffn_rank=0, attention_size=4, ffn_size=2)
        assert tile == 24
        assert ffn_tile_count(attention_size=4, ffn_size=2) * tile == 48
        assert ffn_receive_rows(counts, 0, attention_size=4, ffn_size=2) == 48
