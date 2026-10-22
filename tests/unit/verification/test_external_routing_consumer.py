# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""The reference's routing alignment, checked on its own.

The reference exists to be independent of the candidate, so its alignment must not
share code with the candidate. These tests drive it directly with row markers that
identify a rank and weights that differ per cell: a collective that mixes ranks, or an
alignment that reorders rows, changes which marker lands where, which a uniform fixture
would hide.

Scope: DP-level alignment of routing rows. The collective is a stand-in; the real
NCCL/HCCL behaviour is a device concern.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from afd_plugin.verification.external_routing_consumer import (  # noqa: E402
    _align_routing_independently,
)

TOP_K = 2

# A value no rank-N marker can produce (those stay below 1000), so the two halves of a
# concatenation are never confusable.
_RANK0_MARK = 90000


def _marked(rows: int, rank: int):
    """Routing whose ids and weights both identify the rank and the row."""

    base = torch.arange(rows * TOP_K, dtype=torch.int32).reshape(rows, TOP_K)
    ids = base + rank * 100
    # A distinct value per cell, so swapping two rows changes the values.
    weights = (base.to(torch.float32) + 1.0 + rank * 0.5) / 100.0
    return ids, weights


def test_padding_rows_are_zero_and_real_rows_keep_their_order():
    ids, weights = _marked(rows=3, rank=0)

    padded_ids, padded_weights = _align_routing_independently(
        ids,
        weights,
        pad_size=2,
        gather=None,
    )

    assert padded_ids.shape[0] == 5
    assert torch.equal(padded_ids[:3], ids)
    assert torch.count_nonzero(padded_ids[3:]) == 0
    # Weights are not constant, so reading the wrong rows would change the values.
    assert torch.equal(padded_weights[:3], weights)
    assert torch.count_nonzero(padded_weights[3:]) == 0


def test_gather_receives_ids_and_weights_and_its_results_are_returned():
    """Both tensors must travel through the same collective, in the same call shape."""

    ids, weights = _marked(rows=2, rank=1)
    seen: list[torch.Tensor] = []

    def gather(tensor: torch.Tensor) -> torch.Tensor:
        seen.append(tensor)
        return tensor + 1000

    gathered_ids, gathered_weights = _align_routing_independently(
        ids,
        weights,
        pad_size=0,
        gather=gather,
    )

    assert len(seen) == 2
    assert torch.equal(seen[0], ids)
    assert torch.equal(seen[1], weights)
    # The gathered tensors are what comes back, not the pre-collective ones.
    assert torch.equal(gathered_ids, ids + 1000)
    assert torch.equal(gathered_weights, weights + 1000)


def test_rank_markers_stay_in_rank_order_after_concatenation():
    """A collective that mixed ranks would put a later rank's marker first."""

    def gather(tensor: torch.Tensor) -> torch.Tensor:
        # Rank 0's marker is far outside any rank-1 value, so the two halves of the
        # concatenation can never be confused for one another.
        from_rank_zero = torch.full_like(tensor, _RANK0_MARK)
        return torch.cat([from_rank_zero, tensor], dim=0)

    ids, weights = _marked(rows=2, rank=1)

    gathered_ids, gathered_weights = _align_routing_independently(
        ids,
        weights,
        pad_size=0,
        gather=gather,
    )

    assert gathered_ids.shape[0] == 4
    # Compare whole rows, not one column: a swap between the two halves keeps any
    # single repeated value in place and would slip through.
    assert torch.equal(gathered_ids[:2], torch.full_like(ids, _RANK0_MARK))
    assert torch.equal(gathered_ids[2:], ids)
    assert torch.equal(gathered_weights[2:], weights)
    assert not torch.equal(gathered_weights[:2], gathered_weights[2:])


def _swap_rows(gathered: torch.Tensor) -> torch.Tensor:
    """Reorder the two ranks, the defect the marker test has to catch."""

    half = gathered.shape[0] // 2
    return torch.cat([gathered[half:], gathered[:half]], dim=0)


def test_the_rank_marker_test_fails_when_the_collective_reorders_rows():
    """A mutation check: the previous assertion must not survive a row swap."""

    def gather(tensor: torch.Tensor) -> torch.Tensor:
        from_rank_zero = torch.full_like(tensor, _RANK0_MARK)
        return _swap_rows(torch.cat([from_rank_zero, tensor], dim=0))

    ids, weights = _marked(rows=2, rank=1)

    gathered_ids, _ = _align_routing_independently(
        ids,
        weights,
        pad_size=0,
        gather=gather,
    )

    assert not torch.equal(gathered_ids[2:], ids)


@pytest.mark.parametrize("pad_size", [-1, -8])
def test_a_negative_pad_size_is_rejected(pad_size):
    ids, weights = _marked(rows=1, rank=0)

    with pytest.raises(ValueError, match="negative"):
        _align_routing_independently(ids, weights, pad_size=pad_size, gather=None)


def test_zero_padding_without_a_collective_returns_the_input_unchanged():
    ids, weights = _marked(rows=2, rank=3)

    out_ids, out_weights = _align_routing_independently(
        ids,
        weights,
        pad_size=0,
        gather=None,
    )

    assert torch.equal(out_ids, ids)
    assert torch.equal(out_weights, weights)
