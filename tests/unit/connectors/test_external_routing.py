# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Alignment contract for routing computed on the Attention side.

The module under test needs only torch, but importing it through
``afd_plugin.connectors`` pulls vLLM in through the package ``__init__``. Loading
it by file path keeps these checks runnable on a CPU-only host, which is where
they earn their keep: they found a padding bug that a device run would have shown
as a silent numeric drift.
"""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "afd_plugin"
    / "connectors"
    / "npu"
    / "external_routing.py"
)

# Uneven local row counts: padding to the maximum is what makes them comparable.
LOCAL_ROWS_BY_RANK = (3, 1, 2, 2)
TOP_K = 2
SCALE_COLS = 2
BITS_PER_SCALE = 1

# Valid E8M0 bytes that are not all the padding byte, so an implementation that
# overwrote every scale row with the padding fill would fail these checks.
LEGAL_SCALE_BYTES = (120, 124, 127, 130, 133)


def _load_module():
    spec = importlib.util.spec_from_file_location("afd_external_routing", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def routing_module():
    return _load_module()


@pytest.fixture()
def inputs_type(routing_module):
    return routing_module.AFDExternalRoutingInputs


@pytest.fixture()
def pad_byte(routing_module):
    return routing_module.SCALE_PAD_BYTE


def _scale(rows: int):
    """Legal E8M0 bytes varying per row and column."""

    values = [
        [
            LEGAL_SCALE_BYTES[(row * SCALE_COLS + col) % len(LEGAL_SCALE_BYTES)]
            for col in range(SCALE_COLS)
        ]
        for row in range(rows)
    ]
    return torch.tensor(values, dtype=torch.uint8).view(torch.float8_e8m0fnu)


def _inputs(inputs_type, rows: int, rank: int = 0):
    """Routing whose ids encode the origin rank and row index."""

    ids = (
        torch.arange(rows * TOP_K, dtype=torch.int32).reshape(rows, TOP_K)
        + rank * 1000
    )
    weights = torch.arange(rows * TOP_K, dtype=torch.float32).reshape(rows, TOP_K) + rank
    return inputs_type(topk_ids=ids, topk_weights=weights, scale=_scale(rows))


def _repeat_gather(tensor):
    """Fake collective: the same shard arriving from every rank."""

    return tensor.repeat(len(LOCAL_ROWS_BY_RANK), *([1] * (tensor.dim() - 1)))


def test_padding_appends_inert_rows_and_keeps_valid_ones(inputs_type, pad_byte):
    inputs = _inputs(inputs_type, rows=2)
    padded = inputs.padded(3)

    assert pad_byte == 127
    assert padded.token_count == 5
    assert torch.equal(padded.topk_ids[:2], inputs.topk_ids)
    assert torch.equal(padded.topk_weights[:2], inputs.topk_weights)
    assert torch.equal(
        padded.scale.view(torch.uint8)[:2],
        inputs.scale.view(torch.uint8),
    )
    assert torch.all(padded.topk_ids[2:] == 0)
    assert torch.all(padded.topk_weights[2:] == 0)
    assert torch.all(padded.scale.view(torch.uint8)[2:] == pad_byte)


def test_padding_zero_returns_the_same_object(inputs_type):
    inputs = _inputs(inputs_type, rows=2)
    assert inputs.padded(0) is inputs


def test_gathering_preserves_ids_weights_and_scale_bytes(inputs_type):
    inputs = _inputs(inputs_type, rows=3)
    gathered = inputs.gathered(_repeat_gather)

    assert gathered.token_count == 3 * len(LOCAL_ROWS_BY_RANK)
    assert torch.equal(gathered.topk_ids[:3], inputs.topk_ids)
    assert torch.equal(gathered.topk_weights[:3], inputs.topk_weights)
    assert gathered.scale.dtype == inputs.scale.dtype
    assert torch.equal(
        gathered.scale.view(torch.uint8)[:3],
        inputs.scale.view(torch.uint8),
    )


def test_gathering_without_scale_keeps_scale_absent(inputs_type):
    inputs = inputs_type(
        topk_ids=torch.zeros((2, TOP_K), dtype=torch.int32),
        topk_weights=torch.zeros((2, TOP_K), dtype=torch.float32),
    )
    gathered = inputs.gathered(_repeat_gather)

    assert gathered.scale is None
    assert gathered.token_count == 2 * len(LOCAL_ROWS_BY_RANK)


@pytest.mark.parametrize(("rank", "rows"), list(enumerate(LOCAL_ROWS_BY_RANK)))
def test_padded_routing_still_pairs_with_its_activation_row(
    inputs_type,
    rank,
    rows,
):
    max_rows = max(LOCAL_ROWS_BY_RANK)
    pad_size = max_rows - rows
    padded = _inputs(inputs_type, rows=rows, rank=rank).padded(pad_size)

    # Tag each activation row with its own index, then apply the same padding.
    activation = torch.arange(rows, dtype=torch.float32).reshape(rows, 1).repeat(1, 3)
    activation = torch.cat([activation, torch.zeros(pad_size, 3)], dim=0)

    assert padded.token_count == activation.shape[0]
    for row in range(rows):
        assert activation[row, 0].item() == row
        assert padded.topk_ids[row, 0].item() == rank * 1000 + row * TOP_K
    for row in range(rows, max_rows):
        assert activation[row].abs().max().item() == 0


def test_non_matrix_routing_is_rejected(inputs_type):
    with pytest.raises(ValueError, match="2-D"):
        inputs_type(
            topk_ids=torch.zeros(3, dtype=torch.int32),
            topk_weights=torch.zeros(3, dtype=torch.float32),
        )


def test_float_scale_is_rejected(inputs_type):
    with pytest.raises(ValueError, match="E8M0"):
        inputs_type(
            topk_ids=torch.zeros((2, TOP_K), dtype=torch.int32),
            topk_weights=torch.zeros((2, TOP_K), dtype=torch.float32),
            scale=torch.full((2, SCALE_COLS), 1.0, dtype=torch.float32),
        )


def test_mismatched_topk_shapes_are_rejected(inputs_type):
    with pytest.raises(ValueError, match="same shape"):
        inputs_type(
            topk_ids=torch.zeros((2, 2), dtype=torch.int32),
            topk_weights=torch.zeros((2, 3), dtype=torch.float32),
        )


def test_scale_with_wrong_row_count_is_rejected(inputs_type):
    with pytest.raises(ValueError, match="one row per token"):
        inputs_type(
            topk_ids=torch.zeros((2, TOP_K), dtype=torch.int32),
            topk_weights=torch.zeros((2, TOP_K), dtype=torch.float32),
            scale=_scale(3),
        )


def test_expert_indices_must_not_be_floating(inputs_type):
    with pytest.raises(ValueError, match="expert indices"):
        inputs_type(
            topk_ids=torch.zeros((2, TOP_K), dtype=torch.float32),
            topk_weights=torch.zeros((2, TOP_K), dtype=torch.float32),
        )
