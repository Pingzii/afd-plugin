# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for CAMP2P token-id transport helpers.

These helpers are pure tensor logic, so they are testable without an Ascend
device. The surrounding connector needs ``torch_npu`` and is covered by
``test_camp2p_connector.py`` instead.

The module-level ``vllm`` stub mirrors the pattern used by the compat patch
tests: importing any connector module pulls in ``vllm`` through
``afd_plugin.connectors``, so the import is satisfied with the minimal surface
the helpers' module needs.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import types
from collections.abc import Iterator

import pytest

torch = pytest.importorskip("torch")

_STUB_MODULES = (
    "vllm",
    "vllm.forward_context",
    "vllm.logger",
    "vllm.utils",
    "vllm.utils.torch_utils",
    "vllm.distributed",
    "vllm.distributed.parallel_state",
)


@contextlib.contextmanager
def _vllm_stub() -> Iterator[None]:
    """Expose a minimal ``vllm`` surface for the duration of one import.

    Importing any connector module pulls in ``vllm`` through
    ``afd_plugin.connectors``. This stub supplies only the names the helpers'
    module needs at import time.

    The stub is removed again immediately afterwards. Leaving a partial
    ``vllm`` in ``sys.modules`` would mask the real "vllm is not installed"
    failure for every other test module in the same pytest session, turning a
    clear error into a confusing one.
    """

    missing = [name for name in _STUB_MODULES if name not in sys.modules]
    if not missing:
        yield
        return

    saved = {name: sys.modules.get(name) for name in missing}

    def make(name: str, **attributes: object) -> None:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        sys.modules[name] = module

    make("vllm")
    make(
        "vllm.forward_context",
        DPMetadata=type("DPMetadata", (), {}),
        get_forward_context=lambda: None,
    )
    make("vllm.logger", init_logger=lambda *args, **kwargs: logging.getLogger("test"))
    make("vllm.utils")
    make(
        "vllm.utils.torch_utils",
        direct_register_custom_op=lambda **kwargs: None,
        is_torch_equal_or_newer=lambda *args, **kwargs: True,
    )
    make("vllm.distributed")
    make(
        "vllm.distributed.parallel_state",
        get_pcp_group=None,
        get_tensor_model_parallel_rank=None,
    )
    try:
        yield
    finally:
        for name in missing:
            sys.modules.pop(name, None)
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module


with _vllm_stub():
    from afd_plugin.connectors.npu.camp2p import (  # noqa: E402
        prepare_token_id_transfer,
        received_token_ids,
    )


def test_prepare_token_id_transfer_replicates_ids_across_columns():
    input_ids = torch.tensor([7, 11, 13], dtype=torch.int64)

    ids, scales = prepare_token_id_transfer(
        input_ids,
        topk=2,
        expected_tokens=3,
    )

    assert ids.dtype == torch.int32
    assert tuple(ids.shape) == (3, 2)
    assert ids[:, 0].tolist() == [7, 11, 13]
    assert ids[:, 1].tolist() == [7, 11, 13]
    assert scales.dtype == torch.float32
    assert tuple(scales.shape) == (3, 2)
    # Scales accompany token identity, not routing weights, so they are inert.
    assert torch.count_nonzero(scales) == 0


def test_prepare_token_id_transfer_rejects_token_count_mismatch():
    input_ids = torch.tensor([7, 11], dtype=torch.int32)

    with pytest.raises(ValueError, match="does not match the AFD transfer"):
        prepare_token_id_transfer(input_ids, topk=2, expected_tokens=3)


def test_prepare_token_id_transfer_rejects_non_vector_input():
    input_ids = torch.tensor([[7, 11]], dtype=torch.int32)

    with pytest.raises(ValueError, match="must be one-dimensional"):
        prepare_token_id_transfer(input_ids, topk=2, expected_tokens=1)


def test_prepare_token_id_transfer_rejects_float_input():
    input_ids = torch.tensor([7.0, 11.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="must be an integer tensor"):
        prepare_token_id_transfer(input_ids, topk=2, expected_tokens=2)


def test_prepare_token_id_transfer_rejects_non_positive_topk():
    input_ids = torch.tensor([7], dtype=torch.int32)

    with pytest.raises(ValueError, match="topk must be positive"):
        prepare_token_id_transfer(input_ids, topk=0, expected_tokens=1)


def test_received_token_ids_collapses_replicated_columns():
    sent_ids, _ = prepare_token_id_transfer(
        torch.tensor([7, 11, 13], dtype=torch.int32),
        topk=2,
        expected_tokens=3,
    )

    received = received_token_ids(sent_ids, expected_tokens=3, topk=2)

    assert received.dtype == torch.int32
    assert received.tolist() == [7, 11, 13]


def test_received_token_ids_trims_operator_padded_capacity():
    # The operator works on a padded capacity, so extra trailing rows are normal.
    sent_ids, _ = prepare_token_id_transfer(
        torch.tensor([7, 11, 13, 99, 99], dtype=torch.int32),
        topk=2,
        expected_tokens=5,
    )

    received = received_token_ids(sent_ids, expected_tokens=3, topk=2)

    assert received.tolist() == [7, 11, 13]


def test_received_token_ids_rejects_alignment_shortfall():
    """Misaligned ids must fail loudly rather than route the wrong tokens."""
    sent_ids, _ = prepare_token_id_transfer(
        torch.tensor([7, 11], dtype=torch.int32),
        topk=2,
        expected_tokens=2,
    )

    with pytest.raises(ValueError, match="not aligned with the FFN token layout"):
        received_token_ids(sent_ids, expected_tokens=5, topk=2)


def test_received_token_ids_rejects_wrong_topk():
    sent_ids, _ = prepare_token_id_transfer(
        torch.tensor([7, 11], dtype=torch.int32),
        topk=4,
        expected_tokens=2,
    )

    with pytest.raises(ValueError, match="column count 4 does not"):
        received_token_ids(sent_ids, expected_tokens=2, topk=2)


def test_received_token_ids_rejects_non_matrix_input():
    flat = torch.tensor([7, 11], dtype=torch.int32)

    with pytest.raises(ValueError, match="must be two-dimensional"):
        received_token_ids(flat, expected_tokens=2, topk=1)
