# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for the DSV4 Attention-side Hash id selection helper.

``local_hash_input_ids`` decides which token ids Attention sends towards the FFN
role for a Hash layer. The helper is pure tensor logic, which matters here: an
ids/token misalignment would let a token-keyed router select experts for the
wrong tokens without raising anywhere, so the alignment behaviour is worth
testing without an Ascend device.
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
)


@contextlib.contextmanager
def _vllm_stub() -> Iterator[None]:
    """Expose a minimal ``vllm`` surface for the duration of one import.

    Importing the model package pulls in ``vllm.forward_context``. The stub is
    removed again immediately afterwards so a partial ``vllm`` cannot mask the
    real "vllm is not installed" failure for other test modules in the same
    session.
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
        ForwardContext=type("ForwardContext", (), {}),
        get_forward_context=lambda: None,
    )
    make(
        "vllm.logger",
        init_logger=lambda *args, **kwargs: logging.getLogger("test"),
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
    from afd_plugin.model_executor.models.npu.deepseek_v4_attention_gate import (
        hash_input_ids_from_context,
        local_hash_input_ids,
    )


def _make_model(**attributes: object) -> types.SimpleNamespace:
    return types.SimpleNamespace(**attributes)


def test_returns_global_ids_when_no_flash_comm_split_is_needed():
    ids = torch.tensor([5, 6, 7], dtype=torch.int32)

    result = local_hash_input_ids(
        input_ids=ids,
        router_tokens=3,
        flash_comm_v1_enabled=False,
        pad_size=0,
    )

    assert result.dtype == torch.int64
    assert result.tolist() == [5, 6, 7]


def test_flattens_multi_dimensional_ids():
    ids = torch.tensor([[5, 6], [7, 8]], dtype=torch.int64)

    result = local_hash_input_ids(
        input_ids=ids,
        router_tokens=4,
        flash_comm_v1_enabled=False,
        pad_size=0,
    )

    assert result.tolist() == [5, 6, 7, 8]


def test_rejects_missing_ids():
    with pytest.raises(RuntimeError, match="requires input_ids to send"):
        local_hash_input_ids(
            input_ids=None,
            router_tokens=3,
            flash_comm_v1_enabled=False,
            pad_size=0,
        )


def test_rejects_unalignable_token_count():
    """A count mismatch must fail here, before any cross-role transfer."""
    ids = torch.tensor([5, 6], dtype=torch.int64)

    with pytest.raises(RuntimeError, match="cannot align the ids sent to FFN"):
        local_hash_input_ids(
            input_ids=ids,
            router_tokens=3,
            flash_comm_v1_enabled=False,
            pad_size=0,
        )


def test_applies_flash_comm_padding_and_tp_split(monkeypatch):
    """FlashComm v1 must slice ids exactly like the router logits it mirrors."""
    captured: dict[str, object] = {}

    def fake_split(tensor: torch.Tensor, *, num_partitions: int, **kwargs: object):
        captured["num_partitions"] = num_partitions
        captured["kwargs"] = kwargs
        return list(torch.chunk(tensor, num_partitions))

    distributed = types.ModuleType("vllm.distributed")
    distributed.get_tp_group = lambda: _make_model(world_size=2, rank_in_group=1)
    ascend_distributed = types.ModuleType("vllm_ascend.distributed")
    ascend_utils = types.ModuleType("vllm_ascend.distributed.utils")
    ascend_utils.split_tensor_along_first_dim = fake_split
    ascend_distributed.utils = ascend_utils

    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    monkeypatch.setitem(sys.modules, "vllm_ascend.distributed", ascend_distributed)
    monkeypatch.setitem(sys.modules, "vllm_ascend.distributed.utils", ascend_utils)

    # 4 global ids plus one padded slot gives 5. torch.chunk(5, 2) splits that
    # into [3, 2] rather than evenly, so rank 1 receives the trailing pair.
    result = local_hash_input_ids(
        input_ids=torch.tensor([5, 6, 7, 8], dtype=torch.int64),
        router_tokens=2,
        flash_comm_v1_enabled=True,
        pad_size=1,
    )

    assert captured["num_partitions"] == 2
    assert captured["kwargs"] == {"contiguous_split_chunks": True}
    assert result.tolist() == [8, 0]


def test_context_without_ids_is_an_error_not_a_quiet_fallback():
    """A context with no ids must fail here, not downgrade the transfer.

    The FFN role decides once per run that it expects ids and cannot tell Hash
    layers from non-Hash ones, so "this layer does not need them" is not a case
    that can be answered by sending activations alone.
    """

    with pytest.raises(RuntimeError, match="requires input_ids to send"):
        hash_input_ids_from_context(
            forward_context=_make_model(input_ids=None),
            router_tokens=3,
        )


def test_context_that_omits_the_attribute_is_an_error():
    with pytest.raises(RuntimeError, match="requires input_ids to send"):
        hash_input_ids_from_context(
            forward_context=_make_model(),
            router_tokens=3,
        )


def test_context_ids_are_returned_as_router_aligned_tokens():
    result = hash_input_ids_from_context(
        forward_context=_make_model(
            input_ids=torch.tensor([5, 6, 7], dtype=torch.int32),
            flash_comm_v1_enabled=False,
            pad_size=0,
        ),
        router_tokens=3,
    )

    assert result.tolist() == [5, 6, 7]


def test_context_ids_are_still_validated_against_the_local_token_count():
    """Carrying ids opts into validation; a mismatch must not pass silently."""
    with pytest.raises(RuntimeError, match="cannot align the ids sent to FFN"):
        hash_input_ids_from_context(
            forward_context=_make_model(
                input_ids=torch.tensor([5, 6], dtype=torch.int32),
                flash_comm_v1_enabled=False,
                pad_size=0,
            ),
            router_tokens=3,
        )
