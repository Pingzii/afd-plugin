"""The profile flag must be read inside the custom op, not traced."""

from __future__ import annotations

import torch
import pytest

# Needs a full vLLM install: vllm.utils is what registers custom ops.
pytest.importorskip("vllm.utils")

from afd_plugin.compat.npu import routing_ids  # noqa: E402,F401  registers the op

NUM_EXPERTS = 8
TOKENS = 4
TOPK = 2


def _ids(offset):
    flat = torch.arange(offset, offset + TOKENS * TOPK, dtype=torch.int32)
    return (flat % NUM_EXPERTS).reshape(TOKENS, TOPK)


def _balanced():
    return [i % NUM_EXPERTS for i in range(TOKENS * TOPK)]


def test_balances_in_profile_and_is_inert_otherwise(monkeypatch):
    monkeypatch.setattr(routing_ids, "_afd_in_profile_run", lambda: True)
    profile_ids = _ids(3)
    torch.ops.vllm.afd_balance_routing_ids_(profile_ids, NUM_EXPERTS, False)
    assert profile_ids.flatten().tolist() == _balanced()

    monkeypatch.setattr(routing_ids, "_afd_in_profile_run", lambda: False)
    live_ids = _ids(3)
    expected = live_ids.clone()
    torch.ops.vllm.afd_balance_routing_ids_(live_ids, NUM_EXPERTS, False)
    assert torch.equal(live_ids, expected)


def test_explicit_force_balances_outside_profile(monkeypatch):
    monkeypatch.setattr(routing_ids, "_afd_in_profile_run", lambda: False)
    forced = _ids(5)
    torch.ops.vllm.afd_balance_routing_ids_(forced, NUM_EXPERTS, True)
    assert forced.flatten().tolist() == _balanced()


class _ThroughOp(torch.nn.Module):
    def forward(self, ids):
        torch.ops.vllm.afd_balance_routing_ids_(ids, NUM_EXPERTS, False)
        return ids.clone()


class _PlainPython(torch.nn.Module):
    def forward(self, ids):
        if routing_ids._afd_in_profile_run():
            from afd_plugin.model_executor.models.npu.deepseek_v2_attention_gate import (
                _force_balanced_topk_ids,
            )

            _force_balanced_topk_ids(ids, num_logical_experts=NUM_EXPERTS)
        return ids.clone()


def _compile_once(module):
    compiled = []

    def backend(gm, example_inputs):
        fn = torch._dynamo.backends.registry.lookup_backend("eager")(gm, example_inputs)
        compiled.append((fn, example_inputs))
        return fn

    torch.compile(module, backend=backend)(_ids(9))
    assert len(compiled) == 1, "test setup must compile exactly once"
    return compiled[0]


@pytest.mark.parametrize("module_cls", [_ThroughOp, _PlainPython])
def test_reuse_without_guards_keeps_live_routing(monkeypatch, module_cls):
    monkeypatch.setattr(routing_ids, "_afd_in_profile_run", lambda: True)
    compiled_fn, example_inputs = _compile_once(module_cls())

    monkeypatch.setattr(routing_ids, "_afd_in_profile_run", lambda: False)
    live = _ids(11)
    expected = live.clone()
    compiled_fn(live, *example_inputs[1:])
    if module_cls is _ThroughOp:
        assert torch.equal(live, expected)
    else:
        # Negative control: the traced branch stays frozen on profile balancing.
        assert live.flatten().tolist() == _balanced()
