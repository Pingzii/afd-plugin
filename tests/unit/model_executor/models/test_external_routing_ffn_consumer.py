# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""The sync FFN consumer: what it uses, what it returns, and what it leaves behind.

Three properties matter here and none of them is "each collaborator was called
once":

- the values handed to ``apply`` are the ones ``prepare`` aligned, not the raw
  transport values, so a missing alignment step cannot pass unnoticed;
- ``finalize``'s result is what the caller receives;
- every exit path clears the stash and the experts module's per-call state, and a
  later call cannot inherit routing from an earlier one.

Skipped on a CPU-only host: the consumer needs vLLM-Ascend symbols at call time.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm.forward_context")

from afd_plugin.model_executor.models.npu import deepseek_v4 as adapter  # noqa: E402

TOP_K = 2
HIDDEN = 4
RAW_ID = 7
ALIGNED_ID = 99


class _AllGatherCommImpl:
    """Marker type the consumer accepts."""


class _Mc2CommImpl:
    """Marker type for the other wired communication method."""


class _PrepareFinalize:
    """Carries the stash the prepare patch would read and replace."""

    def __init__(self) -> None:
        self.afd_external_routing = None
        self.prepare_calls = 0


class _CommBehaviour:
    """The flow both wired methods share: record, serve aligned routing, apply."""

    def __init__(self, *, fail_in: str | None = None) -> None:
        self.prepare_finalize = _PrepareFinalize()
        self.fail_in = fail_in
        self.applied: dict[str, object] = {}
        self.events: list[str] = []

    def prepare(self, *, hidden_states, router_logits, replace_allreduce, quant_type):
        del router_logits, replace_allreduce, quant_type
        self.events.append("prepare")
        if self.fail_in == "prepare":
            raise RuntimeError("prepare exploded")
        stash = self.prepare_finalize.afd_external_routing
        assert stash is not None, "the consumer must publish routing before prepare"
        # Deliberately different from the transport values: anything downstream that
        # kept using the raw routing is now visible in the result. The scale travels
        # with the routing, as the real patch publishes it.
        self.prepare_finalize.afd_external_routing = type(stash)(
            topk_ids=torch.full_like(stash.topk_ids, ALIGNED_ID),
            topk_weights=stash.topk_weights,
            scale=stash.scale,
        )
        return SimpleNamespace(
            hidden_states=hidden_states,
            mc2_mask=None,
            # The real patch publishes the stash's scale here, and the consumer installs
            # it on the experts for the MoE to read.
            pertoken_scale=stash.scale,
            padded_hidden_states_shape=hidden_states.shape,
        )

    def finalize(self, *, hidden_states, reduce_results, padded_hidden_states_shape):
        del reduce_results, padded_hidden_states_shape
        self.events.append("finalize")
        if self.fail_in == "finalize":
            raise RuntimeError("finalize exploded")
        return hidden_states


class _Comm(_CommBehaviour, _AllGatherCommImpl):
    """The ALLGATHER case: same behaviour, ALLGATHER marker."""


def _experts(comm: _Comm):
    def apply(*, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        del layer, shared_experts, shared_experts_input
        comm.events.append("apply")
        if comm.fail_in == "apply":
            raise RuntimeError("apply exploded")
        comm.applied["topk_ids"] = topk_ids
        comm.applied["topk_weights"] = topk_weights
        # What the MoE would read to decide whether to quantize its input. Recording
        # it here is how a test sees the scale the consumer installed.
        comm.applied["pertoken_scale"] = experts.ascend_pertoken_scale
        # A marker of the right shape, built without arithmetic on ``x``: adding to an
        # FP8 activation is not implemented on every backend, and the value here only
        # has to be identifiable in the assertion.
        return SimpleNamespace(
            routed_out=torch.ones(
                x.shape[0],
                x.shape[1],
                dtype=torch.float32,
                device=x.device,
            ),
        )

    experts = SimpleNamespace(
        moe_config=SimpleNamespace(is_sequence_parallel=False),
        quant_type="quant",
        quant_method=SimpleNamespace(apply=apply),
        ascend_pertoken_scale="leaked",
        ascend_mc2_mask="leaked",
    )
    return experts


def _layer(comm, experts):
    layer = object.__new__(adapter.AFDDeepseekV4DecoderLayer)
    layer.mlp = SimpleNamespace(experts=SimpleNamespace(routed_experts=experts))
    # Set by the real constructor, and read by the pass-through diagnostic. A fake
    # without it fails before reaching the behaviour under test.
    layer.layer_idx = 0
    return layer


def _install_vllm_ascend(monkeypatch, comm):
    modules = {}
    for name in (
        "vllm_ascend",
        "vllm_ascend.ascend_forward_context",
        "vllm_ascend.ops",
        "vllm_ascend.ops.fused_moe",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        modules[name] = module
    modules["vllm_ascend.ascend_forward_context"]._EXTRA_CTX = SimpleNamespace(
        moe_comm_method=comm,
    )
    method_module = ModuleType("vllm_ascend.ops.fused_moe.moe_comm_method")
    # Both wired methods have to exist: the consumer imports them to test membership,
    # so a fake without MC2CommImpl fails at import rather than at the assertion.
    method_module.AllGatherCommImpl = _AllGatherCommImpl
    method_module.MC2CommImpl = _Mc2CommImpl
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.moe_comm_method",
        method_module,
    )


def _raw_routing():
    return (
        torch.full((2, TOP_K), RAW_ID, dtype=torch.int32),
        torch.full((2, TOP_K), 0.5, dtype=torch.float32),
    )


def test_apply_consumes_the_aligned_routing_not_the_transport_values(monkeypatch):
    """A missing alignment step must change the result, not just the call order."""

    comm = _Comm()
    _install_vllm_ascend(monkeypatch, comm)
    experts = _experts(comm)
    layer = _layer(comm, experts)
    topk_ids, topk_weights = _raw_routing()

    output = layer._compute_external_routing_ffn(
        torch.zeros(2, HIDDEN),
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )

    assert comm.events == ["prepare", "apply", "finalize"]
    assert torch.equal(comm.applied["topk_ids"], torch.full_like(topk_ids, ALIGNED_ID))
    assert torch.equal(comm.applied["topk_weights"], topk_weights)
    # finalize's result is returned unchanged.
    assert torch.equal(output, torch.ones(2, HIDDEN))
    # Per-call state is cleared after a clean run too.
    assert experts.ascend_pertoken_scale is None
    assert experts.ascend_mc2_mask is None
    assert comm.prepare_finalize.afd_external_routing is None


@pytest.mark.parametrize("fail_in", ["prepare", "apply", "finalize"])
def test_every_failure_stage_clears_state(monkeypatch, fail_in):
    comm = _Comm(fail_in=fail_in)
    _install_vllm_ascend(monkeypatch, comm)
    experts = _experts(comm)
    layer = _layer(comm, experts)
    topk_ids, topk_weights = _raw_routing()

    with pytest.raises(RuntimeError, match=f"{fail_in} exploded"):
        layer._compute_external_routing_ffn(
            torch.zeros(2, HIDDEN),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    assert experts.ascend_pertoken_scale is None
    assert experts.ascend_mc2_mask is None
    assert comm.prepare_finalize.afd_external_routing is None


def test_a_later_call_does_not_inherit_the_previous_routing(monkeypatch):
    failing = _Comm(fail_in="prepare")
    _install_vllm_ascend(monkeypatch, failing)
    experts = _experts(failing)
    layer = _layer(failing, experts)
    topk_ids, topk_weights = _raw_routing()
    with pytest.raises(RuntimeError):
        layer._compute_external_routing_ffn(
            torch.zeros(2, HIDDEN),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    healthy = _Comm()
    _install_vllm_ascend(monkeypatch, healthy)
    layer.mlp = SimpleNamespace(experts=SimpleNamespace(routed_experts=_experts(healthy)))

    layer._compute_external_routing_ffn(
        torch.zeros(2, HIDDEN),
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )

    assert healthy.applied["topk_ids"] is not None
    assert healthy.prepare_finalize.afd_external_routing is None


def test_an_unwired_comm_method_is_rejected(monkeypatch):
    """MC2 is wired; the all-to-all paths are not, and must not be guessed at."""

    comm = SimpleNamespace(prepare_finalize=_PrepareFinalize())
    _install_vllm_ascend(monkeypatch, comm)
    layer = _layer(comm, _experts(_Comm()))
    topk_ids, topk_weights = _raw_routing()

    with pytest.raises(RuntimeError, match="ALLGATHER and MC2"):
        layer._compute_external_routing_ffn(
            torch.zeros(2, HIDDEN),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )


class _Mc2Comm(_CommBehaviour, _Mc2CommImpl):
    """The MC2 case: same behaviour, MC2 marker, so only the reduce decision differs."""


@pytest.mark.parametrize("comm_type", [_Comm, _Mc2Comm])
def test_both_wired_comm_methods_run_the_same_flow(monkeypatch, comm_type):
    """MC2 shares the consumer flow, so it must apply and finalize the same way."""

    comm = comm_type()
    _install_vllm_ascend(monkeypatch, comm)
    experts = _experts(comm)
    layer = _layer(comm, experts)
    topk_ids, topk_weights = _raw_routing()

    output = layer._compute_external_routing_ffn(
        torch.zeros(2, HIDDEN),
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )

    assert comm.events == ["prepare", "apply", "finalize"]
    assert torch.equal(comm.applied["topk_ids"], torch.full_like(topk_ids, ALIGNED_ID))
    assert torch.equal(output, torch.ones(2, HIDDEN))
    assert experts.ascend_pertoken_scale is None
    assert experts.ascend_mc2_mask is None
    assert comm.prepare_finalize.afd_external_routing is None


@pytest.mark.parametrize("comm_type", [_Comm, _Mc2Comm])
def test_both_wired_comm_methods_clear_state_after_a_failure(monkeypatch, comm_type):
    comm = comm_type(fail_in="apply")
    _install_vllm_ascend(monkeypatch, comm)
    experts = _experts(comm)
    layer = _layer(comm, experts)
    topk_ids, topk_weights = _raw_routing()

    with pytest.raises(RuntimeError, match="apply exploded"):
        layer._compute_external_routing_ffn(
            torch.zeros(2, HIDDEN),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    assert experts.ascend_pertoken_scale is None
    assert experts.ascend_mc2_mask is None
    assert comm.prepare_finalize.afd_external_routing is None


def _fp8_activation(rows: int = 2):
    return torch.zeros((rows, HIDDEN), dtype=torch.float8_e4m3fn)


def _mx_scale(rows: int = 2, columns: int = -(-HIDDEN // 32)):
    return torch.zeros((rows, columns), dtype=torch.float8_e8m0fnu)


def test_a_quantized_activation_passes_through_on_allgather(monkeypatch):
    """The direct path: the MoE receives the FP8 activation and its scale together."""

    comm = _Comm()
    _install_vllm_ascend(monkeypatch, comm)
    experts = _experts(comm)
    layer = _layer(comm, experts)
    topk_ids, topk_weights = _raw_routing()
    scale = _mx_scale()

    layer._compute_external_routing_ffn(
        _fp8_activation(),
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        activation_scale=scale,
    )

    assert comm.events == ["prepare", "apply", "finalize"]
    # The scale published to the MoE is the one the transport carried.
    assert comm.applied["pertoken_scale"] is scale


def test_a_quantized_activation_is_refused_on_mc2(monkeypatch):
    """MC2 has never been given an FP8 activation; refusing here beats failing inside."""

    comm = _Mc2Comm()
    _install_vllm_ascend(monkeypatch, comm)
    layer = _layer(comm, _experts(comm))
    topk_ids, topk_weights = _raw_routing()

    with pytest.raises(RuntimeError, match="only wired for the ALLGATHER"):
        layer._compute_external_routing_ffn(
            _fp8_activation(),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            activation_scale=_mx_scale(),
        )

    assert comm.events == []


def test_bf16_on_mc2_is_still_allowed(monkeypatch):
    comm = _Mc2Comm()
    _install_vllm_ascend(monkeypatch, comm)
    layer = _layer(comm, _experts(comm))
    topk_ids, topk_weights = _raw_routing()

    layer._compute_external_routing_ffn(
        torch.zeros(2, HIDDEN),
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )

    assert comm.events == ["prepare", "apply", "finalize"]


def test_a_quantized_activation_without_a_scale_is_refused(monkeypatch):
    comm = _Comm()
    _install_vllm_ascend(monkeypatch, comm)
    layer = _layer(comm, _experts(comm))
    topk_ids, topk_weights = _raw_routing()

    with pytest.raises(RuntimeError, match="together"):
        layer._compute_external_routing_ffn(
            _fp8_activation(),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )


def test_a_scale_on_a_model_dtype_activation_is_refused(monkeypatch):
    """Otherwise the MoE would skip quantizing a tensor that is not quantized."""

    comm = _Comm()
    _install_vllm_ascend(monkeypatch, comm)
    layer = _layer(comm, _experts(comm))
    topk_ids, topk_weights = _raw_routing()

    with pytest.raises(RuntimeError, match="together"):
        layer._compute_external_routing_ffn(
            torch.zeros(2, HIDDEN),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            activation_scale=_mx_scale(),
        )


@pytest.mark.parametrize("dtype", [torch.float8_e5m2])
def test_a_non_target_fp8_dtype_is_refused(monkeypatch, dtype):
    """The weight path is defined for E4M3; E5M2 must not ride the same route."""

    comm = _Comm()
    _install_vllm_ascend(monkeypatch, comm)
    layer = _layer(comm, _experts(comm))
    topk_ids, topk_weights = _raw_routing()

    with pytest.raises(RuntimeError, match="E4M3"):
        layer._compute_external_routing_ffn(
            torch.zeros((2, HIDDEN), dtype=dtype),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            activation_scale=_mx_scale(),
        )


def test_a_scale_of_the_wrong_width_is_refused(monkeypatch):
    comm = _Comm()
    _install_vllm_ascend(monkeypatch, comm)
    layer = _layer(comm, _experts(comm))
    topk_ids, topk_weights = _raw_routing()

    with pytest.raises(RuntimeError, match="does not match the activation width"):
        layer._compute_external_routing_ffn(
            _fp8_activation(),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            activation_scale=_mx_scale(columns=64),
        )