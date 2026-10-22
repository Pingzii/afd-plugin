# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""What the reference entry point actually calls.

The alignment helper is covered separately. These tests exercise ``_run_reference``
itself, because the properties that make it a valid control are about which function it
reaches: the captured upstream prepare, the activation passed through unpadded, a
placeholder logits tensor of the legal shape, and nothing left behind when the expert
call fails.

``vllm_ascend`` is faked at the module level, so these run on a CPU host.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")

from afd_plugin.verification import external_routing_consumer as consumer  # noqa: E402

TOP_K = 2
NUM_EXPERTS = 8
LOCAL_ROWS = 3


class _AllGatherCommImpl:
    """Marker the reference checks with isinstance."""


class _Comm(_AllGatherCommImpl):
    def __init__(self):
        self.prepare_finalize = object()
        self.prepare_calls = 0
        self.finalize_calls = 0

    def prepare(self, **kwargs):
        del kwargs
        self.prepare_calls += 1
        raise AssertionError("the reference must not go through comm.prepare")

    def finalize(self, *, hidden_states, reduce_results, padded_hidden_states_shape):
        del reduce_results, padded_hidden_states_shape
        self.finalize_calls += 1
        return hidden_states


class _Upstream:
    """Records how the captured upstream prepare was called."""

    def __init__(self, rows_after_padding):
        self.calls: list[dict] = []
        self.rows_after_padding = rows_after_padding

    def __call__(self, self_arg, *, hidden_states, router_logits, replace_allreduce,
                 quant_type):
        del replace_allreduce, quant_type
        self.calls.append(
            {
                "self": self_arg,
                "rows": int(hidden_states.shape[0]),
                "logits_shape": tuple(router_logits.shape),
                "logits_dtype": router_logits.dtype,
                "logits_nonzero": int(torch.count_nonzero(router_logits)),
            },
        )
        return _PrepareOutput(
            hidden_states=torch.zeros(
                self.rows_after_padding,
                hidden_states.shape[1],
                dtype=hidden_states.dtype,
            ),
            mc2_mask=None,
            padded_hidden_states_shape=hidden_states.shape,
        )


class _PrepareOutput:
    def __init__(self, *, hidden_states, mc2_mask, padded_hidden_states_shape):
        self.hidden_states = hidden_states
        self.mc2_mask = mc2_mask
        self.padded_hidden_states_shape = padded_hidden_states_shape


class _Experts:
    def __init__(self):
        self.moe_config = type(
            "MoEConfig",
            (),
            {"num_experts": NUM_EXPERTS, "is_sequence_parallel": False},
        )()
        self.quant_type = "quant"
        self.ascend_pertoken_scale = "stale"
        self.ascend_mc2_mask = "stale"
        self.applied = None
        self.quant_method = type("QM", (), {"apply": self._apply})()

    def _apply(self, **kwargs):
        self.applied = kwargs
        return type("Fused", (), {"routed_out": kwargs["x"]})()


def _install_fakes(monkeypatch, upstream):
    prepare_module = ModuleType("afd_plugin.compat.patches.npu.external_routing_prepare")
    prepare_module.ORIGINAL_PREPARE_WITH_DP_GROUP = upstream
    monkeypatch.setitem(
        sys.modules,
        "afd_plugin.compat.patches.npu.external_routing_prepare",
        prepare_module,
    )
    method_module = ModuleType("vllm_ascend.ops.fused_moe.moe_comm_method")
    method_module.AllGatherCommImpl = _AllGatherCommImpl
    for name in ("vllm_ascend", "vllm_ascend.ops", "vllm_ascend.ops.fused_moe"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.moe_comm_method",
        method_module,
    )


def _run(comm, experts, *, pad_size=1, publish=True):
    activation = torch.ones(LOCAL_ROWS, 4, dtype=torch.bfloat16)
    topk_ids = torch.arange(LOCAL_ROWS * TOP_K, dtype=torch.int32).reshape(
        LOCAL_ROWS,
        TOP_K,
    )
    topk_weights = torch.full((LOCAL_ROWS, TOP_K), 0.25, dtype=torch.float32)
    return consumer._run_reference(
        object(),
        activation,
        topk_ids,
        topk_weights,
        plan=consumer._Plan(pad_size=pad_size),
        gather=None,
        experts=experts,
        comm=comm,
    )


def test_the_captured_upstream_prepare_runs_with_the_unpadded_activation(monkeypatch):
    """Activation alignment belongs to the upstream call, so it must arrive unpadded."""

    upstream = _Upstream(rows_after_padding=LOCAL_ROWS + 1)
    _install_fakes(monkeypatch, upstream)
    comm, experts = _Comm(), _Experts()

    _run(comm, experts)

    assert len(upstream.calls) == 1
    call = upstream.calls[0]
    # Unpadded: the reference must not gather the activation and then let the prepare
    # gather it again.
    assert call["rows"] == LOCAL_ROWS
    # The instance the upstream method treats as self.
    assert call["self"] is comm.prepare_finalize
    assert call["logits_shape"] == (LOCAL_ROWS, NUM_EXPERTS)
    assert call["logits_dtype"] is torch.float32
    assert call["logits_nonzero"] == 0
    assert comm.prepare_calls == 0
    assert comm.finalize_calls == 1


def test_the_aligned_routing_reaches_apply(monkeypatch):
    """The padded rows are what apply sees, and they come from the independent path."""

    upstream = _Upstream(rows_after_padding=LOCAL_ROWS + 1)
    _install_fakes(monkeypatch, upstream)
    comm, experts = _Comm(), _Experts()

    _run(comm, experts, pad_size=1)

    assert experts.applied is not None
    assert int(experts.applied["topk_ids"].shape[0]) == LOCAL_ROWS + 1
    assert experts.applied["topk_ids"].dtype is torch.int32
    # Padding rows carry no expert choice.
    assert int(torch.count_nonzero(experts.applied["topk_ids"][LOCAL_ROWS:])) == 0


def test_a_failing_apply_leaves_no_per_call_state(monkeypatch):
    upstream = _Upstream(rows_after_padding=LOCAL_ROWS + 1)
    _install_fakes(monkeypatch, upstream)
    comm, experts = _Comm(), _Experts()

    def explode(**kwargs):
        del kwargs
        raise RuntimeError("apply exploded")

    experts.quant_method.apply = explode

    with pytest.raises(RuntimeError, match="apply exploded"):
        _run(comm, experts)

    assert experts.ascend_pertoken_scale is None
    assert experts.ascend_mc2_mask is None


def test_a_missing_capture_is_refused(monkeypatch):
    """Without the control the comparison would be the patch against itself."""

    _install_fakes(monkeypatch, None)
    comm, experts = _Comm(), _Experts()

    with pytest.raises(RuntimeError, match="was not captured"):
        _run(comm, experts)
