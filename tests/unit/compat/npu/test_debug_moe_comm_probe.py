# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU contract probes for the opt-in MoE path probe.

The probe must stay inert unless ``AFD_DEBUG_MOE_COMM=1``, because it is the
only diagnostic the smoke run has and a changed selection would invalidate it.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest

_PROBE_MODULE = "afd_plugin.compat.npu.debug_moe_comm_probe"
_ENV_VAR = "AFD_DEBUG_MOE_COMM"


class _MoECommPolicy(Enum):
    ALLGATHER = "allgather"
    CAPACITY_AND_WORLD_SIZE = "capacity_and_world_size"


class _MoECommType(Enum):
    ALLGATHER = "allgather"
    MC2 = "mc2"


@dataclass(frozen=True)
class _HardwareProfile:
    moe_comm_policy: _MoECommPolicy


class _SelectProbe:
    """Selector stand-in that records every call and returns a fixed type."""

    def __init__(self, selected):
        self.selected = selected
        self.num_tokens: list[int] = []

    def __call__(self, num_tokens, vllm_config, **kwargs):
        self.num_tokens.append(num_tokens)
        return self.selected


def _fake_vllm_ascend(monkeypatch, *, selected=_MoECommType.ALLGATHER):
    profile = _HardwareProfile(_MoECommPolicy.CAPACITY_AND_WORLD_SIZE)
    ascend = ModuleType("vllm_ascend")
    ascend.__path__ = []
    device = ModuleType("vllm_ascend.device")
    device.__path__ = []
    hardware_profile = ModuleType("vllm_ascend.device.hardware_profile")
    hardware_profile.get_current_hardware_profile = lambda: profile
    forward_context = ModuleType("vllm_ascend.ascend_forward_context")
    forward_context.get_current_hardware_profile = lambda: profile
    forward_context.get_mc2_tokens_capacity = lambda: 256
    forward_context.select_moe_comm_method = _SelectProbe(selected)
    ops = ModuleType("vllm_ascend.ops")
    ops.__path__ = []
    fused_moe = ModuleType("vllm_ascend.ops.fused_moe")
    fused_moe.__path__ = []
    moe_comm_method = ModuleType("vllm_ascend.ops.fused_moe.moe_comm_method")
    moe_comm_method.get_moe_comm_method = lambda comm_type: SimpleNamespace()

    ascend.device = device
    ascend.ascend_forward_context = forward_context
    ascend.ops = ops
    device.hardware_profile = hardware_profile
    ops.fused_moe = fused_moe
    fused_moe.moe_comm_method = moe_comm_method
    for name, module in (
        ("vllm_ascend", ascend),
        ("vllm_ascend.device", device),
        ("vllm_ascend.device.hardware_profile", hardware_profile),
        ("vllm_ascend.ascend_forward_context", forward_context),
        ("vllm_ascend.ops", ops),
        ("vllm_ascend.ops.fused_moe", fused_moe),
        ("vllm_ascend.ops.fused_moe.moe_comm_method", moe_comm_method),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return forward_context


def _load_probe_module(monkeypatch):
    monkeypatch.delitem(sys.modules, _PROBE_MODULE, raising=False)
    return importlib.import_module(_PROBE_MODULE)


def test_probe_stays_inert_without_the_environment_variable(monkeypatch):
    monkeypatch.delenv(_ENV_VAR, raising=False)
    monkeypatch.delitem(
        sys.modules,
        "vllm_ascend.ascend_forward_context",
        raising=False,
    )
    module = _load_probe_module(monkeypatch)

    assert not module.install_moe_comm_probe()
    # Inert means untouched: the runtime module was never needed, so a run
    # without the variable cannot observe any difference.
    assert "vllm_ascend.ascend_forward_context" not in sys.modules


def test_probe_reports_each_distinct_selection_once(monkeypatch, capsys):
    monkeypatch.setenv(_ENV_VAR, "1")
    forward_context = _fake_vllm_ascend(monkeypatch)
    module = _load_probe_module(monkeypatch)

    assert module.install_moe_comm_probe()
    forward_context.select_moe_comm_method(3072, object())
    forward_context.select_moe_comm_method(3072, object())
    forward_context.select_moe_comm_method(128, object())

    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert len(lines) == 2
    assert "num_tokens=3072" in lines[0]
    assert "capacity=256" in lines[0]
    assert "CAPACITY_AND_WORLD_SIZE" in lines[0]
    assert "ALLGATHER" in lines[0]
    assert "impl=SimpleNamespace" in lines[0]
    assert "num_tokens=128" in lines[1]


def test_probe_returns_the_selection_unchanged(monkeypatch):
    monkeypatch.setenv(_ENV_VAR, "1")
    forward_context = _fake_vllm_ascend(monkeypatch, selected=_MoECommType.MC2)
    module = _load_probe_module(monkeypatch)

    assert module.install_moe_comm_probe()
    assert forward_context.select_moe_comm_method(64, object()) is _MoECommType.MC2


def test_probe_handles_a_non_moe_selection(monkeypatch, capsys):
    """``select_moe_comm_method`` returns None for non-MoE models."""

    monkeypatch.setenv(_ENV_VAR, "1")
    forward_context = _fake_vllm_ascend(monkeypatch, selected=None)
    module = _load_probe_module(monkeypatch)

    assert module.install_moe_comm_probe()
    assert forward_context.select_moe_comm_method(16, object()) is None
    assert "type=None impl=None" in capsys.readouterr().out
