# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU contract probes for the CAMP2P ALLGATHER pin.

``vllm serve`` spawns the engine core, so a rebind made while the config is built
does not reach the process that forwards tokens. Both install points are
exercised here: the shared config hook and the engine-core child entry.
"""

from __future__ import annotations

import __future__
import ast
import importlib
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[5]
_PATCH_MODULE = "afd_plugin.compat.patches.npu.force_allgather"
_ENGINE_CORE_PATCH_PATH = _REPO_ROOT / "afd_plugin/compat/patches/npu/ascend_config.py"
_CONFIG_HOOK_PATH = _REPO_ROOT / "afd_plugin/compat/patches/config_validation.py"
_CONNECTOR = "CAMP2pAFDConnector"


class _MoECommPolicy(Enum):
    ALLGATHER = "allgather"
    CAPACITY_AND_WORLD_SIZE = "capacity_and_world_size"


@dataclass(frozen=True)
class _HardwareProfile:
    device_type: str
    moe_comm_policy: _MoECommPolicy


def _fake_vllm_ascend(monkeypatch):
    """Install a minimal ``vllm_ascend`` surface around one upstream profile."""

    upstream = _HardwareProfile("A5", _MoECommPolicy.CAPACITY_AND_WORLD_SIZE)
    ascend = ModuleType("vllm_ascend")
    ascend.__path__ = []
    device = ModuleType("vllm_ascend.device")
    device.__path__ = []
    hardware_profile = ModuleType("vllm_ascend.device.hardware_profile")
    hardware_profile.HardwareProfile = _HardwareProfile
    hardware_profile.MoECommPolicy = _MoECommPolicy
    hardware_profile.get_current_hardware_profile = lambda: upstream
    forward_context = ModuleType("vllm_ascend.ascend_forward_context")
    forward_context.get_current_hardware_profile = lambda: upstream
    patch_package = ModuleType("vllm_ascend.patch")
    patch_package.__path__ = []
    platform_package = ModuleType("vllm_ascend.patch.platform")
    platform_package.__path__ = []

    ascend.device = device
    ascend.ascend_forward_context = forward_context
    ascend.patch = patch_package
    device.hardware_profile = hardware_profile
    patch_package.platform = platform_package
    for name, module in (
        ("vllm_ascend", ascend),
        ("vllm_ascend.device", device),
        ("vllm_ascend.device.hardware_profile", hardware_profile),
        ("vllm_ascend.ascend_forward_context", forward_context),
        ("vllm_ascend.patch", patch_package),
        ("vllm_ascend.patch.platform", platform_package),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(
        upstream=upstream,
        forward_context=forward_context,
        platform=platform_package,
    )


def _load_patch_module(monkeypatch):
    monkeypatch.delitem(sys.modules, _PATCH_MODULE, raising=False)
    return importlib.import_module(_PATCH_MODULE)


def _vllm_config(*, connector=_CONNECTOR, extra_config=None):
    afd = {"connector": connector}
    if extra_config is not None:
        afd["connector_extra_config"] = extra_config
    return SimpleNamespace(additional_config={"afd": afd})


def test_install_pins_the_policy_where_the_selector_reads_it(monkeypatch):
    fake = _fake_vllm_ascend(monkeypatch)
    module = _load_patch_module(monkeypatch)

    assert module.apply_afd_force_allgather_patch_if_needed(
        _vllm_config(extra_config={"force_allgather": True}),
    )
    pinned = fake.forward_context.get_current_hardware_profile()
    assert pinned.moe_comm_policy is _MoECommPolicy.ALLGATHER
    # Every unrelated field keeps its upstream value.
    assert pinned.device_type == fake.upstream.device_type


def test_install_leaves_the_upstream_getter_alone_when_off(monkeypatch):
    fake = _fake_vllm_ascend(monkeypatch)
    module = _load_patch_module(monkeypatch)
    upstream_getter = fake.forward_context.get_current_hardware_profile

    assert not module.apply_afd_force_allgather_patch_if_needed(
        _vllm_config(extra_config={"force_allgather": False}),
    )
    assert fake.forward_context.get_current_hardware_profile is upstream_getter


@pytest.mark.parametrize(
    "vllm_config",
    [
        SimpleNamespace(additional_config={}),
        _vllm_config(connector="CAMAsyncAFDConnector"),
    ],
)
def test_install_ignores_runs_that_do_not_ask_for_camp2p(monkeypatch, vllm_config):
    fake = _fake_vllm_ascend(monkeypatch)
    module = _load_patch_module(monkeypatch)
    upstream_getter = fake.forward_context.get_current_hardware_profile

    assert not module.apply_afd_force_allgather_patch_if_needed(vllm_config)
    assert fake.forward_context.get_current_hardware_profile is upstream_getter


def test_install_rejects_a_non_boolean_switch(monkeypatch):
    _fake_vllm_ascend(monkeypatch)
    module = _load_patch_module(monkeypatch)

    with pytest.raises(TypeError, match="force_allgather"):
        module.apply_afd_force_allgather_patch_if_needed(
            _vllm_config(extra_config={"force_allgather": "sometimes"}),
        )


def _load_engine_core_entry():
    """Execute the child entry point with its module imports replaced."""

    tree = ast.parse(_ENGINE_CORE_PATCH_PATH.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_afd_ascend_engine_core"
    ]
    namespace: dict[str, object] = {"apply_afd_ascend_config_patch": lambda: None}
    exec(
        compile(
            tree,
            str(_ENGINE_CORE_PATCH_PATH),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace["run_afd_ascend_engine_core"]


def _load_config_hook(monkeypatch):
    """Execute the shared config hook with Ascend setup replaced."""

    from afd_plugin.config import parse_optional_afd_config

    vllm = ModuleType("vllm")
    vllm.__path__ = []
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(device_type="npu")
    vllm.platforms = platforms
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)

    afd_npu = ModuleType("afd_plugin.compat.npu")
    afd_npu.__path__ = []
    afd_npu.apply_afd_ascend_config_patch_if_needed = lambda: None
    monkeypatch.setitem(sys.modules, "afd_plugin.compat.npu", afd_npu)

    tree = ast.parse(_CONFIG_HOOK_PATH.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_apply_afd_npu_config_patches"
    ]
    namespace: dict[str, object] = {
        "parse_optional_afd_config": parse_optional_afd_config,
    }
    exec(
        compile(
            tree,
            str(_CONFIG_HOOK_PATH),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace["_apply_afd_npu_config_patches"]


def test_engine_core_entry_pins_the_path_before_the_core_runs(monkeypatch):
    """The spawned engine core must hold the pin before it builds any worker."""

    fake = _fake_vllm_ascend(monkeypatch)
    _load_patch_module(monkeypatch)
    observed: list[object] = []

    def _run_engine_core_patch_func(*args, **kwargs):
        observed.append(
            fake.forward_context.get_current_hardware_profile().moe_comm_policy,
        )
        return "engine-core-ran"

    fake.platform.patch_engine_core = SimpleNamespace(
        _run_engine_core_patch_func=_run_engine_core_patch_func,
    )
    entry = _load_engine_core_entry()

    result = entry(vllm_config=_vllm_config(extra_config={"force_allgather": True}))

    assert result == "engine-core-ran"
    assert observed == [_MoECommPolicy.ALLGATHER]


def test_config_hook_pins_the_path_for_the_process_that_built_the_config(monkeypatch):
    fake = _fake_vllm_ascend(monkeypatch)
    _load_patch_module(monkeypatch)
    hook = _load_config_hook(monkeypatch)

    assert hook(_vllm_config(extra_config={"force_allgather": True}))

    pinned = fake.forward_context.get_current_hardware_profile()
    assert pinned.moe_comm_policy is _MoECommPolicy.ALLGATHER
