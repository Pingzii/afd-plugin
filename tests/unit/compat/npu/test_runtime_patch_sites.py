# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Which stage installs which AFD patch.

The routing prepare patches replace the MoE prepare path, so importing them pulls
vLLM-Ascend's MoE internals. Building and validating a config does not need that, and
a config-only process that imported them used to fail outright. They therefore belong
to the worker-init stage, which runs before any model load, warm-up or forward.

The two tests below pin both halves of that split, and the third pins the guard that
keeps a worker from installing them again.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from afd_plugin.compat.npu import runtime

_ROUTING_PATCH_MODULES = (
    "afd_plugin.compat.patches.npu.external_routing_prepare",
    "afd_plugin.compat.patches.npu.external_routing_mc2_prepare",
)


def _install_fake_patch_modules(monkeypatch, calls: list[str]):
    """Replace every patch module the runtime entry points import."""

    def _module(name: str, function: str) -> ModuleType:
        module = ModuleType(name)

        def installer(*args, **kwargs):
            del args, kwargs
            calls.append(function)
            return True

        setattr(module, function, installer)
        return module

    fakes = {
        "afd_plugin.compat.patches.npu.ascend_config": ("apply_afd_ascend_config_patch"),
        "afd_plugin.compat.patches.npu.ascend_platform": (
            "apply_afd_ascend_dbo_config_patch"
        ),
        "afd_plugin.compat.patches.npu.mla_graph": "apply_afd_mla_graph_patch",
        "afd_plugin.compat.patches.npu.external_routing_prepare": (
            "apply_afd_external_routing_prepare_patch"
        ),
        "afd_plugin.compat.patches.npu.external_routing_mc2_prepare": (
            "apply_afd_external_routing_mc2_prepare_patch"
        ),
    }
    for name, function in fakes.items():
        monkeypatch.setitem(sys.modules, name, _module(name, function))


@pytest.fixture
def fresh_runtime(monkeypatch):
    """Give each test an uninstalled runtime and a clean import table."""

    monkeypatch.setattr(runtime, "_PATCHES_APPLIED", False)
    for name in _ROUTING_PATCH_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    calls: list[str] = []
    _install_fake_patch_modules(monkeypatch, calls)
    return calls


def test_config_stage_installs_no_routing_patch(fresh_runtime):
    """A config-only process must not load the MoE prepare internals.

    The import itself is guarded by ``test_config_validation``, which runs this
    entry point with only vLLM-Ascend partially available. Here the question is
    narrower: the config stage must not even reach these installers.
    """

    runtime.apply_afd_ascend_config_patch_if_needed()

    assert "apply_afd_external_routing_prepare_patch" not in fresh_runtime
    assert "apply_afd_external_routing_mc2_prepare_patch" not in fresh_runtime


def test_worker_stage_installs_both_routing_patches(fresh_runtime):
    runtime.apply_afd_ascend_patches_if_needed()

    assert "apply_afd_external_routing_prepare_patch" in fresh_runtime
    assert "apply_afd_external_routing_mc2_prepare_patch" in fresh_runtime


def test_worker_stage_installs_only_once(fresh_runtime):
    runtime.apply_afd_ascend_patches_if_needed()
    runtime.apply_afd_ascend_patches_if_needed()

    assert fresh_runtime.count("apply_afd_external_routing_prepare_patch") == 1
    assert fresh_runtime.count("apply_afd_external_routing_mc2_prepare_patch") == 1
