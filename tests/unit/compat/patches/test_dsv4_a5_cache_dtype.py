# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-safe tests for the DeepSeek-V4 A5 cache dtype resolver."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from afd_plugin.compat.patches.npu import dsv4_a5_cache_dtype

_DSV4_CAPABILITY = object()


def _config(
    *,
    model_type: str = "deepseek_v4",
    cache_dtype: str = "auto",
    indexer_kv_dtype: str = "auto",
):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        attention_config=SimpleNamespace(indexer_kv_dtype=indexer_kv_dtype),
    )


@pytest.fixture
def upstream(monkeypatch):
    """Install fake vLLM-Ascend platform and hardware-profile modules."""

    calls: list[object] = []

    class NPUPlatform:
        @classmethod
        def check_and_update_config(cls, vllm_config):
            calls.append(vllm_config)

    hardware = SimpleNamespace(supports_dsv4_cache=True)

    def supports(capability):
        return hardware.supports_dsv4_cache and capability is _DSV4_CAPABILITY

    platform_mod = types.ModuleType("vllm_ascend.platform")
    platform_mod.NPUPlatform = NPUPlatform

    hardware_mod = types.ModuleType("vllm_ascend.device.hardware_profile")
    hardware_mod.HardwareCapability = SimpleNamespace(
        DSV4_COMPRESSED_CACHE=_DSV4_CAPABILITY,
    )
    hardware_mod.get_current_hardware_profile = lambda: SimpleNamespace(
        supports=supports
    )

    device_mod = types.ModuleType("vllm_ascend.device")
    device_mod.hardware_profile = hardware_mod

    monkeypatch.setitem(sys.modules, "vllm_ascend", types.ModuleType("vllm_ascend"))
    monkeypatch.setitem(sys.modules, "vllm_ascend.device", device_mod)
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.device.hardware_profile", hardware_mod
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", platform_mod)

    return SimpleNamespace(platform_cls=NPUPlatform, calls=calls, hardware=hardware)


def test_auto_fields_are_pinned_on_a5_dsv4(upstream):
    assert dsv4_a5_cache_dtype.apply_afd_dsv4_a5_cache_dtype_patch() is True

    config = _config()
    upstream.platform_cls.check_and_update_config(config)

    assert config.cache_config.cache_dtype == "fp8"
    assert config.attention_config.indexer_kv_dtype == "fp8"
    assert upstream.calls == [config]


def test_explicit_values_are_preserved(upstream):
    dsv4_a5_cache_dtype.apply_afd_dsv4_a5_cache_dtype_patch()

    config = _config(cache_dtype="fp8_e4m3", indexer_kv_dtype="bf16")
    upstream.platform_cls.check_and_update_config(config)

    assert config.cache_config.cache_dtype == "fp8_e4m3"
    assert config.attention_config.indexer_kv_dtype == "bf16"


def test_only_the_auto_field_is_pinned(upstream):
    dsv4_a5_cache_dtype.apply_afd_dsv4_a5_cache_dtype_patch()

    config = _config(cache_dtype="fp8_e4m3")
    upstream.platform_cls.check_and_update_config(config)

    assert config.cache_config.cache_dtype == "fp8_e4m3"
    assert config.attention_config.indexer_kv_dtype == "fp8"


def test_non_a5_hardware_is_untouched(upstream):
    upstream.hardware.supports_dsv4_cache = False
    dsv4_a5_cache_dtype.apply_afd_dsv4_a5_cache_dtype_patch()

    config = _config()
    upstream.platform_cls.check_and_update_config(config)

    assert config.cache_config.cache_dtype == "auto"
    assert config.attention_config.indexer_kv_dtype == "auto"
    assert upstream.calls == [config]


def test_non_dsv4_model_is_untouched(upstream):
    dsv4_a5_cache_dtype.apply_afd_dsv4_a5_cache_dtype_patch()

    config = _config(model_type="qwen3_moe")
    upstream.platform_cls.check_and_update_config(config)

    assert config.cache_config.cache_dtype == "auto"
    assert config.attention_config.indexer_kv_dtype == "auto"


def test_dsv4_variants_share_the_prefix(upstream):
    dsv4_a5_cache_dtype.apply_afd_dsv4_a5_cache_dtype_patch()

    config = _config(model_type="deepseek_v41")
    upstream.platform_cls.check_and_update_config(config)

    assert config.attention_config.indexer_kv_dtype == "fp8"


def test_patch_is_idempotent(upstream):
    assert dsv4_a5_cache_dtype.apply_afd_dsv4_a5_cache_dtype_patch() is True
    first = upstream.platform_cls.check_and_update_config.__func__
    assert dsv4_a5_cache_dtype.apply_afd_dsv4_a5_cache_dtype_patch() is True
    assert upstream.platform_cls.check_and_update_config.__func__ is first


def test_patch_reports_missing_platform(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_ascend", types.ModuleType("vllm_ascend"))
    monkeypatch.delitem(sys.modules, "vllm_ascend.platform", raising=False)
    assert dsv4_a5_cache_dtype.apply_afd_dsv4_a5_cache_dtype_patch() is False
