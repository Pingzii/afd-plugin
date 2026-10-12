# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Resolve the DeepSeek-V4 cache dtypes vLLM-Ascend leaves at ``auto`` on A5.

Upstream source: ``vllm_ascend/platform.py`` at commit
``8d4409d6256d8a6729140ddcc0d1889e3f96cdd6``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)

_ASCEND_CACHE_DTYPE_PATCH_ATTR = "_afd_plugin_dsv4_a5_cache_dtype_patch_state"

_AUTO = "auto"
_CACHE_DTYPE = "fp8"
_DSV4_MODEL_TYPE_PREFIX = "deepseek_v4"


def apply_afd_dsv4_a5_cache_dtype_patch() -> bool:
    """Install the DeepSeek-V4 A5 cache dtype resolver.

    Returns whether this process has installed the wrapper (or had already
    installed it), so callers do not cache a failed early import during plugin
    initialization.
    """

    try:
        from vllm_ascend.platform import NPUPlatform
    except ImportError:
        return False

    if hasattr(NPUPlatform, _ASCEND_CACHE_DTYPE_PATCH_ATTR):
        return True

    original_check_and_update_config = NPUPlatform.check_and_update_config

    # Patch reason: vLLM-Ascend's A5 DeepSeek-V4 DSA path only has the FP8
    # compressed-KV implementation (the BF16 one needs `sparse_flash_mla`,
    # which the ascend950 build does not compile), yet it leaves both cache
    # dtype fields at their default ``"auto"``. DeepSeek-V4 then cannot start
    # at all: ``kv_compress_epilog`` rejects the BF16 cache it is handed, and
    # the SFA indexer cache ends up with a bf16 K row whose element count no
    # longer matches the page size the spec allocates. vLLM documents the
    # intended default for the indexer ("fp8 for the DeepSeek sparse indexer")
    # but vLLM-Ascend resolves it against the model dtype instead.
    # Patch functionality: before vLLM-Ascend normalizes the config, substitute
    # ``"fp8"`` for the DeepSeek-V4 cache dtype fields that are still ``"auto"``
    # on A5. Explicit user values are left untouched.
    # Removal plan: remove once vLLM-Ascend resolves both fields itself, which
    # also removes the need for the equivalent ``--kv-cache-dtype fp8`` and
    # ``--attention-config '{"indexer_kv_dtype": "fp8"}'`` flags.
    # Expansion exception: upstream check_and_update_config is platform-owned
    # normalization; keep narrow original-function delegation so this patch owns
    # only the DeepSeek-V4 cache dtype substitution.
    # Signature: matches upstream; no added parameters.
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        del cls
        # ### PATCH START: DeepSeek-V4 A5 cache dtype resolution
        _resolve_dsv4_a5_cache_dtypes(vllm_config)
        # ### PATCH END: DeepSeek-V4 A5 cache dtype resolution
        original_check_and_update_config(vllm_config)

    NPUPlatform.check_and_update_config = classmethod(check_and_update_config)
    setattr(
        NPUPlatform,
        _ASCEND_CACHE_DTYPE_PATCH_ATTR,
        original_check_and_update_config,
    )
    return True


def _resolve_dsv4_a5_cache_dtypes(vllm_config: VllmConfig) -> None:
    """Substitute ``fp8`` for DeepSeek-V4 cache dtypes left at ``auto`` on A5.

    Leaves the config untouched when vLLM-Ascend does not expose the hardware
    profile or the cache fields this patch keys off. Both fields stay settable
    through their launch flags, so skipping the substitution is recoverable and
    must never fail engine startup.
    """

    try:
        from vllm_ascend.device.hardware_profile import (
            HardwareCapability,
            get_current_hardware_profile,
        )
    except ImportError:
        return

    try:
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        attention_config = vllm_config.attention_config
    except AttributeError:
        return

    if not get_current_hardware_profile().supports(
        HardwareCapability.DSV4_COMPRESSED_CACHE,
    ):
        return

    hf_config = getattr(model_config, "hf_config", None)
    if not str(getattr(hf_config, "model_type", "")).startswith(
        _DSV4_MODEL_TYPE_PREFIX
    ):
        return

    resolved: list[str] = []
    if cache_config.cache_dtype == _AUTO:
        cache_config.cache_dtype = _CACHE_DTYPE
        resolved.append("cache_dtype")
    if attention_config.indexer_kv_dtype == _AUTO:
        attention_config.indexer_kv_dtype = _CACHE_DTYPE
        resolved.append("indexer_kv_dtype")

    if resolved:
        logger.info(
            "AFD resolved DeepSeek-V4 A5 cache dtypes left at 'auto': %s -> %s",
            ", ".join(resolved),
            _CACHE_DTYPE,
        )


__all__ = ["apply_afd_dsv4_a5_cache_dtype_patch"]
