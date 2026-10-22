# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Public Ascend runtime compatibility facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

from afd_plugin.compat.npu.feature_validation import (
    fail_if_unsupported_npu_afd_features,
)
from afd_plugin.compat.npu.forward_context import (
    ascend_forward_context,
)
from afd_plugin.compat.npu.runtime_config import (
    fix_all2all_backend_for_afd,
    npu_afd_num_ubatches,
)
from afd_plugin.config import is_afd_async_dp, parse_optional_afd_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_PATCHES_APPLIED = False
_WORKER_HOOKS_APPLIED = False


def apply_afd_ascend_config_patch_if_needed() -> None:
    """Apply patches required while vLLM builds an AFD NPU config."""

    from afd_plugin.compat.patches.npu.ascend_config import (
        apply_afd_ascend_config_patch,
    )
    from afd_plugin.compat.patches.npu.ascend_platform import (
        apply_afd_ascend_dbo_config_patch,
    )

    apply_afd_ascend_config_patch()
    if not apply_afd_ascend_dbo_config_patch():
        raise RuntimeError(
            "AFD NPU DBO config patch requires vLLM-Ascend NPUPlatform",
        )


def apply_afd_async_dp_engine_patch_if_needed(vllm_config: VllmConfig) -> bool:
    """Finalize the async Attention engine entry point after Ascend patches."""

    afd_config = parse_optional_afd_config(vllm_config, validate=False)
    if (
        afd_config is None
        or afd_config.role != "attention"
        or not is_afd_async_dp(vllm_config)
    ):
        return False

    from afd_plugin.compat.patches.async_dp_engine import (
        apply_async_dp_engine_patch,
    )

    if not apply_async_dp_engine_patch():
        raise RuntimeError(
            "AFD async-DP engine patch requires the pinned vLLM version",
        )
    return True


def apply_afd_ascend_engine_core_config_patch_if_needed(
    vllm_config: VllmConfig,
) -> bool:
    """Prepare AFD sync Attention and FFN DP children for Ascend config."""
    afd_config = parse_optional_afd_config(vllm_config, validate=False)
    if afd_config is None or (
        afd_config.role == "attention" and is_afd_async_dp(vllm_config)
    ):
        return False

    from vllm.v1.engine.core import EngineCoreProc

    from afd_plugin.compat.patches.npu.ascend_config import (
        run_afd_ascend_engine_core,
    )

    EngineCoreProc.run_engine_core = staticmethod(run_afd_ascend_engine_core)
    return True


def apply_afd_ascend_patches_if_needed() -> None:
    """Apply plugin-owned runtime patches after Ascend initialization."""

    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    from afd_plugin.compat.patches.npu.external_routing_mc2_prepare import (
        apply_afd_external_routing_mc2_prepare_patch,
    )
    from afd_plugin.compat.patches.npu.external_routing_prepare import (
        apply_afd_external_routing_prepare_patch,
    )
    from afd_plugin.compat.patches.npu.mla_graph import (
        apply_afd_mla_graph_patch,
    )

    apply_afd_ascend_config_patch_if_needed()
    if not apply_afd_mla_graph_patch():
        raise RuntimeError(
            "AFD NPU MLA graph patch requires the vLLM-Ascend MLA resolver",
        )
    # The routing patches replace the MoE prepare path, so they belong here rather
    # than in the config stage: importing them pulls vLLM-Ascend's MoE internals,
    # which a process that only builds and validates a config does not need. This
    # runs from the worker constructor, before any model load, warm-up or forward,
    # and the guard above keeps it to one install per process.
    apply_afd_external_routing_prepare_patch()
    apply_afd_external_routing_mc2_prepare_patch()
    _PATCHES_APPLIED = True


def apply_afd_worker_runtime_hooks(vllm_config: VllmConfig) -> None:
    """Install the worker-side hooks that need the resolved config.

    These cannot ride on the config-build path. A worker is spawned and its config
    arrives unpickled, so ``VllmConfig.__post_init__`` -- and with it
    ``_apply_afd_npu_config_patches`` -- never runs in the worker process. Anything
    installed there exists in the front-end and the engine core, and is absent from
    the process that actually runs the forward.

    Idempotent per process. Installing the ALLGATHER pin twice is harmless but prints
    its evidence line each time, so a repeated call would make the log ambiguous about
    how many processes hold the pin.

    Called from each worker's ``init_device``, which is before any model load,
    warm-up or forward, and after ``vllm_config`` is available.
    """

    global _WORKER_HOOKS_APPLIED
    if _WORKER_HOOKS_APPLIED:
        return

    from afd_plugin.compat.npu.debug_moe_comm_probe import install_moe_comm_probe
    from afd_plugin.compat.patches.npu.force_allgather import (
        apply_afd_force_allgather_patch_if_needed,
    )

    install_moe_comm_probe()
    apply_afd_force_allgather_patch_if_needed(vllm_config)
    _WORKER_HOOKS_APPLIED = True


__all__ = [
    "apply_afd_async_dp_engine_patch_if_needed",
    "apply_afd_ascend_engine_core_config_patch_if_needed",
    "apply_afd_ascend_config_patch_if_needed",
    "apply_afd_ascend_patches_if_needed",
    "apply_afd_worker_runtime_hooks",
    "ascend_forward_context",
    "fail_if_unsupported_npu_afd_features",
    "fix_all2all_backend_for_afd",
    "npu_afd_num_ubatches",
]
