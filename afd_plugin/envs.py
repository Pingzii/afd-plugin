# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Environment-variable helpers for AFD plugin runtime diagnostics."""

from __future__ import annotations

import os

AFD_FORCE_BALANCED_TOPK_IDS = "AFD_FORCE_BALANCED_TOPK_IDS"
AFD_ASYNC_CAM_LAYERED_GMM = "AFD_ASYNC_CAM_LAYERED_GMM"
# A2E transport quantization mode for the synchronous CAMP2P connector:
# 0 keeps the BF16 activation contract, 4 moves pre-quantized E4M3 activations
# together with their E8M0 MX scales. An explicit
# connector_extra_config.quant_mode wins over this variable.
AFD_NPU_CAMP2P_QUANT_MODE = "AFD_NPU_CAMP2P_QUANT_MODE"
A2E_QUANT_MODE_BF16 = 0
A2E_QUANT_MODE_MXFP8 = 4
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

# In MXFP8 mode the FFN normally restores the model dtype before its MoE runs.
# Setting this hands the transported pair to the all-gather MoE instead, which
# removes the FFN-side quantization. Keep it off by default: the expansion of the
# transported scale has to happen inside the framework's DP gather to stay
# symmetric across ranks, and hooking it any earlier launches the collective on a
# different set of ranks ("execute kernel param invalid" from HcclLaunchAicpuKernel).
AFD_NPU_CAMP2P_MXFP8_DIRECT = "AFD_NPU_CAMP2P_MXFP8_DIRECT"


def force_balanced_topk_ids_enabled() -> bool:
    return os.environ.get(AFD_FORCE_BALANCED_TOPK_IDS, "").lower() in ENV_TRUE_VALUES


def async_cam_layered_gmm_enabled() -> bool:
    return os.environ.get(AFD_ASYNC_CAM_LAYERED_GMM, "").lower() in ENV_TRUE_VALUES


def camp2p_mxfp8_direct_enabled() -> bool:
    """Return whether the FFN may consume the transported MXFP8 pair itself."""

    return os.environ.get(AFD_NPU_CAMP2P_MXFP8_DIRECT, "").lower() in ENV_TRUE_VALUES


def camp2p_quant_mode() -> int:
    """Return the configured A2E transport quantization mode.

    Deployment tooling sets the variable when a launch script needs MXFP8 but
    cannot pass ``connector_extra_config``, because the host validates the
    ``afd`` config section against its own field set.
    """

    raw = os.environ.get(AFD_NPU_CAMP2P_QUANT_MODE, "")
    if not raw.strip():
        return A2E_QUANT_MODE_BF16
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{AFD_NPU_CAMP2P_QUANT_MODE} must be an integer, got {raw!r}",
        ) from exc


__all__ = [
    "AFD_FORCE_BALANCED_TOPK_IDS",
    "AFD_ASYNC_CAM_LAYERED_GMM",
    "AFD_NPU_CAMP2P_MXFP8_DIRECT",
    "AFD_NPU_CAMP2P_QUANT_MODE",
    "A2E_QUANT_MODE_BF16",
    "A2E_QUANT_MODE_MXFP8",
    "async_cam_layered_gmm_enabled",
    "camp2p_mxfp8_direct_enabled",
    "camp2p_quant_mode",
    "force_balanced_topk_ids_enabled",
]
