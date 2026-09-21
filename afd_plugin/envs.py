# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Environment-variable helpers for AFD plugin runtime diagnostics."""

from __future__ import annotations

import os

AFD_FORCE_BALANCED_TOPK_IDS = "AFD_FORCE_BALANCED_TOPK_IDS"
AFD_NPU_GRAPH_DIAGNOSTICS = "AFD_NPU_GRAPH_DIAGNOSTICS"
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def force_balanced_topk_ids_enabled() -> bool:
    return os.environ.get(AFD_FORCE_BALANCED_TOPK_IDS, "").lower() in ENV_TRUE_VALUES


def npu_graph_diagnostics_enabled() -> bool:
    return os.environ.get(AFD_NPU_GRAPH_DIAGNOSTICS, "").lower() in ENV_TRUE_VALUES


__all__ = [
    "AFD_FORCE_BALANCED_TOPK_IDS",
    "AFD_NPU_GRAPH_DIAGNOSTICS",
    "force_balanced_topk_ids_enabled",
    "npu_graph_diagnostics_enabled",
]
