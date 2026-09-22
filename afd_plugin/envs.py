# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Environment-variable helpers for AFD plugin runtime diagnostics."""

from __future__ import annotations

import os

AFD_FORCE_BALANCED_TOPK_IDS = "AFD_FORCE_BALANCED_TOPK_IDS"
AFD_VALIDATE_HASH_TOKEN_IDS = "AFD_VALIDATE_HASH_TOKEN_IDS"
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def force_balanced_topk_ids_enabled() -> bool:
    return os.environ.get(AFD_FORCE_BALANCED_TOPK_IDS, "").lower() in ENV_TRUE_VALUES


def validate_hash_token_ids_enabled() -> bool:
    """Return whether token-keyed routing should validate its ids on the way in.

    The CANN Hash operator reads ``tid2eid[input_ids[row]]`` without a bounds
    check, so one out-of-range id faults an AIV core with "The DDR address of the
    MTE instruction is out of range" and reports only which core died. Checking
    the values on the host side turns that into an error naming the rows, which
    is what separates a stale or partially written id buffer from a table that is
    smaller than the id space.

    The check reads device tensors and therefore synchronises, so it is off by
    default and enabled per run with ``AFD_VALIDATE_HASH_TOKEN_IDS=1``.
    """

    return os.environ.get(AFD_VALIDATE_HASH_TOKEN_IDS, "").lower() in ENV_TRUE_VALUES


__all__ = [
    "AFD_FORCE_BALANCED_TOPK_IDS",
    "AFD_VALIDATE_HASH_TOKEN_IDS",
    "force_balanced_topk_ids_enabled",
    "validate_hash_token_ids_enabled",
]
