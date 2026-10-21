# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Pin the Ascend MoE communication policy to ALLGATHER for AFD experiments.

A5 reports ``MoECommPolicy.CAPACITY_AND_WORLD_SIZE``, so one run selects MC2 for
small decode batches and ALLGATHER for large prefill batches. AFD's
Attention-owned gate and its ALLGATHER prepare patch cover only ALLGATHER, and
hold the path fixed across the prefill/decode split so one implementation is
exercised end to end.

The switch is read from the CAMP2P connector's extra configuration and is off by
default; without it nothing is rebound and the selected path is upstream's.

Two install points, because the processes are separate. The first is
``_apply_afd_npu_config_patches``, which runs while the config is built in the
front-end and again whenever a child revalidates it. The second is
``run_afd_ascend_engine_core``, the spawned engine core's own entry point.
``vllm serve`` forces the ``spawn`` start method, so the engine core starts from a
fresh interpreter and does not inherit a rebind made while the config was built.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING

from vllm_ascend.device.hardware_profile import (
    HardwareProfile,
    MoECommPolicy,
    get_current_hardware_profile as get_upstream_hardware_profile,
)

from afd_plugin.config import (
    CAMP2P_CONNECTOR,
    connector_extra_config_from_source,
    parse_optional_afd_config,
)
from afd_plugin.config_utils import coerce_extra_bool

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_FORCE_ALLGATHER_FIELD = "force_allgather"


# Upstream: vllm_ascend/device/hardware_profile.py at
# 8d4409d6256d8a6729140ddcc0d1889e3f96cdd6.
# Patch reason: the A5 profile policy selects different MoE communication
# implementations for prefill and decode, while the Attention-owned gate and the
# ALLGATHER prepare patch support one path.
# Patch functionality: report a profile whose ``moe_comm_policy`` is ALLGATHER.
# Every other field is copied, so unrelated capability queries keep their
# upstream answer.
# Signature: matches upstream; no added parameters.
# Removal plan: remove once MC2 carries Attention-side routing and the gate works
# on every selected path.
def get_current_hardware_profile() -> HardwareProfile:
    """Return the active hardware profile with the policy pinned to ALLGATHER."""

    profile = get_upstream_hardware_profile()
    # ### PATCH START: AFD pins the MoE communication policy
    if profile.moe_comm_policy is MoECommPolicy.ALLGATHER:
        return profile
    return replace(profile, moe_comm_policy=MoECommPolicy.ALLGATHER)
    # ### PATCH END


def apply_afd_force_allgather_patch_if_needed(
    vllm_config: VllmConfig | None,
) -> bool:
    """Install the ALLGATHER pin when the AFD config asks for it.

    ``CAMP2P connector_extra_config.force_allgather`` is off by default. Without
    it nothing is rebound, so the selected communication path is exactly
    upstream's.

    The rebind targets ``vllm_ascend.ascend_forward_context`` because
    ``select_moe_comm_method`` lives there and reads the policy through that
    module-level name; rebinding it makes every caller agree on one path,
    including callers that imported ``select_moe_comm_method`` at module scope.
    """

    if not _force_allgather_requested(vllm_config):
        return False

    from vllm_ascend import ascend_forward_context

    ascend_forward_context.get_current_hardware_profile = get_current_hardware_profile
    # Printed rather than logged: vLLM configures only its own logger, so
    # afd_plugin INFO records are dropped from a serve log. This line is the
    # evidence that the engine-core process, not just the front-end, holds the
    # pin.
    print(
        f"[AFD-ALLGATHER] pid={os.getpid()} MoE communication path pinned "
        "to ALLGATHER",
        flush=True,
    )
    return True


def _force_allgather_requested(vllm_config: VllmConfig | None) -> bool:
    afd_config = parse_optional_afd_config(vllm_config, validate=False)
    if afd_config is None or afd_config.connector != CAMP2P_CONNECTOR:
        return False
    extra_config = connector_extra_config_from_source(vllm_config)
    return coerce_extra_bool(
        extra_config.get(_FORCE_ALLGATHER_FIELD, False),
        field_name=_FORCE_ALLGATHER_FIELD,
    )


__all__ = [
    "apply_afd_force_allgather_patch_if_needed",
    "get_current_hardware_profile",
]
