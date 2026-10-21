# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Temporary probe that records the selected Ascend MoE communication path.

Enable with ``AFD_DEBUG_MOE_COMM=1``. With the variable unset nothing is
imported into the running process.

This module only observes. It does not change the selection, the prepare path, or
any tensor, so a run with the probe on is comparable to a run with it off.

Why it exists: A5 selects its MoE communication method from
``MoECommPolicy.CAPACITY_AND_WORLD_SIZE``, which can pick MC2 for small decode
batches and ALLGATHER for large prefill batches. AFD's Attention-owned gate needs
one known path, so the selected type, the capacity threshold behind it, and the
resulting implementation object must be recorded before any migration work.

Reported fields per distinct (num_tokens, selection) pair:

- ``num_tokens`` seen by the selector;
- ``capacity`` from ``get_mc2_tokens_capacity()``;
- ``policy`` from the active hardware profile;
- ``type`` the selector returns;
- ``impl`` the class that ``get_moe_comm_method`` would hand back.

Graph replay does not re-run this Python code, so replay alone will not print.
Capture does run it, and so does every eager step, so each distinct batch size
should still appear at least once before its graph is replayed.
"""

from __future__ import annotations

import os

PROBE_ENV_VAR = "AFD_DEBUG_MOE_COMM"

# Reported pairs are deduplicated so a long run prints bounded output. This is
# deliberate process-local state in a temporary probe and is never read back.
_REPORTED: set[tuple[int, str]] = set()


def install() -> bool:
    """Install the probe when the environment variable is set."""

    if os.environ.get(PROBE_ENV_VAR) != "1":
        return False

    import vllm_ascend.ascend_forward_context as forward_context
    from vllm_ascend.device.hardware_profile import get_current_hardware_profile
    from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

    upstream_select = forward_context.select_moe_comm_method

    def probe_select_moe_comm_method(*args, **kwargs):
        # Passthrough on purpose: the probe must not drift if upstream adds a
        # parameter, and it only reads the first positional (``num_tokens``).
        selected = upstream_select(*args, **kwargs)
        num_tokens = args[0] if args else kwargs.get("num_tokens")
        key = (int(num_tokens) if num_tokens is not None else -1, str(selected))
        if key not in _REPORTED:
            _REPORTED.add(key)
            impl = get_moe_comm_method(selected)
            print(
                "[AFD-P0] "
                f"num_tokens={num_tokens} "
                f"capacity={forward_context.get_mc2_tokens_capacity()} "
                f"policy={get_current_hardware_profile().moe_comm_policy} "
                f"type={selected} "
                f"impl={type(impl).__name__}",
                flush=True,
            )
        return selected

    # The selector is what builds the forward context, so patching it observes
    # the path the model will actually use for this step.
    forward_context.select_moe_comm_method = probe_select_moe_comm_method
    return True


__all__ = ["PROBE_ENV_VAR", "install"]
