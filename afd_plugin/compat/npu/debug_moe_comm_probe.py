# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in probe that records the MoE communication path Ascend selects.

Enable with ``AFD_DEBUG_MOE_COMM=1``. With the variable unset nothing is
imported from ``vllm_ascend`` and no symbol is replaced.

Why it exists: A5 reports ``MoECommPolicy.CAPACITY_AND_WORLD_SIZE``, which can
pick MC2 for small decode batches and ALLGATHER for large prefill batches. AFD's
Attention-owned gate supports one path at a time, so a smoke run has to record
the selected type, the capacity threshold behind it, and the implementation
object that will run -- per process, because the engine core is spawned.

The probe only observes. It does not change the selection, the prepare path, or
any tensor, so a run with it enabled is comparable to a run with it disabled.

Reported fields per distinct ``(num_tokens, selection)`` pair: the row count the
selector saw, ``get_mc2_tokens_capacity()``, the active policy, the selected
type, and the class ``get_moe_comm_method`` would hand back.

Graph replay does not re-run this Python code, so replay alone prints nothing.
Capture and every eager step do run it, so each distinct batch size appears at
least once before its graph is replayed.
"""

from __future__ import annotations

import os

PROBE_ENV_VAR = "AFD_DEBUG_MOE_COMM"

# Response pairs are reported once per process so a long run prints bounded
# output. Both are deliberately process-local probe state and are never read
# back; they are not configuration.
_REPORTED: set[tuple[int, str]] = set()
_INSTALLED = False
_CONSUMED: set[str] = set()


def report_consumer_comm_method(comm_method: object) -> None:
    """Report the comm method the FFN consumer is about to use.

    ``install_moe_comm_probe`` reports what the selector returned through the
    process-wide registry. That is not necessarily the object a model ends up
    running: when the model carries its own table, the forward context takes the
    object from there instead. This reports the object reached from the consumer,
    so a run can tell the two apart.

    Returns nothing and changes nothing; a no-op unless the probe is enabled.
    """

    if os.environ.get(PROBE_ENV_VAR) != "1":
        return

    name = type(comm_method).__name__
    if name in _CONSUMED:
        return
    _CONSUMED.add(name)
    print(
        "[AFD-P1] "
        f"pid={os.getpid()} "
        f"consumer_comm_method={name}",
        flush=True,
    )


def install_moe_comm_probe() -> bool:
    """Wrap the Ascend selector when ``AFD_DEBUG_MOE_COMM=1``.

    Returns whether the probe is active in this process. Calling it more than
    once in a process installs a single wrapper.
    """

    if os.environ.get(PROBE_ENV_VAR) != "1":
        return False

    global _INSTALLED
    if _INSTALLED:
        return True

    import vllm_ascend.ascend_forward_context as forward_context
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
            implementation = (
                None if selected is None else type(get_moe_comm_method(selected)).__name__
            )
            print(
                "[AFD-P0] "
                f"pid={os.getpid()} "
                f"num_tokens={num_tokens} "
                f"capacity={forward_context.get_mc2_tokens_capacity()} "
                # Read the policy through the module the selector itself reads.
                # A hardware-profile override such as force_allgather replaces the
                # getter here, so reading the original module would print the
                # unoverridden policy next to an overridden selection.
                f"policy={forward_context.get_current_hardware_profile().moe_comm_policy} "
                f"type={selected} "
                f"impl={implementation}",
                flush=True,
            )
        return selected

    # The selector builds the forward context, so wrapping it observes the path
    # the model will actually use for this step.
    forward_context.select_moe_comm_method = probe_select_moe_comm_method
    _INSTALLED = True
    return True


__all__ = [
    "PROBE_ENV_VAR",
    "install_moe_comm_probe",
    "report_consumer_comm_method",
]
