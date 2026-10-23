# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Run the Attention model's profile pass without compilation.

Upstream source: vllm_ascend/worker/v2/model_runner.py (8d4409d),
``NPUModelRunner.profile_run``.

Patch reason: the profile pass runs with ``in_profile_run=True`` so that MoE
load is spread over every expert instead of whatever the startup inputs happen
to hit. Only the extra MC2 dummy run is wrapped in ``disable_compilation``; the
standard ``super().profile_run()`` that follows is not, so it can be the first
call to compile the Attention model. Compilation traces ``in_profile_run`` as a
constant, and the compiled wrapper does not re-trace when the flag goes back to
False, so the balanced expert ids stay substituted for real requests and
override the live routing.

Patch functionality: hold compilation off for the whole profile pass. The
balanced routing then stays an eager-only behaviour, and the first compile
happens on the formal warmup, where the flag is already False.

Signature: matches upstream; no added parameters.
"""

from __future__ import annotations

from vllm_ascend.ascend_forward_context import (
    MoECommType,
    get_mc2_tokens_capacity,
    override_mrv2_in_profile_run,
    select_moe_comm_method,
)
from vllm_ascend.worker.utils import disable_compilation
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


# Patch reason: see the module docstring. The AFD difference is the guard around
# the whole body; everything inside is the upstream body unchanged.
# Patch functionality: forward with compilation disabled for the whole pass.
# Signature: matches upstream; no added parameters.
def profile_run(self) -> None:
    # ### PATCH START: keep the profile pass out of compilation
    with disable_compilation(self.get_model()):
        # ### PATCH END: keep the profile pass out of compilation
        mc2_tokens_capacity = get_mc2_tokens_capacity()
        with override_mrv2_in_profile_run(True):
            if (
                mc2_tokens_capacity is not None
                and self.max_num_tokens > mc2_tokens_capacity
                and select_moe_comm_method(mc2_tokens_capacity, self.vllm_config)
                in {MoECommType.MC2, MoECommType.FUSED_MC2}
            ):
                # Use a call-scoped bypass because skip_compiled would require runner-specific ForwardContext plumbing.
                with disable_compilation(self.get_model()):
                    self._dummy_run(
                        mc2_tokens_capacity,
                        skip_attn=True,
                        skip_eplb=True,
                        is_profile=True,
                    )
            super(NPUModelRunner, self).profile_run()


def apply_profile_run_eager_patch() -> bool:
    """Install the patch once. Returns whether it is installed."""

    if NPUModelRunner.profile_run is profile_run:
        return True
    NPUModelRunner.profile_run = profile_run
    return True


__all__ = ["apply_profile_run_eager_patch", "profile_run"]
