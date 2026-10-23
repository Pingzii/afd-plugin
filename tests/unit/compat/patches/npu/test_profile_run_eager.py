# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""The profile pass must not compile, and must leave real routing behind.

The patch under test wraps the Attention model's profile pass in
``disable_compilation``. Two things then have to hold at the same time:

* during the profile pass, the balanced expert ids are still substituted, so the
  memory peak is measured against a spread-out MoE load;
* the first compile happens on the formal warmup, with the profile flag already
  off, so the ids the graph carries are live ones.

A test that only checked the second would pass if the patch disabled the
balancing instead of the compilation, which is the failure mode being avoided.

Needs vLLM-Ascend, so it is collected but skipped on a CPU-only host.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm_ascend.worker.v2.model_runner")


class _FakeModel:
    """Stands in for the runner's model, which is what carries ``do_not_compile``."""

    def __init__(self) -> None:
        self.do_not_compile = False
        self.seen_flags: list[bool] = []


class _FakeRunner:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.num_tokens = 0
        self.profile_ran = 0

    def get_model(self) -> _FakeModel:
        return self.model

    def _dummy_run(self, num_tokens, **kwargs) -> None:
        del kwargs
        self.num_tokens = num_tokens
        self.model.seen_flags.append(self.model.do_not_compile)

    def parent_profile_run(self) -> None:
        self.profile_ran += 1
        self.model.seen_flags.append(self.model.do_not_compile)


def test_the_whole_profile_pass_runs_with_compilation_off(monkeypatch):
    """Both the dummy run and the parent profile run see ``do_not_compile``."""

    from afd_plugin.compat.patches.npu import profile_run_eager

    runner = _FakeRunner()
    monkeypatch.setattr(
        profile_run_eager,
        "get_mc2_tokens_capacity",
        lambda: 128,
    )
    monkeypatch.setattr(
        profile_run_eager,
        "select_moe_comm_method",
        lambda *args, **kwargs: profile_run_eager.MoECommType.MC2,
    )
    monkeypatch.setattr(
        profile_run_eager.NPUModelRunner,
        "profile_run",
        lambda self: _FakeRunner.parent_profile_run(self),
    )
    monkeypatch.setattr(runner, "num_tokens", 4096)
    runner.max_num_tokens = 4096

    profile_run_eager.profile_run(runner)

    assert runner.profile_ran == 1
    # Every observation of the flag, including the parent's, is inside the guard.
    assert runner.model.seen_flags, "the profile pass observed nothing"
    assert all(runner.model.seen_flags), (
        "part of the profile pass ran with compilation enabled: "
        f"{runner.model.seen_flags}"
    )
    # The guard is released again, so later compilation is not blocked.
    assert runner.model.do_not_compile is False


def test_the_guard_is_released_when_the_profile_raises(monkeypatch):
    from afd_plugin.compat.patches.npu import profile_run_eager

    runner = _FakeRunner()

    def explode(_self):
        raise RuntimeError("profile exploded")

    monkeypatch.setattr(
        profile_run_eager.NPUModelRunner,
        "profile_run",
        explode,
    )

    with pytest.raises(RuntimeError, match="profile exploded"):
        profile_run_eager.profile_run(runner)

    assert runner.model.do_not_compile is False, (
        "a failed profile left compilation disabled for the process"
    )


def test_the_patch_is_idempotent():
    from afd_plugin.compat.patches.npu import profile_run_eager

    installed = profile_run_eager.NPUModelRunner.profile_run
    assert profile_run_eager.apply_profile_run_eager_patch()
    assert profile_run_eager.apply_profile_run_eager_patch()
    assert profile_run_eager.NPUModelRunner.profile_run is installed
