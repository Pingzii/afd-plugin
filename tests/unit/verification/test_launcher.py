# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""The launcher's device-free decisions: run id, layer choice, exit codes, aggregation.

These decide what a run reports. The bootstrap that loads a model is a device concern;
what is checked here is that a run cannot report success from a subset of ranks, a
mismatched plan, or an unexecuted verification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from afd_plugin.verification.launcher import (
    EXIT_MISMATCH,
    EXIT_NOT_EXECUTED,
    EXIT_ORGANIZATION,
    EXIT_PASS,
    EXIT_UNSUPPORTED,
    aggregate,
    exit_code_for,
    resolve_output_dir,
    resolve_run_id,
    select_layer,
)
from afd_plugin.verification.external_routing_consumer import (
    OutcomeSummary,
    VerificationRecord,
)

DP_SIZE = 4


class _Model:
    def __init__(self, indices):
        self._indices = indices

    def get_experts_layer_indices(self):
        return tuple(self._indices)


def _payload(*, exit_code, plan="same", versions=None):
    return {
        "exit_code": exit_code,
        "plan_signature": plan,
        "versions": versions or {"vllm": "a", "afd-plugin": "b"},
        "reasons": [] if exit_code == EXIT_PASS else [f"exit {exit_code}"],
    }


def _write_all(directory: Path, payload, *, ranks=DP_SIZE):
    for rank in range(ranks):
        (directory / f"rank{rank}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )


def test_the_run_id_is_shared_when_given():
    assert resolve_run_id("run-7") == "run-7"


def test_a_generated_run_id_is_not_empty():
    assert resolve_run_id(None)


def test_the_output_directory_puts_the_run_id_under_the_root(tmp_path):
    assert resolve_output_dir(None, "r1", root=tmp_path) == tmp_path / "verify-runs" / "r1"


def test_an_explicit_output_directory_overrides_the_root(tmp_path):
    assert resolve_output_dir(str(tmp_path / "x"), "r1") == tmp_path / "x" / "r1"


def test_the_first_moe_layer_is_the_default():
    assert select_layer(_Model([3, 5, 7]), None) == 3


def test_an_explicit_layer_must_be_an_moe_layer():
    with pytest.raises(ValueError, match="not an MoE layer"):
        select_layer(_Model([3, 5, 7]), 4)


def test_a_model_without_moe_layers_is_refused():
    with pytest.raises(ValueError, match="no MoE layers"):
        select_layer(_Model([]), None)


def test_the_reported_layer_is_the_model_index_not_a_list_position():
    """A model whose MoE layers start at 3 must report 3, not 0."""

    assert select_layer(_Model([3, 5, 7]), None) == 3


def _record(*, topology_refusal=False, summary=None):
    record = VerificationRecord(versions={}, configuration={})
    record.topology_refusal = topology_refusal
    record.summary = summary
    return record


def test_an_unexecuted_run_maps_to_its_own_code():
    assert exit_code_for(_record()) == EXIT_NOT_EXECUTED


def test_a_topology_refusal_maps_to_unsupported():
    assert exit_code_for(_record(topology_refusal=True)) == EXIT_UNSUPPORTED


def test_a_failed_summary_maps_to_mismatch():
    assert exit_code_for(_record(summary=OutcomeSummary(failed=True))) == EXIT_MISMATCH


def test_a_clean_summary_maps_to_pass():
    assert exit_code_for(_record(summary=OutcomeSummary(failed=False))) == EXIT_PASS


def test_a_missing_rank_file_fails_the_aggregation(tmp_path):
    _write_all(tmp_path, _payload(exit_code=EXIT_PASS), ranks=DP_SIZE - 1)

    code, reasons = aggregate(tmp_path, DP_SIZE)

    assert code == EXIT_ORGANIZATION
    assert any("no result" in reason for reason in reasons)


def test_all_ranks_passing_aggregates_to_pass(tmp_path):
    _write_all(tmp_path, _payload(exit_code=EXIT_PASS))

    code, reasons = aggregate(tmp_path, DP_SIZE)

    assert code == EXIT_PASS
    assert reasons == []


def test_disagreeing_plans_fail_the_aggregation(tmp_path):
    for rank in range(DP_SIZE):
        plan = "same" if rank else "different"
        (tmp_path / f"rank{rank}.json").write_text(
            json.dumps(_payload(exit_code=EXIT_PASS, plan=plan)),
            encoding="utf-8",
        )

    code, reasons = aggregate(tmp_path, DP_SIZE)

    assert code != EXIT_PASS
    assert any("different plans" in reason for reason in reasons)


def test_disagreeing_versions_fail_the_aggregation(tmp_path):
    for rank in range(DP_SIZE):
        versions = {"vllm": "a"} if rank else {"vllm": "z"}
        (tmp_path / f"rank{rank}.json").write_text(
            json.dumps(_payload(exit_code=EXIT_PASS, versions=versions)),
            encoding="utf-8",
        )

    code, reasons = aggregate(tmp_path, DP_SIZE)

    assert code != EXIT_PASS
    assert any("different versions" in reason for reason in reasons)


def test_one_rank_failing_fails_the_aggregation(tmp_path):
    for rank in range(DP_SIZE):
        code = EXIT_MISMATCH if rank == 2 else EXIT_PASS
        (tmp_path / f"rank{rank}.json").write_text(
            json.dumps(_payload(exit_code=code)),
            encoding="utf-8",
        )

    overall, reasons = aggregate(tmp_path, DP_SIZE)

    assert overall == EXIT_MISMATCH
    assert reasons


def test_a_timeout_is_never_a_pass(tmp_path):
    """124 comes from the caller's timeout; it must not aggregate into success."""

    _write_all(tmp_path, _payload(exit_code=124))

    code, _ = aggregate(tmp_path, DP_SIZE)

    assert code != EXIT_PASS
