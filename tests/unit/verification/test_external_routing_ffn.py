# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""The device comparison's bookkeeping, checked without a device.

The comparison itself runs on an NPU host. Everything that decides *whether* a run
passes -- the case matrix, the shape and finiteness gates, and the tolerance check
-- is plain tensor logic, so it is checked here.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "afd_plugin"
    / "verification"
    / "external_routing_ffn.py"
)
_MODULE_NAME = "afd_external_routing_ffn_under_test"


@pytest.fixture(scope="module")
def verification():
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve field types through sys.modules[cls.__module__], so the
    # module must be registered before it executes.
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(_MODULE_NAME, None)
    return module


def test_case_matrix_covers_uneven_rows_and_an_empty_rank(verification):
    cases = verification.DEFAULT_CASES
    names = {case.name for case in cases}

    assert "uniform_bf16" in names
    assert "uneven_bf16" in names
    assert "uneven_fp8" in names
    uneven = next(case for case in cases if case.name == "uneven_bf16")
    assert min(uneven.tokens_per_rank) == 0
    assert len(set(uneven.tokens_per_rank)) > 1
    # Both precisions are represented, because only FP8 carries a scale.
    assert {case.with_scale for case in cases} == {True, False}


def test_gathered_rows_follow_the_padding_rule(verification):
    """dp_size * max(local_rows), not the sum of the local rows."""

    case = verification.RoutingCase(
        name="uneven",
        tokens_per_rank=(3, 1, 2, 2),
        top_k=8,
        with_scale=False,
        note="",
    )

    assert case.dp_size == 4
    assert case.max_rows == 3
    assert case.gathered_rows == 12
    assert case.gathered_rows != sum(case.tokens_per_rank)


def test_identical_routing_passes(verification):
    reference = torch.ones(4, 8, dtype=torch.bfloat16)
    result = verification.compare_routed(reference, reference.clone(), case="same")

    assert result.passed
    assert result.shape_ok and result.finite_ok
    assert result.max_abs_error == 0.0


def test_a_row_count_difference_fails_before_any_error_number(verification):
    result = verification.compare_routed(
        torch.ones(4, 8),
        torch.ones(6, 8),
        case="rows",
    )

    assert not result.passed
    assert not result.shape_ok
    assert "shape" in result.detail


def test_non_finite_candidate_fails(verification):
    candidate = torch.ones(2, 8)
    candidate[0, 0] = float("nan")

    result = verification.compare_routed(torch.ones(2, 8), candidate, case="nan")

    assert not result.passed
    assert result.shape_ok
    assert not result.finite_ok


def test_error_beyond_tolerance_fails_and_reports_magnitudes(verification):
    reference = torch.full((2, 4), 1.0)
    candidate = reference + 0.5

    result = verification.compare_routed(reference, candidate, case="off")

    assert not result.passed
    assert result.max_abs_error == pytest.approx(0.5, rel=1e-6)
    assert result.max_rel_error == pytest.approx(0.5, rel=1e-6)


def test_error_within_tolerance_passes(verification):
    reference = torch.full((2, 4), 1.0)
    candidate = reference + 1e-4

    result = verification.compare_routed(reference, candidate, case="near")

    assert result.passed
    assert result.max_abs_error == pytest.approx(1e-4, rel=1e-3)


def test_report_aggregates_and_renders(verification):
    report = verification.CaseReport()
    report.add(verification.compare_routed(torch.ones(2, 2), torch.ones(2, 2), case="a"))
    assert report.passed
    assert "PASS a" in report.render()

    report.add(verification.compare_routed(torch.ones(2, 2), torch.ones(3, 2), case="b"))
    assert not report.passed
    assert "at least one case failed" in report.render()


def test_dry_run_names_every_case_and_the_reference_requirement(verification):
    rendered = verification.render_dry_run()

    for case in verification.DEFAULT_CASES:
        assert case.name in rendered
    # The independence requirement must survive into the printed plan.
    assert "unpatched prepare" in rendered
    assert "ALLGATHER -> MC2 -> ALLGATHER" in rendered
