# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""The plan and verdict rules, which have to agree across ranks.

These decide whether a verification run is allowed to start and what it reports. They
are pure functions so every rule is exercised without a collective: a rank that reaches
a different answer from the same facts is exactly the failure this file prevents.
"""

from __future__ import annotations

import pytest

from afd_plugin.verification.external_routing_consumer import (
    LAYER_CASES,
    VerificationPlan,
    agree_plan,
    build_plan,
    precheck,
    summarize,
)
from afd_plugin.verification.external_routing_ffn import ComparisonResult

DP_SIZE = 4


def _plan(**overrides) -> VerificationPlan:
    arguments = {"seed": 7, "atol": 1e-2, "rtol": 1e-3}
    arguments.update(overrides)
    return build_plan(DP_SIZE, **arguments)


def _result(case: str, *, passed: bool) -> ComparisonResult:
    return ComparisonResult(
        case=case,
        passed=passed,
        max_abs_error=0.0 if passed else 1.0,
        max_rel_error=0.0 if passed else 1.0,
        shape_ok=True,
        finite_ok=True,
        detail="",
    )


def _all_cases(plan: VerificationPlan, *, passed: bool) -> list[ComparisonResult]:
    return [
        _result(f"{case}/{scaling}", passed=passed)
        for case in plan.case_order
        for scaling, _ in plan.scaling
    ]


def test_the_plan_covers_every_case_at_this_topology():
    plan = _plan()

    assert plan.case_order == tuple(name for name, _ in LAYER_CASES)
    for rows in plan.rows_by_case:
        assert len(rows) == DP_SIZE
    assert plan.seed == 7


def test_a_topology_without_its_own_row_counts_is_refused():
    """Folding a table onto another DP size would reuse a shape nobody chose."""

    with pytest.raises(ValueError, match="instead of reusing another one"):
        build_plan(3, seed=0, atol=1e-2, rtol=1e-3)


def test_a_single_rank_has_nothing_to_align():
    with pytest.raises(ValueError, match="no ranks to align"):
        build_plan(1, seed=0, atol=1e-2, rtol=1e-3)


def test_each_rank_reads_its_own_row_count_from_the_shared_plan():
    """Different row counts per rank are fine; different plans are not."""

    plan = _plan()
    case = plan.case_order[1]

    rows = [plan.rows_for(case, rank) for rank in range(DP_SIZE)]

    assert rows == list(LAYER_CASES[1][1])
    # The uneven case owns an empty rank, which must stay a real case rather than be
    # clamped away.
    assert 0 in rows


def test_matching_plans_agree():
    plan = _plan()

    assert agree_plan(plan, lambda value: [value] * DP_SIZE) is None


def test_a_rank_with_a_different_plan_is_named():
    plan = _plan()
    other = _plan(seed=99).signature()

    def exchange(value):
        return [value, other, value, value]

    reason = agree_plan(plan, exchange)

    assert reason is not None
    assert "rank 1" in reason


def test_an_unknown_comm_method_is_refused():
    assert "AllGatherCommImpl" in precheck(
        comm_type_name="MC2CommImpl",
        dp_size=DP_SIZE,
        tp_size=1,
        pcp_size=1,
        sequence_parallel=False,
        captured=True,
    )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("tp_size", 2, "tp_size"),
        ("pcp_size", 2, "pcp_size"),
        ("sequence_parallel", True, "sequence parallel"),
        ("captured", False, "not captured"),
    ],
)
def test_an_uncovered_topology_is_refused_by_name(field, value, expected):
    arguments = {
        "comm_type_name": "AllGatherCommImpl",
        "dp_size": DP_SIZE,
        "tp_size": 1,
        "pcp_size": 1,
        "sequence_parallel": False,
        "captured": True,
    }
    arguments[field] = value

    reason = precheck(**arguments)

    assert reason is not None
    assert expected in reason


def test_a_supported_topology_passes_the_precheck():
    assert (
        precheck(
            comm_type_name="AllGatherCommImpl",
            dp_size=DP_SIZE,
            tp_size=1,
            pcp_size=1,
            sequence_parallel=False,
            captured=True,
        )
        is None
    )


def test_a_complete_matching_run_passes():
    plan = _plan()

    summary = summarize(plan, _all_cases(plan, passed=True), [])

    assert summary.failed is False
    assert summary.reasons == []


def test_a_skipped_case_fails_the_run():
    """Reporting nothing as a pass is how an entry point goes green having checked
    nothing."""

    plan = _plan()

    summary = summarize(plan, _all_cases(plan, passed=True), ["dp_size is 1"])

    assert summary.failed is True
    assert any("dp_size is 1" in reason for reason in summary.reasons)


def test_a_missing_case_fails_the_run():
    plan = _plan()
    results = _all_cases(plan, passed=True)[:-1]

    summary = summarize(plan, results, [])

    assert summary.failed is True
    assert any("of" in reason for reason in summary.reasons)


def test_one_failing_case_fails_the_run():
    plan = _plan()
    results = _all_cases(plan, passed=True)
    results[1] = _result(results[1].case, passed=False)

    summary = summarize(plan, results, [])

    assert summary.failed is True
    assert any(results[1].case in reason for reason in summary.reasons)
