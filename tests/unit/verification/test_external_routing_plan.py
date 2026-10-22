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
    OutcomeSummary,
    VerificationPlan,
    VerificationRecord,
    agree_plan,
    build_plan,
    decide_together,
    precheck,
    summarize,
)
from afd_plugin.verification.external_routing_consumer import (
    SUPPORTED_COMM_METHODS as _SUPPORTED_COMM_METHODS,
)
from afd_plugin.verification.external_routing_ffn import ComparisonResult

DP_SIZE = 4


def _plan(**overrides) -> VerificationPlan:
    arguments = {"seed": 7, "atol": 1e-2, "rtol": 1e-3}
    arguments.update(overrides)
    return build_plan(DP_SIZE, **arguments)


RANKS = list(range(DP_SIZE))


def _entries(plan: VerificationPlan, *, passed: bool) -> list[tuple[int, str, bool]]:
    """Every rank reporting every planned case with the same verdict."""

    return [
        (rank, f"{case}/{scaling}", passed)
        for rank in RANKS
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


def _echo(vote):
    """Every rank reached the same answer, so the exchange echoes it back."""

    return [vote] * DP_SIZE


def test_every_rank_runs_when_none_refuses_and_the_plans_match():
    plan = _plan()

    refusal = decide_together(
        dp_rank=0,
        precheck_reason=None,
        plan=plan,
        exchange=_echo,
    )

    assert refusal is None


def test_a_rank_that_refuses_stops_the_whole_run():
    """The refusing rank must not exit alone: the others would wait for it."""

    plan = _plan()

    def exchange(vote):
        refuser = type(vote)(rank=2, reason="comm method is MC2CommImpl", plan=None)
        return [vote, vote, refuser, vote]

    refusal = decide_together(
        dp_rank=0,
        precheck_reason=None,
        plan=plan,
        exchange=exchange,
    )

    assert refusal is not None
    assert "rank 2" in refusal
    assert "MC2CommImpl" in refusal


def test_a_local_refusal_still_takes_part_in_the_exchange():
    """Refusing locally must not skip the vote, or the others block forever."""

    plan = _plan()
    seen: list[object] = []

    def exchange(vote):
        seen.append(vote)
        return [vote] * DP_SIZE

    refusal = decide_together(
        dp_rank=1,
        precheck_reason="tp_size 2: routing rows are not sharded by TP here",
        plan=plan,
        exchange=exchange,
    )

    assert seen, "the vote must still be exchanged"
    assert refusal is not None
    assert "rank 1" in refusal


def test_an_unbuildable_plan_travels_as_a_refusal():
    refusal = decide_together(
        dp_rank=3,
        precheck_reason=None,
        plan=None,
        exchange=_echo,
    )

    assert refusal is not None
    assert "rank 3" in refusal


def test_differing_plans_without_a_refusal_are_reported_as_a_plan_mismatch():
    plan = _plan()

    def exchange(vote):
        other = _plan(seed=99)
        mismatched = type(vote)(
            rank=1,
            reason=None,
            plan=other.signature(),
        )
        return [vote, mismatched, vote, vote]

    refusal = decide_together(
        dp_rank=0,
        precheck_reason=None,
        plan=plan,
        exchange=exchange,
    )

    assert refusal is not None
    assert "planned differently" in refusal


def test_an_unknown_comm_method_is_refused():
    reason = precheck(
        comm_type_name="AlltoAllCommImpl",
        dp_size=DP_SIZE,
        tp_size=1,
        pcp_size=1,
        sequence_parallel=False,
        captured=True,
    )

    assert reason is not None
    assert "AlltoAllCommImpl" in reason


@pytest.mark.parametrize("comm_type_name", sorted(_SUPPORTED_COMM_METHODS))
def test_each_wired_comm_method_passes_the_topology_check(comm_type_name):
    """ALLGATHER and MC2 share the consumer flow, so both have to be accepted."""

    assert (
        precheck(
            comm_type_name=comm_type_name,
            dp_size=DP_SIZE,
            tp_size=1,
            pcp_size=1,
            sequence_parallel=False,
            captured=True,
        )
        is None
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

    summary = summarize(plan, _entries(plan, passed=True), [], RANKS)

    assert summary.failed is False
    assert summary.reasons == []


def test_a_skipped_case_fails_the_run():
    """Reporting nothing as a pass is how an entry point goes green having checked
    nothing."""

    plan = _plan()

    summary = summarize(plan, _entries(plan, passed=True), ["dp_size is 1"], RANKS)

    assert summary.failed is True
    assert any("dp_size is 1" in reason for reason in summary.reasons)


def test_a_case_missing_on_one_rank_fails_the_run():
    plan = _plan()
    entries = _entries(plan, passed=True)
    dropped = entries[0]
    entries.remove(dropped)

    summary = summarize(plan, entries, [], RANKS)

    assert summary.failed is True
    assert any(dropped[1] in reason for reason in summary.reasons)


def test_a_case_reported_twice_by_one_rank_fails_the_run():
    """A count-based check would miss this: one duplicate can hide one omission."""

    plan = _plan()
    entries = _entries(plan, passed=True)
    duplicated = entries[0]
    entries = [entry for entry in entries if entry != entries[1]]
    entries.append(duplicated)

    summary = summarize(plan, entries, [], RANKS)

    assert summary.failed is True


def test_a_failing_case_on_a_remote_rank_fails_the_run():
    """A local view cannot see this one, which is the whole reason for the exchange."""

    plan = _plan()
    entries = _entries(plan, passed=True)
    remote = RANKS[-1]
    case = entries[0][1]
    entries = [
        (rank, name, False if (rank == remote and name == case) else passed)
        for rank, name, passed in entries
    ]

    summary = summarize(plan, entries, [], RANKS)

    assert summary.failed is True
    assert any(str(remote) in reason for reason in summary.reasons)


def test_a_case_outside_the_plan_fails_the_run():
    plan = _plan()
    entries = _entries(plan, passed=True)
    entries.append((0, "invented/shape", True))

    summary = summarize(plan, entries, [], RANKS)

    assert summary.failed is True
    assert any("outside the plan" in reason for reason in summary.reasons)


def test_the_record_is_not_passed_before_a_verdict_exists():
    """An unrun entry point must not read as a pass."""

    record = VerificationRecord(versions={}, configuration={})
    record.report.add(
        ComparisonResult(
            case="uniform/configured",
            passed=True,
            max_abs_error=0.0,
            max_rel_error=0.0,
            shape_ok=True,
            finite_ok=True,
            detail="",
        ),
    )

    assert record.passed is False

    record.summary = OutcomeSummary(failed=False)

    assert record.passed is True
