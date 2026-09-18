"""Fake-data conformance tests for regression tracking (issue #24 WP-C).

Exercises ``codevolt_mdf.regression_check.compare_for_regressions`` with
synthetic ``EvaluationOutput`` fixtures built directly (no adapter, no
real or fake model inference, no ``run_evaluator_contract`` call --
this module operates purely on already-produced ``EvaluationOutput``
objects, so its tests construct them directly, mirroring how
``tests/test_evaluator_contract.py`` builds fixtures for the contract
runner it tests).
"""

from __future__ import annotations

import pytest

from codevolt_mdf.evaluator_contract import EvaluationOutput, EvaluationStatus, ExampleResult
from codevolt_mdf.regression_check import (
    ExampleRegressionResult,
    IncomparableEvaluationsError,
    RegressionSeverity,
    compare_for_regressions,
)


def make_output(
    *,
    status: EvaluationStatus = EvaluationStatus.SCORED,
    package_id: str | None = "PKG-1",
    artifact_id: str | None = "art-x",
    results: dict[str, bool] | None = None,
    reason: str = "scored",
) -> EvaluationOutput:
    results = results or {}
    example_results = tuple(
        ExampleResult(example_id=eid, correct=correct, score=1.0 if correct else 0.0)
        for eid, correct in results.items()
    )
    aggregate = (
        sum(r.score for r in example_results) / len(example_results) if example_results else None
    )
    return EvaluationOutput(
        status=status,
        reason=reason,
        artifact_id=artifact_id,
        package_id=package_id,
        aggregate_score=aggregate,
        results=example_results,
    )


# 1. Baseline-only comparison (no previous candidate supplied) ---------------


def test_no_regression_when_candidate_still_passes_everything():
    baseline = make_output(
        artifact_id="baseline", results={"e1": True, "e2": True, "e3": False}
    )
    candidate = make_output(
        artifact_id="cand-1", results={"e1": True, "e2": True, "e3": True}
    )

    report = compare_for_regressions(baseline, candidate)

    assert report.regression_count == 0
    assert report.regressed_example_ids == ()
    # e3 was already failing in baseline -> excluded from "previously passing".
    assert report.previously_passing_count == 2
    by_id = {d.example_id: d for d in report.details}
    assert by_id["e3"].severity == RegressionSeverity.NOT_PREVIOUSLY_PASSING
    assert by_id["e1"].severity == RegressionSeverity.STILL_PASSING
    assert by_id["e2"].severity == RegressionSeverity.STILL_PASSING


def test_regression_detected_when_candidate_breaks_a_previously_passing_example():
    baseline = make_output(artifact_id="baseline", results={"e1": True, "e2": True})
    candidate = make_output(artifact_id="cand-1", results={"e1": True, "e2": False})

    report = compare_for_regressions(baseline, candidate)

    assert report.regression_count == 1
    assert report.regressed_example_ids == ("e2",)
    assert report.previously_passing_count == 2
    by_id = {d.example_id: d for d in report.details}
    assert by_id["e2"] == ExampleRegressionResult(
        example_id="e2",
        severity=RegressionSeverity.REGRESSED,
        baseline_correct=True,
        candidate_correct=False,
        previous_correct=None,
    )


def test_not_a_regression_when_example_was_already_failing_in_baseline():
    baseline = make_output(artifact_id="baseline", results={"e1": False})
    candidate = make_output(artifact_id="cand-1", results={"e1": False})

    report = compare_for_regressions(baseline, candidate)

    assert report.regression_count == 0
    assert report.previously_passing_count == 0
    assert report.details[0].severity == RegressionSeverity.NOT_PREVIOUSLY_PASSING


def test_examples_absent_from_one_side_are_silently_excluded_not_errors():
    baseline = make_output(artifact_id="baseline", results={"e1": True, "e2": True})
    candidate = make_output(artifact_id="cand-1", results={"e1": True})  # e2 dropped

    report = compare_for_regressions(baseline, candidate)

    assert report.regression_count == 0
    assert report.previously_passing_count == 1
    assert {d.example_id for d in report.details} == {"e1"}


# 2. Three-way comparison: baseline + previous accepted candidate + candidate --


def test_three_way_regression_requires_passing_in_both_baseline_and_previous():
    baseline = make_output(artifact_id="baseline", results={"e1": True, "e2": True, "e3": False})
    previous = make_output(artifact_id="prev-1", results={"e1": True, "e2": False, "e3": True})
    candidate = make_output(artifact_id="cand-2", results={"e1": False, "e2": False, "e3": True})

    report = compare_for_regressions(baseline, candidate, previous_candidate=previous)

    # e1: passing in baseline AND previous -> now fails -> regression.
    # e2: passing in baseline but NOT previous -> not "previously passing" (3-way rule).
    # e3: failing in baseline -> not previously passing regardless of previous/candidate.
    assert report.regression_count == 1
    assert report.regressed_example_ids == ("e1",)
    assert report.previously_passing_count == 1
    by_id = {d.example_id: d for d in report.details}
    assert by_id["e2"].severity == RegressionSeverity.NOT_PREVIOUSLY_PASSING
    assert by_id["e3"].severity == RegressionSeverity.NOT_PREVIOUSLY_PASSING
    assert report.has_previous_candidate is True
    assert report.previous_artifact_id == "prev-1"


def test_three_way_no_regression_when_all_still_pass():
    baseline = make_output(artifact_id="baseline", results={"e1": True})
    previous = make_output(artifact_id="prev-1", results={"e1": True})
    candidate = make_output(artifact_id="cand-2", results={"e1": True})

    report = compare_for_regressions(baseline, candidate, previous_candidate=previous)

    assert report.regression_count == 0
    assert report.previously_passing_count == 1


# 3. Tamper / malformed-input rejection paths ---------------------------------


def test_rejects_non_scored_baseline():
    baseline = make_output(status=EvaluationStatus.INVALID, reason="held-out tamper")
    candidate = make_output(results={"e1": True})

    with pytest.raises(IncomparableEvaluationsError, match="baseline"):
        compare_for_regressions(baseline, candidate)


def test_rejects_non_scored_candidate():
    baseline = make_output(results={"e1": True})
    candidate = make_output(status=EvaluationStatus.REJECTED, reason="unsupported artifact")

    with pytest.raises(IncomparableEvaluationsError, match="candidate"):
        compare_for_regressions(baseline, candidate)


def test_rejects_non_scored_previous_candidate():
    baseline = make_output(results={"e1": True})
    candidate = make_output(results={"e1": True})
    previous = make_output(status=EvaluationStatus.INVALID, reason="contaminated")

    with pytest.raises(IncomparableEvaluationsError, match="previous_candidate"):
        compare_for_regressions(baseline, candidate, previous_candidate=previous)


def test_rejects_scored_output_with_zero_results():
    baseline = make_output(results={})  # SCORED but empty -- shouldn't happen, but defend anyway
    candidate = make_output(results={"e1": True})

    with pytest.raises(IncomparableEvaluationsError, match="zero results"):
        compare_for_regressions(baseline, candidate)


def test_rejects_mismatched_package_id_between_baseline_and_candidate():
    baseline = make_output(package_id="PKG-A", results={"e1": True})
    candidate = make_output(package_id="PKG-B", results={"e1": True})

    with pytest.raises(IncomparableEvaluationsError, match="different held-out sets"):
        compare_for_regressions(baseline, candidate)


def test_rejects_mismatched_package_id_for_previous_candidate():
    baseline = make_output(package_id="PKG-A", results={"e1": True})
    candidate = make_output(package_id="PKG-A", results={"e1": True})
    previous = make_output(package_id="PKG-OTHER", results={"e1": True})

    with pytest.raises(IncomparableEvaluationsError, match="different held-out sets"):
        compare_for_regressions(baseline, candidate, previous_candidate=previous)


# 4. Measurement-only boundary: no accept/reject authority --------------------


def test_regression_report_exposes_no_accept_reject_field():
    """RegressionReport must only expose measurement fields -- no verdict,
    accept/reject/promote field or method, per docs/ARCHITECTURE.md core
    contracts #3/#6 and docs/EVALUATION_POLICY.md's decision boundary."""
    baseline = make_output(results={"e1": True})
    candidate = make_output(results={"e1": False})
    report = compare_for_regressions(baseline, candidate)

    field_names = set(report.__dataclass_fields__.keys())
    forbidden = {"accepted", "rejected", "decision", "promote", "promoted", "verdict"}
    assert field_names.isdisjoint(forbidden)


def test_compare_for_regressions_module_has_no_import_relationship_with_trainer_or_promotion():
    """AST-import test, same pattern as test_evaluator_contract.py's,
    proving this module never imports the trainer side or anything
    promotion-shaped."""
    import ast

    import codevolt_mdf.regression_check as regression_check_module

    with open(regression_check_module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())

    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert not any("trl_adapter" in name for name in imported_modules)
    assert not any("trainer_contract" in name for name in imported_modules)
    assert not any("promot" in name for name in imported_modules)
