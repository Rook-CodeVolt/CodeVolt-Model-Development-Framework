"""Regression tracking across candidate revisions (issue #24 WP-C).

This module is a *measurement/reporting* capability only. It compares a
candidate's already-produced ``EvaluationOutput`` (from
``evaluator_contract.run_evaluator_contract``) against both the
immutable baseline's ``EvaluationOutput`` and the previous accepted
candidate's ``EvaluationOutput``, restricted to examples that were
*previously passing*, and reports which of those have regressed.

It does not run any evaluation itself (no adapter call, no model
inference), does not compute an aggregate accept/reject verdict, and
has no promotion authority of any kind -- consistent with
``docs/ARCHITECTURE.md`` core contracts #3 ("An evaluator produces
measurements; it does not promote the candidate") and #6 ("Promotion
is a separate governed action"), and with ``docs/EVALUATION_POLICY.md``'s
"Regression checks for retained capabilities" being one of several
*required evaluation dimensions* that a later, separate acceptance
decision consumes -- this module produces exactly one of those
dimensions' evidence, nothing more.

Deliberately not a new ``TaskType``: WP-A's task types describe how one
``HeldOutExample`` is scored *within* a single evaluator run.
Regression tracking compares *across* two or three already-completed
runs' ``EvaluationOutput`` objects -- there is no new example-scoring
mode to add, so this is additive stand-alone logic that only *consumes*
existing ``evaluator_contract`` types (``EvaluationOutput``,
``EvaluationStatus``, ``ExampleResult``). Nothing in
``evaluator_contract.py`` changes shape; ``CONTRACT_VERSION`` stays
unchanged. See ``docs/decisions/0009-regression-tracking-across-candidate-revisions.md``.

Design summary:

- ``RegressionStatus`` distinguishes "cannot compare" (``invalid``, e.g.
  a run wasn't ``SCORED``, or the two runs don't share a held-out
  identity) from "compared" outcomes.
- ``compare_for_regressions`` takes a *baseline* ``EvaluationOutput``,
  an optional *previous-accepted-candidate* ``EvaluationOutput``, and a
  *candidate* ``EvaluationOutput``, all for the *same held-out set*
  (verified by matching ``package_id`` and by matching the example-id
  sets present in ``results`` -- not by trusting a caller-supplied
  label). It restricts attention to examples that were passing in
  baseline (and, when supplied, also passing in the previous
  candidate), and reports, per such example, whether the new candidate
  now fails it.
- No thresholds, no aggregate pass/fail verdict over the whole
  comparison: the caller (a separate, later governed acceptance/
  promotion step) decides what a `RegressionReport` implies. This
  module raises no exception and returns no field that could be read
  as "reject" or "accept" -- only counts and per-example detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .evaluator_contract import EvaluationOutput, EvaluationStatus, ExampleResult


class RegressionCheckError(Exception):
    """Base class for every error this module defines."""


class IncomparableEvaluationsError(RegressionCheckError):
    """Two ``EvaluationOutput`` objects cannot be meaningfully compared.

    Raised when a supplied ``EvaluationOutput`` was not ``SCORED``
    (e.g. ``INVALID``/``REJECTED`` -- there is no trustworthy per-example
    evidence to compare), or when the baseline/previous/candidate runs
    do not share the same ``package_id`` (comparing across different
    held-out sets would produce a meaningless, silently-wrong report,
    the same class of contamination-adjacent trust problem
    ``evaluator_contract.py`` guards against for a single run).
    """


class RegressionSeverity(str, Enum):
    """Per-example comparison outcome for a previously-passing example."""

    STILL_PASSING = "still_passing"
    """Passing in baseline (and previous candidate, if supplied) and
    still passing in the new candidate. Not a regression."""

    REGRESSED = "regressed"
    """Was passing in baseline (and previous candidate, if supplied);
    the new candidate now fails it."""

    NOT_PREVIOUSLY_PASSING = "not_previously_passing"
    """Excluded from the "previously passing" universe: it was already
    failing in baseline (or, when a previous candidate is supplied, in
    the previous candidate) so a candidate failing it too is not a new
    regression. Reported for transparency, never counted as a
    regression."""


@dataclass(frozen=True)
class ExampleRegressionResult:
    """Per-example regression-comparison detail for one held-out example."""

    example_id: str
    severity: RegressionSeverity
    baseline_correct: bool
    candidate_correct: bool
    previous_correct: bool | None = None
    """``None`` when no previous-accepted-candidate run was supplied
    (baseline-only comparison)."""


@dataclass(frozen=True)
class RegressionReport:
    """Measurement-only regression report. Carries no accept/reject verdict.

    ``regressed_example_ids`` and ``regression_count`` are the two
    fields a later, separate acceptance/promotion decision would read
    to apply *its own* threshold -- this report does not apply one
    itself, and this module exposes no method that could.
    """

    package_id: str | None
    artifact_id: str | None
    baseline_artifact_id: str | None
    previous_artifact_id: str | None
    previously_passing_count: int
    regression_count: int
    regressed_example_ids: tuple[str, ...]
    details: tuple[ExampleRegressionResult, ...]

    @property
    def has_previous_candidate(self) -> bool:
        return self.previous_artifact_id is not None


def _require_scored(label: str, evaluation: EvaluationOutput) -> None:
    if evaluation.status != EvaluationStatus.SCORED:
        raise IncomparableEvaluationsError(
            f"{label} EvaluationOutput has status={evaluation.status!r}, not "
            f"{EvaluationStatus.SCORED!r}; only a SCORED run carries trustworthy "
            f"per-example results to compare (reason={evaluation.reason!r})"
        )
    if not evaluation.results:
        raise IncomparableEvaluationsError(
            f"{label} EvaluationOutput is SCORED but has zero results; nothing to compare"
        )


def _require_same_package(baseline: EvaluationOutput, other: EvaluationOutput, label: str) -> None:
    if other.package_id != baseline.package_id:
        raise IncomparableEvaluationsError(
            f"{label} EvaluationOutput has package_id={other.package_id!r}, which does "
            f"not match baseline's package_id={baseline.package_id!r}; refusing to "
            f"compare regressions across two different held-out sets"
        )


def _results_by_id(evaluation: EvaluationOutput) -> dict[str, ExampleResult]:
    return {r.example_id: r for r in evaluation.results}


def compare_for_regressions(
    baseline: EvaluationOutput,
    candidate: EvaluationOutput,
    previous_candidate: EvaluationOutput | None = None,
) -> RegressionReport:
    """Compare ``candidate`` against ``baseline`` (and optionally ``previous_candidate``).

    Restricts the comparison to examples that were previously passing:

    - When ``previous_candidate`` is ``None``: examples correct in
      ``baseline``.
    - When ``previous_candidate`` is supplied: examples correct in
      *both* ``baseline`` and ``previous_candidate`` -- a candidate that
      already regressed relative to the previous candidate is not this
      function's concern to re-report; it reports only newly-broken
      examples relative to the full retained-capability set.

    Only ``example_id``s present in *all* evaluations being compared
    are considered comparable; an example present in baseline but
    absent from candidate's results (e.g. a held-out set that grew or
    shrank between runs) is silently excluded from the report rather
    than raising -- held-out sets evolving between revisions is
    expected and is not, on its own, evidence of tamper the way a
    ``package_id`` mismatch would be.

    Raises ``IncomparableEvaluationsError`` if any supplied evaluation
    is not ``SCORED``, has zero results, or was scored against a
    different ``package_id`` than baseline. Never raises for "no
    regressions found" -- that is a normal, successful comparison
    (``regression_count == 0``).
    """
    _require_scored("baseline", baseline)
    _require_scored("candidate", candidate)
    _require_same_package(baseline, candidate, "candidate")
    if previous_candidate is not None:
        _require_scored("previous_candidate", previous_candidate)
        _require_same_package(baseline, previous_candidate, "previous_candidate")

    baseline_results = _results_by_id(baseline)
    candidate_results = _results_by_id(candidate)
    previous_results = _results_by_id(previous_candidate) if previous_candidate else None

    comparable_ids = set(baseline_results) & set(candidate_results)
    if previous_results is not None:
        comparable_ids &= set(previous_results)

    details: list[ExampleRegressionResult] = []
    regressed: list[str] = []
    previously_passing = 0

    for example_id in sorted(comparable_ids):
        baseline_correct = baseline_results[example_id].correct
        candidate_correct = candidate_results[example_id].correct
        previous_correct = previous_results[example_id].correct if previous_results else None

        was_previously_passing = baseline_correct and (
            previous_correct is None or previous_correct
        )
        if not was_previously_passing:
            details.append(
                ExampleRegressionResult(
                    example_id=example_id,
                    severity=RegressionSeverity.NOT_PREVIOUSLY_PASSING,
                    baseline_correct=baseline_correct,
                    candidate_correct=candidate_correct,
                    previous_correct=previous_correct,
                )
            )
            continue

        previously_passing += 1
        if candidate_correct:
            details.append(
                ExampleRegressionResult(
                    example_id=example_id,
                    severity=RegressionSeverity.STILL_PASSING,
                    baseline_correct=baseline_correct,
                    candidate_correct=candidate_correct,
                    previous_correct=previous_correct,
                )
            )
        else:
            regressed.append(example_id)
            details.append(
                ExampleRegressionResult(
                    example_id=example_id,
                    severity=RegressionSeverity.REGRESSED,
                    baseline_correct=baseline_correct,
                    candidate_correct=candidate_correct,
                    previous_correct=previous_correct,
                )
            )

    return RegressionReport(
        package_id=baseline.package_id,
        artifact_id=candidate.artifact_id,
        baseline_artifact_id=baseline.artifact_id,
        previous_artifact_id=(previous_candidate.artifact_id if previous_candidate else None),
        previously_passing_count=previously_passing,
        regression_count=len(regressed),
        regressed_example_ids=tuple(regressed),
        details=tuple(details),
    )
