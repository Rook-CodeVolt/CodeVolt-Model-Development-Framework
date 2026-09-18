# ADR-0009: Regression tracking across candidate revisions

- Status: accepted
- Date: 2026-09-18

## Context

Issue #24 scoped evaluation-suite depth into WP-A (task-type breadth,
ADR-0007), WP-B (safety/red-team probing, ADR-0008), and WP-C
(regression tracking) as independently gateable work packages. This ADR
covers WP-C only.

`docs/EVALUATION_POLICY.md` names "Regression checks for retained
capabilities" as one of the required evaluation dimensions a later,
separate acceptance decision consumes. Nothing in the framework
automated this before WP-C: a new candidate's `EvaluationOutput` could
be inspected in isolation, but there was no way to answer "did this
candidate stop passing an example that used to pass, either against the
immutable baseline or against the previous accepted candidate?" without
a human diffing two result sets by hand.

## Decision

Add a new, stand-alone module, `src/codevolt_mdf/regression_check.py`,
that compares already-produced `EvaluationOutput` objects — it does not
run any evaluation itself and calls no adapter.

`compare_for_regressions(baseline, candidate, previous_candidate=None)`:

- Accepts a *baseline* `EvaluationOutput`, a *candidate*
  `EvaluationOutput`, and an optional *previous-accepted-candidate*
  `EvaluationOutput`, all expected to be for the same held-out set.
  Sameness is verified structurally (matching `package_id` and matching
  the example-id sets actually present in `results`), not by trusting a
  caller-supplied label.
- Restricts attention to examples that were **previously passing** —
  passing in baseline (and, when a previous candidate is supplied, also
  passing in that previous candidate) — and reports, per such example,
  whether the new candidate now fails it. Examples that were already
  failing are excluded: this is a regression check, not a general
  diff.
- Raises `IncomparableEvaluationsError` for non-`SCORED` status, empty
  result sets, or mismatched `package_id` — fails closed rather than
  silently comparing incomparable runs.
- Produces **no aggregate pass/fail verdict, no threshold, no
  accept/reject field, and no promotion authority of any kind**. It
  returns counts and per-example detail only; the caller (a separate,
  later governed acceptance/promotion step) decides what the resulting
  `RegressionReport` implies. This follows `docs/ARCHITECTURE.md` core
  contracts #3 ("An evaluator produces measurements; it does not
  promote the candidate") and #6 ("Promotion is a separate governed
  action").

### Why a stand-alone module, not a new `TaskType`

WP-A's task types (`exact_match` / `multiple_choice` /
`format_conformance`) and WP-B's `safety_probe` describe how one
`HeldOutExample` is scored *within a single evaluator run*. Regression
tracking compares *across* two or three already-completed runs'
`EvaluationOutput` objects — there is no new example-scoring mode to
add. This is additive, stand-alone logic that only *consumes* existing
`evaluator_contract` types (`EvaluationOutput`, `EvaluationStatus`,
`ExampleResult`). Nothing in `evaluator_contract.py` changes shape;
`CONTRACT_VERSION` stays unchanged.

## Testing

- `tests/test_regression_check.py`: 14 fake-data tests covering
  no-regression, regression-detected, already-failing-examples-excluded,
  absent-example handling, 3-way (baseline + previous-candidate)
  comparison, and tamper/malformed-input rejection paths (non-`SCORED`
  status, empty results, mismatched `package_id`), plus an AST-import
  structural-separation test and a test asserting no verdict/accept/
  reject field exists anywhere on the result type.
- `tests/test_regression_check_real.py`: two real-adapter integration
  tests running `HFLocalCausalLMEvaluatorAdapter` against the pinned
  `HuggingFaceTB/SmolLM2-135M` checkpoint through
  `run_evaluator_contract` twice, feeding the real `EvaluationOutput`
  objects into `compare_for_regressions` — one confirming no false
  regressions are reported on two identical real runs, one confirming a
  genuinely perturbed candidate input is correctly detected as a real
  regression.

## Out of scope

- Any threshold, aggregate verdict, or wiring into a promotion/
  accept-reject decision path. `compare_for_regressions` output is
  evidence for a separate governed step, never an automatic gate.
- Persisting evaluation history/run storage — this module takes
  already-in-hand `EvaluationOutput` objects; how a caller retrieves the
  baseline/previous-candidate outputs it wants to compare is out of
  scope for this ADR.
