"""Real-adapter integration test for regression tracking (issue #24 WP-C).

``tests/test_regression_check.py`` exercises ``compare_for_regressions``
against synthetic ``EvaluationOutput`` fixtures only. This file closes that
gap: it runs the REAL ``HFLocalCausalLMEvaluatorAdapter``
(``codevolt_mdf.hf_local_evaluator_adapter``) against the pinned local
Hugging Face checkpoint (``HuggingFaceTB/SmolLM2-135M``) through the same
``run_evaluator_contract`` runner the rest of this suite uses -- twice, to
produce two real ``EvaluationOutput`` objects, one used as "baseline" and
one as "candidate" -- and feeds those real objects through
``compare_for_regressions`` to prove the module works on genuine scored
evidence, not just hand-built fixtures.

No training happens anywhere in this file. As in
``tests/test_hf_local_evaluator_adapter.py``, the pristine untrained base
checkpoint is used as a stand-in artifact purely to exercise the real
inference + regression-comparison path end-to-end; this file makes no
pilot-evidence claim.

Skips cleanly (does not fail) if transformers/torch aren't installed or the
pinned local snapshot hasn't been fetched -- identical skip discipline to
``tests/test_hf_local_evaluator_adapter.py``'s ``_require_pinned_model``
(and, upstream of that, ``tests/test_trl_adapter.py``'s trl-not-installed
skips).
"""

from __future__ import annotations

import pytest
from held_out_eval import HeldOutExclusionRegistry

from codevolt_mdf.evaluator_contract import (
    EvaluationStatus,
    HeldOutExample,
    HeldOutSet,
    run_evaluator_contract,
)
from codevolt_mdf.hf_local_evaluator_adapter import (
    PINNED_MODEL_REPO,
    HFLocalCausalLMEvaluatorAdapter,
)
from codevolt_mdf.regression_check import RegressionSeverity, compare_for_regressions

# Same pinned revision as tests/test_hf_local_evaluator_adapter.py -- keeping
# this test's provenance claim checkable by content, not by a trusted name.
PINNED_MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"


def _resolve_pinned_model_path():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None
    try:
        return snapshot_download(
            repo_id=PINNED_MODEL_REPO,
            revision=PINNED_MODEL_REVISION,
            local_files_only=True,
        )
    except Exception:  # noqa: BLE001 - any resolution failure means "not available locally"
        return None


def _require_pinned_model():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    path = _resolve_pinned_model_path()
    if path is None:
        pytest.skip(
            f"pinned local snapshot {PINNED_MODEL_REPO}@{PINNED_MODEL_REVISION} "
            "not present in the local Hugging Face cache; fetch it with "
            "`hf download` to run real-inference tests"
        )
    return path


def _make_held_out(package_id: str) -> HeldOutSet:
    return HeldOutSet.create(
        package_id,
        [
            HeldOutExample(example_id="arith-0000", input="2 + 2 = ", expected="4"),
            HeldOutExample(example_id="arith-0001", input="The opposite of hot is", expected="cold"),
            HeldOutExample(example_id="arith-0002", input="5 + 3 = ", expected="8"),
        ],
    )


def test_compare_for_regressions_runs_on_two_real_adapter_evaluation_outputs(tmp_path):
    """End-to-end: real adapter run x2 -> real EvaluationOutput objects ->
    compare_for_regressions -- not fixtures.

    The same untrained checkpoint is scored twice against the same
    held-out set to stand in for "baseline" and "candidate": since
    decoding is greedy/deterministic (see
    test_real_inference_is_deterministic_given_same_artifact_and_example),
    this is expected to produce a real, genuine RegressionReport with zero
    regressions -- proving the comparison module consumes real
    EvaluationOutput shapes correctly end-to-end, not merely that it
    accepts hand-built fixtures.
    """
    model_path = _require_pinned_model()
    held_out = _make_held_out("P-regression-real")

    baseline_adapter = HFLocalCausalLMEvaluatorAdapter(max_new_tokens=6)
    baseline_output = run_evaluator_contract(
        baseline_adapter,
        "baseline-checkpoint",
        model_path,
        held_out,
        HeldOutExclusionRegistry(),
        tmp_path / "baseline",
    )
    assert baseline_output.status == EvaluationStatus.SCORED

    candidate_adapter = HFLocalCausalLMEvaluatorAdapter(max_new_tokens=6)
    candidate_output = run_evaluator_contract(
        candidate_adapter,
        "candidate-checkpoint",
        model_path,
        held_out,
        HeldOutExclusionRegistry(),
        tmp_path / "candidate",
    )
    assert candidate_output.status == EvaluationStatus.SCORED

    report = compare_for_regressions(baseline_output, candidate_output)

    # Same artifact scored twice under greedy decoding -> identical
    # per-example correctness -> no regressions, by construction.
    assert report.regression_count == 0
    assert report.regressed_example_ids == ()
    assert report.package_id == "P-regression-real"
    assert report.baseline_artifact_id == "baseline-checkpoint"
    assert report.artifact_id == "candidate-checkpoint"
    assert len(report.details) == len(held_out.examples)
    for detail in report.details:
        assert detail.severity in (
            RegressionSeverity.STILL_PASSING,
            RegressionSeverity.NOT_PREVIOUSLY_PASSING,
        )


def test_compare_for_regressions_detects_a_real_regression_with_perturbed_candidate_input(tmp_path):
    """Same real adapter, but the candidate run is scored against a
    deliberately mismatched ``expected`` value for one example, forcing
    that example's real ``correct`` to flip to ``False`` -- proving
    ``compare_for_regressions`` reports a genuine REGRESSED example from
    real (not synthetic) scored evidence, not just from hand-built
    ``ExampleResult`` fixtures.
    """
    model_path = _require_pinned_model()
    package_id = "P-regression-real-perturbed"

    baseline_held_out = HeldOutSet.create(
        package_id,
        [HeldOutExample(example_id="opposite-0000", input="The opposite of hot is", expected="cold")],
    )

    baseline_adapter = HFLocalCausalLMEvaluatorAdapter(max_new_tokens=6)
    baseline_output = run_evaluator_contract(
        baseline_adapter,
        "baseline-checkpoint",
        model_path,
        baseline_held_out,
        HeldOutExclusionRegistry(),
        tmp_path / "baseline",
    )
    assert baseline_output.status == EvaluationStatus.SCORED
    if baseline_output.results[0].correct is not True:
        pytest.skip(
            "untrained pinned checkpoint did not happen to answer "
            "'The opposite of hot is' -> 'cold' correctly on this run; "
            "this example is not 'previously passing' so a real "
            "regression cannot be demonstrated against it -- not a "
            "failure of this module"
        )

    # Same example_id, same input, but an expected value the real model's
    # actual greedy output (which we just confirmed matched "cold" above)
    # cannot also match -- a genuine, real scoring failure on a
    # previously-passing example, not a fabricated one.
    candidate_held_out = HeldOutSet.create(
        package_id,
        [
            HeldOutExample(
                example_id="opposite-0000",
                input="The opposite of hot is",
                expected="definitely-not-what-the-model-outputs",
            )
        ],
    )

    candidate_adapter = HFLocalCausalLMEvaluatorAdapter(max_new_tokens=6)
    candidate_output = run_evaluator_contract(
        candidate_adapter,
        "candidate-checkpoint",
        model_path,
        candidate_held_out,
        HeldOutExclusionRegistry(),
        tmp_path / "candidate",
    )
    assert candidate_output.status == EvaluationStatus.SCORED
    assert candidate_output.results[0].correct is False

    report = compare_for_regressions(baseline_output, candidate_output)

    assert report.regression_count == 1
    assert report.regressed_example_ids == ("opposite-0000",)
    assert report.details[0].severity == RegressionSeverity.REGRESSED
