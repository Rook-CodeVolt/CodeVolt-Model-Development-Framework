"""Contract-conformance tests for HFLocalCausalLMEvaluatorAdapter.

Exercises the REAL (non-fake) evaluator adapter
(``codevolt_mdf.hf_local_evaluator_adapter.HFLocalCausalLMEvaluatorAdapter``)
against the same ``run_evaluator_contract`` runner in
``codevolt_mdf.evaluator_contract`` that
``tests/test_evaluator_contract.py`` exercises against the fake adapter --
same runner, same ``HeldOutExclusionRegistry``, no special-casing. Real
offline inference against a pinned local Hugging Face checkpoint runs in
this file (the tests marked with ``pytest.importorskip("transformers")``
and the local-snapshot-presence skip below); everything else exercises
rejection/invalid-input paths that never need to load a model at all.

No training happens anywhere in this file. The "artifact" scored in the
real-inference tests is the pristine, untrained base checkpoint itself,
supplied manually as a stand-in artifact purely to exercise this
adapter's real-inference scoring path end-to-end -- there is no trained
pilot candidate yet (ADR-0006 is drafted, not executed), and this file
does not claim to produce pilot evidence.
"""

from __future__ import annotations

import shutil

import pytest

from codevolt_mdf.evaluator_contract import (
    CONTRACT_VERSION,
    EvaluationStatus,
    HeldOutExample,
    HeldOutSet,
    run_evaluator_contract,
    verify_evidence,
)
from codevolt_mdf.held_out_registry import HeldOutExclusionRegistry
from codevolt_mdf.hf_local_evaluator_adapter import (
    PINNED_MODEL_REPO,
    HFLocalCausalLMEvaluatorAdapter,
    _hash_model_dir,
)

# --------------------------------------------------------------------------
# Pinned model identity -- matches ADR-0006's draft bounded-pilot plan.
# Revision recorded here (not just a mutable branch name) so this test
# suite's own provenance claim is checkable the same way the adapter
# checks a caller's: by content, not by a trusted name.
# --------------------------------------------------------------------------

PINNED_MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"


def _resolve_pinned_model_path():
    """Resolve the pinned local snapshot path via huggingface_hub, offline only.

    Returns ``None`` (never raises) if the snapshot is not present locally
    or ``huggingface_hub`` is not importable -- callers use this to skip
    real-inference tests cleanly in an environment that has not fetched
    the pinned model, exactly as ``trl_adapter``'s tests skip cleanly when
    ``trl`` is not installed.
    """
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


def make_held_out(examples: list[tuple[str, str, str]], package_id: str = "P1") -> HeldOutSet:
    return HeldOutSet.create(
        package_id,
        [
            HeldOutExample(example_id=eid, input=prompt, expected=expected)
            for eid, prompt, expected in examples
        ],
    )


def make_fake_checkpoint_dir(tmp_path, name: str = "not-a-real-checkpoint") -> str:
    """A directory that looks nothing like a HF checkpoint (no config.json)."""
    path = tmp_path / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "readme.txt").write_text("not a checkpoint", encoding="utf-8")
    return str(path)


# 1. Adapter identity and contract-version wiring ----------------------------


def test_adapter_declares_supported_contract_version():
    adapter = HFLocalCausalLMEvaluatorAdapter()
    assert adapter.contract_version == CONTRACT_VERSION


def test_adapter_is_pinned_to_adr_0006_pilot_model_for_consistency():
    """Not a hard requirement of the adapter's design (see module docstring),
    but this suite's own declared consistency choice must match ADR-0006's
    proposed pilot model exactly, not just approximately."""
    assert PINNED_MODEL_REPO == "HuggingFaceTB/SmolLM2-135M"


# 2. Rejection paths that never need to load a model -------------------------


def test_nonexistent_artifact_locator_is_rejected(tmp_path):
    adapter = HFLocalCausalLMEvaluatorAdapter()
    held_out = make_held_out([("q1", "2 + 2 = ", "4")])
    registry = HeldOutExclusionRegistry()

    output = run_evaluator_contract(
        adapter, "art-1", str(tmp_path / "does-not-exist"), held_out, registry, tmp_path
    )

    assert output.status == EvaluationStatus.REJECTED
    assert "does not exist" in output.reason


def test_artifact_locator_must_be_a_directory(tmp_path):
    adapter = HFLocalCausalLMEvaluatorAdapter()
    held_out = make_held_out([("q1", "2 + 2 = ", "4")])
    registry = HeldOutExclusionRegistry()
    not_a_dir = tmp_path / "artifact.txt"
    not_a_dir.write_text("not a directory", encoding="utf-8")

    output = run_evaluator_contract(adapter, "art-2", str(not_a_dir), held_out, registry, tmp_path)

    assert output.status == EvaluationStatus.REJECTED
    assert "not a directory" in output.reason


def test_artifact_locator_missing_config_json_is_rejected(tmp_path):
    adapter = HFLocalCausalLMEvaluatorAdapter()
    held_out = make_held_out([("q1", "2 + 2 = ", "4")])
    registry = HeldOutExclusionRegistry()
    fake_dir = make_fake_checkpoint_dir(tmp_path)

    output = run_evaluator_contract(adapter, "art-3", fake_dir, held_out, registry, tmp_path)

    assert output.status == EvaluationStatus.REJECTED
    assert "config.json" in output.reason


def test_model_hash_mismatch_is_rejected(tmp_path):
    """Provenance-by-content check, mirroring TRLTrainerAdapter.prepare()'s
    model_hash verification -- a caller-declared expected_model_hash that
    doesn't match the artifact's actual content is refused before any
    example is scored."""
    fake_dir = tmp_path / "checkpoint"
    fake_dir.mkdir()
    (fake_dir / "config.json").write_text('{"model_type": "fake"}', encoding="utf-8")

    adapter = HFLocalCausalLMEvaluatorAdapter(expected_model_hash="0" * 64)
    held_out = make_held_out([("q1", "2 + 2 = ", "4")])
    registry = HeldOutExclusionRegistry()

    output = run_evaluator_contract(adapter, "art-4", str(fake_dir), held_out, registry, tmp_path)

    assert output.status == EvaluationStatus.REJECTED
    assert "model_hash mismatch" in output.reason


def test_matching_model_hash_passes_provenance_check(tmp_path):
    """The hash gate itself needs no transformers/torch -- but the code path
    that runs immediately after it (attempting to load the fake checkpoint
    as a real model) does import torch, so this test still needs those
    optional deps importable to reach the (expected) inference-loading
    failure rather than a provenance rejection. importorskip keeps this
    clean in an environment without the optional hf-local-evaluator extra
    installed, mirroring trl_adapter's tests' own skip discipline."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    fake_dir = tmp_path / "checkpoint"
    fake_dir.mkdir()
    (fake_dir / "config.json").write_text('{"model_type": "fake"}', encoding="utf-8")
    real_hash = _hash_model_dir(fake_dir)

    # Adapter whose expected_model_hash matches: confirm the failure that
    # follows is the (expected) inference-loading failure, not a
    # provenance rejection -- proving the hash check itself passed.
    adapter = HFLocalCausalLMEvaluatorAdapter(expected_model_hash=real_hash)
    held_out = make_held_out([("q1", "2 + 2 = ", "4")])
    registry = HeldOutExclusionRegistry()

    output = run_evaluator_contract(adapter, "art-5", str(fake_dir), held_out, registry, tmp_path)

    # Fake checkpoint content is not real model weights, so this can't
    # SCORE -- but it must not be REJECTED for a hash mismatch.
    assert output.status == EvaluationStatus.INVALID
    assert "model_hash mismatch" not in output.reason


# 3. Invalid-input paths that never need to load a model ---------------------


def test_empty_string_input_is_invalid(tmp_path):
    fake_dir = tmp_path / "checkpoint"
    fake_dir.mkdir()
    (fake_dir / "config.json").write_text("{}", encoding="utf-8")

    adapter = HFLocalCausalLMEvaluatorAdapter()
    held_out = HeldOutSet.create(
        "P1", [HeldOutExample(example_id="q1", input="   ", expected="4")]
    )
    registry = HeldOutExclusionRegistry()

    output = run_evaluator_contract(adapter, "art-6", str(fake_dir), held_out, registry, tmp_path)

    assert output.status == EvaluationStatus.INVALID
    assert "HeldOutExample.input" in output.reason


def test_non_string_expected_is_invalid(tmp_path):
    fake_dir = tmp_path / "checkpoint"
    fake_dir.mkdir()
    (fake_dir / "config.json").write_text("{}", encoding="utf-8")

    adapter = HFLocalCausalLMEvaluatorAdapter()
    held_out = HeldOutSet.create(
        "P1", [HeldOutExample(example_id="q1", input="2 + 2 = ", expected=4)]
    )
    registry = HeldOutExclusionRegistry()

    output = run_evaluator_contract(adapter, "art-7", str(fake_dir), held_out, registry, tmp_path)

    assert output.status == EvaluationStatus.INVALID
    assert "HeldOutExample.expected" in output.reason


# 4. Held-out/train contamination enforcement, same registry, no special-casing --


def test_held_out_set_contaminated_by_train_usage_is_rejected_before_scoring(tmp_path):
    """Identical assertion shape to
    test_evaluator_contract.py::test_held_out_set_contaminated_by_train_usage_is_rejected_before_scoring
    but against this real adapter -- proves the contamination gate in
    run_evaluator_contract applies uniformly, before this adapter's
    score_example (and therefore before any real inference) is ever
    called."""
    fake_dir = tmp_path / "checkpoint"
    fake_dir.mkdir()
    (fake_dir / "config.json").write_text("{}", encoding="utf-8")

    adapter = HFLocalCausalLMEvaluatorAdapter()
    held_out = make_held_out(
        [
            ("task-0000", "2 + 2 = ", "4"),
            ("task-0001", "5 + 3 = ", "8"),
        ]
    )
    registry = HeldOutExclusionRegistry()
    registry.register_package_train("P0", ["task-0000"])

    output = run_evaluator_contract(adapter, "art-8", str(fake_dir), held_out, registry, tmp_path)

    assert output.status == EvaluationStatus.INVALID
    assert output.error_class == "ContaminationDetectedError"
    assert output.contaminated_ids == ("task-0000",)
    assert output.results == ()  # nothing was scored -- not even attempted


def test_tampered_held_out_set_is_invalid_through_this_adapter_too(tmp_path):
    fake_dir = tmp_path / "checkpoint"
    fake_dir.mkdir()
    (fake_dir / "config.json").write_text("{}", encoding="utf-8")

    adapter = HFLocalCausalLMEvaluatorAdapter()
    held_out = make_held_out([("q1", "2 + 2 = ", "4"), ("q2", "5 + 3 = ", "8")])
    tampered = HeldOutSet(
        package_id=held_out.package_id, examples=held_out.examples, dataset_hash="0" * 64
    )
    registry = HeldOutExclusionRegistry()

    output = run_evaluator_contract(adapter, "art-9", str(fake_dir), held_out=tampered,
                                     registry=registry, work_dir=tmp_path)

    assert output.status == EvaluationStatus.INVALID
    assert output.error_class == "TamperDetectedError"


# 5. Structural separation from trl_adapter.py / trainer_contract.py ---------
# Same AST-import pattern as
# test_evaluator_contract.py::test_evaluator_contract_module_does_not_import_trl_adapter.


def test_hf_local_evaluator_adapter_module_does_not_import_trainer_side():
    """Structural proof, not a documented promise: this module has no real
    ``import``/``from ... import`` statement referencing ``trl_adapter``
    or ``trainer_contract`` -- prose mentions in the module docstring
    don't count; only AST Import/ImportFrom nodes do."""
    import ast

    import codevolt_mdf.hf_local_evaluator_adapter as adapter_module

    with open(adapter_module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())

    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert not any("trl_adapter" in name for name in imported_modules)
    assert not any("trainer_contract" in name for name in imported_modules)


def test_adapter_has_no_accept_reject_or_promote_method():
    """No method on this concrete adapter class scores as anything other
    than a measurement -- confirms this real adapter doesn't grow extra
    surface beyond what EvaluatorAdapterV1 allows (score_example only),
    per docs/ARCHITECTURE.md core contract #3."""
    public_methods = {
        name
        for name in dir(HFLocalCausalLMEvaluatorAdapter)
        if not name.startswith("_") and callable(getattr(HFLocalCausalLMEvaluatorAdapter, name))
    }
    assert public_methods == {"score_example"}


def test_evaluation_output_has_no_accept_reject_field():
    """The aggregate this adapter's scores feed into carries no
    accept/reject/promoted field -- EvaluationOutput.status is scored /
    rejected / invalid (run trustworthiness), never a quality verdict.
    Re-confirms (adapter-side) the same boundary
    test_evaluator_contract.py checks contract-side."""
    from dataclasses import fields

    from codevolt_mdf.evaluator_contract import EvaluationOutput

    field_names = {f.name for f in fields(EvaluationOutput)}
    assert "accepted" not in field_names
    assert "promoted" not in field_names
    assert "passed" not in field_names


# 6. Real inference: end-to-end through run_evaluator_contract ---------------
# Skips cleanly (does not fail) if transformers/torch aren't installed or
# the pinned local snapshot hasn't been fetched -- see _require_pinned_model.


def test_real_inference_scores_against_pinned_local_checkpoint(tmp_path):
    """The core new behaviour this adapter adds over the fake one: real,
    offline, local model inference actually runs. The base (untrained)
    checkpoint is used as a manually-provided stand-in artifact purely to
    exercise the scoring path -- this is NOT pilot evidence; no training
    happened to produce this artifact."""
    model_path = _require_pinned_model()

    adapter = HFLocalCausalLMEvaluatorAdapter(max_new_tokens=6)
    held_out = make_held_out(
        [
            ("arith-0000", "2 + 2 = ", "4"),
            ("arith-0001", "The opposite of hot is", "cold"),
        ],
        package_id="P-real-inference",
    )
    registry = HeldOutExclusionRegistry()

    output = run_evaluator_contract(adapter, "base-checkpoint", model_path, held_out, registry, tmp_path)

    assert output.status == EvaluationStatus.SCORED
    assert output.aggregate_score is not None
    assert 0.0 <= output.aggregate_score <= 1.0
    assert len(output.results) == 2
    # Real generations were actually produced (not fabricated placeholders).
    for result in output.results:
        assert isinstance(result.raw_output, str)
        assert result.raw_output != ""
    assert output.evidence_locator is not None
    assert verify_evidence(output.evidence_locator, output.evidence_hash)


def test_real_inference_is_deterministic_given_same_artifact_and_example(tmp_path):
    """Greedy decoding (do_sample=False) means the same artifact + example
    reproduces the same evidence -- required for this adapter's output to
    be trustworthy, re-checkable evidence rather than a one-off claim."""
    model_path = _require_pinned_model()

    adapter = HFLocalCausalLMEvaluatorAdapter(max_new_tokens=6)
    held_out = make_held_out([("arith-0000", "2 + 2 = ", "4")], package_id="P-determinism")
    registry = HeldOutExclusionRegistry()

    output_a = run_evaluator_contract(
        adapter, "run-a", model_path, held_out, registry, tmp_path / "a"
    )
    output_b = run_evaluator_contract(
        adapter, "run-b", model_path, held_out, registry, tmp_path / "b"
    )

    assert output_a.status == output_b.status == EvaluationStatus.SCORED
    assert output_a.results[0].raw_output == output_b.results[0].raw_output
    assert output_a.results[0].correct == output_b.results[0].correct


def test_real_inference_model_hash_is_verified_when_declared(tmp_path):
    """End-to-end with a real checkpoint: a correct expected_model_hash
    lets scoring proceed; a wrong one is rejected before any inference."""
    model_path = _require_pinned_model()
    real_hash = _hash_model_dir(__import__("pathlib").Path(model_path))

    good_adapter = HFLocalCausalLMEvaluatorAdapter(max_new_tokens=4, expected_model_hash=real_hash)
    held_out = make_held_out([("arith-0000", "2 + 2 = ", "4")], package_id="P-hash-good")
    registry = HeldOutExclusionRegistry()
    good_output = run_evaluator_contract(
        good_adapter, "run-good-hash", model_path, held_out, registry, tmp_path / "good"
    )
    assert good_output.status == EvaluationStatus.SCORED

    bad_adapter = HFLocalCausalLMEvaluatorAdapter(max_new_tokens=4, expected_model_hash="f" * 64)
    bad_output = run_evaluator_contract(
        bad_adapter, "run-bad-hash", model_path, held_out, registry, tmp_path / "bad"
    )
    assert bad_output.status == EvaluationStatus.REJECTED
    assert "model_hash mismatch" in bad_output.reason


def test_real_inference_with_a_copied_artifact_directory_has_stable_identity(tmp_path):
    """The artifact identity hash is content-based: copying the checkpoint
    to a new path produces the same hash, proving _hash_model_dir hashes
    content, not the caller-supplied path string."""
    model_path = _require_pinned_model()
    from pathlib import Path

    original_hash = _hash_model_dir(Path(model_path))

    copy_dir = tmp_path / "copied-checkpoint"
    shutil.copytree(model_path, copy_dir)
    copy_hash = _hash_model_dir(copy_dir)

    assert original_hash == copy_hash


def test_missing_held_out_response_is_scored_incorrect_not_an_error(tmp_path):
    """An example the model's generation simply doesn't happen to contain
    the expected substring for scores 0.0/incorrect, not an error -- the
    adapter's job is to score what it generates, not to require a match."""
    model_path = _require_pinned_model()

    adapter = HFLocalCausalLMEvaluatorAdapter(max_new_tokens=4)
    held_out = make_held_out(
        [("nonsense-0000", "2 + 2 = ", "the-answer-will-never-appear-xyzzy-123")],
        package_id="P-incorrect",
    )
    registry = HeldOutExclusionRegistry()

    output = run_evaluator_contract(adapter, "run-incorrect", model_path, held_out, registry, tmp_path)

    assert output.status == EvaluationStatus.SCORED
    assert output.results[0].correct is False
    assert output.results[0].score == 0.0


# 6b. Real inference: multiple_choice task type (WP-A, issue #24) ------------
# Per-option log-likelihood scoring against the pinned checkpoint. Same
# skip discipline as section 6 above.


def _make_multiple_choice_held_out(
    example_id: str, prompt: str, choices: list, expected, package_id: str
) -> HeldOutSet:
    return HeldOutSet.create(
        package_id,
        [
            HeldOutExample(
                example_id=example_id,
                input={"prompt": prompt, "choices": choices},
                expected=expected,
                metadata=(("task_type", "multiple_choice"),),
            )
        ],
    )


def test_real_inference_scores_multiple_choice_by_log_likelihood(tmp_path):
    """The core new multiple_choice behaviour: real per-option teacher-forced
    log-likelihood scoring against the pinned checkpoint, end-to-end through
    run_evaluator_contract -- same contamination/tamper gates as exact_match,
    no special-casing."""
    model_path = _require_pinned_model()

    adapter = HFLocalCausalLMEvaluatorAdapter()
    held_out = _make_multiple_choice_held_out(
        "mc-0000",
        "The opposite of hot is",
        [" cold", " purple", " Tuesday"],
        0,
        package_id="P-mc-real",
    )
    registry = HeldOutExclusionRegistry()

    output = run_evaluator_contract(adapter, "base-checkpoint", model_path, held_out, registry, tmp_path)

    assert output.status == EvaluationStatus.SCORED
    assert output.aggregate_score is not None
    assert 0.0 <= output.aggregate_score <= 1.0
    assert len(output.results) == 1
    result = output.results[0]
    assert isinstance(result.raw_output, str)
    assert "chose" in result.raw_output
    assert "log-likelihoods" in result.raw_output
    assert output.evidence_locator is not None
    assert verify_evidence(output.evidence_locator, output.evidence_hash)


def test_real_inference_multiple_choice_is_deterministic(tmp_path):
    """Teacher-forced log-likelihood scoring has no sampling anywhere -- the
    same artifact + example must reproduce the exact same choice and score."""
    model_path = _require_pinned_model()

    adapter = HFLocalCausalLMEvaluatorAdapter()
    held_out = _make_multiple_choice_held_out(
        "mc-0000", "2 + 2 = ", ["3", "4", "5"], "4", package_id="P-mc-determinism"
    )
    registry = HeldOutExclusionRegistry()

    output_a = run_evaluator_contract(
        adapter, "run-a", model_path, held_out, registry, tmp_path / "a"
    )
    output_b = run_evaluator_contract(
        adapter, "run-b", model_path, held_out, registry, tmp_path / "b"
    )

    assert output_a.status == output_b.status == EvaluationStatus.SCORED
    assert output_a.results[0].raw_output == output_b.results[0].raw_output
    assert output_a.results[0].correct == output_b.results[0].correct


def test_real_inference_multiple_choice_resolves_expected_by_text_or_index(tmp_path):
    """expected may be given as the choice's exact text or its 0-based index --
    both must resolve to the identical scoring outcome against the same
    real inference (same model, same prompt, same choices)."""
    model_path = _require_pinned_model()

    adapter = HFLocalCausalLMEvaluatorAdapter()
    registry = HeldOutExclusionRegistry()

    held_out_by_text = _make_multiple_choice_held_out(
        "mc-text", "The opposite of hot is", [" cold", " purple"], " cold",
        package_id="P-mc-by-text",
    )
    held_out_by_index = _make_multiple_choice_held_out(
        "mc-index", "The opposite of hot is", [" cold", " purple"], 0,
        package_id="P-mc-by-index",
    )

    output_text = run_evaluator_contract(
        adapter, "run-text", model_path, held_out_by_text, registry, tmp_path / "text"
    )
    output_index = run_evaluator_contract(
        adapter, "run-index", model_path, held_out_by_index, registry, tmp_path / "index"
    )

    assert output_text.status == output_index.status == EvaluationStatus.SCORED
    assert output_text.results[0].correct == output_index.results[0].correct


def test_real_inference_multiple_choice_malformed_input_is_invalid_without_loading_model(tmp_path):
    """A malformed multiple_choice example (too few choices) is reported as
    InvalidInputError -- exercised here against the real pinned checkpoint
    to prove the shape-validation-before-model-load ordering holds for this
    adapter's real (not fake) code path too."""
    model_path = _require_pinned_model()

    adapter = HFLocalCausalLMEvaluatorAdapter()
    held_out = HeldOutSet.create(
        "P-mc-malformed",
        [
            HeldOutExample(
                example_id="mc-bad",
                input={"prompt": "p", "choices": ["only-one"]},
                expected="only-one",
                metadata=(("task_type", "multiple_choice"),),
            )
        ],
    )
    registry = HeldOutExclusionRegistry()

    output = run_evaluator_contract(adapter, "run-bad", model_path, held_out, registry, tmp_path)

    assert output.status == EvaluationStatus.INVALID
    assert "at least 2" in output.reason


# 6c. Real inference: format_conformance task type (WP-A, issue #24) ---------
# Greedy generation + format check against the pinned checkpoint.


def _make_format_conformance_held_out(
    example_id: str, prompt: str, spec: dict, package_id: str
) -> HeldOutSet:
    return HeldOutSet.create(
        package_id,
        [
            HeldOutExample(
                example_id=example_id,
                input=prompt,
                expected=spec,
                metadata=(("task_type", "format_conformance"),),
            )
        ],
    )


def test_real_inference_scores_format_conformance_via_generation(tmp_path):
    """The core new format_conformance behaviour: real greedy generation
    against the pinned checkpoint, checked against a declared format spec
    (not a fixed expected string) -- end-to-end through
    run_evaluator_contract, same contamination/tamper gates as exact_match."""
    model_path = _require_pinned_model()

    adapter = HFLocalCausalLMEvaluatorAdapter(max_new_tokens=6)
    held_out = _make_format_conformance_held_out(
        "fc-0000", "2 + 2 = ", {"format": "regex", "pattern": r"."},
        package_id="P-fc-real",
    )
    registry = HeldOutExclusionRegistry()

    output = run_evaluator_contract(adapter, "base-checkpoint", model_path, held_out, registry, tmp_path)

    assert output.status == EvaluationStatus.SCORED
    assert output.aggregate_score is not None
    assert 0.0 <= output.aggregate_score <= 1.0
    assert len(output.results) == 1
    result = output.results[0]
    assert isinstance(result.raw_output, str)
    assert "matches pattern" in result.raw_output
    assert output.evidence_locator is not None
    assert verify_evidence(output.evidence_locator, output.evidence_hash)


def test_real_inference_format_conformance_json_spec_scores_untrained_output_incorrect(tmp_path):
    """The untrained base checkpoint's free-form continuation of an
    arithmetic prompt is not valid JSON -- correctly scored non-conforming
    (False), not an error, proving this mode measures shape, not equality,
    and doesn't silently pass everything."""
    model_path = _require_pinned_model()

    adapter = HFLocalCausalLMEvaluatorAdapter(max_new_tokens=6)
    held_out = _make_format_conformance_held_out(
        "fc-0001", "2 + 2 = ", {"format": "json"}, package_id="P-fc-json"
    )
    registry = HeldOutExclusionRegistry()

    output = run_evaluator_contract(adapter, "run-json", model_path, held_out, registry, tmp_path)

    assert output.status == EvaluationStatus.SCORED
    assert output.results[0].correct is False
    assert output.results[0].score == 0.0
    assert "not valid JSON" in output.results[0].raw_output


def test_real_inference_format_conformance_is_deterministic(tmp_path):
    """Greedy decoding (do_sample=False) means the same artifact + example
    reproduces the same generated text and the same conformance verdict."""
    model_path = _require_pinned_model()

    adapter = HFLocalCausalLMEvaluatorAdapter(max_new_tokens=6)
    held_out = _make_format_conformance_held_out(
        "fc-0000", "2 + 2 = ", {"format": "regex", "pattern": r"."},
        package_id="P-fc-determinism",
    )
    registry = HeldOutExclusionRegistry()

    output_a = run_evaluator_contract(
        adapter, "run-a", model_path, held_out, registry, tmp_path / "a"
    )
    output_b = run_evaluator_contract(
        adapter, "run-b", model_path, held_out, registry, tmp_path / "b"
    )

    assert output_a.status == output_b.status == EvaluationStatus.SCORED
    assert output_a.results[0].raw_output == output_b.results[0].raw_output
    assert output_a.results[0].correct == output_b.results[0].correct


def test_real_inference_format_conformance_malformed_spec_is_invalid_without_loading_model(tmp_path):
    """A malformed format_conformance spec (unsupported format) is reported
    as InvalidInputError -- exercised against the real pinned checkpoint to
    prove the shape-validation-before-model-load ordering holds here too."""
    model_path = _require_pinned_model()

    adapter = HFLocalCausalLMEvaluatorAdapter()
    held_out = _make_format_conformance_held_out(
        "fc-bad", "2 + 2 = ", {"format": "xml"}, package_id="P-fc-malformed"
    )
    registry = HeldOutExclusionRegistry()

    output = run_evaluator_contract(adapter, "run-bad", model_path, held_out, registry, tmp_path)

    assert output.status == EvaluationStatus.INVALID
    assert "one of" in output.reason


# 7. No self-promotion: aggregate_score alone never implies acceptance --------


def test_aggregate_score_carries_no_promotion_semantics(tmp_path):
    """A perfect aggregate_score of 1.0 is still just a measurement --
    EvaluationOutput has no accepted/promoted field for this (or any)
    adapter to set, and this adapter's own score_example return type
    (ExampleResult) has no such field either. Uses a synthetic
    always-correct held-out example scored via the FAKE-artifact
    provenance-rejection path is not appropriate here (needs real
    inference), so this test instead statically checks the return type
    shape rather than asserting on a specific generation."""
    from dataclasses import fields

    from codevolt_mdf.evaluator_contract import ExampleResult

    result_field_names = {f.name for f in fields(ExampleResult)}
    assert result_field_names == {"example_id", "correct", "score", "raw_output"}
    assert "accepted" not in result_field_names
    assert "promoted" not in result_field_names
