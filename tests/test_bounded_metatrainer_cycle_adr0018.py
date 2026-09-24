"""Tests for the ADR-0018 gated DPO runner
(examples/pilot-metatrainer-v2/run_bounded_cycle_adr0018.py).

Same scope boundary as tests/test_bounded_metatrainer_cycle_adr0014.py: proves
the runner's gate-verification/hash-checking logic and its dry-run/--execute
split without ever exercising real training. No test in this file calls
``DPOTrainerAdapter.train()`` or performs a live cycle.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from codevolt_mdf.trl_adapter import _hash_path_identity

RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/pilot-metatrainer-v2/run_bounded_cycle_adr0018.py"
)


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("adr0018_runner_test", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _head(runner) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=runner.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


# 1. Gate schema / role-completeness -----------------------------------------


def test_fabricated_legacy_approval_strings_are_rejected(runner, tmp_path):
    gate = {
        "approved": True,
        "implementation_sha": _head(runner),
        "maya_review_ref": "approved",
        "owner_authorization_ref": "yes",
        "dataset_admission_ref": "placeholder",
        "dataset_licence": "whatever",
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(SystemExit, match="schema mismatch"):
        runner._load_gate(path)


def test_execution_gate_requires_all_three_exact_approval_roles(runner, tmp_path):
    gate = {
        "schema_version": 1,
        "implementation_sha": _head(runner),
        "dataset_hash": runner.EXPECTED_FILE_HASHES["preference_pairs.jsonl"],
        "dataset_licence": "internal-use-only",
        "approvals": {"security": {}},
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(SystemExit, match="requires exact security, owner, and dataset"):
        runner._load_gate(path)


def test_gate_bound_to_wrong_commit_sha_is_rejected(runner, tmp_path):
    gate = {
        "schema_version": 1,
        "implementation_sha": "0" * 40,
        "dataset_hash": runner.EXPECTED_FILE_HASHES["preference_pairs.jsonl"],
        "dataset_licence": "internal-use-only",
        "approvals": {role: {} for role in runner.APPROVAL_ROLES},
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(SystemExit, match="but checkout HEAD is"):
        runner._load_gate(path)


def test_gate_dataset_hash_mismatch_is_rejected(runner, tmp_path):
    gate = {
        "schema_version": 1,
        "implementation_sha": _head(runner),
        "dataset_hash": "f" * 64,
        "dataset_licence": "internal-use-only",
        "approvals": {role: {} for role in runner.APPROVAL_ROLES},
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(SystemExit, match="not bound to the locked training dataset"):
        runner._load_gate(path)


# 2. Fail-closed trust root / unsigned refusal -------------------------------


def test_placeholder_approval_bundle_cannot_pass_without_trusted_signers(
    runner, tmp_path, monkeypatch
):
    # Same rationale as the ADR-0014/0016 runner tests: an explicit empty
    # trust-root fixture keeps this invariant under test regardless of how
    # many keys are currently admitted in this checkout's live file.
    empty_signers = tmp_path / "empty_allowed_signers"
    empty_signers.write_text(
        "# no signer keys admitted in this fixture\n", encoding="utf-8"
    )
    monkeypatch.setattr(runner, "APPROVAL_ALLOWED_SIGNERS_PATH", empty_signers)
    approval = {"document": "placeholder", "signature": "placeholder", "document_sha256": "0" * 64}
    gate = {
        "schema_version": 1,
        "implementation_sha": _head(runner),
        "dataset_hash": runner.EXPECTED_FILE_HASHES["preference_pairs.jsonl"],
        "dataset_licence": "internal-use-only",
        "approvals": {role: dict(approval) for role in runner.APPROVAL_ROLES},
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(SystemExit, match="trust root has no admitted signer keys"):
        runner._load_gate(path)


def test_placeholder_approval_bundle_rejected_once_trust_root_has_keys(runner):
    # Companion to the fail-closed test above: once the live trust root (as
    # committed in this repo) has at least one admitted signer key, a
    # placeholder/garbage approval bundle must still be rejected -- just at
    # the per-role document/signature verification step instead of at the
    # trust-root-emptiness check.
    trusted_signer_lines = [
        line
        for line in runner.APPROVAL_ALLOWED_SIGNERS_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not trusted_signer_lines:
        pytest.skip("no signer keys admitted yet in this checkout")
    approval = {"document": "placeholder", "signature": "placeholder", "document_sha256": "0" * 64}
    with pytest.raises(SystemExit, match="approval document/signature is missing"):
        runner._verify_signed_approval(
            "security",
            approval,
            head=_head(runner),
            dataset_licence="internal-use-only",
        )


def test_unsigned_approval_bundle_never_verifies_even_with_matching_fields(
    runner, tmp_path
):
    # A document with every field matching exactly, but backed by a fake
    # ("test-signature") signature file, must still fail at the
    # ssh-keygen -Y verify step -- this proves the runner never trusts a
    # document's own self-reported correctness, only a real signature
    # checked against the trust root.
    document = {
        "schema_version": 1,
        "role": "security",
        "approver_id": "maya-security",
        "decision": "approved",
        "implementation_sha": _head(runner),
        "dataset_hash": runner.EXPECTED_FILE_HASHES["preference_pairs.jsonl"],
        "dataset_licence": "internal-use-only",
        "run_id": runner.RUN_ID,
        "scope": [
            "evaluator-process-containment-v1",
            runner.HOST_CONTAINMENT_SCOPE,
        ],
    }
    document_path = tmp_path / "security.json"
    document_path.write_text(json.dumps(document), encoding="utf-8")
    signature_path = tmp_path / "security.json.sig"
    signature_path.write_text("not-a-real-signature", encoding="utf-8")
    approval = {
        "document": str(document_path),
        "signature": str(signature_path),
        "document_sha256": runner._sha256(document_path),
    }
    with pytest.raises(SystemExit, match="signature is not trusted or valid"):
        runner._verify_signed_approval(
            "security",
            approval,
            head=_head(runner),
            dataset_licence="internal-use-only",
        )


# 3. Security approval scope requirements ------------------------------------


def _security_approval(runner, tmp_path, scope):
    document = {
        "schema_version": 1,
        "role": "security",
        "approver_id": "maya-security",
        "decision": "approved",
        "implementation_sha": _head(runner),
        "dataset_hash": runner.EXPECTED_FILE_HASHES["preference_pairs.jsonl"],
        "dataset_licence": "internal-use-only",
        "run_id": runner.RUN_ID,
        "scope": scope,
    }
    document_path = tmp_path / "security.json"
    document_path.write_text(json.dumps(document), encoding="utf-8")
    signature_path = tmp_path / "security.json.sig"
    signature_path.write_text("test-signature", encoding="utf-8")
    return document, {
        "document": str(document_path),
        "signature": str(signature_path),
        "document_sha256": runner._sha256(document_path),
    }


def test_security_approval_requires_complete_cycle_host_containment_scope(runner, tmp_path):
    _, approval = _security_approval(
        runner, tmp_path, ["evaluator-process-containment-v1"]
    )
    with pytest.raises(SystemExit, match="complete-cycle-host-containment-v1"):
        runner._verify_signed_approval(
            "security",
            approval,
            head=_head(runner),
            dataset_licence="internal-use-only",
        )


def test_security_approval_accepts_both_containment_scopes(
    runner, tmp_path, monkeypatch
):
    document, approval = _security_approval(
        runner,
        tmp_path,
        [
            "evaluator-process-containment-v1",
            "complete-cycle-host-containment-v1",
        ],
    )
    head = _head(runner)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, b"", b""),
    )
    assert (
        runner._verify_signed_approval(
            "security",
            approval,
            head=head,
            dataset_licence="internal-use-only",
        )
        == document
    )


# 4. Config values equal ADR-0018's pinned values ----------------------------


def test_pinned_hyperparameters_match_adr0018_exactly(runner):
    # Every value below is taken directly from ADR-0018's own "Every DPO
    # hyperparameter, explicit, with beta given a real justification" table
    # (docs/decisions/ADR-0018-dpo-execution-config.md, item 3) -- this test
    # locks the runner's constants to that document's recorded decision, not
    # to whatever might seem reasonable independently.
    assert runner.BETA == 0.3
    assert runner.REFERENCE_FREE is False
    assert runner.LEARNING_RATE == 5e-7
    assert runner.MAX_STEPS == 22
    assert runner.BATCH_SIZE == 1
    assert runner.SAVE_STEPS == 11
    assert runner.SAVE_TOTAL_LIMIT == 2
    assert runner.MAX_PROMPT_LENGTH == 512
    assert runner.MAX_LENGTH == 1024
    assert runner.SEED == 20260923


def test_pinned_resource_budget_matches_adr0018_item_6(runner, tmp_path):
    budget = runner._budget(tmp_path)
    assert budget.max_memory_mb == 20480
    assert budget.max_wall_seconds == 1800
    assert budget.max_cpu_seconds == 3600
    assert budget.max_storage_mb == 1024
    assert budget.network_policy == "offline"


def test_pinned_model_and_reference_model_identity_match_adr0018_item_1(runner):
    assert runner.EXPECTED_MODEL_HASH == (
        "43752b3f39894c0122d9a94f3b4e64ad2d76e43d25c2a08aa360ad17a1a0145c"
    )
    assert runner.EXPECTED_REFERENCE_MODEL_HASH == runner.EXPECTED_MODEL_HASH
    assert runner.REFERENCE_MODEL_PATH == runner.MODEL_PATH
    assert runner.MODEL_REVISION == "12fd25f77366fa6b3b4b768ec3050bf629380bac"


def test_pinned_dataset_hashes_match_adr0018_item_2(runner):
    assert runner.EXPECTED_FILE_HASHES == {
        "preference_pairs.jsonl": (
            "e6c1d20884eda08aac354a49ba4514ec44adfe565bb70e03bd9526a3f8da5b33"
        ),
        "generate_pairs.py": (
            "9a8d63cb44d30cb03ac9b4ae5e9bc23a12e7e914df4470bde8c7b338cafed054"
        ),
        "validate_dataset.py": (
            "8b352dd666d4cf42a101833af4bec775f47ae3be5d4e611c16f317b246216144"
        ),
        "DATASET_CARD.md": (
            "af2ac8b088b56d484db21e5265d9f4c22d05dd7228e1fe9b3285bc37a2e4cb5d"
        ),
        "VALIDATION_REPORT.json": (
            "68059d3f5c7b558b5821de66aa620df17024448aee6e22689fd2dd81678967fd"
        ),
    }


# 5. Hash-mismatch refusal ----------------------------------------------------


def _fake_model_dir(tmp_path, name: str, content: bytes) -> tuple[Path, str]:
    """Build a tiny local stand-in "model checkpoint" directory and return its
    (path, real content hash), using the same ``_hash_path_identity`` the
    runner itself calls -- so these tests exercise the runner's actual
    hash-mismatch detection logic against real, present-on-disk content
    instead of a real Hugging Face snapshot. Same discipline as
    tests/test_trl_adapter.py's ``make_local_model``/
    tests/test_dpo_adapter.py's ``make_local_reference_model``, applied here
    to this runner's model-identity check in ``validate_plan()`` rather than
    to the adapter's own ``prepare()``.
    """
    model_dir = tmp_path / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "weights.bin").write_bytes(content)
    return model_dir, _hash_path_identity(model_dir)


def _patch_valid_model_paths(runner, tmp_path, monkeypatch) -> None:
    """Point MODEL_PATH/REFERENCE_MODEL_PATH at small local fake checkpoint
    directories whose real content hash matches EXPECTED_MODEL_HASH/
    EXPECTED_REFERENCE_MODEL_HASH exactly, so validate_plan()'s own
    model-identity checks pass without the real HF snapshot present on the
    host -- for tests that need to reach validate_plan() logic beyond those
    two checks without also exercising hash-mismatch detection itself.
    """
    model_dir, model_hash = _fake_model_dir(tmp_path, "policy-model", b"fake-policy-weights")
    ref_dir, ref_hash = _fake_model_dir(tmp_path, "reference-model", b"fake-reference-weights")
    monkeypatch.setattr(runner, "MODEL_PATH", model_dir)
    monkeypatch.setattr(runner, "EXPECTED_MODEL_HASH", model_hash)
    monkeypatch.setattr(runner, "REFERENCE_MODEL_PATH", ref_dir)
    monkeypatch.setattr(runner, "EXPECTED_REFERENCE_MODEL_HASH", ref_hash)


def test_missing_or_hash_invalid_retention_suite_fails_closed(runner, monkeypatch):
    specs = {name: dict(value) for name, value in runner.EVALUATION_SUITES.items()}
    specs["capability_retention"]["file_hash"] = "0" * 64
    monkeypatch.setattr(runner, "EVALUATION_SUITES", specs)
    with pytest.raises(SystemExit, match="capability_retention suite hash mismatch"):
        runner._load_suite("capability_retention")


def test_locked_suites_include_meta_capability_and_safety_with_registries(runner):
    suites, registry = runner._evaluation_suites(runner._load_train_records())
    assert set(suites) == {"meta_trainer", "capability_retention", "safety"}
    assert {name: len(suite.examples) for name, suite in suites.items()} == {
        "meta_trainer": 48,
        "capability_retention": 10,
        "safety": 15,
    }
    for suite in suites.values():
        assert not registry.check_held_out_not_trained(suite.example_ids)


def test_dataset_hash_mismatch_is_detected_by_validate_plan(runner, tmp_path, monkeypatch):
    bad_hashes = dict(runner.EXPECTED_FILE_HASHES)
    bad_hashes["preference_pairs.jsonl"] = "1" * 64
    monkeypatch.setattr(runner, "EXPECTED_FILE_HASHES", bad_hashes)
    with pytest.raises(SystemExit, match="dataset/support hash mismatch"):
        runner.validate_plan(tmp_path)


def test_model_hash_mismatch_is_detected_by_validate_plan(runner, tmp_path, monkeypatch):
    # Exercises the real detection logic against a small local fake
    # "model checkpoint" directory (real content, really hashed via
    # ``_hash_path_identity``, same as tests/test_trl_adapter.py's
    # ``make_local_model``) instead of requiring the real
    # SmolLM2-135M-Instruct snapshot to be present on the host -- CI runners
    # never have it cached, so a hard dependency on it here would make this
    # detection path untested in CI rather than genuinely exercised.
    model_dir, _real_hash = _fake_model_dir(tmp_path, "policy-model", b"fake-policy-weights")
    monkeypatch.setattr(runner, "MODEL_PATH", model_dir)
    monkeypatch.setattr(runner, "EXPECTED_MODEL_HASH", "2" * 64)  # deliberately wrong
    with pytest.raises(SystemExit, match="model snapshot content hash mismatch"):
        runner.validate_plan(tmp_path)


def test_reference_model_hash_mismatch_is_detected_by_validate_plan(
    runner, tmp_path, monkeypatch
):
    # DPO-specific: the reference model gets its own independent hash check,
    # per ADR-0018 item 1 / Maya's ADR-0017 gate (b)(3). A tampered/wrong
    # reference-model hash must fail closed exactly like the policy model's.
    # The policy model is pointed at a local fake checkpoint that validates
    # cleanly first, so the failure under test is genuinely the reference
    # model's own check, not an accidental earlier one -- same local-fake-
    # checkpoint approach as the policy-model hash-mismatch test above, no
    # real snapshot required.
    model_dir, model_hash = _fake_model_dir(tmp_path, "policy-model", b"fake-policy-weights")
    monkeypatch.setattr(runner, "MODEL_PATH", model_dir)
    monkeypatch.setattr(runner, "EXPECTED_MODEL_HASH", model_hash)
    ref_dir, _real_ref_hash = _fake_model_dir(tmp_path, "reference-model", b"fake-reference-weights")
    monkeypatch.setattr(runner, "REFERENCE_MODEL_PATH", ref_dir)
    monkeypatch.setattr(runner, "EXPECTED_REFERENCE_MODEL_HASH", "3" * 64)  # deliberately wrong
    with pytest.raises(SystemExit, match="reference model snapshot content hash mismatch"):
        runner.validate_plan(tmp_path)


def test_dependency_version_pin_mismatch_is_detected(runner, tmp_path, monkeypatch):
    # validate_plan() reaches this check only after both model-identity
    # checks pass, so point MODEL_PATH/REFERENCE_MODEL_PATH at valid local
    # fake checkpoints first (same helper the two hash-mismatch tests use)
    # rather than requiring the real snapshot.
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    bad_versions = dict(runner.EXPECTED_VERSIONS)
    bad_versions["trl"] = "0.0.0"
    # _version_map() itself has no logic of its own -- it just imports and
    # reports whatever trl/transformers/datasets/accelerate/torch happen to
    # be installed. Faking its return value (rather than requiring the real,
    # large, optional trl-adapter extra to be installed in this environment,
    # which CI runners are not provisioned with) lets this test genuinely
    # exercise validate_plan()'s own comparison-and-refuse logic
    # immediately below it, instead of skipping that logic untested.
    monkeypatch.setattr(runner, "_version_map", lambda: bad_versions)
    with pytest.raises(SystemExit, match="dependency pin mismatch"):
        runner.validate_plan(tmp_path)


# 6. Dry-run produces a plan with no training --------------------------------


def test_dry_run_validate_plan_reports_pass_with_no_training_called(runner, tmp_path):
    # This is the one test in this file that genuinely needs the real
    # SmolLM2-135M-Instruct snapshot (validate_plan()'s full success path
    # also calls the real DPOTrainerAdapter.prepare(), _runtime_config()
    # (needs a real trl.DPOConfig), and _renderer_id() (needs a real
    # tokenizer's chat template) -- none of which a small local fake
    # directory can stand in for, unlike the hash-mismatch tests above
    # which fail before reaching any of that). Gated on both the optional
    # trl-adapter extra being importable and the exact pinned snapshot
    # actually being present in the local Hugging Face cache, same
    # discipline as tests/test_regression_check_real.py's
    # ``_require_pinned_model``.
    pytest.importorskip("trl")
    pytest.importorskip("transformers")
    if not runner.MODEL_PATH.is_dir():
        pytest.skip(
            f"pinned local snapshot at {runner.MODEL_PATH} not present in the "
            "local Hugging Face cache; fetch it with `hf download` to run this "
            "real-snapshot dry-run test"
        )
    result = runner.validate_plan(tmp_path)
    assert result["status"] == "PASS"
    assert result["training_called"] is False
    assert result["train_count"] == 22
    assert result["held_out_count"] == 48
    assert result["hyperparameters"]["beta"] == 0.3
    assert result["hyperparameters"]["reference_free"] is False
    assert result["model_hash"] == runner.EXPECTED_MODEL_HASH
    assert result["reference_model_hash"] == runner.EXPECTED_REFERENCE_MODEL_HASH
    assert result.get("execution_blockers")


def test_main_without_execute_flag_never_touches_adapter_train(runner, tmp_path, monkeypatch):
    # Locks in the dry-run/--execute split at the argparse boundary: without
    # --execute, main() must call validate_plan() (prepare()-level only) and
    # must never import/construct anything that would call train().
    calls = {"validate_plan": 0}

    def _fake_validate_plan(scratch_root):
        calls["validate_plan"] += 1
        return {"status": "PASS", "training_called": False}

    monkeypatch.setattr(runner, "validate_plan", _fake_validate_plan)
    monkeypatch.setattr(
        sys, "argv", ["run_bounded_cycle_adr0018.py", "--scratch-root", str(tmp_path / "scratch")]
    )
    exit_code = runner.main()
    assert exit_code == 0
    assert calls["validate_plan"] == 1


def test_execute_without_review_gate_is_rejected(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_bounded_cycle_adr0018.py",
            "--execute",
            "--scratch-root",
            str(tmp_path / "scratch"),
        ],
    )
    with pytest.raises(SystemExit, match="--execute requires --review-gate"):
        runner.main()


def test_execute_rejects_non_reviewed_scratch_root(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_bounded_cycle_adr0018.py",
            "--execute",
            "--scratch-root",
            str(tmp_path / "not-the-reviewed-root"),
            "--review-gate",
            str(tmp_path / "gate.json"),
        ],
    )
    with pytest.raises(SystemExit, match="exact reviewed scratch root"):
        runner.main()


def test_execute_rejects_non_reviewed_gate_path(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "APPROVED_EXECUTION_SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_bounded_cycle_adr0018.py",
            "--execute",
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--review-gate",
            str(tmp_path / "not-the-reviewed-gate.json"),
        ],
    )
    with pytest.raises(SystemExit, match="exact reviewed gate path"):
        runner.main()


# 7. Host containment profile -------------------------------------------------


def test_host_containment_profile_denies_network_and_out_of_root_writes(
    runner, tmp_path
):
    profile = runner._host_containment_profile(tmp_path / "run")
    assert "(deny network*)" in profile
    assert "(deny file-write*)" in profile
    assert f'(subpath "{tmp_path / "run"}")' in profile
    assert '(literal "/dev/null")' in profile
