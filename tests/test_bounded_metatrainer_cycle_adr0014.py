from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/pilot-metatrainer-v2/run_bounded_cycle_adr0014.py"
)


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("adr0014_runner_test", RUNNER_PATH)
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
        "dataset_hash": runner.EXPECTED_FILE_HASHES["train.jsonl"],
        "dataset_licence": "internal-use-only",
        "approvals": {"security": {}},
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(SystemExit, match="requires exact security, owner, and dataset"):
        runner._load_gate(path)


def test_placeholder_approval_bundle_cannot_pass_without_trusted_signers(
    runner, tmp_path, monkeypatch
):
    # This exercises the "trust root is empty" fail-closed invariant in isolation
    # from the repo's live approval_allowed_signers file. That file is meant to
    # gain real admitted keys over time via independently reviewed trust-root
    # PRs (ADR-0013), so asserting on its live contents here would make this
    # test's pass/fail depend on unrelated trust-root-admission state instead of
    # on the code path it names. An explicit empty trust-root fixture keeps the
    # invariant under test regardless of how many keys are currently admitted.
    empty_signers = tmp_path / "empty_allowed_signers"
    empty_signers.write_text(
        "# no signer keys admitted in this fixture\n", encoding="utf-8"
    )
    monkeypatch.setattr(runner, "APPROVAL_ALLOWED_SIGNERS_PATH", empty_signers)
    approval = {"document": "placeholder", "signature": "placeholder", "document_sha256": "0" * 64}
    gate = {
        "schema_version": 1,
        "implementation_sha": _head(runner),
        "dataset_hash": runner.EXPECTED_FILE_HASHES["train.jsonl"],
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
    # placeholder/garbage approval bundle must still be rejected -- just at the
    # per-role document/signature verification step instead of at the
    # trust-root-emptiness check. This documents and locks in the intended
    # validation order: empty-trust-root check first, then per-role signature
    # verification once real keys are admitted.
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
        "meta_trainer": 20,
        "capability_retention": 10,
        "safety": 15,
    }
    for suite in suites.values():
        assert not registry.check_held_out_not_trained(suite.example_ids)


def _security_approval(runner, tmp_path, scope):
    document = {
        "schema_version": 1,
        "role": "security",
        "approver_id": "maya-security",
        "decision": "approved",
        "implementation_sha": _head(runner),
        "dataset_hash": runner.EXPECTED_FILE_HASHES["train.jsonl"],
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


def test_host_containment_profile_denies_network_and_out_of_root_writes(
    runner, tmp_path
):
    profile = runner._host_containment_profile(tmp_path / "run")
    assert "(deny network*)" in profile
    assert "(deny file-write*)" in profile
    assert f'(subpath "{tmp_path / "run"}")' in profile
    assert '(literal "/dev/null")' in profile
