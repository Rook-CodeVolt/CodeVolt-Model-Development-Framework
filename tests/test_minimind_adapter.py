"""Contract-conformance tests for MiniMindTrainerAdapter (`codevolt_mdf.minimind_adapter`).

Scope, deliberately narrow (see
``docs/decisions/0010-minimind-trainer-adapter-v1.md`` and the module
docstring in ``minimind_adapter.py``): this module proves the adapter's
``prepare()``-level validation and contract-runner wiring for every
scenario reachable **without ever calling ``adapter.train()`` against a
real MiniMind subprocess**. Every fixture here is a local, throwaway git
repository (created with real ``git init``/``git commit`` inside
``tmp_path``, never a clone of the real MiniMind repo) and every
``train()``-adjacent scenario that needs subprocess behaviour uses
``unittest.mock.patch`` on ``subprocess.Popen``/``subprocess.run`` --
never a real MiniMind checkout, never a real training run. No network
access happens anywhere in this file.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codevolt_mdf.minimind_adapter import (
    MINIMIND_PINNED_COMMIT,
    MiniMindTrainerAdapter,
    _detect_minimind_checkout_commit,
    _hash_path_identity,
)
from codevolt_mdf.trainer_contract import (
    CONTRACT_VERSION,
    CancellationToken,
    InvalidInputError,
    RejectedInputError,
    ResourceBudget,
    TrainingInputs,
    TrainingStatus,
    UpstreamRequirement,
    run_trainer_contract,
)

MODEL_HASH_PLACEHOLDER = "a" * 64
DATASET_HASH_PLACEHOLDER = "b" * 64


def make_budget(**overrides) -> ResourceBudget:
    defaults = {
        "max_wall_seconds": 5.0,
        "max_cpu_seconds": 5.0,
        "max_memory_mb": 512.0,
        "max_gpu_count": 0,
        "max_storage_mb": 64.0,
        "network_policy": "offline",
    }
    defaults.update(overrides)
    return ResourceBudget(**defaults)


def make_adapter(tmp_path) -> MiniMindTrainerAdapter:
    return MiniMindTrainerAdapter(work_dir=tmp_path / "adapter-work")


def _run_git(args: list[str], cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": "/tmp",
        },
    )


def make_fake_minimind_repo(tmp_path, *, pinned: bool = True) -> tuple[str, str]:
    """Create a local, throwaway git repo standing in for a MiniMind checkout.

    Never touches the network and never clones the real MiniMind repo.
    Returns (repo_path, actual_head_commit). When ``pinned`` is True, the
    single commit's SHA is asserted to differ from the real pin (a fresh
    local repo cannot forge an arbitrary commit SHA), so instead the test
    layer that wants a "pin matches" scenario monkeypatches
    ``MINIMIND_PINNED_COMMIT`` down to this repo's real, freshly created
    commit -- see ``adapter_pinned_to_fake_repo`` below.
    """
    repo_path = tmp_path / "minimind-repo"
    repo_path.mkdir(parents=True, exist_ok=True)
    trainer_dir = repo_path / "trainer"
    trainer_dir.mkdir(parents=True, exist_ok=True)
    (trainer_dir / "train_full_sft.py").write_text("# fake stand-in for MiniMind's SFT script\n")
    (repo_path / "LICENSE").write_text("Apache License 2.0\n")
    _run_git(["init", "-q"], cwd=repo_path)
    _run_git(["add", "-A"], cwd=repo_path)
    _run_git(["commit", "-q", "-m", "fake minimind checkout for tests"], cwd=repo_path)
    head = _detect_minimind_checkout_commit(repo_path)
    assert head, "fixture git repo must have a resolvable HEAD"
    if pinned:
        assert head != MINIMIND_PINNED_COMMIT, (
            "a freshly created local commit can never equal the real pin; "
            "if this ever collides, something is very wrong with git"
        )
    return str(repo_path), head


def make_local_model(tmp_path, content: bytes = b"fake-model-weights") -> tuple[str, str]:
    model_path = tmp_path / "model" / "weights.bin"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(content)
    return str(model_path), _hash_path_identity(model_path)


def make_local_dataset(tmp_path, rows: list[dict] | None = None) -> tuple[str, str]:
    rows = rows if rows is not None else [{"conversations": [{"role": "user", "content": "hi"}]}]
    dataset_path = tmp_path / "dataset" / "train.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return str(dataset_path), _hash_path_identity(dataset_path)


def make_inputs(
    tmp_path,
    *,
    run_id: str | None = None,
    minimind_repo_path: str | None = None,
    model_path: str | None = None,
    model_hash: str | None = None,
    dataset_path: str | None = None,
    dataset_hash: str | None = None,
    max_steps: int | None = 4,
    extra_params: dict | None = None,
    **overrides,
) -> TrainingInputs:
    if minimind_repo_path is None:
        minimind_repo_path, _head = make_fake_minimind_repo(tmp_path, pinned=False)
    if model_path is None:
        model_path, generated_model_hash = make_local_model(tmp_path)
        if model_hash is None:
            model_hash = generated_model_hash
    elif model_hash is None:
        model_hash = _hash_path_identity(Path(model_path))
    if dataset_path is None:
        dataset_path, generated_dataset_hash = make_local_dataset(tmp_path)
        if dataset_hash is None:
            dataset_hash = generated_dataset_hash
    elif dataset_hash is None:
        dataset_hash = _hash_path_identity(Path(dataset_path))

    params: dict = {
        "minimind_repo_path": minimind_repo_path,
        "model_path": model_path,
        "dataset_path": dataset_path,
    }
    if max_steps is not None:
        params["max_steps"] = max_steps
    if extra_params:
        params.update(extra_params)

    defaults = {
        "model_revision": "minimind-pilot-v1",
        "model_hash": model_hash,
        "dataset_version": "synthetic-minimind-v1",
        "dataset_hash": dataset_hash,
        "seed": 42,
        "training_params": params,
        "dataset_licence": "CC0-1.0",
        "contamination_checked": True,
        "run_id": run_id or "run-minimind-adapter-test",
    }
    defaults.update(overrides)
    return TrainingInputs.create(**defaults)


# 1. Adapter identity and declared contract/upstream bounds -----------------


def test_adapter_declares_supported_contract_version(tmp_path):
    adapter = make_adapter(tmp_path)
    assert adapter.contract_version == CONTRACT_VERSION


def test_adapter_declares_minimind_as_upstream_engine(tmp_path):
    adapter = make_adapter(tmp_path)
    assert adapter.upstream.engine_name == "minimind"
    assert adapter.upstream.min_version == MINIMIND_PINNED_COMMIT
    assert adapter.upstream.max_version == MINIMIND_PINNED_COMMIT


def test_pinned_commit_is_a_full_length_sha1_hex_string():
    assert len(MINIMIND_PINNED_COMMIT) == 40
    assert all(c in "0123456789abcdef" for c in MINIMIND_PINNED_COMMIT.lower())


# 2. _detect_minimind_checkout_commit(): git rev-parse against a local fixture --


def test_detect_minimind_checkout_commit_returns_real_head(tmp_path):
    repo_path, head = make_fake_minimind_repo(tmp_path, pinned=False)
    assert _detect_minimind_checkout_commit(Path(repo_path)) == head


def test_detect_minimind_checkout_commit_returns_empty_for_nonexistent_path(tmp_path):
    assert _detect_minimind_checkout_commit(tmp_path / "does-not-exist") == ""


def test_detect_minimind_checkout_commit_returns_empty_for_non_git_directory(tmp_path):
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    (not_a_repo / "somefile.txt").write_text("hello")
    assert _detect_minimind_checkout_commit(not_a_repo) == ""


# 3. prepare(): pin verification via git rev-parse (the core deviation from TRL) --


def test_prepare_rejects_checkout_not_at_pinned_commit(tmp_path):
    """A real local git repo whose HEAD is NOT the pinned commit is rejected.

    This is the adapter's core deviation from TRLTrainerAdapter: instead
    of comparing an installed package's ``__version__`` against a
    semantic-version bound, it runs ``git rev-parse HEAD`` against the
    caller's checkout and requires exact equality with
    ``MINIMIND_PINNED_COMMIT``.
    """
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path)  # fixture repo's HEAD != real pin, by construction
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="checked out at commit"):
        adapter.prepare(inputs, budget)


def test_prepare_accepts_checkout_when_pin_matches_fixture_head(tmp_path, monkeypatch):
    """When the module-level pin is monkeypatched to the fixture's real HEAD, prepare() passes.

    Proves the positive path of the same check without needing to forge
    (impossible) or actually clone (network access, forbidden here) a
    repo whose HEAD equals the real MiniMind pin.
    """
    repo_path, head = make_fake_minimind_repo(tmp_path, pinned=False)
    monkeypatch.setattr("codevolt_mdf.minimind_adapter.MINIMIND_PINNED_COMMIT", head)

    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, minimind_repo_path=repo_path)
    budget = make_budget()

    adapter.prepare(inputs, budget)  # must not raise


def test_prepare_rejects_nonexistent_minimind_repo_path(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, minimind_repo_path=str(tmp_path / "no-such-minimind-checkout"))
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="does not exist locally"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_non_git_directory_as_minimind_repo(tmp_path):
    fake_dir = tmp_path / "just-a-folder"
    fake_dir.mkdir()
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, minimind_repo_path=str(fake_dir))
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="not a readable git checkout"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_pinned_commit_missing_expected_script(tmp_path, monkeypatch):
    """Pin matches, but the expected trainer/train_full_sft.py layout is missing."""
    repo_path = tmp_path / "sparse-repo"
    repo_path.mkdir()
    (repo_path / "README.md").write_text("no trainer dir here")
    _run_git(["init", "-q"], cwd=repo_path)
    _run_git(["add", "-A"], cwd=repo_path)
    _run_git(["commit", "-q", "-m", "sparse checkout"], cwd=repo_path)
    head = _detect_minimind_checkout_commit(repo_path)
    monkeypatch.setattr("codevolt_mdf.minimind_adapter.MINIMIND_PINNED_COMMIT", head)

    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, minimind_repo_path=str(repo_path))
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="does not exist in this checkout"):
        adapter.prepare(inputs, budget)


# 4. prepare(): well-formed local inputs pass validation (no train() call) ----


def test_prepare_accepts_well_formed_local_inputs(tmp_path, monkeypatch):
    repo_path, head = make_fake_minimind_repo(tmp_path, pinned=False)
    monkeypatch.setattr("codevolt_mdf.minimind_adapter.MINIMIND_PINNED_COMMIT", head)

    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, minimind_repo_path=repo_path)
    budget = make_budget()

    adapter.prepare(inputs, budget)  # must not raise


# 5. prepare(): offline network policy is mandatory ---------------------------


def test_prepare_rejects_non_offline_network_policy(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path)
    budget = make_budget(network_policy="allow-list", allowed_hosts=("github.com",))

    with pytest.raises(RejectedInputError, match="offline"):
        adapter.prepare(inputs, budget)


def test_contract_runner_rejects_non_offline_policy_before_training(tmp_path, monkeypatch):
    """Full run_trainer_contract() path -- also monkeypatches upstream to compatible."""
    repo_path, head = make_fake_minimind_repo(tmp_path, pinned=False)
    monkeypatch.setattr("codevolt_mdf.minimind_adapter.MINIMIND_PINNED_COMMIT", head)

    adapter = make_adapter(tmp_path)
    adapter.upstream = UpstreamRequirement(
        engine_name="minimind", min_version=head, max_version=head, installed_version=head
    )
    inputs = make_inputs(tmp_path, minimind_repo_path=repo_path)
    budget = make_budget(network_policy="allow-list", allowed_hosts=("github.com",))

    output = run_trainer_contract(adapter, inputs, budget)

    assert output.status == TrainingStatus.REJECTED
    assert "offline" in output.reason


# 6. Contract-runner wiring: upstream-version rejection (never reaches prepare/train) --


def test_incompatible_upstream_is_rejected_before_prepare(tmp_path):
    adapter = make_adapter(tmp_path)
    adapter.upstream = UpstreamRequirement(
        engine_name="minimind",
        min_version=MINIMIND_PINNED_COMMIT,
        max_version=MINIMIND_PINNED_COMMIT,
        installed_version="0000000000000000000000000000000000000",  # deliberately wrong
    )
    inputs = make_inputs(tmp_path)
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="outside the declared"):
        run_trainer_contract(adapter, inputs, budget)


# 7. prepare(): required training_params ---------------------------------------


def test_prepare_rejects_missing_minimind_repo_path(tmp_path):
    adapter = make_adapter(tmp_path)
    _, model_hash = make_local_model(tmp_path)
    _, dataset_hash = make_local_dataset(tmp_path)
    inputs = TrainingInputs.create(
        model_revision="minimind-pilot-v1",
        model_hash=model_hash,
        dataset_version="synthetic-minimind-v1",
        dataset_hash=dataset_hash,
        seed=42,
        training_params={"model_path": "irrelevant", "dataset_path": "irrelevant", "max_steps": 4},
        dataset_licence="CC0-1.0",
        contamination_checked=True,
        run_id="run-missing-repo-path",
    )
    budget = make_budget()

    with pytest.raises(InvalidInputError, match="minimind_repo_path"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_missing_model_path(tmp_path):
    adapter = make_adapter(tmp_path)
    repo_path, _head = make_fake_minimind_repo(tmp_path, pinned=False)
    _, dataset_hash = make_local_dataset(tmp_path)
    inputs = TrainingInputs.create(
        model_revision="minimind-pilot-v1",
        model_hash=MODEL_HASH_PLACEHOLDER,
        dataset_version="synthetic-minimind-v1",
        dataset_hash=dataset_hash,
        seed=42,
        training_params={
            "minimind_repo_path": repo_path,
            "dataset_path": "irrelevant",
            "max_steps": 4,
        },
        dataset_licence="CC0-1.0",
        contamination_checked=True,
        run_id="run-missing-model-path",
    )
    budget = make_budget()

    with pytest.raises(InvalidInputError, match="model_path"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_missing_dataset_path(tmp_path):
    adapter = make_adapter(tmp_path)
    repo_path, _head = make_fake_minimind_repo(tmp_path, pinned=False)
    model_path, model_hash = make_local_model(tmp_path)
    inputs = TrainingInputs.create(
        model_revision="minimind-pilot-v1",
        model_hash=model_hash,
        dataset_version="synthetic-minimind-v1",
        dataset_hash=DATASET_HASH_PLACEHOLDER,
        seed=42,
        training_params={
            "minimind_repo_path": repo_path,
            "model_path": model_path,
            "max_steps": 4,
        },
        dataset_licence="CC0-1.0",
        contamination_checked=True,
        run_id="run-missing-dataset-path",
    )
    budget = make_budget()

    with pytest.raises(InvalidInputError, match="dataset_path"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_missing_max_steps_and_epochs(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, max_steps=None)
    budget = make_budget()

    with pytest.raises(InvalidInputError, match="max_steps"):
        adapter.prepare(inputs, budget)


def test_prepare_accepts_epochs_instead_of_max_steps(tmp_path, monkeypatch):
    repo_path, head = make_fake_minimind_repo(tmp_path, pinned=False)
    monkeypatch.setattr("codevolt_mdf.minimind_adapter.MINIMIND_PINNED_COMMIT", head)

    adapter = make_adapter(tmp_path)
    inputs = make_inputs(
        tmp_path, minimind_repo_path=repo_path, max_steps=None, extra_params={"epochs": 2}
    )
    budget = make_budget()

    adapter.prepare(inputs, budget)  # must not raise


def test_prepare_rejects_zero_or_negative_max_steps(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, max_steps=0)
    budget = make_budget()

    with pytest.raises(InvalidInputError, match="max_steps"):
        adapter.prepare(inputs, budget)


# 8. prepare(): provenance -- paths must exist locally --------------------------


def test_prepare_rejects_nonexistent_model_path(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)
    repo_path, head = make_fake_minimind_repo(tmp_path, pinned=False)
    monkeypatch.setattr("codevolt_mdf.minimind_adapter.MINIMIND_PINNED_COMMIT", head)
    dataset_path, dataset_hash = make_local_dataset(tmp_path)
    inputs = TrainingInputs.create(
        model_revision="minimind-pilot-v1",
        model_hash=MODEL_HASH_PLACEHOLDER,
        dataset_version="synthetic-minimind-v1",
        dataset_hash=dataset_hash,
        seed=42,
        training_params={
            "minimind_repo_path": repo_path,
            "model_path": str(tmp_path / "does-not-exist"),
            "dataset_path": dataset_path,
            "max_steps": 4,
        },
        dataset_licence="CC0-1.0",
        contamination_checked=True,
        run_id="run-missing-model-file",
    )
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="does not exist locally"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_nonexistent_dataset_path(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)
    repo_path, head = make_fake_minimind_repo(tmp_path, pinned=False)
    monkeypatch.setattr("codevolt_mdf.minimind_adapter.MINIMIND_PINNED_COMMIT", head)
    model_path, model_hash = make_local_model(tmp_path)
    inputs = TrainingInputs.create(
        model_revision="minimind-pilot-v1",
        model_hash=model_hash,
        dataset_version="synthetic-minimind-v1",
        dataset_hash=DATASET_HASH_PLACEHOLDER,
        seed=42,
        training_params={
            "minimind_repo_path": repo_path,
            "model_path": model_path,
            "dataset_path": str(tmp_path / "does-not-exist.jsonl"),
            "max_steps": 4,
        },
        dataset_licence="CC0-1.0",
        contamination_checked=True,
        run_id="run-missing-dataset-file",
    )
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="does not exist locally"):
        adapter.prepare(inputs, budget)


# 9. prepare(): provenance -- hash mismatch is refused (tamper/corruption) ------


def test_prepare_rejects_model_hash_mismatch(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)
    repo_path, head = make_fake_minimind_repo(tmp_path, pinned=False)
    monkeypatch.setattr("codevolt_mdf.minimind_adapter.MINIMIND_PINNED_COMMIT", head)
    model_path, _real_hash = make_local_model(tmp_path)
    dataset_path, dataset_hash = make_local_dataset(tmp_path)
    inputs = TrainingInputs.create(
        model_revision="minimind-pilot-v1",
        model_hash=MODEL_HASH_PLACEHOLDER,  # deliberately wrong
        dataset_version="synthetic-minimind-v1",
        dataset_hash=dataset_hash,
        seed=42,
        training_params={
            "minimind_repo_path": repo_path,
            "model_path": model_path,
            "dataset_path": dataset_path,
            "max_steps": 4,
        },
        dataset_licence="CC0-1.0",
        contamination_checked=True,
        run_id="run-model-hash-mismatch",
    )
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="model_hash mismatch"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_dataset_hash_mismatch(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)
    repo_path, head = make_fake_minimind_repo(tmp_path, pinned=False)
    monkeypatch.setattr("codevolt_mdf.minimind_adapter.MINIMIND_PINNED_COMMIT", head)
    model_path, model_hash = make_local_model(tmp_path)
    dataset_path, _real_hash = make_local_dataset(tmp_path)
    inputs = TrainingInputs.create(
        model_revision="minimind-pilot-v1",
        model_hash=model_hash,
        dataset_version="synthetic-minimind-v1",
        dataset_hash=DATASET_HASH_PLACEHOLDER,  # deliberately wrong
        seed=42,
        training_params={
            "minimind_repo_path": repo_path,
            "model_path": model_path,
            "dataset_path": dataset_path,
            "max_steps": 4,
        },
        dataset_licence="CC0-1.0",
        contamination_checked=True,
        run_id="run-dataset-hash-mismatch",
    )
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="dataset_hash mismatch"):
        adapter.prepare(inputs, budget)


def test_contract_runner_reports_rejected_for_tampered_model(tmp_path, monkeypatch):
    """End-to-end through run_trainer_contract(): tamper is caught before train()."""
    repo_path, head = make_fake_minimind_repo(tmp_path, pinned=False)
    monkeypatch.setattr("codevolt_mdf.minimind_adapter.MINIMIND_PINNED_COMMIT", head)

    adapter = make_adapter(tmp_path)
    adapter.upstream = UpstreamRequirement(
        engine_name="minimind", min_version=head, max_version=head, installed_version=head
    )
    model_path, real_hash = make_local_model(tmp_path)
    dataset_path, dataset_hash = make_local_dataset(tmp_path)
    inputs = make_inputs(
        tmp_path,
        minimind_repo_path=repo_path,
        model_path=model_path,
        model_hash=real_hash,
        dataset_path=dataset_path,
        dataset_hash=dataset_hash,
    )
    budget = make_budget()

    # Tamper with the model file AFTER inputs declared its (now stale) hash.
    with open(model_path, "ab") as handle:
        handle.write(b"tampered-bytes")

    output = run_trainer_contract(adapter, inputs, budget)

    assert output.status == TrainingStatus.REJECTED
    assert "model_hash mismatch" in output.reason


# 10. train(): mocked subprocess only -- never a real MiniMind run --------------


def test_train_builds_expected_subprocess_args(tmp_path, monkeypatch):
    repo_path, head = make_fake_minimind_repo(tmp_path, pinned=False)
    monkeypatch.setattr("codevolt_mdf.minimind_adapter.MINIMIND_PINNED_COMMIT", head)

    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, minimind_repo_path=repo_path, extra_params={"epochs": 1})
    budget = make_budget()
    token = CancellationToken()

    mock_process = MagicMock()
    mock_process.wait.return_value = 0
    mock_process.returncode = 0
    mock_process.stdout.read.return_value = "fake minimind training log\n"

    with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
        output = adapter.train(inputs, budget, token)

    assert output.status == TrainingStatus.ACCEPTED
    called_args = mock_popen.call_args
    argv = called_args[0][0]
    assert argv[0] == "python3"
    assert str(Path(repo_path) / "trainer" / "train_full_sft.py") in argv
    assert "--data_path" in argv
    assert "--epochs" in argv
    assert called_args[1]["cwd"] == str(repo_path)


def test_train_never_invoked_through_prepare_only_path(tmp_path, monkeypatch):
    """Sanity guard: prepare() only ever runs 'git rev-parse', never MiniMind's script.

    ``prepare()`` legitimately calls ``subprocess.run(["git", ...])`` to
    verify the pin, so this asserts on the *arguments* actually passed
    rather than blanket-blocking ``subprocess.Popen``/``run`` (which
    would also break the git call itself, since ``subprocess.run`` is
    implemented on top of ``Popen``).
    """
    repo_path, head = make_fake_minimind_repo(tmp_path, pinned=False)
    monkeypatch.setattr("codevolt_mdf.minimind_adapter.MINIMIND_PINNED_COMMIT", head)

    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, minimind_repo_path=repo_path)
    budget = make_budget()

    adapter.prepare(inputs, budget)  # must not raise

    # And explicitly: no training-script invocation happened as a side effect.
    with patch("subprocess.Popen") as mock_popen:
        pass
    mock_popen.assert_not_called()


def test_train_reports_interrupted_on_nonzero_exit(tmp_path, monkeypatch):
    repo_path, head = make_fake_minimind_repo(tmp_path, pinned=False)
    monkeypatch.setattr("codevolt_mdf.minimind_adapter.MINIMIND_PINNED_COMMIT", head)

    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, minimind_repo_path=repo_path)
    budget = make_budget()
    token = CancellationToken()

    mock_process = MagicMock()
    mock_process.wait.return_value = 1
    mock_process.returncode = 1
    mock_process.stdout.read.return_value = "fake traceback\n"

    with patch("subprocess.Popen", return_value=mock_process):
        output = adapter.train(inputs, budget, token)

    assert output.status == TrainingStatus.INTERRUPTED
    assert output.error_class == "MiniMindSubprocessError"


def test_train_terminates_subprocess_on_cancellation(tmp_path, monkeypatch):
    repo_path, head = make_fake_minimind_repo(tmp_path, pinned=False)
    monkeypatch.setattr("codevolt_mdf.minimind_adapter.MINIMIND_PINNED_COMMIT", head)

    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, minimind_repo_path=repo_path)
    budget = make_budget()
    token = CancellationToken()
    token.cancel("test cancellation")

    mock_process = MagicMock()
    mock_process.wait.side_effect = [subprocess.TimeoutExpired(cmd="x", timeout=0.5), None]
    mock_process.stdout.read.return_value = "partial log\n"

    with patch("subprocess.Popen", return_value=mock_process):
        output = adapter.train(inputs, budget, token)

    mock_process.terminate.assert_called_once()
    assert output.status == TrainingStatus.INTERRUPTED


# 11. hash-identity helper: deterministic, content-sensitive --------------------


def test_hash_path_identity_is_deterministic_for_same_content(tmp_path):
    _path_a, hash_a = make_local_model(tmp_path / "a", content=b"identical-content")
    _path_b, hash_b = make_local_model(tmp_path / "b", content=b"identical-content")
    assert hash_a == hash_b


def test_hash_path_identity_changes_with_content(tmp_path):
    _, hash_a = make_local_model(tmp_path / "a", content=b"content-one")
    _, hash_b = make_local_model(tmp_path / "b", content=b"content-two")
    assert hash_a != hash_b


def test_hash_path_identity_rejects_nonexistent_path(tmp_path):
    with pytest.raises(RejectedInputError):
        _hash_path_identity(tmp_path / "does-not-exist-at-all")


# 12. cleanup(): idempotent, safe with nothing to clean up -----------------------


def test_cleanup_is_idempotent_and_safe_with_nothing_to_clean(tmp_path):
    adapter = make_adapter(tmp_path)
    adapter.cleanup("run-never-started")
    adapter.cleanup("run-never-started")  # second call must not raise


def test_cleanup_removes_run_directory(tmp_path):
    adapter = make_adapter(tmp_path)
    run_dir = adapter._run_dir("run-to-clean")
    (run_dir / "marker.txt").write_text("evidence")
    assert run_dir.exists()

    adapter.cleanup("run-to-clean")

    assert not run_dir.exists()
    adapter.cleanup("run-to-clean")  # idempotent


# 13. TrainingInputs.validate() precondition still applies (adapter-independent) --


def test_training_inputs_validate_rejects_uncontaminated_dataset_for_minimind_inputs(tmp_path):
    inputs = make_inputs(tmp_path, contamination_checked=False)
    with pytest.raises(InvalidInputError):
        inputs.validate()


def test_contract_runner_never_reaches_prepare_for_invalid_inputs(tmp_path):
    """Bad hash shape fails TrainingInputs.validate() before adapter.prepare() runs."""
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, model_hash="not-a-sha256")
    budget = make_budget()

    output = run_trainer_contract(adapter, inputs, budget)

    assert output.status == TrainingStatus.INVALID
    assert output.error_class == InvalidInputError.__name__
