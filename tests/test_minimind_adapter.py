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
import shutil
import stat
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codevolt_mdf.minimind_adapter import (
    MINIMIND_DEFAULT_HIDDEN_SIZE,
    MINIMIND_DEFAULT_NUM_HIDDEN_LAYERS,
    MINIMIND_DEFAULT_USE_MOE,
    MINIMIND_FROM_WEIGHT_STAGED_NAME,
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
    assert "--save_dir" in argv
    assert "--from_weight" in argv
    # Regression guard for escalated from Maya's PR #62 re-review: --from_weight must be the fixed staged NAME, never the
    # literal model_path -- MiniMind's own --from_weight resolution treats
    # it as a name/prefix resolved against init_model()'s own hardcoded
    # save_dir, not a literal path. See test_train_stages_from_weight_checkpoint_at_expected_path
    # below for the real-filesystem proof the staged file actually lands
    # where MiniMind's own resolution formula looks for it.
    from_weight_index = argv.index("--from_weight")
    assert argv[from_weight_index + 1] == MINIMIND_FROM_WEIGHT_STAGED_NAME
    assert inputs.params["model_path"] not in argv
    # Regression guard for issue #47: these flags do not exist in the
    # real pinned train_full_sft.py and must never be constructed.
    assert "--out_dir" not in argv
    assert "--model_path" not in argv
    assert "--max_steps" not in argv
    # Regression guard for issue #46/ADR-0011: the
    # subprocess MUST run with cwd inside a per-run "shadow trainer
    # directory", never the checkout root and never <repo>/trainer
    # directly. MiniMind's own trainer/trainer_utils.py:init_model()
    # defaults tokenizer_path to '../model' (relative to cwd), and
    # trainer/train_full_sft.py's train_epoch() unconditionally also
    # writes a second, CLI-flag-independent checkpoint to
    # '../checkpoints' -- both resolve outside the checkout when cwd is
    # the shadow trainer dir under this run's own work_dir, closing the
    # write-escape gap structurally (see test_train_creates_isolated_shadow_directory_layout
    # below for the real-filesystem proof this data-copy actually happens).
    actual_cwd = Path(called_args[1]["cwd"])
    assert actual_cwd != Path(repo_path)
    assert actual_cwd != Path(repo_path) / "trainer"
    assert actual_cwd.name == "trainer"
    assert actual_cwd.parent.name == "mm_shadow"


def test_train_creates_isolated_shadow_directory_layout(tmp_path, monkeypatch):
    """Real-filesystem proof (no mocking of shutil/Path) that train() copies
    trainer/, model/, and dataset/ into a per-run shadow directory before
    invoking the subprocess, and that the shadow checkpoints dir exists
    ready to receive MiniMind's hardcoded '../checkpoints' write.

    This is the regression guard for the exact defect Maya's live-execution
    review found (issue #46/ADR-0011): a fake MiniMind
    checkout fixture here includes model/ and dataset/ directories (the
    real pinned commit has both), and this test fails if either is missing
    from the shadow copy -- exactly the gap a first fix attempt had (it
    copied trainer/ and model/ but not dataset/, which
    trainer/train_full_sft.py's own `from dataset.lm_dataset import
    SFTDataset` import needs at the real pinned commit).
    """
    repo_path, head = make_fake_minimind_repo(tmp_path, pinned=False)
    monkeypatch.setattr("codevolt_mdf.minimind_adapter.MINIMIND_PINNED_COMMIT", head)

    repo_path_obj = Path(repo_path)
    (repo_path_obj / "model").mkdir(parents=True, exist_ok=True)
    (repo_path_obj / "model" / "tokenizer_config.json").write_text("{}")
    (repo_path_obj / "dataset").mkdir(parents=True, exist_ok=True)
    (repo_path_obj / "dataset" / "lm_dataset.py").write_text("# fake lm_dataset stand-in\n")

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
    shadow_trainer_dir = Path(mock_popen.call_args[1]["cwd"])
    shadow_dir = shadow_trainer_dir.parent

    assert (shadow_dir / "trainer" / "train_full_sft.py").is_file()
    assert (shadow_dir / "model" / "tokenizer_config.json").is_file()
    assert (shadow_dir / "dataset" / "lm_dataset.py").is_file()
    assert (shadow_dir / "checkpoints").is_dir()

    # The shadow copies must be real files, not symlinks back into the
    # checkout: a symlinked shadow dir would make MiniMind's own
    # '../checkpoints'/'../model' relative-path resolution walk back
    # through the symlink target's real parent (the actual checkout),
    # completely defeating containment. Verified directly against a real
    # symlink-based variant of this fix during: it still wrote
    # checkpoints into the real checkout.
    assert not (shadow_dir / "trainer").is_symlink()
    assert not (shadow_dir / "model").is_symlink()
    assert not (shadow_dir / "dataset").is_symlink()


def test_build_subprocess_args_uses_default_hidden_size_when_unset() -> None:
    """Existing default behaviour (768-hidden-size) is preserved.

    When ``training_params`` does not specify
    ``hidden_size``/``num_hidden_layers``/``use_moe``, the constructed
    args must still use the adapter's ``MINIMIND_DEFAULT_*`` fallback
    constants -- callers that predate issue #67 Finding 1's fix keep
    training at exactly the same configuration as before.
    """
    adapter = MiniMindTrainerAdapter(work_dir=Path("/tmp/does-not-need-to-exist-for-this-check"))
    args = adapter._build_subprocess_args(
        python_executable="python3",
        script_path=Path("/tmp/does-not-need-to-exist-for-this-check/train_full_sft.py"),
        params={"dataset_path": "irrelevant.jsonl", "max_steps": 4},
        checkpoint_dir=Path("/tmp/does-not-need-to-exist-for-this-check/checkpoints"),
        seed=42,
    )
    assert "--hidden_size" in args
    assert args[args.index("--hidden_size") + 1] == str(MINIMIND_DEFAULT_HIDDEN_SIZE)
    assert "--num_hidden_layers" in args
    assert args[args.index("--num_hidden_layers") + 1] == str(MINIMIND_DEFAULT_NUM_HIDDEN_LAYERS)
    assert "--use_moe" in args
    assert args[args.index("--use_moe") + 1] == ("1" if MINIMIND_DEFAULT_USE_MOE else "0")


def test_build_subprocess_args_forwards_caller_hidden_size_num_layers_and_moe() -> None:
    """Issue #67 Finding 1's direct regression guard.

    A caller supplying non-default ``hidden_size``/``num_hidden_layers``/
    ``use_moe`` in ``training_params`` must see those exact values
    forwarded as ``--hidden_size``/``--num_hidden_layers``/``--use_moe``
    -- previously these keys were silently ignored and every real run
    trained at the script's own 768-hidden-size default regardless of
    what the caller asked for.
    """
    adapter = MiniMindTrainerAdapter(work_dir=Path("/tmp/does-not-need-to-exist-for-this-check"))
    args = adapter._build_subprocess_args(
        python_executable="python3",
        script_path=Path("/tmp/does-not-need-to-exist-for-this-check/train_full_sft.py"),
        params={
            "dataset_path": "irrelevant.jsonl",
            "max_steps": 4,
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "use_moe": True,
        },
        checkpoint_dir=Path("/tmp/does-not-need-to-exist-for-this-check/checkpoints"),
        seed=42,
    )
    assert args[args.index("--hidden_size") + 1] == "512"
    assert args[args.index("--num_hidden_layers") + 1] == "8"
    assert args[args.index("--use_moe") + 1] == "1"


def test_train_reaches_adr0011_minimind2_small_config_end_to_end(tmp_path, monkeypatch):
    """Proves ADR-0011's proposed minimind2-small config is now reachable.

    End-to-end (through the real, non-mocked ``_build_subprocess_args``
    and ``_stage_from_weight_checkpoint`` call sites inside ``train()``,
    only ``subprocess.Popen`` itself mocked) proof that a caller
    requesting ADR-0011's proposed minimind2-small pilot config
    (``hidden_size=512, num_hidden_layers=8, use_moe=False``, ~26M
    params) actually gets that config threaded all the way through to
    the constructed CLI invocation AND to the --from_weight staging path
    (which must resolve using the SAME hidden_size, per MiniMind's own
    ``f'{save_dir}/{from_weight}_{hidden_size}{moe_suffix}.pth'``
    formula) -- not silently downgraded to the script's 768-hidden-size
    default. This is the exact gap Maya's issue #67 Finding 1 review
    comment identified as invalidating ADR-0011's resource-limits table.
    """
    repo_path, head = make_fake_minimind_repo(tmp_path, pinned=False)
    monkeypatch.setattr("codevolt_mdf.minimind_adapter.MINIMIND_PINNED_COMMIT", head)

    model_path, model_hash = make_local_model(tmp_path, content=b"minimind2-small-checkpoint")
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(
        tmp_path,
        minimind_repo_path=repo_path,
        model_path=model_path,
        model_hash=model_hash,
        extra_params={
            "epochs": 1,
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "use_moe": False,
        },
    )
    budget = make_budget()
    token = CancellationToken()

    mock_process = MagicMock()
    mock_process.wait.return_value = 0
    mock_process.returncode = 0
    mock_process.stdout.read.return_value = "fake minimind training log\n"

    with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
        output = adapter.train(inputs, budget, token)

    assert output.status == TrainingStatus.ACCEPTED
    constructed_args = mock_popen.call_args[0][0]
    assert constructed_args[constructed_args.index("--hidden_size") + 1] == "512"
    assert constructed_args[constructed_args.index("--num_hidden_layers") + 1] == "8"
    assert constructed_args[constructed_args.index("--use_moe") + 1] == "0"

    # The --from_weight staged file must be resolvable using hidden_size=512
    # (ADR-0011's proposed config), not the MINIMIND_DEFAULT_HIDDEN_SIZE=768
    # fallback -- MiniMind's own init_model() would otherwise fail to find
    # the checkpoint at the filename it actually looks for.
    shadow_trainer_dir = Path(mock_popen.call_args[1]["cwd"])
    shadow_dir = shadow_trainer_dir.parent
    expected_staged_path = (
        shadow_dir / "out" / f"{MINIMIND_FROM_WEIGHT_STAGED_NAME}_512.pth"
    )
    assert expected_staged_path.is_file()
    assert expected_staged_path.read_bytes() == b"minimind2-small-checkpoint"
    # And the 768-hidden-size default filename must NOT exist -- proving
    # this run genuinely used the caller's 512 override, not the fallback.
    default_staged_path = (
        shadow_dir / "out" / f"{MINIMIND_FROM_WEIGHT_STAGED_NAME}_{MINIMIND_DEFAULT_HIDDEN_SIZE}.pth"
    )
    assert not default_staged_path.exists()


def test_train_stages_from_weight_checkpoint_at_expected_path(tmp_path, monkeypatch):
    """Real-filesystem proof (no mocking of shutil) that train() stages the

    model_path checkpoint at the EXACT path MiniMind's own
    trainer/trainer_utils.py:init_model() resolution formula
    (f'{save_dir}/{from_weight}_{hidden_size}{moe_suffix}.pth', with
    init_model()'s own hardcoded save_dir='../out' default -- never this
    adapter's --save_dir flag) will look for it, relative to the real
    subprocess cwd (the shadow trainer dir). This is the direct regression
    guard for Maya's PR #62 re-review escalation: the
    pre-fix adapter passed model_path literally as --from_weight, which
    MiniMind's own script cannot resolve to any real file at all.
    """
    repo_path, head = make_fake_minimind_repo(tmp_path, pinned=False)
    monkeypatch.setattr("codevolt_mdf.minimind_adapter.MINIMIND_PINNED_COMMIT", head)

    model_path, model_hash = make_local_model(tmp_path, content=b"real-checkpoint-bytes")
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(
        tmp_path,
        minimind_repo_path=repo_path,
        model_path=model_path,
        model_hash=model_hash,
        extra_params={"epochs": 1},
    )
    budget = make_budget()
    token = CancellationToken()

    mock_process = MagicMock()
    mock_process.wait.return_value = 0
    mock_process.returncode = 0
    mock_process.stdout.read.return_value = "fake minimind training log\n"

    with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
        output = adapter.train(inputs, budget, token)

    assert output.status == TrainingStatus.ACCEPTED
    shadow_trainer_dir = Path(mock_popen.call_args[1]["cwd"])
    shadow_dir = shadow_trainer_dir.parent

    # This is exactly what init_model(lm_config, 'codevolt-staged-from-weight',
    # device=...)'s own weight_path formula computes, relative to a cwd of
    # shadow_trainer_dir: f'../out/{from_weight}_{hidden_size}{moe_suffix}.pth'.
    moe_suffix = "_moe" if MINIMIND_DEFAULT_USE_MOE else ""
    expected_staged_path = (
        shadow_dir / "out" / f"{MINIMIND_FROM_WEIGHT_STAGED_NAME}_{MINIMIND_DEFAULT_HIDDEN_SIZE}{moe_suffix}.pth"
    )
    assert expected_staged_path.is_file()
    assert expected_staged_path.read_bytes() == b"real-checkpoint-bytes"

    # Not a symlink, for the same containment reason as the trainer/model/
    # dataset shadow copies (see test_train_creates_isolated_shadow_directory_layout).
    assert not expected_staged_path.is_symlink()


def test_train_omits_from_weight_staging_when_model_path_unset(tmp_path, monkeypatch):
    """No staged file and no --from_weight flag when model_path is absent.

    ``model_path`` is currently required by ``prepare()``, but
    ``_build_subprocess_args``/``train()`` both branch on
    ``params.get("model_path")`` independently of that -- this proves the
    "unset" branch does not, e.g., stage a stray empty file or crash.
    """
    repo_path, head = make_fake_minimind_repo(tmp_path, pinned=False)
    monkeypatch.setattr("codevolt_mdf.minimind_adapter.MINIMIND_PINNED_COMMIT", head)

    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, minimind_repo_path=repo_path, extra_params={"epochs": 1})
    # Directly rebuild training_params without model_path, bypassing
    # make_inputs' default (which always sets one) -- exercises the
    # params.get("model_path") falsy branch in both _build_subprocess_args
    # and train()'s staging call.
    params = dict(inputs.training_params)
    del params["model_path"]
    inputs = TrainingInputs.create(
        model_revision=inputs.model_revision,
        model_hash=inputs.model_hash,
        dataset_version=inputs.dataset_version,
        dataset_hash=inputs.dataset_hash,
        seed=inputs.seed,
        training_params=params,
        dataset_licence=inputs.dataset_licence,
        contamination_checked=inputs.contamination_checked,
        run_id=inputs.run_id,
    )
    budget = make_budget()
    token = CancellationToken()

    mock_process = MagicMock()
    mock_process.wait.return_value = 0
    mock_process.returncode = 0
    mock_process.stdout.read.return_value = "fake minimind training log\n"

    with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
        output = adapter.train(inputs, budget, token)

    assert output.status == TrainingStatus.ACCEPTED
    argv = mock_popen.call_args[0][0]
    assert "--from_weight" not in argv
    shadow_trainer_dir = Path(mock_popen.call_args[1]["cwd"])
    shadow_dir = shadow_trainer_dir.parent
    assert not (shadow_dir / "out").exists()


def test_prepare_rejects_directory_model_path(tmp_path, monkeypatch):
    """MiniMind's --from_weight resolution has no directory-loading path.

    A directory model_path can never be staged into the single-file shape
    MiniMind's init_model() resolution expects, so this is rejected in
    prepare() (cheapest correct layer) rather than failing opaquely deep
    inside a real subprocess invocation.
    """
    repo_path, head = make_fake_minimind_repo(tmp_path, pinned=False)
    monkeypatch.setattr("codevolt_mdf.minimind_adapter.MINIMIND_PINNED_COMMIT", head)

    model_dir = tmp_path / "model-as-directory"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"fake-weights")
    model_hash = _hash_path_identity(model_dir)

    adapter = make_adapter(tmp_path)
    inputs = make_inputs(
        tmp_path,
        minimind_repo_path=repo_path,
        model_path=str(model_dir),
        model_hash=model_hash,
    )
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="is a directory"):
        adapter.prepare(inputs, budget)


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


def test_cleanup_removes_shadow_dir_with_read_only_copies(tmp_path, caplog):
    """Regression guard for the shadow-dir permission leak.

    ``shutil.copytree``'s default ``copy2`` copy function preserves the
    real checkout's read-only permission bits onto the ``mm_shadow/``
    copies made in ``train()`` (see the module docstring there). This
    reproduces that exact filesystem state directly -- a real read-only
    file inside a real ``mm_shadow/`` tree, no mocking of ``shutil`` or
    ``os`` -- for an INTERRUPTED-style run outcome (the caller always
    calls ``adapter.cleanup(run_id)`` regardless of the run's terminal
    status; see ``trainer_contract.run_trainer_contract``), and asserts
    the directory is actually gone from disk afterwards, not merely that
    some ``rmtree`` call was made.
    """
    adapter = make_adapter(tmp_path)
    run_dir = adapter._run_dir("run-interrupted")
    shadow_dir = run_dir / "mm_shadow"
    shadow_trainer_dir = shadow_dir / "trainer"
    shadow_trainer_dir.mkdir(parents=True)

    ro_file = shadow_trainer_dir / "dataset.md"
    ro_file.write_text("read-only shadow copy content")
    # Mirror shutil.copytree(..., copy2=True)'s effect on a shadow copy of
    # a read-only real checkout: strip owner/group/other write bits.
    ro_file.chmod(stat.S_IREAD)
    shadow_trainer_dir.chmod(stat.S_IREAD | stat.S_IEXEC)

    # Directly confirm the failure mode this test guards against: a plain
    # shutil.rmtree with no special handling really does raise
    # PermissionError against this exact fixture, matching Maya's live
    # confirmation in the review.
    with pytest.raises(PermissionError):
        shutil.rmtree(run_dir)
    # Restore so the fixture is in a known state for the real assertion
    # below (the failed rmtree above may have partially removed entries
    # it *could* delete before hitting the read-only one).
    shadow_trainer_dir.chmod(stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
    ro_file.chmod(stat.S_IWRITE | stat.S_IREAD)
    ro_file.write_text("read-only shadow copy content")
    ro_file.chmod(stat.S_IREAD)
    shadow_trainer_dir.chmod(stat.S_IREAD | stat.S_IEXEC)

    with caplog.at_level("WARNING"):
        adapter.cleanup("run-interrupted")

    assert not run_dir.exists(), "cleanup() must remove read-only shadow copies, not leak them"
    assert not any(
        "failed to fully remove run directory" in record.message for record in caplog.records
    ), "cleanup() succeeded and must not also log a leak warning"


def test_cleanup_logs_warning_when_removal_genuinely_cannot_succeed(tmp_path, monkeypatch, caplog):
    """If removal still fails after the restore-and-retry pass, cleanup() must
    warn with the leaked path rather than silently swallowing the failure
    (the defect: ``ignore_errors=True`` gave zero operator visibility).
    """
    adapter = make_adapter(tmp_path)
    run_dir = adapter._run_dir("run-unremovable")
    (run_dir / "marker.txt").write_text("evidence")

    monkeypatch.setattr(
        "codevolt_mdf.minimind_adapter._rmtree_best_effort", lambda path: False
    )

    with caplog.at_level("WARNING"):
        adapter.cleanup("run-unremovable")

    assert any(
        "failed to fully remove run directory" in record.message
        and "run-unremovable" in record.message
        for record in caplog.records
    )


def test_cleanup_idempotent_on_double_call_after_read_only_removal(tmp_path):
    """No regression to the existing idempotent-double-cleanup behaviour
    once a previously read-only shadow tree has actually been removed.
    """
    adapter = make_adapter(tmp_path)
    run_dir = adapter._run_dir("run-double-clean")
    shadow_trainer_dir = run_dir / "mm_shadow" / "trainer"
    shadow_trainer_dir.mkdir(parents=True)
    ro_file = shadow_trainer_dir / "readonly.py"
    ro_file.write_text("x = 1\n")
    ro_file.chmod(stat.S_IREAD)
    shadow_trainer_dir.chmod(stat.S_IREAD | stat.S_IEXEC)

    adapter.cleanup("run-double-clean")
    assert not run_dir.exists()

    adapter.cleanup("run-double-clean")  # second call on an already-gone path must not raise
    assert not run_dir.exists()


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
