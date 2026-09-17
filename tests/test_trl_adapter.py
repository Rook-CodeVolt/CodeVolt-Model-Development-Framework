"""Contract-conformance tests for TRLTrainerAdapter (`codevolt_mdf.trl_adapter`).

Scope, deliberately narrow (see ``docs/decisions/0005-trl-trainer-adapter-v1.md``
and the module docstring in ``trl_adapter.py``): this module proves the
adapter's ``prepare()``-level validation and contract-runner wiring for
every scenario reachable **without ever calling ``adapter.train()``** --
no real TRL training run happens anywhere in this file. That is a hard
boundary this work package does not cross: the fake-adapter-style
success/rejection/invalid-input conformance pattern from
``test_trainer_contract.py`` is reused everywhere the contract allows
proving the behaviour without training (rejection, invalid-input,
upstream-version-incompatibility, provenance/hash mismatch, offline-policy
enforcement), and is explicitly NOT reused for the scenarios that would
require an actual run (success/timeout/cancellation/resource-overrun/
checkpoint-resume/tamper against a real trained artifact) -- those remain
untested against this adapter until the separately gated real pilot,
per the ADR's explicit "what is NOT covered" list.

Where a check genuinely requires the ``trl`` package importable (e.g.
confirming the installed version the environment actually has falls
inside the adapter's declared compatibility bound), the test uses
``pytest.importorskip("trl")`` rather than failing hard in an environment
that has not installed the optional real-engine dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codevolt_mdf.trainer_contract import (
    CONTRACT_VERSION,
    InvalidInputError,
    RejectedInputError,
    ResourceBudget,
    TrainingInputs,
    TrainingStatus,
    UpstreamRequirement,
    run_trainer_contract,
)
from codevolt_mdf.trl_adapter import (
    TRL_MAX_VERSION,
    TRL_MIN_VERSION,
    TRLTrainerAdapter,
    _detect_installed_trl_version,
    _hash_path_identity,
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


def make_adapter(tmp_path) -> TRLTrainerAdapter:
    return TRLTrainerAdapter(work_dir=tmp_path / "adapter-work")


def make_local_model(tmp_path, content: bytes = b"fake-model-weights") -> tuple[str, str]:
    """Create a tiny local file standing in for a model checkpoint; return (path, hash)."""
    model_path = tmp_path / "model" / "weights.bin"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(content)
    return str(model_path), _hash_path_identity(model_path)


def make_local_dataset(tmp_path, rows: list[dict] | None = None) -> tuple[str, str]:
    """Create a tiny local JSONL dataset file; return (path, hash)."""
    rows = rows if rows is not None else [{"text": "hello world"}]
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
    model_path: str | None = None,
    model_hash: str | None = None,
    dataset_path: str | None = None,
    dataset_hash: str | None = None,
    max_steps: int | None = 4,
    extra_params: dict | None = None,
    **overrides,
) -> TrainingInputs:
    # model_path/dataset_path and their hashes are generated independently
    # of each other so a caller can override just the hash (to deliberately
    # create a mismatch) while keeping a real, existing path, or vice versa.
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

    params: dict = {"model_path": model_path, "dataset_path": dataset_path}
    if max_steps is not None:
        params["max_steps"] = max_steps
    if extra_params:
        params.update(extra_params)

    defaults = {
        "model_revision": "trl-pilot-v1",
        "model_hash": model_hash,
        "dataset_version": "synthetic-trl-v1",
        "dataset_hash": dataset_hash,
        "seed": 42,
        "training_params": params,
        "dataset_licence": "CC0-1.0",
        "contamination_checked": True,
        "run_id": run_id or "run-trl-adapter-test",
    }
    defaults.update(overrides)
    return TrainingInputs.create(**defaults)


# 1. Adapter identity and declared contract/upstream bounds -----------------


def test_adapter_declares_supported_contract_version(tmp_path):
    adapter = make_adapter(tmp_path)
    assert adapter.contract_version == CONTRACT_VERSION


def test_adapter_declares_trl_as_upstream_engine(tmp_path):
    adapter = make_adapter(tmp_path)
    assert adapter.upstream.engine_name == "trl"
    assert adapter.upstream.min_version == TRL_MIN_VERSION
    assert adapter.upstream.max_version == TRL_MAX_VERSION


def test_detect_installed_trl_version_returns_zero_when_unimportable(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "trl":
            raise ImportError("simulated: trl not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    assert _detect_installed_trl_version() == "0.0.0"


def test_installed_trl_version_is_within_declared_bounds():
    """If trl is installed in this environment, it must satisfy the adapter's own bound.

    This is the concrete evidence backing the ADR's version-bound claim:
    it fails loudly if the installed trl drifts outside [min, max] without
    the ADR being revisited, rather than only being asserted in prose.
    """
    trl = pytest.importorskip("trl")
    req = UpstreamRequirement(
        engine_name="trl",
        min_version=TRL_MIN_VERSION,
        max_version=TRL_MAX_VERSION,
        installed_version=trl.__version__,
    )
    assert req.is_compatible(), (
        f"installed trl=={trl.__version__} is outside the adapter's declared bound "
        f"[{TRL_MIN_VERSION}, {TRL_MAX_VERSION}]"
    )


def test_trl_api_surface_matches_adapter_expectations():
    """Confirms SFTTrainer/SFTConfig/TrainerCallback still expose what train() calls.

    Import-only + signature-inspection check -- proves the adapter's
    train() implementation is not calling into a renamed/removed API,
    without ever instantiating a trainer or running training.
    """
    trl = pytest.importorskip("trl")
    import inspect

    from trl import SFTConfig, SFTTrainer

    sft_config_params = set(inspect.signature(SFTConfig.__init__).parameters)
    for required in (
        "output_dir",
        "max_steps",
        "save_steps",
        "save_strategy",
        "learning_rate",
        "per_device_train_batch_size",
        "seed",
        "report_to",
        "logging_steps",
    ):
        assert required in sft_config_params, f"SFTConfig missing expected param {required!r}"

    sft_trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    for required in ("model", "args", "train_dataset", "peft_config", "callbacks"):
        assert required in sft_trainer_params, f"SFTTrainer missing expected param {required!r}"
    del trl  # imported only to trigger importorskip


# 2. Contract-runner wiring: upstream-version rejection (never reaches prepare/train) --


def test_incompatible_upstream_version_is_rejected_before_prepare(tmp_path):
    """The contract runner raises RejectedInputError directly for an incompatible
    upstream version (unlike prepare()-level rejections, which it catches and
    turns into a TrainingOutput(status=REJECTED)) -- this is existing
    trainer_contract.run_trainer_contract() behaviour (see its unconditional
    `if not adapter.upstream.is_compatible(): raise RejectedInputError(...)`
    before the try/except block), not something this adapter controls.
    """
    adapter = make_adapter(tmp_path)
    adapter.upstream = UpstreamRequirement(
        engine_name="trl",
        min_version=TRL_MIN_VERSION,
        max_version=TRL_MAX_VERSION,
        installed_version="0.1.0",  # deliberately below TRL_MIN_VERSION
    )
    inputs = make_inputs(tmp_path)
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="outside the declared"):
        run_trainer_contract(adapter, inputs, budget)


def test_incompatible_upstream_version_above_max_is_rejected(tmp_path):
    adapter = make_adapter(tmp_path)
    adapter.upstream = UpstreamRequirement(
        engine_name="trl",
        min_version=TRL_MIN_VERSION,
        max_version=TRL_MAX_VERSION,
        installed_version="99.0.0",  # deliberately above TRL_MAX_VERSION
    )
    inputs = make_inputs(tmp_path)
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="outside the declared"):
        run_trainer_contract(adapter, inputs, budget)


# 3. prepare(): well-formed local inputs pass validation (no train() call) ----


def test_prepare_accepts_well_formed_local_inputs(tmp_path):
    """Proves prepare() alone -- never run_trainer_contract() -- to avoid train()."""
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path)
    budget = make_budget()

    adapter.prepare(inputs, budget)  # must not raise


# 4. prepare(): offline network policy is mandatory ---------------------------


def test_prepare_rejects_non_offline_network_policy(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path)
    budget = make_budget(network_policy="allow-list", allowed_hosts=("huggingface.co",))

    with pytest.raises(RejectedInputError, match="offline"):
        adapter.prepare(inputs, budget)


def test_contract_runner_rejects_non_offline_policy_before_training(tmp_path):
    """Full run_trainer_contract() path -- requires trl importable to clear the
    upstream-version gate and actually reach adapter.prepare()'s own rejection.
    """
    pytest.importorskip("trl")
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path)
    budget = make_budget(network_policy="allow-list", allowed_hosts=("huggingface.co",))

    output = run_trainer_contract(adapter, inputs, budget)

    assert output.status == TrainingStatus.REJECTED
    assert "offline" in output.reason


# 5. prepare(): required training_params ---------------------------------------


def test_prepare_rejects_missing_model_path(tmp_path):
    adapter = make_adapter(tmp_path)
    _, dataset_hash = make_local_dataset(tmp_path)
    inputs = TrainingInputs.create(
        model_revision="trl-pilot-v1",
        model_hash=MODEL_HASH_PLACEHOLDER,
        dataset_version="synthetic-trl-v1",
        dataset_hash=dataset_hash,
        seed=42,
        training_params={"dataset_path": "irrelevant", "max_steps": 4},
        dataset_licence="CC0-1.0",
        contamination_checked=True,
        run_id="run-missing-model-path",
    )
    budget = make_budget()

    with pytest.raises(InvalidInputError, match="model_path"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_missing_dataset_path(tmp_path):
    adapter = make_adapter(tmp_path)
    model_path, model_hash = make_local_model(tmp_path)
    inputs = TrainingInputs.create(
        model_revision="trl-pilot-v1",
        model_hash=model_hash,
        dataset_version="synthetic-trl-v1",
        dataset_hash=DATASET_HASH_PLACEHOLDER,
        seed=42,
        training_params={"model_path": model_path, "max_steps": 4},
        dataset_licence="CC0-1.0",
        contamination_checked=True,
        run_id="run-missing-dataset-path",
    )
    budget = make_budget()

    with pytest.raises(InvalidInputError, match="dataset_path"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_missing_max_steps(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, max_steps=None)
    budget = make_budget()

    with pytest.raises(InvalidInputError, match="max_steps"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_zero_or_negative_max_steps(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, max_steps=0)
    budget = make_budget()

    with pytest.raises(InvalidInputError, match="max_steps"):
        adapter.prepare(inputs, budget)


# 6. prepare(): provenance -- paths must exist locally --------------------------


def test_prepare_rejects_nonexistent_model_path(tmp_path):
    adapter = make_adapter(tmp_path)
    _, dataset_hash = make_local_dataset(tmp_path)
    dataset_path, dataset_hash = make_local_dataset(tmp_path)
    inputs = TrainingInputs.create(
        model_revision="trl-pilot-v1",
        model_hash=MODEL_HASH_PLACEHOLDER,
        dataset_version="synthetic-trl-v1",
        dataset_hash=dataset_hash,
        seed=42,
        training_params={
            "model_path": str(tmp_path / "does-not-exist"),
            "dataset_path": dataset_path,
            "max_steps": 4,
        },
        dataset_licence="CC0-1.0",
        contamination_checked=True,
        run_id="run-missing-model-file",
    )
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="does not exist"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_nonexistent_dataset_path(tmp_path):
    adapter = make_adapter(tmp_path)
    model_path, model_hash = make_local_model(tmp_path)
    inputs = TrainingInputs.create(
        model_revision="trl-pilot-v1",
        model_hash=model_hash,
        dataset_version="synthetic-trl-v1",
        dataset_hash=DATASET_HASH_PLACEHOLDER,
        seed=42,
        training_params={
            "model_path": model_path,
            "dataset_path": str(tmp_path / "does-not-exist.jsonl"),
            "max_steps": 4,
        },
        dataset_licence="CC0-1.0",
        contamination_checked=True,
        run_id="run-missing-dataset-file",
    )
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="does not exist"):
        adapter.prepare(inputs, budget)


# 7. prepare(): provenance -- hash mismatch is refused (tamper/corruption) ------


def test_prepare_rejects_model_hash_mismatch(tmp_path):
    adapter = make_adapter(tmp_path)
    model_path, _real_hash = make_local_model(tmp_path)
    dataset_path, dataset_hash = make_local_dataset(tmp_path)
    inputs = TrainingInputs.create(
        model_revision="trl-pilot-v1",
        model_hash=MODEL_HASH_PLACEHOLDER,  # deliberately wrong
        dataset_version="synthetic-trl-v1",
        dataset_hash=dataset_hash,
        seed=42,
        training_params={"model_path": model_path, "dataset_path": dataset_path, "max_steps": 4},
        dataset_licence="CC0-1.0",
        contamination_checked=True,
        run_id="run-model-hash-mismatch",
    )
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="model_hash mismatch"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_dataset_hash_mismatch(tmp_path):
    adapter = make_adapter(tmp_path)
    model_path, model_hash = make_local_model(tmp_path)
    dataset_path, _real_hash = make_local_dataset(tmp_path)
    inputs = TrainingInputs.create(
        model_revision="trl-pilot-v1",
        model_hash=model_hash,
        dataset_version="synthetic-trl-v1",
        dataset_hash=DATASET_HASH_PLACEHOLDER,  # deliberately wrong
        seed=42,
        training_params={"model_path": model_path, "dataset_path": dataset_path, "max_steps": 4},
        dataset_licence="CC0-1.0",
        contamination_checked=True,
        run_id="run-dataset-hash-mismatch",
    )
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="dataset_hash mismatch"):
        adapter.prepare(inputs, budget)


def test_contract_runner_reports_invalid_or_rejected_for_tampered_model(tmp_path):
    """End-to-end through run_trainer_contract(): tamper is caught before train().

    Requires trl importable to clear the upstream-version gate first.
    """
    pytest.importorskip("trl")
    adapter = make_adapter(tmp_path)
    model_path, real_hash = make_local_model(tmp_path)
    dataset_path, dataset_hash = make_local_dataset(tmp_path)
    inputs = make_inputs(
        tmp_path,
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


# 8. prepare(): LoRA requested without peft installed ---------------------------


def test_prepare_rejects_lora_when_peft_unimportable(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "peft":
            raise ImportError("simulated: peft not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, extra_params={"use_lora": True})
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="peft"):
        adapter.prepare(inputs, budget)


# 9. hash-identity helper: deterministic, content-sensitive ---------------------


def test_hash_path_identity_is_deterministic_for_same_content(tmp_path):
    _path_a, hash_a = make_local_model(tmp_path / "a", content=b"identical-content")
    _path_b, hash_b = make_local_model(tmp_path / "b", content=b"identical-content")
    assert hash_a == hash_b


def test_hash_path_identity_changes_with_content(tmp_path):
    _, hash_a = make_local_model(tmp_path / "a", content=b"content-one")
    _, hash_b = make_local_model(tmp_path / "b", content=b"content-two")
    assert hash_a != hash_b


def test_hash_path_identity_hashes_directory_manifest(tmp_path):
    dir_a = tmp_path / "model_dir_a"
    dir_a.mkdir()
    (dir_a / "config.json").write_bytes(b'{"a": 1}')
    (dir_a / "weights.bin").write_bytes(b"weights")

    dir_b = tmp_path / "model_dir_b"
    dir_b.mkdir()
    (dir_b / "config.json").write_bytes(b'{"a": 1}')
    (dir_b / "weights.bin").write_bytes(b"weights")

    assert _hash_path_identity(dir_a) == _hash_path_identity(dir_b)

    # Changing one file's content changes the directory hash.
    (dir_b / "weights.bin").write_bytes(b"different-weights")
    assert _hash_path_identity(dir_a) != _hash_path_identity(dir_b)


def test_hash_path_identity_rejects_nonexistent_path(tmp_path):
    with pytest.raises(RejectedInputError):
        _hash_path_identity(tmp_path / "does-not-exist-at-all")


# 10. cleanup(): idempotent, safe with nothing to clean up -----------------------


def test_cleanup_is_idempotent_and_safe_with_nothing_to_clean(tmp_path):
    adapter = make_adapter(tmp_path)
    # No run ever created a directory for this run_id.
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


# 11. TrainingInputs.validate() precondition still applies (adapter-independent) --


def test_training_inputs_validate_rejects_uncontaminated_dataset_for_trl_inputs(tmp_path):
    inputs = make_inputs(tmp_path, contamination_checked=False)
    with pytest.raises(InvalidInputError):
        inputs.validate()


def test_contract_runner_never_reaches_prepare_for_invalid_inputs(tmp_path):
    """Bad hash shape fails TrainingInputs.validate() before adapter.prepare() runs.

    Requires trl importable to clear the upstream-version gate first (that
    gate runs even earlier than TrainingInputs.validate(), so without trl
    installed the run would be REJECTED for the wrong reason before ever
    reaching the INVALID check this test is actually about).
    """
    pytest.importorskip("trl")
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, model_hash="not-a-sha256")
    budget = make_budget()

    output = run_trainer_contract(adapter, inputs, budget)

    assert output.status == TrainingStatus.INVALID
    assert output.error_class == InvalidInputError.__name__
