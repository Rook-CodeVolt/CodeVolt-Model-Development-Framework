"""Contract-conformance tests for DPOTrainerAdapter (`codevolt_mdf.dpo_adapter`).

Same scope boundary as `tests/test_trl_adapter.py`: proves the adapter's
`prepare()`-level validation and contract-runner wiring for every scenario
reachable **without ever calling `adapter.train()`** -- no real DPO training
run happens anywhere in this file, per ADR-0017's "design/dataset-shape
proposal only; grants no training authority" scope.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codevolt_mdf.dpo_adapter import DPOTrainerAdapter
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
from codevolt_mdf.trl_adapter import TRL_MAX_VERSION, TRL_MIN_VERSION, _hash_path_identity

MODEL_HASH_PLACEHOLDER = "a" * 64
DATASET_HASH_PLACEHOLDER = "b" * 64
REFERENCE_HASH_PLACEHOLDER = "c" * 64


def make_budget(**overrides) -> ResourceBudget:
    defaults = {
        "max_wall_seconds": 5.0,
        "max_cpu_seconds": 5.0,
        "max_memory_mb": 1024.0,
        "max_gpu_count": 0,
        "max_storage_mb": 64.0,
        "network_policy": "offline",
    }
    defaults.update(overrides)
    return ResourceBudget(**defaults)


def make_adapter(tmp_path) -> DPOTrainerAdapter:
    return DPOTrainerAdapter(work_dir=tmp_path / "adapter-work")


def make_local_model(tmp_path, content: bytes = b"fake-model-weights") -> tuple[str, str]:
    model_path = tmp_path / "model" / "weights.bin"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(content)
    return str(model_path), _hash_path_identity(model_path)


def make_local_reference_model(tmp_path, content: bytes = b"fake-reference-weights") -> tuple[str, str]:
    ref_path = tmp_path / "reference-model" / "weights.bin"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_bytes(content)
    return str(ref_path), _hash_path_identity(ref_path)


def make_local_preference_dataset(tmp_path, rows: list[dict] | None = None) -> tuple[str, str]:
    """Create a tiny local prompt/chosen/rejected JSONL dataset; return (path, hash)."""
    rows = rows if rows is not None else [
        {"prompt": "Is the sky blue?", "chosen": "Yes, typically during clear daytime conditions.", "rejected": "The sky is green."}
    ]
    dataset_path = tmp_path / "dataset" / "preference_pairs.jsonl"
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
    reference_model_path: str | None = None,
    reference_model_hash: str | None = None,
    max_steps: int | None = 4,
    beta: float | None = 0.1,
    reference_free: bool | None = False,
    extra_params: dict | None = None,
    **overrides,
) -> TrainingInputs:
    if model_path is None:
        model_path, generated_model_hash = make_local_model(tmp_path)
        if model_hash is None:
            model_hash = generated_model_hash
    elif model_hash is None:
        model_hash = _hash_path_identity(Path(model_path))

    if dataset_path is None:
        dataset_path, generated_dataset_hash = make_local_preference_dataset(tmp_path)
        if dataset_hash is None:
            dataset_hash = generated_dataset_hash
    elif dataset_hash is None:
        dataset_hash = _hash_path_identity(Path(dataset_path))

    if reference_model_path is None:
        reference_model_path, generated_ref_hash = make_local_reference_model(tmp_path)
        if reference_model_hash is None:
            reference_model_hash = generated_ref_hash
    elif reference_model_hash is None:
        reference_model_hash = _hash_path_identity(Path(reference_model_path))

    params: dict = {"model_path": model_path, "dataset_path": dataset_path}
    if reference_model_path is not None:
        params["reference_model_path"] = reference_model_path
    if reference_model_hash is not None:
        params["reference_model_hash"] = reference_model_hash
    if max_steps is not None:
        params["max_steps"] = max_steps
    if beta is not None:
        params["beta"] = beta
    if reference_free is not None:
        params["reference_free"] = reference_free
    if extra_params:
        params.update(extra_params)

    defaults = {
        "model_revision": "trl-dpo-pilot-v1",
        "model_hash": model_hash,
        "dataset_version": "synthetic-dpo-v1",
        "dataset_hash": dataset_hash,
        "seed": 42,
        "training_params": params,
        "dataset_licence": "CC0-1.0",
        "contamination_checked": True,
        "run_id": run_id or "run-dpo-adapter-test",
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


def test_dpo_api_surface_matches_adapter_expectations():
    """Confirms DPOTrainer/DPOConfig still expose what train() calls.

    Import-only + signature-inspection check -- proves the adapter's
    train() implementation is not calling into a renamed/removed API,
    without ever instantiating a trainer or running training.
    """
    trl = pytest.importorskip("trl")
    import inspect

    from trl import DPOConfig, DPOTrainer

    dpo_config_params = set(inspect.signature(DPOConfig.__init__).parameters)
    for required in (
        "output_dir",
        "max_steps",
        "save_steps",
        "save_strategy",
        "save_total_limit",
        "learning_rate",
        "per_device_train_batch_size",
        "seed",
        "report_to",
        "logging_steps",
        "beta",
        "reference_free",
        "max_prompt_length",
        "max_length",
    ):
        assert required in dpo_config_params, f"DPOConfig missing expected param {required!r}"

    dpo_trainer_params = set(inspect.signature(DPOTrainer.__init__).parameters)
    for required in ("model", "ref_model", "args", "train_dataset", "callbacks"):
        assert required in dpo_trainer_params, f"DPOTrainer missing expected param {required!r}"
    del trl  # imported only to trigger importorskip


# 2. Contract-runner wiring: upstream-version rejection ----------------------


def test_incompatible_upstream_version_is_rejected_before_prepare(tmp_path):
    adapter = make_adapter(tmp_path)
    adapter.upstream = UpstreamRequirement(
        engine_name="trl",
        min_version=TRL_MIN_VERSION,
        max_version=TRL_MAX_VERSION,
        installed_version="0.1.0",
    )
    inputs = make_inputs(tmp_path)
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="outside the declared"):
        run_trainer_contract(adapter, inputs, budget)


# 3. prepare(): well-formed local inputs pass validation ---------------------


def test_prepare_accepts_well_formed_local_inputs(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path)
    budget = make_budget()

    adapter.prepare(inputs, budget)  # must not raise


def test_prepare_accepts_reference_free_true_but_still_requires_reference_artifact(tmp_path):
    """reference_free=True does not waive reference_model_path/hash verification.

    See DPOTrainerAdapter class docstring's reference_free note: this
    adapter always verifies a reference-model artifact's provenance,
    regardless of the eventual reference_free training configuration.
    """
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, reference_free=True)
    budget = make_budget()

    adapter.prepare(inputs, budget)  # must not raise -- reference still required+verified


# 4. prepare(): offline network policy is mandatory ---------------------------


def test_prepare_rejects_non_offline_network_policy(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path)
    budget = make_budget(network_policy="allow-list", allowed_hosts=("huggingface.co",))

    with pytest.raises(RejectedInputError, match="offline"):
        adapter.prepare(inputs, budget)


# 5. prepare(): required training_params -------------------------------------


def test_prepare_rejects_missing_model_path(tmp_path):
    adapter = make_adapter(tmp_path)
    _, dataset_hash = make_local_preference_dataset(tmp_path)
    _, ref_hash = make_local_reference_model(tmp_path)
    inputs = TrainingInputs.create(
        model_revision="trl-dpo-pilot-v1",
        model_hash=MODEL_HASH_PLACEHOLDER,
        dataset_version="synthetic-dpo-v1",
        dataset_hash=dataset_hash,
        seed=42,
        training_params={
            "dataset_path": "irrelevant",
            "reference_model_path": "irrelevant",
            "reference_model_hash": ref_hash,
            "max_steps": 4,
            "beta": 0.1,
            "reference_free": False,
        },
        dataset_licence="CC0-1.0",
        contamination_checked=True,
        run_id="run-missing-model-path",
    )
    budget = make_budget()

    with pytest.raises(InvalidInputError, match="model_path"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_missing_reference_model_path(tmp_path):
    adapter = make_adapter(tmp_path)
    model_path, model_hash = make_local_model(tmp_path)
    _, dataset_hash = make_local_preference_dataset(tmp_path)
    inputs = TrainingInputs.create(
        model_revision="trl-dpo-pilot-v1",
        model_hash=model_hash,
        dataset_version="synthetic-dpo-v1",
        dataset_hash=dataset_hash,
        seed=42,
        training_params={
            "model_path": model_path,
            "dataset_path": "irrelevant-but-nonempty",
            "reference_model_hash": REFERENCE_HASH_PLACEHOLDER,
            "max_steps": 4,
            "beta": 0.1,
            "reference_free": False,
        },
        dataset_licence="CC0-1.0",
        contamination_checked=True,
        run_id="run-missing-reference-path",
    )
    budget = make_budget()

    with pytest.raises(InvalidInputError, match="reference_model_path"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_missing_reference_model_hash(tmp_path):
    adapter = make_adapter(tmp_path)
    model_path, model_hash = make_local_model(tmp_path)
    _, dataset_hash = make_local_preference_dataset(tmp_path)
    reference_model_path, _real_ref_hash = make_local_reference_model(tmp_path)
    inputs = TrainingInputs.create(
        model_revision="trl-dpo-pilot-v1",
        model_hash=model_hash,
        dataset_version="synthetic-dpo-v1",
        dataset_hash=dataset_hash,
        seed=42,
        training_params={
            "model_path": model_path,
            "dataset_path": "irrelevant-but-nonempty",
            "reference_model_path": reference_model_path,
            "max_steps": 4,
            "beta": 0.1,
            "reference_free": False,
        },
        dataset_licence="CC0-1.0",
        contamination_checked=True,
        run_id="run-missing-reference-hash",
    )
    budget = make_budget()

    with pytest.raises(InvalidInputError, match="reference_model_hash"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_malformed_reference_model_hash(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, reference_model_hash="not-a-sha256")
    budget = make_budget()

    with pytest.raises(InvalidInputError, match="reference_model_hash"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_missing_max_steps(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, max_steps=None)
    budget = make_budget()

    with pytest.raises(InvalidInputError, match="max_steps"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_missing_beta(tmp_path):
    """beta has NO library-default fallback in this adapter (unlike TRLTrainerAdapter's
    learning_rate) -- the security reviewer's gate 4 requires an explicit, reviewed value."""
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, beta=None)
    budget = make_budget()

    with pytest.raises(InvalidInputError, match="beta"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_zero_or_negative_beta(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, beta=0.0)
    budget = make_budget()

    with pytest.raises(InvalidInputError, match="beta"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_missing_reference_free(tmp_path):
    """reference_free has NO library-default fallback in this adapter -- the
    same gate-4 rationale as beta."""
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, reference_free=None)
    budget = make_budget()

    with pytest.raises(InvalidInputError, match="reference_free"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_non_bool_reference_free(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, extra_params={"reference_free": "yes"})
    budget = make_budget()

    with pytest.raises(InvalidInputError, match="reference_free"):
        adapter.prepare(inputs, budget)


# 6. prepare(): use_lora is explicitly rejected (full-parameter DPO only) ----


def test_prepare_rejects_use_lora(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs(tmp_path, extra_params={"use_lora": True})
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="use_lora"):
        adapter.prepare(inputs, budget)


# 7. prepare(): provenance -- reference model hash mismatch is refused --------


def test_prepare_rejects_reference_model_hash_mismatch(tmp_path):
    adapter = make_adapter(tmp_path)
    model_path, model_hash = make_local_model(tmp_path)
    dataset_path, dataset_hash = make_local_preference_dataset(tmp_path)
    reference_model_path, _real_ref_hash = make_local_reference_model(tmp_path)
    inputs = TrainingInputs.create(
        model_revision="trl-dpo-pilot-v1",
        model_hash=model_hash,
        dataset_version="synthetic-dpo-v1",
        dataset_hash=dataset_hash,
        seed=42,
        training_params={
            "model_path": model_path,
            "dataset_path": dataset_path,
            "reference_model_path": reference_model_path,
            "reference_model_hash": REFERENCE_HASH_PLACEHOLDER,  # deliberately wrong
            "max_steps": 4,
            "beta": 0.1,
            "reference_free": False,
        },
        dataset_licence="CC0-1.0",
        contamination_checked=True,
        run_id="run-reference-hash-mismatch",
    )
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="reference_model_hash mismatch"):
        adapter.prepare(inputs, budget)


def test_prepare_rejects_nonexistent_reference_model_path(tmp_path):
    adapter = make_adapter(tmp_path)
    model_path, model_hash = make_local_model(tmp_path)
    dataset_path, dataset_hash = make_local_preference_dataset(tmp_path)
    inputs = TrainingInputs.create(
        model_revision="trl-dpo-pilot-v1",
        model_hash=model_hash,
        dataset_version="synthetic-dpo-v1",
        dataset_hash=dataset_hash,
        seed=42,
        training_params={
            "model_path": model_path,
            "dataset_path": dataset_path,
            "reference_model_path": str(tmp_path / "does-not-exist-reference"),
            "reference_model_hash": REFERENCE_HASH_PLACEHOLDER,
            "max_steps": 4,
            "beta": 0.1,
            "reference_free": False,
        },
        dataset_licence="CC0-1.0",
        contamination_checked=True,
        run_id="run-missing-reference-file",
    )
    budget = make_budget()

    with pytest.raises(RejectedInputError, match="does not exist"):
        adapter.prepare(inputs, budget)


def test_contract_runner_reports_rejected_for_tampered_reference_model(tmp_path):
    """End-to-end through run_trainer_contract(): reference-model tamper is caught
    before train(), same discipline as the existing model_hash tamper test."""
    pytest.importorskip("trl")
    adapter = make_adapter(tmp_path)
    model_path, model_hash = make_local_model(tmp_path)
    dataset_path, dataset_hash = make_local_preference_dataset(tmp_path)
    reference_model_path, reference_model_hash = make_local_reference_model(tmp_path)
    inputs = make_inputs(
        tmp_path,
        model_path=model_path,
        model_hash=model_hash,
        dataset_path=dataset_path,
        dataset_hash=dataset_hash,
        reference_model_path=reference_model_path,
        reference_model_hash=reference_model_hash,
    )
    budget = make_budget()

    # Tamper with the reference model file AFTER inputs declared its (now stale) hash.
    with open(reference_model_path, "ab") as handle:
        handle.write(b"tampered-reference-bytes")

    output = run_trainer_contract(adapter, inputs, budget)

    assert output.status == TrainingStatus.REJECTED
    assert "reference_model_hash mismatch" in output.reason


# 8. prepare(): preference-pair dataset schema check --------------------------


def test_prepare_rejects_dataset_missing_prompt_chosen_rejected_columns(tmp_path):
    adapter = make_adapter(tmp_path)
    dataset_path, dataset_hash = make_local_preference_dataset(
        tmp_path, rows=[{"prompt": "only a prompt, no chosen/rejected"}]
    )
    inputs = make_inputs(tmp_path, dataset_path=dataset_path, dataset_hash=dataset_hash)
    budget = make_budget()

    with pytest.raises(InvalidInputError, match="prompt.*chosen.*rejected|chosen.*rejected"):
        adapter.prepare(inputs, budget)


def test_prepare_accepts_well_formed_preference_pair_dataset(tmp_path):
    adapter = make_adapter(tmp_path)
    dataset_path, dataset_hash = make_local_preference_dataset(
        tmp_path,
        rows=[
            {"prompt": "p1", "chosen": "c1", "rejected": "r1"},
            {"prompt": "p2", "chosen": "c2", "rejected": "r2"},
        ],
    )
    inputs = make_inputs(tmp_path, dataset_path=dataset_path, dataset_hash=dataset_hash)
    budget = make_budget()

    adapter.prepare(inputs, budget)  # must not raise


# 9. cleanup(): idempotent, safe with nothing to clean up ---------------------


def test_cleanup_is_idempotent_and_safe_with_nothing_to_clean(tmp_path):
    adapter = make_adapter(tmp_path)
    adapter.cleanup("run-never-started")
    adapter.cleanup("run-never-started")


def test_cleanup_removes_run_directory(tmp_path):
    adapter = make_adapter(tmp_path)
    run_dir = adapter._run_dir("run-to-clean")
    (run_dir / "marker.txt").write_text("evidence")
    assert run_dir.exists()

    adapter.cleanup("run-to-clean")

    assert not run_dir.exists()
    adapter.cleanup("run-to-clean")  # idempotent
