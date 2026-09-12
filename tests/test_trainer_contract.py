"""Contract-conformance tests for TrainerAdapterContract v1.

Every test in this module exercises the FAKE deterministic adapter
(``codevolt_mdf.fake_adapter.FakeTrainerAdapter``) against the contract
runner in ``codevolt_mdf.trainer_contract``. No real training engine is
imported or invoked; nothing here trains a real model.

The eight cases below satisfy issue #7's evidence-plan step 2:
success, rejection, invalid-input, timeout, cancellation, resource-overrun,
checkpoint/resume, and tamper.
"""

from __future__ import annotations

import threading

import pytest

from codevolt_mdf.fake_adapter import FakeTrainerAdapter
from codevolt_mdf.trainer_contract import (
    CONTRACT_VERSION,
    CancellationToken,
    InvalidInputError,
    ResourceBudget,
    TrainingInputs,
    TrainingStatus,
    run_trainer_contract,
)

MODEL_HASH = "a" * 64
DATASET_HASH = "b" * 64


def make_inputs(scenario: str = "success", run_id: str | None = None, **overrides) -> TrainingInputs:
    defaults = {
        "model_revision": "fake-tiny-v1",
        "model_hash": MODEL_HASH,
        "dataset_version": "synthetic-v1",
        "dataset_hash": DATASET_HASH,
        "seed": 42,
        "training_params": {"scenario": scenario, "epochs": 1},
        "dataset_licence": "CC0-1.0",
        "contamination_checked": True,
        "run_id": run_id or f"run-{scenario}",
    }
    defaults.update(overrides)
    return TrainingInputs.create(**defaults)


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


def make_adapter(tmp_path) -> FakeTrainerAdapter:
    return FakeTrainerAdapter(work_dir=tmp_path / "adapter-work")


def test_contract_version_is_declared():
    assert CONTRACT_VERSION == "1.0.0"


# 1. success -----------------------------------------------------------


def test_success_case_produces_accepted_output_with_evidence(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs("success")
    budget = make_budget()

    output = run_trainer_contract(adapter, inputs, budget)

    assert output.status == TrainingStatus.ACCEPTED
    assert output.artifact_id == f"artifact-{inputs.run_id}"
    assert output.evidence_locator is not None
    assert output.evidence_hash is not None
    assert output.resource_usage is not None
    assert not output.resource_usage.exceeds(budget)


# 2. rejection -----------------------------------------------------------


def test_rejection_case_unsupported_model_revision(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs("success", model_revision="not-a-real-revision")
    budget = make_budget()

    output = run_trainer_contract(adapter, inputs, budget)

    assert output.status == TrainingStatus.REJECTED
    assert "not-a-real-revision" in output.reason


# 3. invalid input -----------------------------------------------------------


def test_invalid_input_case_bad_hash_never_reaches_adapter(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs("success", model_hash="not-a-sha256")
    budget = make_budget()

    output = run_trainer_contract(adapter, inputs, budget)

    assert output.status == TrainingStatus.INVALID
    assert output.error_class == InvalidInputError.__name__


def test_training_inputs_validate_rejects_uncontaminated_dataset():
    inputs = make_inputs("success", contamination_checked=False)
    with pytest.raises(InvalidInputError):
        inputs.validate()


# 4. timeout -----------------------------------------------------------


def test_timeout_case_is_interrupted_by_wall_clock_deadline(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs("timeout")
    # Deliberately tiny deadline; the fake adapter's timeout scenario loops
    # far longer than this, so the contract runner must intervene.
    budget = make_budget(max_wall_seconds=0.2, max_cpu_seconds=0.2)

    output = run_trainer_contract(adapter, inputs, budget)

    assert output.status == TrainingStatus.INTERRUPTED
    assert "max_wall_seconds" in output.reason


# 5. cancellation -----------------------------------------------------------


def test_cancellation_case_stops_promptly_when_token_is_cancelled(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs("cancel")
    budget = make_budget(max_wall_seconds=30.0, max_cpu_seconds=30.0)
    token = CancellationToken()

    # Cancel shortly after the run starts, well inside the wall-clock
    # deadline, to prove cancellation is distinct from a timeout.
    timer = threading.Timer(0.1, token.cancel, kwargs={"reason": "user requested stop"})
    timer.start()
    try:
        output = run_trainer_contract(adapter, inputs, budget, cancel_token=token)
    finally:
        timer.cancel()

    assert output.status == TrainingStatus.INTERRUPTED
    assert "user requested stop" in output.reason


# 6. resource overrun -----------------------------------------------------------


def test_resource_overrun_case_is_caught_post_hoc(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs("resource_overrun")
    budget = make_budget(max_memory_mb=64.0, max_storage_mb=1.0, max_gpu_count=0)

    output = run_trainer_contract(adapter, inputs, budget)

    assert output.status == TrainingStatus.INTERRUPTED
    assert "resource budget exceeded" in output.reason
    assert output.resource_usage is not None
    violations = output.resource_usage.exceeds(budget)
    assert "max_memory_mb" in violations
    assert "max_gpu_count" in violations


# 7. checkpoint / resume -----------------------------------------------------------


def test_checkpoint_resume_case_completes_after_safe_halt(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs("checkpoint_resume")
    budget = make_budget()

    first = run_trainer_contract(adapter, inputs, budget)
    assert first.status == TrainingStatus.INTERRUPTED
    assert first.checkpoint is not None
    assert first.checkpoint.step == 1

    second = run_trainer_contract(adapter, inputs, budget, resume_from=first.checkpoint)
    assert second.status == TrainingStatus.ACCEPTED
    assert "resumed from step 1" in second.reason
    assert second.evidence_locator is not None


def test_checkpoint_resume_case_rejects_tampered_checkpoint_state(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs("checkpoint_resume")
    budget = make_budget()

    first = run_trainer_contract(adapter, inputs, budget)
    assert first.checkpoint is not None

    # Corrupt the checkpoint file on disk before resuming.
    from pathlib import Path

    Path(first.checkpoint.state_locator).write_text('{"step": 999, "seed": 0}')

    second = run_trainer_contract(adapter, inputs, budget, resume_from=first.checkpoint)
    assert second.status == TrainingStatus.INVALID


# 8. tamper -----------------------------------------------------------


def test_tamper_case_evidence_hash_mismatch_is_detected(tmp_path):
    adapter = make_adapter(tmp_path)
    inputs = make_inputs("tamper")
    budget = make_budget()

    output = run_trainer_contract(adapter, inputs, budget)

    assert output.status == TrainingStatus.INVALID
    assert "tamper" in output.reason.lower()
