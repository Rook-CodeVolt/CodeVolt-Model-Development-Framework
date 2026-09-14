"""Contract-conformance tests for TrainerAdapterContract v1.1.

Every test in this module exercises the FAKE deterministic adapter
(``codevolt_mdf.fake_adapter.FakeTrainerAdapter``) against the contract
runner in ``codevolt_mdf.trainer_contract``, with two exceptions (see
section 9 below) that exercise TEST-ONLY non-conforming adapters
(``codevolt_mdf.testing_adapters``) to prove OS-level enforcement.
No real training engine is imported or invoked; nothing here trains a
real model.

The eight cases in sections 1-8 satisfy issue #7's evidence-plan step 2:
success, rejection, invalid-input, timeout, cancellation, resource-overrun,
checkpoint/resume, and tamper. Section 9 (two more cases) proves the v1.1
OS-level process-isolation enforcement (see
docs/decisions/0003-trainer-contract-os-level-enforcement.md): a real
SIGKILL of a non-cooperative adapter, and OS-measured usage overriding a
lying adapter's fake self-report.
"""

from __future__ import annotations

import threading
import time

import pytest

from codevolt_mdf import process_isolation
from codevolt_mdf.fake_adapter import FakeTrainerAdapter
from codevolt_mdf.testing_adapters import LyingAdapter, RunawayAdapter
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
    assert CONTRACT_VERSION == "1.1.0"


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
    # As of v1.1 the contract runner overrides the adapter's self-reported
    # wall/cpu/memory with OS-measured figures before this check (see
    # docs/decisions/0003-trainer-contract-os-level-enforcement.md); this
    # fake scenario does negligible real work, so max_memory_mb genuinely
    # is not exceeded and must NOT appear here -- only the two dimensions
    # that remain adapter self-reported (gpu_count, storage) still trigger
    # a real post-hoc violation.
    assert "max_memory_mb" not in violations
    assert "max_gpu_count" in violations
    assert "max_storage_mb" in violations


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


# 9. OS-level enforcement: non-conforming adapters ------------------------
#
# The eight cases above all exercise the well-behaved FakeTrainerAdapter.
# The two tests below use the TEST-ONLY non-conforming adapters in
# ``codevolt_mdf.testing_adapters`` to prove the v1.1 process-isolation
# enforcement is a real OS-level backstop, not just decoration around a
# cooperative adapter. See docs/decisions/0003-trainer-contract-os-level-
# enforcement.md.


def test_runaway_adapter_is_sigkilled_within_bound():
    """A non-cooperative adapter that never checks the cancellation token
    and never returns must still be terminated with a real SIGKILL within
    a bounded time -- proven directly against process_isolation, since
    the fake adapter never exercises this path."""
    budget = make_budget(max_wall_seconds=2.0, max_cpu_seconds=100.0)
    inputs = make_inputs("smoke", run_id="runaway-1")
    token = CancellationToken()

    start = time.monotonic()
    output, exc, measured = process_isolation.run_in_isolated_process(
        RunawayAdapter(), inputs, budget, None, token
    )
    elapsed = time.monotonic() - start

    assert elapsed < 10, f"runaway adapter was not killed promptly: {elapsed}s"
    assert measured.killed_for_timeout is True
    assert output is None
    assert exc is None


def test_lying_adapter_self_report_is_overridden_by_os_measurement():
    """An adapter that burns real CPU but self-reports a fake, tiny
    ResourceUsage must not have that lie trusted: the contract runner
    replaces wall/CPU/memory with OS-measured figures before checking the
    budget. Exercised at two levels: process_isolation directly (to see
    the raw measured vs. self-reported numbers) and through
    run_trainer_contract (to prove the override actually reaches the
    budget check, not just the raw measurement plumbing)."""
    inputs = make_inputs("smoke", run_id="lying-1")
    budget = make_budget(max_wall_seconds=5.0, max_cpu_seconds=100.0)
    lying = LyingAdapter(burn_seconds=0.3)

    output, exc, measured = process_isolation.run_in_isolated_process(
        lying, inputs, budget, None, CancellationToken()
    )

    assert exc is None
    assert output is not None
    # The adapter's own self-report is the deceptive tiny value.
    assert output.resource_usage.cpu_seconds < 0.01
    # The OS actually measured real work, distinct from the self-report.
    assert measured.cpu_seconds > 0.05 or measured.wall_seconds > 0.05

    # And run_trainer_contract must use the OS measurement, not the lie,
    # when it evaluates the run.
    lying2 = LyingAdapter(burn_seconds=0.3)
    contract_output = run_trainer_contract(lying2, inputs, budget)
    assert contract_output.status == TrainingStatus.ACCEPTED
    assert contract_output.resource_usage.cpu_seconds > 0.05 or (
        contract_output.resource_usage.wall_seconds > 0.05
    )
