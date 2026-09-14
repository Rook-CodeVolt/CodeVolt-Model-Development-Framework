"""Deterministic FAKE trainer adapter used to exercise TrainerAdapterContract.

This adapter never imports or calls a real training engine. It is
deterministic given ``inputs.seed`` and a ``scenario`` training parameter,
and exists solely so the contract in ``trainer_contract.py`` has something
concrete and safe to test against. It grants no training, data-admission,
deployment, or publication authority.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .trainer_contract import (
    CancellationToken,
    CheckpointHandle,
    InvalidInputError,
    RejectedInputError,
    ResourceBudget,
    ResourceUsage,
    TrainingInputs,
    TrainingOutput,
    TrainingStatus,
    UpstreamRequirement,
)

SUPPORTED_MODEL_REVISIONS = ("fake-tiny-v1", "fake-tiny-v2")


@dataclass
class FakeTrainerAdapter:
    """Deterministic, in-process, stdlib-only trainer adapter.

    ``work_dir`` is where the fake adapter writes evidence/checkpoint
    files; callers typically point it at a pytest ``tmp_path``.
    """

    work_dir: Path
    name: str = "fake-deterministic-v1"
    contract_version: str = "1.0.0"
    upstream: UpstreamRequirement = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if self.upstream is None:
            self.upstream = UpstreamRequirement(
                engine_name="fake-engine",
                min_version="1.0.0",
                max_version="1.99.99",
                installed_version="1.0.0",
            )

    # -- contract methods ---------------------------------------------

    def prepare(self, inputs: TrainingInputs, budget: ResourceBudget) -> None:
        params = inputs.params
        scenario = params.get("scenario", "success")
        if scenario not in _SCENARIOS:
            raise InvalidInputError(f"unknown scenario {scenario!r}")
        if inputs.model_revision not in SUPPORTED_MODEL_REVISIONS:
            raise RejectedInputError(
                f"model revision {inputs.model_revision!r} is not in the fake adapter's "
                f"supported set {SUPPORTED_MODEL_REVISIONS}"
            )
        if budget.network_policy != "offline":
            raise RejectedInputError("fake adapter only accepts an offline network policy")

    def train(
        self,
        inputs: TrainingInputs,
        budget: ResourceBudget,
        cancel_token: CancellationToken,
        resume_from: CheckpointHandle | None = None,
    ) -> TrainingOutput:
        scenario = inputs.params.get("scenario", "success")
        handler = _SCENARIOS[scenario]
        return handler(self, inputs, budget, cancel_token, resume_from)

    def cleanup(self, run_id: str) -> None:
        run_dir = self.work_dir / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

    # -- helpers ---------------------------------------------------------

    def _run_dir(self, run_id: str) -> Path:
        run_dir = self.work_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _write_evidence(self, inputs: TrainingInputs, step: int) -> tuple[str, str]:
        """Write a deterministic evidence file; return (locator, sha256)."""
        run_dir = self._run_dir(inputs.run_id)
        evidence_path = run_dir / "evidence.json"
        payload = {
            "run_id": inputs.run_id,
            "model_revision": inputs.model_revision,
            "dataset_version": inputs.dataset_version,
            "seed": inputs.seed,
            "step": step,
            "params": inputs.params,
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        evidence_path.write_bytes(raw)
        return str(evidence_path), hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------
# Scenario handlers -- each is a pure function of (adapter, inputs, budget,
# cancel_token, resume_from) so behaviour is deterministic and inspectable.
# --------------------------------------------------------------------------


def _scenario_success(adapter, inputs, budget, cancel_token, resume_from):
    locator, digest = adapter._write_evidence(inputs, step=1)
    usage = ResourceUsage(
        wall_seconds=0.01,
        cpu_seconds=0.01,
        memory_mb_peak=16.0,
        gpu_count_used=0,
        storage_mb_used=0.01,
        network_calls=0,
    )
    return TrainingOutput(
        status=TrainingStatus.ACCEPTED,
        reason="fake training completed deterministically",
        artifact_id=f"artifact-{inputs.run_id}",
        evidence_locator=locator,
        evidence_hash=digest,
        resource_usage=usage,
    )


def _scenario_timeout(adapter, inputs, budget, cancel_token, resume_from):
    # Deliberately loop far longer than any reasonable budget, checking the
    # cancellation token cooperatively as the contract requires. The
    # contract runner's own timeout will fire first in the timeout test
    # (budget.max_wall_seconds is set very small); this handler still
    # exits promptly once cancelled so it never leaks a hung thread.
    steps = 0
    while steps < 100000:
        if cancel_token.wait(0.01):
            return TrainingOutput(
                status=TrainingStatus.INTERRUPTED,
                reason=f"cancelled after {steps} steps: {cancel_token.reason}",
            )
        steps += 1
    return TrainingOutput(status=TrainingStatus.INTERRUPTED, reason="exhausted step budget")


def _scenario_cancel(adapter, inputs, budget, cancel_token, resume_from):
    # Identical loop to the timeout scenario, but exercised by tests that
    # cancel the token directly (user-initiated cancellation) rather than
    # letting the wall-clock deadline expire.
    return _scenario_timeout(adapter, inputs, budget, cancel_token, resume_from)


def _scenario_resource_overrun(adapter, inputs, budget, cancel_token, resume_from):
    locator, digest = adapter._write_evidence(inputs, step=1)
    # As of v1.1, wall/CPU/memory in this self-report are discarded by the
    # contract runner and replaced with OS-measured figures (see
    # docs/decisions/0003-trainer-contract-os-level-enforcement.md), so a
    # self-reported wall/CPU/memory lie no longer fools the budget check --
    # this scenario does negligible real work, so the OS-measured wall/CPU/
    # memory legitimately stay under budget despite the fake numbers below.
    # gpu_count_used and storage_mb_used remain adapter self-reported even
    # under v1.1 (see process_isolation's module docstring, "What this
    # does NOT enforce"), so this scenario still exercises a genuine
    # post-hoc multi-dimension violation via those two dimensions.
    usage = ResourceUsage(
        wall_seconds=budget.max_wall_seconds * 100,
        cpu_seconds=budget.max_cpu_seconds * 100,
        memory_mb_peak=budget.max_memory_mb * 100,
        gpu_count_used=budget.max_gpu_count + 1,
        storage_mb_used=budget.max_storage_mb * 100,
    )
    return TrainingOutput(
        status=TrainingStatus.ACCEPTED,
        reason="fake training completed but used far more than its budget",
        artifact_id=f"artifact-{inputs.run_id}",
        evidence_locator=locator,
        evidence_hash=digest,
        resource_usage=usage,
    )


def _scenario_checkpoint_resume(adapter, inputs, budget, cancel_token, resume_from):
    total_steps = 4
    start_step = resume_from.step if resume_from is not None else 0
    if resume_from is None:
        # First call: do one step of deterministic "work", then safe-halt
        # and checkpoint rather than running to completion. This models a
        # planned safe-stop (e.g. preemption) rather than a crash.
        step = start_step + 1
        run_dir = adapter._run_dir(inputs.run_id)
        state_path = run_dir / "checkpoint.json"
        state = {"step": step, "seed": inputs.seed}
        raw = json.dumps(state, sort_keys=True).encode("utf-8")
        state_path.write_bytes(raw)
        checkpoint = CheckpointHandle(
            checkpoint_id=f"ckpt-{inputs.run_id}-{step}",
            step=step,
            state_locator=str(state_path),
            state_hash=hashlib.sha256(raw).hexdigest(),
        )
        return TrainingOutput(
            status=TrainingStatus.INTERRUPTED,
            reason=f"safe-halted after step {step}/{total_steps}; checkpoint saved",
            checkpoint=checkpoint,
        )

    # Resumed call: verify the checkpoint's integrity, then run to completion.
    state_path = Path(resume_from.state_locator)
    raw = state_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != resume_from.state_hash:
        raise InvalidInputError("checkpoint state hash mismatch on resume")
    state = json.loads(raw)
    if state["step"] != resume_from.step:
        raise InvalidInputError("checkpoint state does not match its own handle")

    locator, digest = adapter._write_evidence(inputs, step=total_steps)
    usage = ResourceUsage(
        wall_seconds=0.01,
        cpu_seconds=0.01,
        memory_mb_peak=16.0,
        gpu_count_used=0,
        storage_mb_used=0.01,
    )
    return TrainingOutput(
        status=TrainingStatus.ACCEPTED,
        reason=f"resumed from step {resume_from.step} and completed step {total_steps}",
        artifact_id=f"artifact-{inputs.run_id}",
        evidence_locator=locator,
        evidence_hash=digest,
        resource_usage=usage,
    )


def _scenario_tamper(adapter, inputs, budget, cancel_token, resume_from):
    locator, digest = adapter._write_evidence(inputs, step=1)
    # Simulate post-write corruption/tampering: mutate the evidence file
    # on disk *after* computing its hash, then (honestly) report the
    # pre-tamper hash as the declared evidence hash, exactly as a trainer
    # would if it hashed evidence at write time. The contract runner
    # recomputes the hash from the file it finds and must detect the
    # mismatch rather than trust the adapter's claim.
    path = Path(locator)
    path.write_bytes(path.read_bytes() + b"\ntampered")
    usage = ResourceUsage(
        wall_seconds=0.01,
        cpu_seconds=0.01,
        memory_mb_peak=16.0,
        gpu_count_used=0,
        storage_mb_used=0.01,
    )
    return TrainingOutput(
        status=TrainingStatus.ACCEPTED,
        reason="fake training completed (evidence tampered with after write for the test)",
        artifact_id=f"artifact-{inputs.run_id}",
        evidence_locator=locator,
        evidence_hash=digest,
        resource_usage=usage,
    )


_SCENARIOS = {
    "success": _scenario_success,
    "timeout": _scenario_timeout,
    "cancel": _scenario_cancel,
    "resource_overrun": _scenario_resource_overrun,
    "checkpoint_resume": _scenario_checkpoint_resume,
    "tamper": _scenario_tamper,
}
