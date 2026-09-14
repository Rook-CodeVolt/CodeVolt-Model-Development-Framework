"""TEST-ONLY non-conforming adapters for trainer-contract enforcement tests.

Neither adapter here is a real training engine and neither is exercised
by any production code path (`core.py`'s `run_experiment` flow, the CLI,
or `fake_adapter.py`'s deterministic scenarios). They exist solely so
`tests/test_trainer_contract.py` can prove the contract runner's
OS-level enforcement (``codevolt_mdf.process_isolation``) actually works
against an adapter that does *not* cooperate, rather than only ever
testing the well-behaved ``FakeTrainerAdapter``.

- ``RunawayAdapter``: never checks the cancellation token and never
  returns on its own. Proves the contract runner can SIGKILL a truly
  non-conforming adapter's process within a bounded time.
- ``LyingAdapter``: does real, measurable CPU work but self-reports a
  fake, deceptively low ``ResourceUsage``. Proves the contract runner's
  budget check uses OS-measured usage, not the adapter's self-report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .trainer_contract import (
    CancellationToken,
    CheckpointHandle,
    ResourceBudget,
    ResourceUsage,
    TrainingInputs,
    TrainingOutput,
    TrainingStatus,
    UpstreamRequirement,
)

_TEST_ONLY_UPSTREAM = UpstreamRequirement(
    engine_name="test-only-non-conforming",
    min_version="1.0.0",
    max_version="1.99.99",
    installed_version="1.0.0",
)


@dataclass
class RunawayAdapter:
    """TEST-ONLY: ignores cancellation and spins forever, burning real CPU."""

    name: str = "test-only-runaway"
    contract_version: str = "1.1.0"
    upstream: UpstreamRequirement = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.upstream is None:
            self.upstream = _TEST_ONLY_UPSTREAM

    def prepare(self, inputs: TrainingInputs, budget: ResourceBudget) -> None:
        return None

    def train(
        self,
        inputs: TrainingInputs,
        budget: ResourceBudget,
        cancel_token: CancellationToken,
        resume_from: CheckpointHandle | None = None,
    ) -> TrainingOutput:
        # Deliberately does NOT check cancel_token. A conforming adapter
        # must poll it; this one is the "non-conforming" test double that
        # proves the OS-level kill is a real backstop, not decoration.
        deadline = time.monotonic() + 300  # long enough to always be killed first
        total = 0
        while time.monotonic() < deadline:
            total += 1  # real CPU work, not a sleep
        return TrainingOutput(status=TrainingStatus.ACCEPTED, reason="unreachable")

    def cleanup(self, run_id: str) -> None:
        return None


@dataclass
class LyingAdapter:
    """TEST-ONLY: burns real CPU but self-reports fake, deceptively low usage."""

    burn_seconds: float = 0.5
    name: str = "test-only-lying"
    contract_version: str = "1.1.0"
    upstream: UpstreamRequirement = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.upstream is None:
            self.upstream = _TEST_ONLY_UPSTREAM

    def prepare(self, inputs: TrainingInputs, budget: ResourceBudget) -> None:
        return None

    def train(
        self,
        inputs: TrainingInputs,
        budget: ResourceBudget,
        cancel_token: CancellationToken,
        resume_from: CheckpointHandle | None = None,
    ) -> TrainingOutput:
        # Real, measurable CPU-bound work -- not reflected in the lie below.
        deadline = time.monotonic() + self.burn_seconds
        total = 0
        while time.monotonic() < deadline:
            total += 1
        # Self-report a fake, deceptively tiny usage. If the contract
        # runner trusted this, the run would be wrongly ACCEPTED.
        lying_usage = ResourceUsage(
            wall_seconds=0.001,
            cpu_seconds=0.001,
            memory_mb_peak=1.0,
            gpu_count_used=0,
            storage_mb_used=0.0,
        )
        return TrainingOutput(
            status=TrainingStatus.ACCEPTED,
            reason=f"lying adapter burned ~{self.burn_seconds}s of real CPU (total={total})",
            artifact_id=f"artifact-{inputs.run_id}",
            resource_usage=lying_usage,
        )

    def cleanup(self, run_id: str) -> None:
        return None
