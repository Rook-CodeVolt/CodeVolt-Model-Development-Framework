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
- ``SubprocessSpawningAdapter``: spawns a real, genuinely long-running
  grandchild OS process (``subprocess.Popen``) and never waits on it,
  then hangs. Proves ``_kill_group``/``os.setsid`` process-GROUP
  termination reaches adapter-spawned subprocesses, not just the direct
  child the contract runner tracks.
- ``SetsidEscapingAdapter``: spawns a grandchild that itself calls
  ``os.setsid()`` before hanging, leaving the isolated child's process
  group entirely. Proves the ``_kill_pid_tree`` ppid-lineage walk added
  in ``docs/decisions/0004-pid-tree-walk-setsid-escape-fix.md`` reaches
  a descendant that has escaped process-group-based termination, which
  ``SubprocessSpawningAdapter`` alone does not exercise.
- ``MaliciousExceptionAdapter``: raises an exception whose class defines
  a malicious ``__reduce__`` that would run ``os.system(...)`` the
  instant it is unpickled. Proves the IPC sanitisation in
  ``process_isolation`` (exceptions are reduced to plain strings before
  crossing the ``multiprocessing.Queue``, never pickled/unpickled as the
  original object) actually prevents that code from running in the
  parent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .evaluator_contract import ExampleResult, HeldOutExample
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
class RunawayEvaluatorAdapter:
    """TEST-ONLY evaluator that never returns, for containment tests."""

    name: str = "test-only-runaway-evaluator"
    contract_version: str = "1.0.0"

    def score_example(
        self, artifact_id: str, artifact_locator: str, example: HeldOutExample
    ) -> ExampleResult:
        deadline = time.monotonic() + 300
        total = 0
        while time.monotonic() < deadline:
            total += 1
        return ExampleResult(example.example_id, False, 0.0, str(total))


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


@dataclass
class SubprocessSpawningAdapter:
    """TEST-ONLY: spawns a real, long-running grandchild process and hangs.

    Never waits on the subprocess and never checks ``cancel_token``, so
    the only way the contract runner can stop this adapter is by killing
    its OS process. The point of this adapter is what happens to the
    grandchild it spawned: it writes the grandchild's pid to
    ``pid_file`` (in the shared filesystem, so the parent test process
    can read it back) before hanging, so the test can verify the
    grandchild -- not just the tracked direct child -- is also dead
    after the contract runner kills the adapter.
    """

    pid_file: str = ""
    name: str = "test-only-subprocess-spawning"
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
        import subprocess

        # A genuinely long-running grandchild: not waited on, not reaped.
        grandchild = subprocess.Popen(["sleep", "300"])
        if self.pid_file:
            with open(self.pid_file, "w") as f:
                f.write(str(grandchild.pid))
        # Deliberately does NOT check cancel_token and does NOT wait() on
        # the grandchild: only an OS-level kill of the whole process
        # group can end this run.
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            time.sleep(0.05)
        return TrainingOutput(status=TrainingStatus.ACCEPTED, reason="unreachable")

    def cleanup(self, run_id: str) -> None:
        return None


@dataclass
class SetsidEscapingAdapter:
    """TEST-ONLY: spawns a grandchild that calls ``os.setsid()`` to escape the group.

    Unlike ``SubprocessSpawningAdapter`` (whose ``sleep 300`` grandchild
    stays in the isolated child's process group and dies with a plain
    ``os.killpg``), this adapter's grandchild is a short Python one-liner
    that calls ``os.setsid()`` on startup -- becoming the leader of a
    brand-new OS process group and session -- before it busy-loops. That
    is precisely the disclosed residual gap in
    ``docs/decisions/0003-trainer-contract-os-level-enforcement.md``:
    ``os.setsid()`` changes process-group/session membership but leaves
    ``ppid`` lineage untouched, so a pure ``os.killpg`` sweep no longer
    reaches it once it has escaped, while a ``ppid``-lineage walk
    (``_kill_pid_tree``) still does. Writes the grandchild's pid to
    ``pid_file`` before detaching so the test can verify it via the real
    OS, not an internal accounting flag.
    """

    pid_file: str = ""
    name: str = "test-only-setsid-escaping"
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
        import subprocess
        import sys

        # A grandchild that detaches into its own session/process group
        # (os.setsid()) before busy-looping, so it is no longer a member
        # of the isolated child's group by the time any kill signal
        # arrives.
        script = (
            "import os, time\n"
            "os.setsid()\n"
            "deadline = time.monotonic() + 300\n"
            "while time.monotonic() < deadline:\n"
            "    time.sleep(0.05)\n"
        )
        grandchild = subprocess.Popen([sys.executable, "-c", script])
        if self.pid_file:
            with open(self.pid_file, "w") as f:
                f.write(str(grandchild.pid))
        # Deliberately does NOT check cancel_token and does NOT wait() on
        # the grandchild: only an OS-level kill that reaches beyond the
        # process group (a ppid-lineage walk) can end this run.
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            time.sleep(0.05)
        return TrainingOutput(status=TrainingStatus.ACCEPTED, reason="unreachable")

    def cleanup(self, run_id: str) -> None:
        return None


@dataclass
class FailingAdapter:
    """TEST-ONLY: a well-behaved adapter whose train() raises a real error.

    Unlike ``MaliciousExceptionAdapter``, this raises a plain, ordinary
    ``ValueError`` -- no crafted ``__reduce__``, nothing adversarial. It
    proves the IPC sanitisation added to close the pickle exploit (see
    ``_MaliciousReduceException`` above and
    ``docs/decisions/0003-trainer-contract-os-level-enforcement.md``) does
    not also mangle or swallow a genuine, legitimate adapter failure: the
    exact exception type name and message must still cross the child ->
    parent boundary intact.
    """

    name: str = "test-only-failing"
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
        raise ValueError("something went wrong")

    def cleanup(self, run_id: str) -> None:
        return None


class _MaliciousReduceException(Exception):
    """TEST-ONLY: an exception whose ``__reduce__`` runs code on unpickle.

    If this object (rather than a plain-string summary of it) were ever
    pickled by the child and unpickled by the (trusted) parent, unpickling
    it would execute ``os.system(...)`` in the parent's context. Used to
    prove the IPC sanitisation in ``process_isolation`` (exceptions are
    reduced to plain strings before crossing the ``multiprocessing.Queue``)
    actually prevents that -- not merely happens to avoid it.
    """

    def __init__(self, message: str, marker_path: str) -> None:
        super().__init__(message)
        self.marker_path = marker_path

    def __reduce__(self):
        import os

        # If unpickled, this touches marker_path via a real shell command.
        return (os.system, (f"touch {self.marker_path}",))


@dataclass
class MaliciousExceptionAdapter:
    """TEST-ONLY: raises a crafted exception with a malicious ``__reduce__``.

    Proves that unpickling the raw child exception object in the parent
    (the pre-fix behaviour) is closed off: the contract runner must never
    let the child's raw exception cross the ``multiprocessing.Queue``
    boundary, since unpickling an attacker-controlled object executes
    arbitrary code in the (trusted) parent process the instant
    ``Queue.get`` deserialises it.
    """

    marker_path: str = ""
    name: str = "test-only-malicious-exception"
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
        raise _MaliciousReduceException("crafted exception with malicious __reduce__", self.marker_path)

    def cleanup(self, run_id: str) -> None:
        return None
