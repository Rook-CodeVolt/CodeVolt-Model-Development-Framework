"""TrainerAdapterContract v1: the first real trainer-adapter boundary.

This module defines an executable, versioned contract that a trainer
adapter must satisfy before it is admitted into CodeVolt MDF. It replaces
the informal two-method ``TrainerAdapter``/``EvaluatorAdapter`` Protocol in
``core.py`` as the boundary a *real* training engine integration must cross.

Design constraints (see docs/TRAINER_ADAPTER_CONTRACT.md for the full
rationale):

- Dependency-light: standard library only (``multiprocessing``,
  ``resource``, ``subprocess`` -- all stdlib). As of v1.1, the contract
  runner executes ``adapter.train(...)`` in a real child OS process
  (see ``codevolt_mdf.process_isolation``), so timeout/cancellation is
  enforced with ``SIGKILL`` rather than a cooperative-only request, and
  resource usage is measured from the OS, not self-reported by the
  adapter. See ``docs/TRAINER_ADAPTER_CONTRACT.md`` ("OS-level
  enforcement") for exactly what is and is not covered.
- The contract governs a *trainer*. It never evaluates, scores, or
  promotes its own output (see ``docs/ARCHITECTURE.md`` core contract #2).
- Nothing in this module trains a real model or grants training,
  data-admission, deployment, or publication authority. It defines a
  boundary and ships a deterministic FAKE adapter to exercise it.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

CONTRACT_VERSION = "1.1.0"
"""Semantic version of TrainerAdapterContract. Bump on any breaking change
to the shapes or semantics below; keep old versions documented so contract
changes stay diffable (see docs/decisions/0002-trainer-adapter-contract-v1.md
and docs/decisions/0003-trainer-contract-os-level-enforcement.md).
"""

SUPPORTED_CONTRACT_VERSIONS: tuple[str, ...] = ("1.0.0", CONTRACT_VERSION)
"""v1.1 only changes runner enforcement (real process isolation), not the
``TrainerAdapterV1`` Protocol's method shapes, so an adapter written
against 1.0.0 remains admissible unchanged; it now simply runs inside a
real child process instead of a cooperative thread."""


# --------------------------------------------------------------------------
# Error classes
# --------------------------------------------------------------------------


class TrainerContractError(Exception):
    """Base class for every error this contract defines."""


class InvalidInputError(TrainerContractError):
    """Inputs are malformed, incomplete, or fail static validation.

    Maps to run status ``invalid``: the run never started meaningful work.
    """


class RejectedInputError(TrainerContractError):
    """Inputs are well-formed but the adapter declines to run them.

    E.g. unsupported model revision, incompatible upstream engine version,
    disallowed network/filesystem policy. Maps to run status ``rejected``.
    """


class ResourceBudgetExceededError(TrainerContractError):
    """Measured resource usage exceeded the declared budget.

    Maps to run status ``interrupted``; the run is halted and rolled back.
    """


class TrainerTimeoutError(TrainerContractError):
    """The adapter did not return within ``budget.max_wall_seconds``.

    Maps to run status ``interrupted``.
    """


class TrainerCancelledError(TrainerContractError):
    """The run was cancelled cooperatively via a ``CancellationToken``.

    Maps to run status ``interrupted``.
    """


class TamperDetectedError(TrainerContractError):
    """Evidence recovered after the run does not match its declared hash.

    Maps to run status ``invalid``: the evidence cannot be trusted.
    """


class CheckpointError(TrainerContractError):
    """A checkpoint could not be created, read, or resumed from safely."""


# --------------------------------------------------------------------------
# Status and typed outputs
# --------------------------------------------------------------------------


class TrainingStatus(str, Enum):
    """Outcome of a single trainer-adapter run.

    Distinct from ``core.Decision.status`` (an *evaluation acceptance*
    decision). This status describes whether the *trainer run itself*
    produced trustworthy evidence, not whether the artifact is good.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INVALID = "invalid"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class UpstreamRequirement:
    """Declares the upstream engine/dependency compatibility bounds.

    A FAKE adapter has no real upstream dependency; it declares itself as
    its own upstream with a fixed version so the contract's compatibility
    check has something concrete to exercise. A real integration MUST
    declare the actual engine name and an inclusive ``[min_version,
    max_version]`` bound and implement ``installed_version`` for real.
    """

    engine_name: str
    min_version: str
    max_version: str
    installed_version: str

    def is_compatible(self) -> bool:
        return _version_tuple(self.min_version) <= _version_tuple(self.installed_version) <= _version_tuple(
            self.max_version
        )


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for chunk in version.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


@dataclass(frozen=True)
class ResourceBudget:
    """Immutable resource ceiling declared before a run starts."""

    max_wall_seconds: float
    max_cpu_seconds: float
    max_memory_mb: float
    max_gpu_count: int
    max_storage_mb: float
    network_policy: str = "offline"  # "offline" | "allow-list"
    allowed_hosts: tuple[str, ...] = ()
    filesystem_root: str | None = None  # adapter may only read/write under this root

    def __post_init__(self) -> None:
        if self.network_policy not in ("offline", "allow-list"):
            raise InvalidInputError(
                f"network_policy must be 'offline' or 'allow-list', got {self.network_policy!r}"
            )
        if self.network_policy == "allow-list" and not self.allowed_hosts:
            raise InvalidInputError("allow-list network policy requires at least one allowed host")
        for field_name in ("max_wall_seconds", "max_cpu_seconds", "max_memory_mb", "max_storage_mb"):
            if getattr(self, field_name) <= 0:
                raise InvalidInputError(f"{field_name} must be positive")
        if self.max_gpu_count < 0:
            raise InvalidInputError("max_gpu_count cannot be negative")


@dataclass(frozen=True)
class ResourceUsage:
    """Measured resource consumption reported after a run."""

    wall_seconds: float
    cpu_seconds: float
    memory_mb_peak: float
    gpu_count_used: int
    storage_mb_used: float
    network_calls: int = 0

    def exceeds(self, budget: ResourceBudget) -> tuple[str, ...]:
        """Return the names of every budget dimension this usage violates."""
        violations = []
        if self.wall_seconds > budget.max_wall_seconds:
            violations.append("max_wall_seconds")
        if self.cpu_seconds > budget.max_cpu_seconds:
            violations.append("max_cpu_seconds")
        if self.memory_mb_peak > budget.max_memory_mb:
            violations.append("max_memory_mb")
        if self.gpu_count_used > budget.max_gpu_count:
            violations.append("max_gpu_count")
        if self.storage_mb_used > budget.max_storage_mb:
            violations.append("max_storage_mb")
        return tuple(violations)


@dataclass(frozen=True)
class TrainingInputs:
    """Exact immutable inputs to a trainer-adapter run.

    ``training_params`` is accepted as any mapping but stored as a sorted
    tuple of items so the dataclass stays hashable/frozen and the params
    cannot be mutated after construction; use ``.params`` to read them back
    as a plain dict.
    """

    model_revision: str
    model_hash: str
    dataset_version: str
    dataset_hash: str
    seed: int
    training_params: tuple[tuple[str, Any], ...]
    dataset_licence: str
    contamination_checked: bool
    run_id: str

    @classmethod
    def create(
        cls,
        *,
        model_revision: str,
        model_hash: str,
        dataset_version: str,
        dataset_hash: str,
        seed: int,
        training_params: Mapping[str, Any],
        dataset_licence: str,
        contamination_checked: bool,
        run_id: str,
    ) -> TrainingInputs:
        return cls(
            model_revision=model_revision,
            model_hash=model_hash,
            dataset_version=dataset_version,
            dataset_hash=dataset_hash,
            seed=seed,
            training_params=tuple(sorted(training_params.items())),
            dataset_licence=dataset_licence,
            contamination_checked=contamination_checked,
            run_id=run_id,
        )

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.training_params)

    def validate(self) -> None:
        for name, value in (
            ("model_revision", self.model_revision),
            ("model_hash", self.model_hash),
            ("dataset_version", self.dataset_version),
            ("dataset_hash", self.dataset_hash),
            ("dataset_licence", self.dataset_licence),
            ("run_id", self.run_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise InvalidInputError(f"{name} is required and must be a non-empty string")
        if not _looks_like_sha256(self.model_hash):
            raise InvalidInputError("model_hash must be a 64-character hex sha256 digest")
        if not _looks_like_sha256(self.dataset_hash):
            raise InvalidInputError("dataset_hash must be a 64-character hex sha256 digest")
        if not isinstance(self.seed, int):
            raise InvalidInputError("seed must be an int")
        if not self.contamination_checked:
            raise InvalidInputError(
                "dataset must have a recorded contamination check before training"
            )


def _looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


@dataclass(frozen=True)
class CheckpointHandle:
    """Identity of a resumable checkpoint."""

    checkpoint_id: str
    step: int
    state_locator: str
    state_hash: str


@dataclass(frozen=True)
class TrainingOutput:
    """Typed output of a trainer-adapter run."""

    status: TrainingStatus
    reason: str
    artifact_id: str | None = None
    checkpoint: CheckpointHandle | None = None
    evidence_locator: str | None = None
    evidence_hash: str | None = None
    resource_usage: ResourceUsage | None = None
    error_class: str | None = None


# --------------------------------------------------------------------------
# Cancellation
# --------------------------------------------------------------------------


class CancellationToken:
    """Cooperative cancellation signal, stdlib-only (``threading.Event``).

    Adapters are still written against this simple API for portability
    and to keep ``TrainerAdapterV1`` unchanged. As of v1.1 the contract
    runner additionally enforces cancellation at the OS-process level:
    if the adapter does not honour this token within
    ``process_isolation.DEFAULT_KILL_GRACE_SECONDS`` of it being set, the
    runner escalates to ``SIGKILL`` on the child process. Cooperative
    checking (``is_cancelled()``/``wait(interval)``) is still the polite,
    low-latency path; it is no longer the *only* backstop. See
    docs/TRAINER_ADAPTER_CONTRACT.md, "OS-level enforcement".
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = "cancelled"

    def cancel(self, reason: str = "cancelled") -> None:
        self._reason = reason
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float) -> bool:
        """Sleep up to ``timeout`` seconds, returning early if cancelled."""
        return self._event.wait(timeout)

    @property
    def reason(self) -> str:
        return self._reason


# --------------------------------------------------------------------------
# Adapter protocol
# --------------------------------------------------------------------------


class TrainerAdapterV1(Protocol):
    """Contract a trainer adapter must satisfy to be admitted (v1)."""

    name: str
    contract_version: str
    upstream: UpstreamRequirement

    def prepare(self, inputs: TrainingInputs, budget: ResourceBudget) -> None:
        """Validate inputs/budget/policy before any work starts.

        Must raise ``InvalidInputError`` for malformed inputs and
        ``RejectedInputError`` for well-formed inputs this adapter declines
        (unsupported revision, incompatible upstream version, disallowed
        network policy, filesystem boundary violation, etc). Must not
        perform training work.
        """
        ...

    def train(
        self,
        inputs: TrainingInputs,
        budget: ResourceBudget,
        cancel_token: CancellationToken,
        resume_from: CheckpointHandle | None = None,
    ) -> TrainingOutput:
        """Execute (or resume) a bounded training run and return evidence."""
        ...

    def cleanup(self, run_id: str) -> None:
        """Roll back partial state after a failed/interrupted/invalid run.

        Must be safe to call more than once and safe to call when there is
        nothing to clean up.
        """
        ...


# --------------------------------------------------------------------------
# Contract runner: enforces the boundary around any conforming adapter
# --------------------------------------------------------------------------


def run_trainer_contract(
    adapter: TrainerAdapterV1,
    inputs: TrainingInputs,
    budget: ResourceBudget,
    cancel_token: CancellationToken | None = None,
    resume_from: CheckpointHandle | None = None,
) -> TrainingOutput:
    """Execute one trainer-adapter run inside the contract boundary.

    Enforces: contract-version compatibility, input validation, a
    process-level wall-clock timeout and cancellation (the adapter runs
    in a real child OS process that is SIGKILLed on timeout, overrun, or
    cancellation -- see ``codevolt_mdf.process_isolation``), OS-measured
    resource-budget checks (not adapter self-reports), and evidence-hash
    verification (tamper detection). Always calls ``adapter.cleanup``
    when the run does not end ``ACCEPTED``.
    """
    if adapter.contract_version not in SUPPORTED_CONTRACT_VERSIONS:
        raise RejectedInputError(
            f"adapter declares contract_version={adapter.contract_version!r}, "
            f"runner supports {SUPPORTED_CONTRACT_VERSIONS}"
        )
    if not adapter.upstream.is_compatible():
        raise RejectedInputError(
            f"adapter upstream {adapter.upstream.engine_name} "
            f"{adapter.upstream.installed_version} is outside the declared "
            f"compatibility bound [{adapter.upstream.min_version}, {adapter.upstream.max_version}]"
        )

    try:
        inputs.validate()
    except InvalidInputError as exc:
        adapter.cleanup(inputs.run_id)
        return TrainingOutput(status=TrainingStatus.INVALID, reason=str(exc), error_class=type(exc).__name__)
    try:
        adapter.prepare(inputs, budget)
    except RejectedInputError as exc:
        adapter.cleanup(inputs.run_id)
        return TrainingOutput(status=TrainingStatus.REJECTED, reason=str(exc), error_class=type(exc).__name__)
    except InvalidInputError as exc:
        adapter.cleanup(inputs.run_id)
        return TrainingOutput(status=TrainingStatus.INVALID, reason=str(exc), error_class=type(exc).__name__)

    token = cancel_token or CancellationToken()

    # Imported lazily to keep trainer_contract.py importable even in
    # environments where multiprocessing's spawn method is restricted
    # (e.g. certain sandboxes); process_isolation is stdlib-only so this
    # is purely to avoid a hard import-time dependency cycle risk.
    from .process_isolation import run_in_isolated_process

    output_payload, exc_payload, measured = run_in_isolated_process(
        adapter, inputs, budget, resume_from, token
    )

    if measured.killed_for_timeout:
        adapter.cleanup(inputs.run_id)  # timed out with no checkpoint recorded: nothing to resume
        return TrainingOutput(
            status=TrainingStatus.INTERRUPTED,
            reason=(
                f"training exceeded max_wall_seconds={budget.max_wall_seconds} "
                f"(elapsed={measured.wall_seconds}s); adapter process was SIGKILLed"
                f"{_pid_tree_walk_reason_suffix(measured)}"
            ),
            error_class=TrainerTimeoutError.__name__,
            resource_usage=_measured_usage_as_resource_usage(measured),
        )

    if measured.killed_for_overrun:
        adapter.cleanup(inputs.run_id)
        return TrainingOutput(
            status=TrainingStatus.INTERRUPTED,
            reason=(
                "live-measured resource usage exceeded budget while running; "
                "adapter process was SIGKILLed "
                f"(cpu_seconds={measured.cpu_seconds}, memory_mb_peak={measured.memory_mb_peak})"
                f"{_pid_tree_walk_reason_suffix(measured)}"
            ),
            error_class=ResourceBudgetExceededError.__name__,
            resource_usage=_measured_usage_as_resource_usage(measured),
        )

    if exc_payload is not None:
        adapter.cleanup(inputs.run_id)
        exc = exc_payload
        if isinstance(exc, RejectedInputError):
            return TrainingOutput(status=TrainingStatus.REJECTED, reason=str(exc), error_class=type(exc).__name__)
        if isinstance(exc, InvalidInputError):
            return TrainingOutput(status=TrainingStatus.INVALID, reason=str(exc), error_class=type(exc).__name__)
        # A ChildProcessError is the parent-reconstructed stand-in for a
        # child-raised exception whose real class is never trusted/
        # instantiated (see process_isolation.py); its own class name
        # ("ChildProcessError") is not useful to a caller, so report the
        # original child-side exception type name it carries instead.
        from .process_isolation import ChildProcessError

        reported_error_class = (
            exc.original_type_name if isinstance(exc, ChildProcessError) else type(exc).__name__
        )
        return TrainingOutput(
            status=TrainingStatus.INTERRUPTED,
            reason=str(exc),
            error_class=reported_error_class,
        )

    output = output_payload
    if output is None:
        adapter.cleanup(inputs.run_id)
        return TrainingOutput(
            status=TrainingStatus.INTERRUPTED,
            reason="adapter process exited without reporting a result",
            error_class=TrainerTimeoutError.__name__,
        )

    # v1.1: overwrite wall/cpu/memory with OS-measured figures, discarding
    # the adapter's self-report for those three dimensions. gpu_count_used
    # and network_calls remain adapter-reported -- see process_isolation's
    # module docstring ("What this does NOT enforce").
    if output.resource_usage is not None:
        measured_usage = replace(
            output.resource_usage,
            wall_seconds=measured.wall_seconds,
            cpu_seconds=measured.cpu_seconds,
            memory_mb_peak=measured.memory_mb_peak,
            storage_mb_used=(
                measured.storage_mb_used
                if measured.storage_mb_used is not None
                else output.resource_usage.storage_mb_used
            ),
        )
        output = replace(output, resource_usage=measured_usage)

    if output.status == TrainingStatus.ACCEPTED and output.resource_usage is not None:
        violations = output.resource_usage.exceeds(budget)
        if violations:
            adapter.cleanup(inputs.run_id)
            return TrainingOutput(
                status=TrainingStatus.INTERRUPTED,
                reason=f"resource budget exceeded: {', '.join(violations)}",
                error_class=ResourceBudgetExceededError.__name__,
                resource_usage=output.resource_usage,
            )

    if output.status == TrainingStatus.ACCEPTED and output.evidence_locator is not None:
        actual_hash = _hash_evidence(output.evidence_locator)
        if output.evidence_hash is not None and actual_hash != output.evidence_hash:
            adapter.cleanup(inputs.run_id)
            return TrainingOutput(
                status=TrainingStatus.INVALID,
                reason=(
                    f"evidence at {output.evidence_locator} does not match its declared hash "
                    "(tamper or corruption detected)"
                ),
                error_class=TamperDetectedError.__name__,
                evidence_locator=output.evidence_locator,
            )

    if output.status != TrainingStatus.ACCEPTED and output.checkpoint is None:
        # Roll back partial state -- unless the adapter left a checkpoint
        # behind for a future resume (a safe-halt is not a failure to
        # discard; it is deliberately preserved evidence + resumable state).
        adapter.cleanup(inputs.run_id)

    return output


def _measured_usage_as_resource_usage(measured: Any) -> ResourceUsage:
    """Build a ``ResourceUsage`` purely from OS measurement (no adapter self-report)."""
    return ResourceUsage(
        wall_seconds=measured.wall_seconds,
        cpu_seconds=measured.cpu_seconds,
        memory_mb_peak=measured.memory_mb_peak,
        gpu_count_used=0,
        storage_mb_used=measured.storage_mb_used or 0.0,
    )


def _pid_tree_walk_reason_suffix(measured: Any) -> str:
    """Append evidence-bundle-visible text when the pid-tree kill walk degraded.

    Surfaces both originally-silent failure modes flagged in Maya's PR #14
    review (issue #7's approved Layer 1 design, step 3): the underlying
    ``ps`` call failing/timing out during the kill, and the bounded
    walk-and-kill loop exhausting all its passes without confirming a
    full reap. Returns an empty string when no kill ran or the walk fully
    confirmed the reap, so the common, non-degraded case is unchanged.
    See ``process_isolation.PidTreeWalkOutcome`` and
    ``docs/decisions/0004-pid-tree-walk-setsid-escape-fix.md``.
    """
    outcome = getattr(measured, "pid_tree_walk_outcome", None)
    if outcome is None or not outcome.degraded:
        return ""
    parts = []
    if outcome.ps_call_failed:
        parts.append("'ps' call failed/timed out on at least one pass")
    if outcome.exhausted_without_confirmed_reap:
        parts.append("walk exhausted all passes without confirming a full reap")
    return (
        " [pid-tree-walk degraded: " + "; ".join(parts) + " -- a descendant may "
        "not have been individually confirmed killed by the ppid-lineage walk; "
        "the redundant os.killpg group-kill still ran]"
    )



def _hash_evidence(locator: str) -> str:
    path = Path(locator)
    return hashlib.sha256(path.read_bytes()).hexdigest()
