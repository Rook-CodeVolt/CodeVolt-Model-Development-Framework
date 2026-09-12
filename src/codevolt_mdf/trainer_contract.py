"""TrainerAdapterContract v1: the first real trainer-adapter boundary.

This module defines an executable, versioned contract that a trainer
adapter must satisfy before it is admitted into CodeVolt MDF. It replaces
the informal two-method ``TrainerAdapter``/``EvaluatorAdapter`` Protocol in
``core.py`` as the boundary a *real* training engine integration must cross.

Design constraints (see docs/TRAINER_ADAPTER_CONTRACT.md for the full
rationale):

- Dependency-light: standard library only. Timeout/cancellation is
  therefore *cooperative* -- Python cannot force-kill a thread, so the
  contract runner enforces a wall-clock deadline and signals a
  ``CancellationToken``; a conforming adapter MUST check the token
  frequently and return promptly. This limitation is documented, not
  hidden: see ``docs/TRAINER_ADAPTER_CONTRACT.md`` ("Timeout semantics").
- The contract governs a *trainer*. It never evaluates, scores, or
  promotes its own output (see ``docs/ARCHITECTURE.md`` core contract #2).
- Nothing in this module trains a real model or grants training,
  data-admission, deployment, or publication authority. It defines a
  boundary and ships a deterministic FAKE adapter to exercise it.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

CONTRACT_VERSION = "1.0.0"
"""Semantic version of TrainerAdapterContract. Bump on any breaking change
to the shapes or semantics below; keep old versions documented so contract
changes stay diffable (see docs/decisions/0002-trainer-adapter-contract-v1.md).
"""

SUPPORTED_CONTRACT_VERSIONS: tuple[str, ...] = (CONTRACT_VERSION,)


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

    Python threads cannot be force-killed without extra-stdlib tooling, so
    this contract requires *cooperative* cancellation: a conforming adapter
    must poll ``is_cancelled()`` or ``wait(interval)`` frequently inside any
    loop and return an ``INTERRUPTED`` ``TrainingOutput`` promptly once set.
    This is a documented limitation, not a hidden gap -- see
    docs/TRAINER_ADAPTER_CONTRACT.md, "Timeout semantics".
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

    Enforces: contract-version compatibility, input validation, the
    wall-clock timeout (cooperative cancellation), post-hoc resource-budget
    checks against reported usage, and evidence-hash verification (tamper
    detection). Always calls ``adapter.cleanup`` when the run does not end
    ``ACCEPTED``.
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
    result: dict[str, TrainingOutput] = {}
    error: dict[str, BaseException] = {}

    def _target() -> None:
        try:
            result["output"] = adapter.train(inputs, budget, token, resume_from)
        except BaseException as exc:  # noqa: BLE001 - surfaced to caller below
            error["exc"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    start = time.monotonic()
    thread.start()
    thread.join(timeout=budget.max_wall_seconds)

    if thread.is_alive():
        token.cancel(reason="timeout")
        thread.join(timeout=budget.max_wall_seconds)
        adapter.cleanup(inputs.run_id)  # timed out with no checkpoint recorded: nothing to resume
        elapsed = round(time.monotonic() - start, 6)
        return TrainingOutput(
            status=TrainingStatus.INTERRUPTED,
            reason=f"training exceeded max_wall_seconds={budget.max_wall_seconds} (elapsed={elapsed}s)",
            error_class=TrainerTimeoutError.__name__,
        )

    if "exc" in error:
        adapter.cleanup(inputs.run_id)
        exc = error["exc"]
        if isinstance(exc, RejectedInputError):
            return TrainingOutput(status=TrainingStatus.REJECTED, reason=str(exc), error_class=type(exc).__name__)
        if isinstance(exc, InvalidInputError):
            return TrainingOutput(status=TrainingStatus.INVALID, reason=str(exc), error_class=type(exc).__name__)
        return TrainingOutput(
            status=TrainingStatus.INTERRUPTED,
            reason=str(exc),
            error_class=type(exc).__name__,
        )

    output = result["output"]

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


def _hash_evidence(locator: str) -> str:
    path = Path(locator)
    return hashlib.sha256(path.read_bytes()).hexdigest()
