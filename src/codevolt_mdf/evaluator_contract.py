"""EvaluatorAdapterContract v1: an independent-evaluation harness.

This module defines an executable, versioned contract for issue #7 step
4 ("Independent evaluation: evaluate the candidate through a separately
configured evaluator with a hidden held-out set inaccessible to the
trainer"). It is a deliberately separate component from
``trainer_contract.py``/``trl_adapter.py`` -- nothing here imports or is
imported by ``trl_adapter.py`` -- consistent with
``docs/ARCHITECTURE.md``'s core contracts #2/#3 ("A trainer produces an
artifact; it does not decide whether the artifact is good" / "An
evaluator produces measurements; it does not promote the candidate") and
``docs/TRAINER_ADAPTER_CONTRACT.md``'s "Trainer / evaluator / security-
review role separation" section.

Structural held-out isolation, not just a documented promise: a
``HeldOutSet`` is never constructed from or passed through anything the
trainer adapter touches. ``TrainerAdapterV1.prepare()``/``train()`` take
a ``TrainingInputs``/``ResourceBudget`` pair that has no held-out-set
field at all (see ``trainer_contract.py``) -- there is no shared object
an evaluator and a trainer could both receive that would let a held-out
set leak into the trainer's inputs by accident. The evaluator harness in
this module is the only code path that ever constructs or reads a
``HeldOutSet``.

Held-out exclusion is enforced, not only declared: before scoring
anything, ``run_evaluator_contract`` calls
``held_out_registry.HeldOutExclusionRegistry.check_held_out_not_trained``
against every example id in the held-out set being evaluated. If any id
is already registered as train data by *any* package (including a
different package than the one under evaluation), the run is rejected
as ``INVALID`` before a single example is scored -- this closes the
exact contamination class issue #11 found (67 ids used as train by one
package silently accepted as held-out by another), reusing that
registry's bidirectional, per-package design rather than re-declaring a
weaker one-directional check here.

Design constraints, mirroring ``trainer_contract.py``:

- Dependency-light: standard library only.
- This contract governs only *evaluation*. It never trains, never
  decides promotion (``docs/ARCHITECTURE.md`` core contract #6:
  "Promotion is a separate governed action"), and never mutates a
  ``TrainerAdapterV1``'s inputs or state.
- Ships a deterministic FAKE evaluator adapter
  (``fake_evaluator_adapter.py``) for contract tests; no real
  model-inference engine is wired in by this work package, matching
  the same "contract + tests only, no real run" scoping ADR-0005 used
  for the TRL trainer adapter.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from .held_out_registry import HeldOutExclusionRegistry

CONTRACT_VERSION = "1.0.0"
"""Semantic version of EvaluatorAdapterContract. Bump on any breaking
change to the shapes or semantics below."""

SUPPORTED_CONTRACT_VERSIONS: tuple[str, ...] = (CONTRACT_VERSION,)


# --------------------------------------------------------------------------
# Error classes
# --------------------------------------------------------------------------


class EvaluatorContractError(Exception):
    """Base class for every error this contract defines."""


class InvalidInputError(EvaluatorContractError):
    """Inputs are malformed, incomplete, or fail static validation.

    Maps to run status ``invalid``: the run never started meaningful
    scoring work.
    """


class RejectedInputError(EvaluatorContractError):
    """Inputs are well-formed but the adapter declines to score them.

    E.g. unsupported artifact format, incompatible upstream engine
    version. Maps to run status ``rejected``.
    """


class ContaminationDetectedError(EvaluatorContractError):
    """One or more held-out example ids are already registered as train data.

    Maps to run status ``invalid``: the held-out claim cannot be
    trusted, so the evidence this run would produce is not admissible
    evidence of held-out performance. See ``held_out_registry.py``.
    """


class TamperDetectedError(EvaluatorContractError):
    """A held-out set or evidence bundle does not match its declared hash.

    Maps to run status ``invalid``.
    """


# --------------------------------------------------------------------------
# Status and typed outputs
# --------------------------------------------------------------------------


class EvaluationStatus(str, Enum):
    """Outcome of a single evaluator-adapter run.

    Distinct from ``core.Decision.status`` (an acceptance decision
    applying a threshold) and from ``trainer_contract.TrainingStatus``
    (whether a *training* run produced trustworthy evidence). This
    status describes whether *this evaluation run* produced trustworthy
    measurements -- promotion/acceptance against a threshold remains a
    separate, later governed action per ``docs/ARCHITECTURE.md`` core
    contract #6.
    """

    SCORED = "scored"
    REJECTED = "rejected"
    INVALID = "invalid"


@dataclass(frozen=True)
class HeldOutExample:
    """One hidden held-out evaluation example.

    ``example_id`` must correspond to the same ``source_task_id``-style
    identifier space the held-out exclusion registry tracks, so
    contamination checks are meaningful. ``input``/``expected`` are
    opaque to this contract (an adapter interprets them); ``metadata``
    is a sorted tuple of items for hashability, mirroring
    ``TrainingInputs.training_params``.
    """

    example_id: str
    input: Any
    expected: Any
    metadata: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class HeldOutSet:
    """A hidden held-out set for one package, with a content hash.

    ``dataset_hash`` makes the held-out set's content independently
    checkable, the same way ``TrainingInputs.model_hash``/
    ``dataset_hash`` make trainer inputs checkable by content rather
    than by a trusted name -- a held-out set handed to an evaluator
    must match the hash the caller declared for it, so a corrupted or
    substituted held-out file is caught before scoring, not silently
    scored as if it were the real one.
    """

    package_id: str
    examples: tuple[HeldOutExample, ...]
    dataset_hash: str

    @classmethod
    def create(cls, package_id: str, examples: list[HeldOutExample]) -> HeldOutSet:
        ordered = tuple(sorted(examples, key=lambda e: e.example_id))
        return cls(package_id=package_id, examples=ordered, dataset_hash=_hash_examples(ordered))

    @property
    def example_ids(self) -> frozenset[str]:
        return frozenset(example.example_id for example in self.examples)

    def validate(self) -> None:
        if not self.package_id.strip():
            raise InvalidInputError("HeldOutSet.package_id is required and must be non-empty")
        if not self.examples:
            raise InvalidInputError("HeldOutSet.examples must be non-empty")
        ids = [example.example_id for example in self.examples]
        if len(ids) != len(set(ids)):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise InvalidInputError(f"HeldOutSet contains duplicate example_id(s): {duplicates}")
        for example in self.examples:
            if not example.example_id.strip():
                raise InvalidInputError("every HeldOutExample.example_id must be non-empty")
        actual_hash = _hash_examples(self.examples)
        if actual_hash != self.dataset_hash:
            raise TamperDetectedError(
                f"HeldOutSet declared dataset_hash={self.dataset_hash!r} but its actual "
                f"content hashes to {actual_hash!r} (tamper or corruption detected)"
            )


def _hash_examples(examples: tuple[HeldOutExample, ...]) -> str:
    payload = [
        {
            "example_id": example.example_id,
            "input": example.input,
            "expected": example.expected,
            "metadata": list(example.metadata),
        }
        for example in sorted(examples, key=lambda e: e.example_id)
    ]
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class TaskType(str, Enum):
    """Scoring-mode vocabulary a ``HeldOutExample`` may declare via its metadata.

    Purely additive, dispatch-only information -- adding a member here
    never changes ``HeldOutExample``, ``ExampleResult``, or
    ``EvaluationOutput``'s shape, so it does not require a
    ``CONTRACT_VERSION`` bump (see ``docs/decisions/0007-evaluator-task-type-breadth.md``).
    An adapter that does not implement a given ``TaskType`` is free to
    raise ``InvalidInputError`` for it; the contract itself does not
    require every adapter to support every mode.
    """

    EXACT_MATCH = "exact_match"
    """The original mode: case-insensitive, whitespace-normalized
    containment of ``expected`` (a string) in the artifact's output."""

    MULTIPLE_CHOICE = "multiple_choice"
    """Per-option likelihood scoring: ``input`` names a prompt and a
    list of candidate choices, ``expected`` names the correct one
    (by exact text or index); an adapter picks the highest-likelihood
    choice and scores whether it matches. See ``scoring_modes.py``."""

    FORMAT_CONFORMANCE = "format_conformance"
    """Structured-output conformance: ``expected`` is a format spec
    (e.g. ``{"format": "json"}``); an adapter checks whether the
    artifact's output conforms to that spec, not whether it equals a
    specific string. See ``scoring_modes.py``."""

    SAFETY_PROBE = "safety_probe"
    """Safety/red-team probe scoring (WP-B, issue #24): ``input`` is a
    fixed, synthetic probe prompt; ``expected`` is a spec dict naming a
    ``probe_type`` (``refusal_appropriateness``,
    ``harmful_instruction_compliance``, or ``pii_leakage``) plus an
    optional expected-refusal flag and/or disallowed-content patterns.
    An adapter checks the artifact's output against that spec via
    ``scoring_modes.check_safety_probe``. Like every ``TaskType``, this
    produces a measurement only -- never a pass/fail promotion/
    accept-reject decision (see ``docs/EVALUATION_POLICY.md`` and
    ``docs/decisions/0008-safety-probe-suite.md``)."""


def task_type_of(example: HeldOutExample) -> str:
    """Read the declared scoring-mode task type from ``example.metadata``.

    Defaults to ``TaskType.EXACT_MATCH.value`` when no ``"task_type"``
    key is present in ``metadata`` -- so every ``HeldOutExample`` built
    before this function existed (including every fixture in
    ``tests/test_evaluator_contract.py`` and
    ``tests/test_hf_local_evaluator_adapter.py``) keeps scoring via the
    original exact-match/containment path with zero changes. This is
    the mechanism that lets new scoring modes be additive: a caller
    opts in to a new mode by adding one ``("task_type", ...)`` entry to
    ``metadata``, nothing else about ``HeldOutExample`` changes.
    """
    for key, value in example.metadata:
        if key == "task_type":
            return str(value)
    return TaskType.EXACT_MATCH.value


@dataclass(frozen=True)
class ExampleResult:
    """Per-example scoring result."""

    example_id: str
    correct: bool
    score: float
    raw_output: str = ""


@dataclass(frozen=True)
class EvaluationOutput:
    """Typed output of an evaluator-adapter run."""

    status: EvaluationStatus
    reason: str
    artifact_id: str | None = None
    package_id: str | None = None
    aggregate_score: float | None = None
    results: tuple[ExampleResult, ...] = ()
    evidence_locator: str | None = None
    evidence_hash: str | None = None
    error_class: str | None = None
    contaminated_ids: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# Adapter protocol
# --------------------------------------------------------------------------


class EvaluatorAdapterV1(Protocol):
    """Contract an evaluator adapter must satisfy to be admitted (v1).

    Deliberately narrow: an adapter scores one example at a time and
    has no method that trains, mutates a checkpoint, or decides
    promotion. It never receives a ``TrainerAdapterV1`` or
    ``TrainingInputs`` -- the only thing it is ever given is an
    artifact identity/locator and one ``HeldOutExample`` at a time.
    """

    name: str
    contract_version: str

    def score_example(
        self, artifact_id: str, artifact_locator: str, example: HeldOutExample
    ) -> ExampleResult:
        """Score one held-out example against the artifact. Must not train or mutate it."""
        ...


# --------------------------------------------------------------------------
# Contract runner: enforces the boundary around any conforming adapter
# --------------------------------------------------------------------------


def run_evaluator_contract(
    adapter: EvaluatorAdapterV1,
    artifact_id: str,
    artifact_locator: str,
    held_out: HeldOutSet,
    registry: HeldOutExclusionRegistry,
    work_dir: Path,
) -> EvaluationOutput:
    """Execute one evaluator-adapter run against a hidden held-out set.

    Enforces, in order:

    1. Contract-version compatibility.
    2. ``held_out.validate()`` -- shape, uniqueness, and its own
       content-hash tamper check.
    3. Bidirectional held-out/train contamination check against
       ``registry`` (issue #11's pattern) -- rejects before any
       example is scored if any held-out id is already registered as
       train data by any package.
    4. Per-example scoring via ``adapter.score_example`` (never given
       the whole held-out set at once, and never given a trainer
       object or trainer inputs).
    5. Evidence write + hash, so the evidence bundle is independently
       checkable the same way a trainer run's evidence is.

    Never calls or depends on ``TrainerAdapterV1``/``trl_adapter.py``,
    and never mutates ``registry`` -- it is a read-only precondition
    check here; registering a package's held-out ids into the registry
    (``register_package_held_out``) is a separate, earlier step at
    dataset-split time, owned by the data-governance/split tooling this
    contract does not perform.
    """
    if adapter.contract_version not in SUPPORTED_CONTRACT_VERSIONS:
        raise RejectedInputError(
            f"adapter declares contract_version={adapter.contract_version!r}, "
            f"runner supports {SUPPORTED_CONTRACT_VERSIONS}"
        )

    try:
        held_out.validate()
    except (InvalidInputError, TamperDetectedError) as exc:
        return EvaluationOutput(
            status=EvaluationStatus.INVALID,
            reason=str(exc),
            artifact_id=artifact_id,
            package_id=held_out.package_id,
            error_class=type(exc).__name__,
        )

    contaminated = registry.check_held_out_not_trained(held_out.example_ids)
    if contaminated:
        exc = ContaminationDetectedError(
            f"{len(contaminated)} held-out example id(s) in package "
            f"{held_out.package_id!r} are already registered as train data by another "
            f"package in the held-out exclusion registry; refusing to score against a "
            f"contaminated held-out set: {sorted(contaminated)[:10]}"
            + (" ..." if len(contaminated) > 10 else "")
        )
        return EvaluationOutput(
            status=EvaluationStatus.INVALID,
            reason=str(exc),
            artifact_id=artifact_id,
            package_id=held_out.package_id,
            error_class=type(exc).__name__,
            contaminated_ids=tuple(sorted(contaminated)),
        )

    results: list[ExampleResult] = []
    for example in held_out.examples:
        try:
            result = adapter.score_example(artifact_id, artifact_locator, example)
        except RejectedInputError as exc:
            return EvaluationOutput(
                status=EvaluationStatus.REJECTED,
                reason=str(exc),
                artifact_id=artifact_id,
                package_id=held_out.package_id,
                error_class=type(exc).__name__,
            )
        except InvalidInputError as exc:
            return EvaluationOutput(
                status=EvaluationStatus.INVALID,
                reason=str(exc),
                artifact_id=artifact_id,
                package_id=held_out.package_id,
                error_class=type(exc).__name__,
            )
        if result.example_id != example.example_id:
            return EvaluationOutput(
                status=EvaluationStatus.INVALID,
                reason=(
                    f"adapter.score_example returned result for "
                    f"example_id={result.example_id!r} but was called with "
                    f"example_id={example.example_id!r}"
                ),
                artifact_id=artifact_id,
                package_id=held_out.package_id,
                error_class="MismatchedExampleIdError",
            )
        results.append(result)

    aggregate_score = sum(r.score for r in results) / len(results)
    evidence_locator, evidence_hash = _write_evidence(
        work_dir, artifact_id, held_out.package_id, results, aggregate_score
    )
    return EvaluationOutput(
        status=EvaluationStatus.SCORED,
        reason=f"scored {len(results)} held-out example(s)",
        artifact_id=artifact_id,
        package_id=held_out.package_id,
        aggregate_score=aggregate_score,
        results=tuple(results),
        evidence_locator=evidence_locator,
        evidence_hash=evidence_hash,
    )


def _write_evidence(
    work_dir: Path,
    artifact_id: str,
    package_id: str,
    results: list[ExampleResult],
    aggregate_score: float,
) -> tuple[str, str]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = work_dir / f"evaluation-{artifact_id}-{package_id}.json"
    payload = {
        "artifact_id": artifact_id,
        "package_id": package_id,
        "aggregate_score": aggregate_score,
        "results": [
            {
                "example_id": r.example_id,
                "correct": r.correct,
                "score": r.score,
                "raw_output": r.raw_output,
            }
            for r in results
        ],
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    evidence_path.write_bytes(raw)
    return str(evidence_path), hashlib.sha256(raw).hexdigest()


def verify_evidence(evidence_locator: str, evidence_hash: str) -> bool:
    """Independently re-hash an evidence file and compare to its declared hash."""
    path = Path(evidence_locator)
    if not path.exists():
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == evidence_hash
