"""Dependency-light reference implementation of the CodeVolt experiment contract."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "experiment.schema.json"


class ContractError(ValueError):
    """Raised when an experiment cannot be evaluated safely."""


def _load_schema() -> dict[str, Any]:
    try:
        return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Unable to load experiment schema: {exc}") from exc


def _require_object_with_keys(payload: Any, required: list[str], where: str) -> None:
    """Fail closed: `where` must be a JSON object containing every key in `required`.

    This is a minimal, stdlib-only stand-in for JSON Schema's `type: object` +
    `required` keywords -- the only shape the current schema actually declares.
    It deliberately does not implement the full JSON Schema spec (no new
    dependency is added; see docs/ARCHITECTURE.md's dependency-light policy).
    """
    if not isinstance(payload, dict):
        raise ContractError(f"'{where}' must be an object")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ContractError(f"'{where}' is missing required field(s): {', '.join(missing)}")


def validate_manifest_schema(payload: Any, schema: dict[str, Any] | None = None) -> None:
    """Validate `payload` against schemas/experiment.schema.json (fail closed).

    Enforces exactly the shape the schema declares: top-level required keys,
    and for each top-level object property, its own required sub-keys. This
    covers the manifests this framework accepts today; it is not a general
    JSON Schema engine.
    """
    schema = schema if schema is not None else _load_schema()
    top_required = schema.get("required", [])
    _require_object_with_keys(payload, top_required, "manifest")

    properties = schema.get("properties", {})
    for key, subschema in properties.items():
        if key not in payload:
            continue  # already reported above if it was required
        sub_required = subschema.get("required")
        if sub_required:
            _require_object_with_keys(payload[key], sub_required, key)


@dataclass(frozen=True)
class Experiment:
    name: str
    owner: str
    seed: int
    trainer: str
    evaluator: str
    baseline_score: float
    candidate_score: float
    minimum_score: float
    minimum_improvement: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Experiment:
        validate_manifest_schema(payload)
        try:
            metadata = payload["experiment"]
            training = payload["training"]
            evaluation = payload["evaluation"]
            demo = payload["demo_scores"]
            return cls(
                name=str(metadata["name"]),
                owner=str(metadata["owner"]),
                seed=int(metadata["seed"]),
                trainer=str(training["adapter"]),
                evaluator=str(evaluation["adapter"]),
                baseline_score=float(demo["baseline"]),
                candidate_score=float(demo["candidate"]),
                minimum_score=float(evaluation["acceptance"]["minimum_score"]),
                minimum_improvement=float(
                    evaluation["acceptance"]["minimum_improvement"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"Invalid experiment manifest: {exc}") from exc

    def validate(self) -> None:
        if not self.name.strip() or not self.owner.strip():
            raise ContractError("Experiment name and owner are required")
        if not 0 <= self.baseline_score <= 1 or not 0 <= self.candidate_score <= 1:
            raise ContractError("Demo scores must be between 0 and 1")
        if self.minimum_improvement < 0:
            raise ContractError("Minimum improvement cannot be negative")


@dataclass(frozen=True)
class Decision:
    status: str
    reason: str
    baseline_score: float
    candidate_score: float
    improvement: float


@dataclass(frozen=True)
class LearningEvent:
    event_id: str
    category: str
    observed_behaviour: str
    expected_behaviour: str
    evidence: str


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    action: str
    hypothesis: str
    requires_approval: bool = True


class TrainerAdapter(Protocol):
    name: str

    def train(self, experiment: Experiment) -> dict[str, Any]: ...


class EvaluatorAdapter(Protocol):
    name: str

    def evaluate(self, artifact: dict[str, Any]) -> float: ...


class DeterministicTrainer:
    """Safe demonstration adapter; replace with a real engine integration."""

    name = "deterministic-demo"

    def train(self, experiment: Experiment) -> dict[str, Any]:
        return {"kind": "demo", "score": experiment.candidate_score, "seed": experiment.seed}


class DeterministicEvaluator:
    name = "deterministic-demo"

    def evaluate(self, artifact: dict[str, Any]) -> float:
        score = artifact.get("score")
        if not isinstance(score, (int, float)):
            raise ContractError("Evaluator received an artifact without a numeric score")
        return float(score)


def decide(experiment: Experiment, candidate_score: float) -> Decision:
    improvement = round(candidate_score - experiment.baseline_score, 10)
    accepted = (
        candidate_score >= experiment.minimum_score
        and improvement >= experiment.minimum_improvement
    )
    return Decision(
        status="accepted" if accepted else "rejected",
        reason=(
            "Candidate satisfied every acceptance threshold"
            if accepted
            else "Candidate did not satisfy every acceptance threshold"
        ),
        baseline_score=experiment.baseline_score,
        candidate_score=candidate_score,
        improvement=improvement,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(manifest_path: Path, output_root: Path) -> Path:
    raw = manifest_path.read_bytes()
    payload = json.loads(raw)
    experiment = Experiment.from_dict(payload)
    experiment.validate()
    if experiment.trainer != "deterministic-demo" or experiment.evaluator != "deterministic-demo":
        raise ContractError("v0.1 ships only the deterministic-demo adapters")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{experiment.name}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact = DeterministicTrainer().train(experiment)
    score = DeterministicEvaluator().evaluate(artifact)
    decision = decide(experiment, score)

    _write_json(run_dir / "manifest.lock.json", payload)
    _write_json(run_dir / "baseline.json", {"score": experiment.baseline_score})
    _write_json(run_dir / "evaluation.json", {"score": score})
    _write_json(run_dir / "decision.json", asdict(decision))
    _write_json(
        run_dir / "provenance.json",
        {
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "framework_version": "0.1.0",
        },
    )
    return run_dir
