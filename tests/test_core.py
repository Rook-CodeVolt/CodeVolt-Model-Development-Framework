import json
from pathlib import Path

import pytest

from codevolt_mdf.core import ContractError, Experiment, decide, run_experiment

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "experiment.schema.json"

# Issue #6 minimal repro: missing model, dataset, experiment.budget, training.parameters
# (and, per the reconciled schema, missing nothing else -- demo_scores IS present here).
ISSUE_6_REPRO_MANIFEST = {
    "experiment": {"name": "schema-runtime-gap", "owner": "CodeVolt", "seed": 42},
    "training": {"adapter": "deterministic-demo"},
    "evaluation": {
        "adapter": "deterministic-demo",
        "acceptance": {"minimum_score": 0.7, "minimum_improvement": 0.05},
    },
    "demo_scores": {"baseline": 0.65, "candidate": 0.73},
}


def sample(candidate=0.73):
    """A manifest satisfying every schemas/experiment.schema.json required field."""
    return {
        "experiment": {
            "name": "proof",
            "owner": "CodeVolt",
            "seed": 42,
            "budget": {"max_hours": 1, "max_vram_gb": 1},
        },
        "model": {"base": "demonstration-only", "revision": "immutable-demo-v1"},
        "dataset": {"name": "synthetic-demonstration", "sha256": "not-applicable"},
        "training": {"adapter": "deterministic-demo", "parameters": {}},
        "evaluation": {
            "adapter": "deterministic-demo",
            "acceptance": {"minimum_score": 0.7, "minimum_improvement": 0.05},
        },
        "demo_scores": {"baseline": 0.65, "candidate": candidate},
    }


def test_accepts_candidate_meeting_every_gate():
    experiment = Experiment.from_dict(sample())
    assert decide(experiment, experiment.candidate_score).status == "accepted"


def test_rejects_regression():
    experiment = Experiment.from_dict(sample(candidate=0.64))
    assert decide(experiment, experiment.candidate_score).status == "rejected"


def test_missing_evidence_is_invalid():
    payload = sample()
    del payload["demo_scores"]
    with pytest.raises(ContractError):
        Experiment.from_dict(payload)


def test_run_writes_auditable_bundle(tmp_path):
    manifest = tmp_path / "experiment.json"
    manifest.write_text(json.dumps(sample()))
    run_dir = run_experiment(manifest, tmp_path / "runs")
    assert {p.name for p in run_dir.iterdir()} == {
        "manifest.lock.json",
        "baseline.json",
        "evaluation.json",
        "decision.json",
        "provenance.json",
    }


def test_issue_6_repro_manifest_rejected_by_from_dict():
    """A manifest missing model, dataset, experiment.budget, and training.parameters
    must be rejected, never accepted, per schemas/experiment.schema.json."""
    with pytest.raises(ContractError):
        Experiment.from_dict(ISSUE_6_REPRO_MANIFEST)


def test_issue_6_repro_manifest_rejected_by_run_before_output_created(tmp_path):
    manifest = tmp_path / "experiment.json"
    manifest.write_text(json.dumps(ISSUE_6_REPRO_MANIFEST))
    output_root = tmp_path / "runs"
    with pytest.raises(ContractError):
        run_experiment(manifest, output_root)
    # Fail closed: no run directory/output must be created on schema-invalid input.
    assert not output_root.exists() or not any(output_root.iterdir())


def test_schema_and_runtime_agree_on_required_demo_scores():
    schema = json.loads(SCHEMA_PATH.read_text())
    assert "demo_scores" in schema["required"], (
        "schema must require demo_scores to match runtime's Experiment.from_dict"
    )
    assert schema["properties"]["demo_scores"]["required"] == ["baseline", "candidate"]


def test_valid_full_manifest_matching_schema_is_accepted():
    """A manifest satisfying every schema-required field (the honest, full contract)
    must still be accepted -- this guards against over-tightening the fix."""
    experiment = Experiment.from_dict(sample())
    assert decide(experiment, experiment.candidate_score).status == "accepted"
