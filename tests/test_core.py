import json

import pytest

from codevolt_mdf.core import ContractError, Experiment, decide, run_experiment


def sample(candidate=0.73):
    return {
        "experiment": {"name": "proof", "owner": "CodeVolt", "seed": 42},
        "training": {"adapter": "deterministic-demo"},
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
