#!/usr/bin/env python3
"""Generate examples/pilot-adr0022/items_c2.json (ADR-0022 section 3).

12 items: 6 config-validity, 6 command-produces-expected-artifact.

Config-validity items SUPPLY a fixed manifest and ask the candidate to
classify it (pass / fail-with-named-check) -- this matches section 3's
"each item supplies (or has the model produce) an experiment manifest
... 3 items are constructed to be valid ... 3 are each constructed to
fail exactly one specific, named check" framing directly: the item's
own construction is what supplies the valid/invalid manifest, and the
model's job is to classify it.

Command-produces-expected-artifact items instead give a natural-
language description of the target manifest values (name, seed, model/
dataset identity, adapter, minimum_score/minimum_improvement, baseline/
candidate demo scores) and ask the candidate to EMIT the manifest as a
fenced JSON code block -- matching section 3's own two-stage scoring
mechanism literally ("the candidate's/reference's raw text response is
searched for a fenced code block ... taken as the candidate manifest
text and parsed as JSON"), which only makes sense if the candidate is
producing the manifest itself, not merely classifying a supplied one.
Gold decision.json is computed from the prompt's own declared target
values via decide(), then independently reproduced by actually running
codevolt_mdf.core.run_experiment (the real CLI path) against a manifest
built from those exact values -- not merely hand-computed -- per
section 3's "then independently reproduced by actually running the
real CLI against the item's own manifest before the item is admitted"
requirement. A model that emits a syntactically different but
semantically equivalent manifest is scored on the two-stage outcome
(parse+validate, then decision.json match), not on textual similarity
to any single reference manifest, matching section 3's explicit rule.
"""
import hashlib
import json
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from codevolt_mdf.core import Experiment, ContractError, validate_manifest_schema, decide, run_experiment

OUT = Path(__file__).resolve().parent / "items_c2.json"


def base_manifest(name, seed, baseline, candidate, minscore, minimp):
    return {
        "experiment": {"name": name, "owner": "adr0022-c2-author", "seed": seed, "budget": {"max_runs": 1}},
        "model": {"base": "deterministic-demo", "revision": "n/a"},
        "dataset": {"name": "adr0022-c2-demo-dataset", "sha256": "0" * 64},
        "training": {"adapter": "deterministic-demo", "parameters": {}},
        "evaluation": {"adapter": "deterministic-demo", "acceptance": {"minimum_score": minscore, "minimum_improvement": minimp}},
        "demo_scores": {"baseline": baseline, "candidate": candidate},
    }


items = []

# -- 6 config-validity items: 3 valid, 3 each failing exactly one check --

valid_1 = base_manifest("adr0022-c2-valid-01", 201, 0.2, 0.55, 0.4, 0.2)
valid_2 = base_manifest("adr0022-c2-valid-02", 202, 0.3, 0.9, 0.5, 0.5)
valid_3 = base_manifest("adr0022-c2-valid-03", 203, 0.5, 0.5, 0.5, 0.0)

fail_missing_key = base_manifest("adr0022-c2-fail-missing-evaluation", 204, 0.2, 0.55, 0.4, 0.2)
del fail_missing_key["evaluation"]

fail_demo_score_range = base_manifest("adr0022-c2-fail-demo-score-out-of-range", 205, 0.5, 1.5, 0.4, 0.2)

fail_negative_improvement = base_manifest("adr0022-c2-fail-negative-min-improvement", 206, 0.4, 0.55, 0.4, -0.1)


CONFIG_VALIDITY_PROMPT_TEMPLATE = (
    "You are checking a CodeVolt experiment manifest against this "
    "repository's own real contract checks (schemas/experiment.schema.json "
    "plus Experiment.validate()). Manifest:\n\n{manifest_json}\n\n"
    "On the first line, answer with exactly one of: PASS, or FAIL "
    "followed by a colon and the single specific violated-check name "
    "(one of: missing_required_key, demo_score_out_of_range, "
    "negative_minimum_improvement). Nothing else on the first line."
)

CMD_ARTIFACT_PROMPT_TEMPLATE = (
    "Author a CodeVolt experiment manifest (JSON, matching "
    "schemas/experiment.schema.json) for the following experiment, and "
    "respond with exactly one fenced code block (```json ... ``` or "
    "``` ... ```) containing that manifest and nothing else outside the "
    "fence:\n\n"
    "- experiment.name: \"{name}\"\n"
    "- experiment.owner: \"{owner}\"\n"
    "- experiment.seed: {seed}\n"
    "- experiment.budget: any object (e.g. {{\"max_runs\": 1}})\n"
    "- model.base: \"deterministic-demo\", model.revision: \"n/a\"\n"
    "- dataset.name: \"{dataset_name}\", dataset.sha256: \"{dataset_sha256}\"\n"
    "- training.adapter: \"deterministic-demo\", training.parameters: {{}}\n"
    "- evaluation.adapter: \"deterministic-demo\"\n"
    "- evaluation.acceptance.minimum_score: {minscore}\n"
    "- evaluation.acceptance.minimum_improvement: {minimp}\n"
    "- demo_scores.baseline: {baseline}\n"
    "- demo_scores.candidate: {candidate}"
)


def _validate_pass(manifest):
    validate_manifest_schema(manifest)
    exp = Experiment.from_dict(manifest)
    exp.validate()
    return "pass"


def _validate_fail(manifest):
    try:
        validate_manifest_schema(manifest)
        exp = Experiment.from_dict(manifest)
        exp.validate()
    except ContractError as exc:
        return str(exc)
    raise AssertionError("expected ContractError, manifest unexpectedly passed validation")


VIOLATED_CHECK_TOKENS = {
    "missing_required_key": "missing required top-level key: evaluation",
    "demo_score_out_of_range": "demo_scores value outside [0, 1]",
    "negative_minimum_improvement": "negative minimum_improvement",
}

config_validity_specs = [
    ("adr0022-c2-config-001", valid_1, "pass", None),
    ("adr0022-c2-config-002", valid_2, "pass", None),
    ("adr0022-c2-config-003", valid_3, "pass", None),
    ("adr0022-c2-config-004", fail_missing_key, "fail", "missing_required_key"),
    ("adr0022-c2-config-005", fail_demo_score_range, "fail", "demo_score_out_of_range"),
    ("adr0022-c2-config-006", fail_negative_improvement, "fail", "negative_minimum_improvement"),
]

for example_id, manifest, expected_outcome, violated_check_token in config_validity_specs:
    if expected_outcome == "pass":
        gold_detail = _validate_pass(manifest)
    else:
        gold_detail = _validate_fail(manifest)
    items.append({
        "example_id": example_id,
        "sub_bucket": "config_validity",
        "input": CONFIG_VALIDITY_PROMPT_TEMPLATE.format(
            manifest_json=json.dumps(manifest, indent=2, sort_keys=True)
        ),
        "expected": {
            "outcome": expected_outcome,
            "violated_check": violated_check_token,
            "gold_detail": gold_detail,
            "manifest": manifest,
        },
        "metadata": {
            "task_type": "format_conformance",
            "answer_format": "first_line_PASS_or_FAIL_colon_check",
            "source": (
                "src/codevolt_mdf/core.py validate_manifest_schema/Experiment.validate, "
                "re-run against this exact manifest to derive gold_detail"
            ),
        },
    })

# -- 6 command-produces-expected-artifact items: 3 accept, 3 reject, --
# -- at least one boundary in each half; the candidate authors the ----
# -- manifest itself from a natural-language target description ------

cmd_specs = [
    ("adr0022-c2-cmd-accept-001", 301, 0.20, 0.55, 0.40, 0.20),
    ("adr0022-c2-cmd-accept-002", 302, 0.30, 0.90, 0.50, 0.50),
    # boundary: candidate_score == minimum_score
    ("adr0022-c2-cmd-accept-boundary-003", 303, 0.30, 0.60, 0.60, 0.10),
    ("adr0022-c2-cmd-reject-004", 304, 0.50, 0.55, 0.60, 0.10),
    ("adr0022-c2-cmd-reject-005", 305, 0.50, 0.60, 0.50, 0.30),
    # boundary: improvement exactly at the minimum, but candidate_score
    # still below minimum_score -> both conditions required -> rejected.
    ("adr0022-c2-cmd-reject-boundary-006", 306, 0.50, 0.55, 0.60, 0.05),
]

for example_id, seed, baseline, candidate, minscore, minimp in cmd_specs:
    manifest = base_manifest(example_id, seed, baseline, candidate, minscore, minimp)
    exp = Experiment.from_dict(manifest)
    exp.validate()
    decision = decide(exp, candidate)

    # Independently reproduce via the real CLI path (run_experiment), not
    # merely the hand-computed decide() call, per ADR-0022 section 3's
    # "then independently reproduced by actually running the real CLI
    # against the item's own manifest before the item is admitted".
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        run_dir = run_experiment(manifest_path, tmp_path / "output")
        real_decision = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
        assert real_decision["status"] == decision.status, (example_id, real_decision, decision)
        assert real_decision["improvement"] == decision.improvement, (example_id, real_decision, decision)

    prompt = CMD_ARTIFACT_PROMPT_TEMPLATE.format(
        name=manifest["experiment"]["name"],
        owner=manifest["experiment"]["owner"],
        seed=manifest["experiment"]["seed"],
        dataset_name=manifest["dataset"]["name"],
        dataset_sha256=manifest["dataset"]["sha256"],
        minscore=minscore,
        minimp=minimp,
        baseline=baseline,
        candidate=candidate,
    )

    items.append({
        "example_id": example_id,
        "sub_bucket": "command_produces_expected_artifact",
        "input": prompt,
        "expected": {
            "decision_status": decision.status,
            "decision_improvement": decision.improvement,
            "decision_baseline_score": decision.baseline_score,
            "decision_candidate_score": decision.candidate_score,
            "reference_manifest": manifest,
        },
        "metadata": {
            "task_type": "format_conformance",
            "scoring": "two_stage_parse_then_cli_dry_run",
            "manifest_extraction": "first fenced code block (```json or bare ```) in raw model output",
            "answer_format": "single_fenced_json_manifest",
            "source": (
                "codevolt_mdf.core.decide(), independently reproduced by "
                "actually running run_experiment against a manifest built "
                "from this item's own declared target values"
            ),
        },
    })

assert len(items) == 12
ids = [i["example_id"] for i in items]
assert len(ids) == len(set(ids))
config_count = sum(1 for i in items if i["sub_bucket"] == "config_validity")
cmd_count = sum(1 for i in items if i["sub_bucket"] == "command_produces_expected_artifact")
assert config_count == 6 and cmd_count == 6

payload = {"capability": "C2", "count": len(items), "items": items}
raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
OUT.write_text(raw, encoding="utf-8")
print(OUT)
print(hashlib.sha256(raw.encode("utf-8")).hexdigest())
