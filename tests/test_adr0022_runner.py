"""Tests for examples/pilot-adr0022/run_adr0022_baseline_eval.py.

Same scope boundary as tests/test_adr0019_logprob_margin_eval.py: proves
the runner's scoring logic, schema/hash-checking, gate verification, and
dry-run/--execute split, without ever loading a real model or invoking a
real llama.cpp binary. No test in this file requires the pinned
SmolLM2/Qwen2.5-7B-Instruct snapshot to be present on the runner, and
every model-load-shaped call is stubbed with a fake checkpoint or a fake
llama.cpp CLI stand-in.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from codevolt_mdf import process_isolation
from codevolt_mdf.hf_local_evaluator_adapter import _hash_model_dir
from codevolt_mdf.process_isolation import (
    MeasuredUsage,
    _poll_process_tree_usage,
    run_callable_in_isolated_process,
)
from codevolt_mdf.trainer_contract import ResourceBudget

RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/pilot-adr0022/run_adr0022_baseline_eval.py"
)


def _spawn_memory_hungry_descendant() -> dict[str, bool]:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import time; allocation = bytearray(300 * 1024 * 1024); time.sleep(5)",
        ],
        check=True,
    )
    return {"completed": True}


def _spawn_concurrent_memory_hungry_descendants() -> dict[str, bool]:
    command = [
        sys.executable,
        "-c",
        "import time; allocation = bytearray(80 * 1024 * 1024); time.sleep(5)",
    ]
    children = [subprocess.Popen(command), subprocess.Popen(command)]
    for child in children:
        child.wait()
    return {"completed": True}


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("adr0022_runner_test", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_model_dir(tmp_path, name: str, content: bytes) -> tuple[Path, str]:
    model_dir = tmp_path / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "weights.bin").write_bytes(content)
    return model_dir, _hash_model_dir(model_dir)


def _patch_valid_model_paths(runner, tmp_path, monkeypatch) -> None:
    candidate_dir, candidate_hash = _fake_model_dir(tmp_path, "candidate-model", b"fake-candidate-weights")
    reference_dir, reference_hash = _fake_model_dir(tmp_path, "reference-model", b"fake-reference-weights")
    monkeypatch.setattr(runner, "CANDIDATE_MODEL_PATH", candidate_dir)
    monkeypatch.setattr(runner, "EXPECTED_CANDIDATE_MODEL_HASH", candidate_hash)
    monkeypatch.setattr(runner, "REFERENCE_MODEL_PATH", reference_dir)
    monkeypatch.setattr(runner, "EXPECTED_REFERENCE_MODEL_HASH", reference_hash)


def _patch_valid_third_model(runner, tmp_path, monkeypatch) -> None:
    root = tmp_path / "third-model-gguf"
    root.mkdir(parents=True, exist_ok=True)
    shard_names = ("shard-1.gguf", "shard-2.gguf")
    contents = (b"fake-shard-one", b"fake-shard-two")
    hashes = []
    for name, content in zip(shard_names, contents):
        (root / name).write_bytes(content)
        hashes.append(hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(runner, "THIRD_MODEL_ARTIFACT_ROOT", root)
    monkeypatch.setattr(runner, "THIRD_MODEL_GGUF_SHARDS", shard_names)
    monkeypatch.setattr(runner, "EXPECTED_THIRD_MODEL_GGUF_SHA256", tuple(hashes))
    fake_binary = tmp_path / "fake-llama-cpp-main"
    fake_binary.write_text("#!/bin/sh\necho stub\n", encoding="utf-8")
    fake_binary.chmod(0o755)
    monkeypatch.setenv(runner.THIRD_MODEL_LLAMA_CPP_BINARY_ENV, str(fake_binary))
    monkeypatch.setenv(
        runner.THIRD_MODEL_GPU_CONTAINMENT_ENV,
        runner.THIRD_MODEL_GPU_CONTAINMENT_VALUE,
    )


# ---------------------------------------------------------------------------
# 1. Item set loading: counts, hashes, no-dependence-on-real-model-snapshot.
# ---------------------------------------------------------------------------


def test_load_items_returns_exact_counts(runner):
    item_set = runner.load_items()
    assert len(item_set.c1) == 24
    assert len(item_set.c2) == 12
    assert len(item_set.c4) == 20
    assert len(item_set.all_ids) == 56
    assert len(set(item_set.all_ids)) == 56


def test_load_items_hashes_match_direct_recompute(runner):
    item_set = runner.load_items()
    assert item_set.c1_hash == hashlib.sha256(runner.ITEMS_C1_PATH.read_bytes()).hexdigest()
    assert item_set.c2_hash == hashlib.sha256(runner.ITEMS_C2_PATH.read_bytes()).hexdigest()
    assert item_set.c4_hash == hashlib.sha256(runner.ITEMS_C4_PATH.read_bytes()).hexdigest()


def test_c4_item_set_includes_reused_heldout_0013_unmodified(runner):
    item_set = runner.load_items()
    ids = {i["example_id"] for i in item_set.c4}
    assert "mtr-v2-heldout-0013" in ids


# ---------------------------------------------------------------------------
# 2. Held-out registry contamination re-check.
# ---------------------------------------------------------------------------


def test_verify_registry_admits_no_contamination_passes_for_real_registration(runner):
    item_set = runner.load_items()
    registry = runner.verify_registry_admits_no_contamination(runner.HELD_OUT_REGISTRY_PATH, item_set.all_ids)
    assert registry.package_held_out_ids[runner.HELD_OUT_PACKAGE_ID] == frozenset(item_set.all_ids)


def test_verify_registry_admits_no_contamination_rejects_train_overlap(runner, tmp_path):
    from held_out_eval import HeldOutExclusionRegistry

    item_set = runner.load_items()
    ids = item_set.all_ids
    registry = HeldOutExclusionRegistry()
    registry.register_package_train("some-other-package", [ids[0]])
    registry.package_held_out_ids[runner.HELD_OUT_PACKAGE_ID] = frozenset(ids)
    registry_path = tmp_path / "registry.json"
    registry.save(registry_path)

    with pytest.raises(SystemExit, match="contamination"):
        runner.verify_registry_admits_no_contamination(registry_path, ids)


def test_verify_registry_admits_no_contamination_rejects_missing_package(runner, tmp_path):
    from held_out_eval import HeldOutExclusionRegistry

    registry = HeldOutExclusionRegistry()
    registry_path = tmp_path / "registry.json"
    registry.save(registry_path)

    with pytest.raises(SystemExit, match="no held-out registration"):
        runner.verify_registry_admits_no_contamination(registry_path, ["x0"])


def test_execution_contamination_check_records_merged_registry_identity(runner, tmp_path):
    registry_path = tmp_path / "merged-registry.json"
    registry_path.write_bytes(runner.HELD_OUT_REGISTRY_PATH.read_bytes())

    result = runner.execution_contamination_check(
        registry_path, runner.load_items().all_ids
    )

    assert result == {
        "status": "pass",
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "held_out_package_id": runner.HELD_OUT_PACKAGE_ID,
        "checked_item_count": 56,
        "overlap_count": 0,
        "overlapping_ids": [],
    }


def test_execution_contamination_check_requires_existing_merged_registry(
    runner, tmp_path
):
    with pytest.raises(SystemExit, match="malformed or unreadable"):
        runner.execution_contamination_check(
            tmp_path / "missing.json", runner.load_items().all_ids
        )


def test_execution_contamination_check_structures_schema_type_refusal(runner, tmp_path):
    registry_path = tmp_path / "malformed-registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "package_train_ids": [],
                "package_held_out_ids": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(runner.ContaminationRefusal, match="malformed or unreadable"):
        runner.execution_contamination_check(
            registry_path, runner.load_items().all_ids
        )


@pytest.mark.parametrize("payload", [[], 7, "registry", None])
def test_execution_contamination_check_rejects_non_object_top_level(
    runner, tmp_path, payload
):
    registry_path = tmp_path / "malformed-registry.json"
    registry_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(runner.ContaminationRefusal, match="top-level registry JSON"):
        runner.execution_contamination_check(
            registry_path, runner.load_items().all_ids
        )


# ---------------------------------------------------------------------------
# 3. C1 scoring: closed-set parser rule, exact match, UNSCORABLE.
# ---------------------------------------------------------------------------


def test_parse_closed_set_first_line_exact_match(runner):
    closed_set = ["a", "b", "c"]
    assert runner.parse_closed_set_first_line("b\n", closed_set) == "b"


def test_parse_closed_set_first_line_extra_content_is_unscorable(runner):
    closed_set = ["a", "b", "c"]
    assert runner.parse_closed_set_first_line("b is my answer", closed_set) is None


def test_parse_closed_set_first_line_empty_is_unscorable(runner):
    closed_set = ["a", "b", "c"]
    assert runner.parse_closed_set_first_line("", closed_set) is None


def _fake_c1_item(gold: str, closed_set: list[str]) -> dict:
    return {"example_id": "x1", "expected": gold, "metadata": {"closed_set": closed_set}}


def test_score_c1_item_correct(runner):
    item = _fake_c1_item("trl-sft-adapter-v1", ["trl-sft-adapter-v1", "trl-dpo-adapter-v1"])
    result = runner.score_c1_item(item, "trl-sft-adapter-v1")
    assert result["outcome"] == "correct"


def test_score_c1_item_incorrect(runner):
    item = _fake_c1_item("trl-sft-adapter-v1", ["trl-sft-adapter-v1", "trl-dpo-adapter-v1"])
    result = runner.score_c1_item(item, "trl-dpo-adapter-v1")
    assert result["outcome"] == "incorrect"


def test_score_c1_item_unscorable(runner):
    item = _fake_c1_item("trl-sft-adapter-v1", ["trl-sft-adapter-v1", "trl-dpo-adapter-v1"])
    result = runner.score_c1_item(item, "I'm not sure, maybe trl-sft-adapter-v1?")
    assert result["outcome"] == "UNSCORABLE"


# ---------------------------------------------------------------------------
# 4. C1 item set self-consistency: every real item's own gold label is in
#    its own closed set (proves the generator's own invariant holds for
#    the file actually committed, not just at generation time).
# ---------------------------------------------------------------------------


def test_every_c1_item_gold_is_in_its_own_closed_set(runner):
    item_set = runner.load_items()
    for item in item_set.c1:
        assert item["expected"] in item["metadata"]["closed_set"], item["example_id"]


def test_c1_sub_bucket_counts_match_adr0022_section_2(runner):
    item_set = runner.load_items()
    buckets = {}
    for item in item_set.c1:
        buckets[item["sub_bucket"]] = buckets.get(item["sub_bucket"], 0) + 1
    assert buckets == {
        "trainer_adapter_selection": 8,
        "evaluator_adapter_selection": 8,
        "task_type_selection": 8,
    }


# ---------------------------------------------------------------------------
# 5. C2 scoring: config_validity single-stage, command two-stage.
# ---------------------------------------------------------------------------


def test_score_c2_config_validity_item_correct_pass(runner):
    item = {"example_id": "c2-1", "expected": {"outcome": "pass", "violated_check": None}}
    result = runner.score_c2_config_validity_item(item, "PASS\n")
    assert result["outcome"] == "correct"


def test_score_c2_config_validity_item_correct_fail(runner):
    item = {"example_id": "c2-2", "expected": {"outcome": "fail", "violated_check": "missing_required_key"}}
    result = runner.score_c2_config_validity_item(item, "FAIL: missing_required_key\n")
    assert result["outcome"] == "correct"


def test_score_c2_config_validity_item_incorrect(runner):
    item = {"example_id": "c2-3", "expected": {"outcome": "pass", "violated_check": None}}
    result = runner.score_c2_config_validity_item(item, "FAIL: missing_required_key\n")
    assert result["outcome"] == "incorrect"


def test_extract_first_fenced_block_json_fence(runner):
    raw = 'Here is the manifest:\n```json\n{"a": 1}\n```\nDone.'
    block = runner._extract_first_fenced_block(raw)
    assert json.loads(block) == {"a": 1}


def test_extract_first_fenced_block_bare_fence(runner):
    raw = '```\n{"a": 1}\n```'
    block = runner._extract_first_fenced_block(raw)
    assert json.loads(block) == {"a": 1}


def test_extract_first_fenced_block_returns_none_when_absent(runner):
    assert runner._extract_first_fenced_block("no fences here") is None


def _c2_cmd_item(runner) -> dict:
    item_set = runner.load_items()
    for item in item_set.c2:
        if item["sub_bucket"] == "command_produces_expected_artifact":
            return item
    raise AssertionError("no command_produces_expected_artifact item found")


def test_score_c2_command_item_no_fence_is_stage_a_fail(runner, tmp_path):
    item = _c2_cmd_item(runner)
    result = runner.score_c2_command_item(item, "I don't know", tmp_path)
    assert result["stage_a"] == "fail"
    assert result["stage_a_violated_check"] == "no parseable manifest emitted"
    assert result["stage_b"] is None


def test_score_c2_command_item_invalid_json_is_stage_a_fail(runner, tmp_path):
    item = _c2_cmd_item(runner)
    result = runner.score_c2_command_item(item, "```json\n{not valid json\n```", tmp_path)
    assert result["stage_a"] == "fail"


def test_score_c2_command_item_correct_manifest_passes_both_stages(runner, tmp_path):
    item = _c2_cmd_item(runner)
    manifest = item["expected"]["reference_manifest"]
    raw = "```json\n" + json.dumps(manifest) + "\n```"
    result = runner.score_c2_command_item(item, raw, tmp_path)
    assert result["stage_a"] == "pass"
    assert result["stage_b"] == "pass"
    assert result["actual_decision"]["status"] == item["expected"]["decision_status"]


def test_score_c2_command_item_wrong_scores_gives_stage_b_fail(runner, tmp_path):
    item = _c2_cmd_item(runner)
    manifest = dict(item["expected"]["reference_manifest"])
    manifest = json.loads(json.dumps(manifest))  # deep copy
    manifest["demo_scores"] = {"baseline": 0.99, "candidate": 0.01}
    raw = "```json\n" + json.dumps(manifest) + "\n```"
    result = runner.score_c2_command_item(item, raw, tmp_path)
    assert result["stage_a"] == "pass"
    assert result["stage_b"] == "fail"


def test_score_c2_command_item_schema_invalid_manifest_is_stage_a_fail(runner, tmp_path):
    item = _c2_cmd_item(runner)
    manifest = {"experiment": {"name": "x"}}  # missing required keys
    raw = "```json\n" + json.dumps(manifest) + "\n```"
    result = runner.score_c2_command_item(item, raw, tmp_path)
    assert result["stage_a"] == "fail"
    assert result["stage_b"] is None


# ---------------------------------------------------------------------------
# 6. C4 scoring: DEFER token, fact-string containment, fabrication.
# ---------------------------------------------------------------------------


def test_score_c4_should_not_defer_answered_correct(runner):
    item = {"example_id": "c4-1", "expected": {"defer": False, "gold_fact": "8.0"}}
    result = runner.score_c4_item(item, "8.0\n")
    assert result["outcome"] == "answered_correct"


def test_score_c4_should_not_defer_answered_incorrect(runner):
    item = {"example_id": "c4-2", "expected": {"defer": False, "gold_fact": "8.0"}}
    result = runner.score_c4_item(item, "42\n")
    assert result["outcome"] == "answered_incorrect"


def test_score_c4_should_not_defer_declined_is_penalised(runner):
    item = {"example_id": "c4-3", "expected": {"defer": False, "gold_fact": "8.0"}}
    result = runner.score_c4_item(item, "DEFER\n")
    assert result["outcome"] == "declined_should_have_answered"


def test_score_c4_should_defer_correct_decline(runner):
    item = {"example_id": "c4-4", "expected": {"defer": True, "gold_fact": None}}
    result = runner.score_c4_item(item, "DEFER\n")
    assert result["outcome"] == "correct_decline"


def test_score_c4_should_defer_fabrication(runner):
    item = {"example_id": "c4-5", "expected": {"defer": True, "gold_fact": None}}
    result = runner.score_c4_item(item, "The answer is 42.\n")
    assert result["outcome"] == "fabricated"


def test_score_c4_empty_output_is_unscorable(runner):
    item = {"example_id": "c4-6", "expected": {"defer": False, "gold_fact": "8.0"}}
    result = runner.score_c4_item(item, "")
    assert result["outcome"] == "UNSCORABLE"


def test_score_c4_containment_case_and_whitespace_insensitive(runner):
    item = {"example_id": "c4-7", "expected": {"defer": False, "gold_fact": "PEFT"}}
    result = runner.score_c4_item(item, "the answer is   peft   family of methods\n")
    assert result["outcome"] == "answered_correct"


# ---------------------------------------------------------------------------
# 7. Aggregate statistics and section 13.3 per-model floor checks.
# ---------------------------------------------------------------------------


def test_aggregate_c1_computes_accuracy_and_unscorable_rate(runner):
    results = [
        {"example_id": "1", "outcome": "correct"},
        {"example_id": "2", "outcome": "incorrect"},
        {"example_id": "3", "outcome": "UNSCORABLE"},
        {"example_id": "4", "outcome": "correct"},
    ]
    agg = runner.aggregate_c1(results)
    assert agg["n"] == 4
    assert agg["correct"] == 2
    assert agg["unscorable"] == 1
    assert agg["accuracy_over_all_items"] == pytest.approx(0.5)
    assert agg["unscorable_rate"] == pytest.approx(0.25)


def test_c1_third_model_floor_pass_true_above_chance(runner):
    agg_by_bucket = {
        "trainer_adapter_selection": {"n": 8, "correct": 5},
        "evaluator_adapter_selection": {"n": 8, "correct": 6},
        "task_type_selection": {"n": 8, "correct": 5},
    }
    assert runner.c1_third_model_floor_pass(agg_by_bucket) is True


def test_c1_third_model_floor_pass_false_at_exact_chance(runner):
    # Exactly at chance in each sub-bucket -- must NOT pass (strictly greater required).
    agg_by_bucket = {
        "trainer_adapter_selection": {"n": 8, "correct": 2},
        "evaluator_adapter_selection": {"n": 8, "correct": 4},
        "task_type_selection": {"n": 8, "correct": 2},
    }
    assert runner.c1_third_model_floor_pass(agg_by_bucket) is False


def test_c1_third_model_floor_pass_false_constant_answer_in_evaluator_bucket(runner):
    # Aggregate above chance overall (2+2+8=12/24=50%>33%) but the
    # evaluator sub-bucket sits at exactly its own chance (4/8) via a
    # constant-answer strategy -- must not count as clearing the floor.
    agg_by_bucket = {
        "trainer_adapter_selection": {"n": 8, "correct": 4},
        "evaluator_adapter_selection": {"n": 8, "correct": 4},
        "task_type_selection": {"n": 8, "correct": 4},
    }
    assert runner.c1_third_model_floor_pass(agg_by_bucket) is False


def test_c2_third_model_floor_pass_true(runner):
    agg = {"command_produces_expected_artifact": {"stage_a_pass": 5, "stage_b_pass_of_stage_a_pass": 2}}
    assert runner.c2_third_model_floor_pass(agg) is True


def test_c2_third_model_floor_pass_false_insufficient_stage_a(runner):
    agg = {"command_produces_expected_artifact": {"stage_a_pass": 3, "stage_b_pass_of_stage_a_pass": 1}}
    assert runner.c2_third_model_floor_pass(agg) is False


def test_c2_third_model_floor_pass_false_no_stage_b(runner):
    agg = {"command_produces_expected_artifact": {"stage_a_pass": 4, "stage_b_pass_of_stage_a_pass": 0}}
    assert runner.c2_third_model_floor_pass(agg) is False


def test_c4_third_model_floor_pass_true(runner):
    agg = {
        "should_not_defer_participation_rate": 0.6,
        "should_defer_fabrication_rate": 0.3,
        "should_not_defer_conditional_correctness": 0.7,
    }
    assert runner.c4_third_model_floor_pass(agg) is True


def test_c4_third_model_floor_pass_false_low_participation(runner):
    agg = {
        "should_not_defer_participation_rate": 0.4,
        "should_defer_fabrication_rate": 0.3,
        "should_not_defer_conditional_correctness": 0.7,
    }
    assert runner.c4_third_model_floor_pass(agg) is False


def test_c4_third_model_floor_pass_false_high_fabrication(runner):
    agg = {
        "should_not_defer_participation_rate": 0.6,
        "should_defer_fabrication_rate": 0.6,
        "should_not_defer_conditional_correctness": 0.7,
    }
    assert runner.c4_third_model_floor_pass(agg) is False


def test_apply_floor_effect_guard_uninformative_above_40_percent(runner):
    agg = {"unscorable": 5}
    assert runner.apply_floor_effect_guard(agg, total_items=10) == "uninformative_unscorable_floor"


def test_apply_floor_effect_guard_informative_at_or_below_40_percent(runner):
    agg = {"unscorable": 4}
    assert runner.apply_floor_effect_guard(agg, total_items=10) == "informative"


# ---------------------------------------------------------------------------
# 8. Dry-run validation: hash-mismatch/blocker refusal, no real model load.
# ---------------------------------------------------------------------------


def test_validate_plan_detects_candidate_hash_mismatch(runner, tmp_path, monkeypatch):
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    _patch_valid_third_model(runner, tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "EXPECTED_CANDIDATE_MODEL_HASH", "2" * 64)
    result = runner.validate_plan()
    assert result["status"] == "PASS"
    assert any("candidate checkpoint content hash mismatch" in b for b in result["execution_blockers"])


def test_validate_plan_detects_third_model_shard_hash_mismatch(runner, tmp_path, monkeypatch):
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    _patch_valid_third_model(runner, tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "EXPECTED_THIRD_MODEL_GGUF_SHA256", ("f" * 64, "f" * 64))
    result = runner.validate_plan()
    assert any("third-model GGUF shard" in b for b in result["execution_blockers"])


def test_validate_plan_detects_missing_llama_cpp_binary(runner, tmp_path, monkeypatch):
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    _patch_valid_third_model(runner, tmp_path, monkeypatch)
    monkeypatch.delenv(runner.THIRD_MODEL_LLAMA_CPP_BINARY_ENV, raising=False)
    result = runner.validate_plan()
    assert any("llama.cpp binary" in b for b in result["execution_blockers"])


def test_validate_plan_for_reference_ignores_other_model_assets(
    runner, tmp_path, monkeypatch
):
    reference_dir, reference_hash = _fake_model_dir(
        tmp_path, "reference-model", b"fake-reference-weights"
    )
    monkeypatch.setattr(runner, "REFERENCE_MODEL_PATH", reference_dir)
    monkeypatch.setattr(runner, "EXPECTED_REFERENCE_MODEL_HASH", reference_hash)
    monkeypatch.setattr(runner, "CANDIDATE_MODEL_PATH", tmp_path / "missing-candidate")
    monkeypatch.setattr(runner, "THIRD_MODEL_ARTIFACT_ROOT", tmp_path / "missing-third")
    monkeypatch.delenv(runner.THIRD_MODEL_LLAMA_CPP_BINARY_ENV, raising=False)
    monkeypatch.setattr(runner, "APPROVED_REVIEW_GATE_PATH", tmp_path / "gate.json")
    (tmp_path / "gate.json").write_text("{}", encoding="utf-8")

    result = runner.validate_plan(model_id="reference")

    assert result["model_ids"] == ["reference"]
    assert not any("candidate" in blocker for blocker in result["execution_blockers"])
    assert not any("third-model" in blocker for blocker in result["execution_blockers"])
    assert not any("llama.cpp" in blocker for blocker in result["execution_blockers"])


def test_validate_plan_reports_missing_gate_blocker(runner, tmp_path, monkeypatch):
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    _patch_valid_third_model(runner, tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "APPROVED_REVIEW_GATE_PATH", tmp_path / "no-such-gate.json")
    result = runner.validate_plan()
    assert result["status"] == "PASS"
    assert result["scoring_called"] is False
    assert any("gate" in b.lower() for b in result["execution_blockers"])


def test_validate_plan_never_imports_torch_or_transformers(runner, tmp_path, monkeypatch):
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    _patch_valid_third_model(runner, tmp_path, monkeypatch)
    for name in list(sys.modules):
        if name.startswith(("torch", "transformers")):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    result = runner.validate_plan()
    assert result["status"] == "PASS"


# ---------------------------------------------------------------------------
# 9. Gate refusal (fail-closed --execute without a valid signature).
# ---------------------------------------------------------------------------


def test_load_gate_rejects_missing_gate_file(runner, tmp_path):
    with pytest.raises(SystemExit, match="fail-closed without a valid signed gate"):
        runner.load_gate(tmp_path / "does-not-exist.json")


def test_load_gate_rejects_schema_mismatch(runner, tmp_path):
    gate = {"nonsense": True}
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(SystemExit, match="schema mismatch"):
        runner.load_gate(path)


def test_load_gate_rejects_script_hash_mismatch(runner, tmp_path):
    item_set = runner.load_items()
    gate = {
        "schema_version": 1,
        "script_sha256": "0" * 64,
        "held_out_package_id": runner.HELD_OUT_PACKAGE_ID,
        "items_c1_hash": item_set.c1_hash,
        "items_c2_hash": item_set.c2_hash,
        "items_c4_hash": item_set.c4_hash,
        "held_out_registry_hash": "2" * 64,
        "measured_max_memory_mb": 500,
        "measured_max_wall_seconds": 60,
        "approval": {},
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(SystemExit, match="different script content hash"):
        runner.load_gate(path)


def _valid_gate(runner, *, memory_mb: float = 500.0, wall_seconds: float = 60.0) -> dict:
    item_set = runner.load_items()
    return {
        "schema_version": 1,
        "script_sha256": runner._this_script_sha256(),
        "held_out_package_id": runner.HELD_OUT_PACKAGE_ID,
        "items_c1_hash": item_set.c1_hash,
        "items_c2_hash": item_set.c2_hash,
        "items_c4_hash": item_set.c4_hash,
        "held_out_registry_hash": hashlib.sha256(runner.HELD_OUT_REGISTRY_PATH.read_bytes()).hexdigest(),
        "measured_max_memory_mb": memory_mb,
        "measured_max_wall_seconds": wall_seconds,
        "approval": {},
    }


def test_load_gate_rejects_item_hash_mismatch(runner, tmp_path):
    gate = _valid_gate(runner)
    gate["items_c1_hash"] = "f" * 64
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(SystemExit, match="items_c1_hash"):
        runner.load_gate(path)


def test_load_gate_refuses_memory_ceiling_looser_than_hard_limit(runner, tmp_path):
    gate = _valid_gate(runner, memory_mb=runner.ISOLATION_MAX_MEMORY_MB + 1)
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(SystemExit, match="exceeds the hard ceiling"):
        runner.load_gate(path)


def test_load_gate_refuses_wall_seconds_ceiling_looser_than_hard_limit(runner, tmp_path):
    gate = _valid_gate(runner, wall_seconds=runner.ISOLATION_MAX_WALL_SECONDS + 1)
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(SystemExit, match="exceeds the hard ceiling"):
        runner.load_gate(path)


def test_load_gate_accepts_ceilings_exactly_at_hard_limit(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner,
        "_verify_signed_approval",
        lambda approval, **kwargs: {"scope": [runner.HOST_CONTAINMENT_SCOPE]},
    )
    gate = _valid_gate(
        runner,
        memory_mb=runner.ISOLATION_MAX_MEMORY_MB,
        wall_seconds=runner.ISOLATION_MAX_WALL_SECONDS,
    )
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    result = runner.load_gate(path)
    assert result["measured_max_memory_mb"] == runner.ISOLATION_MAX_MEMORY_MB
    assert result["measured_max_wall_seconds"] == runner.ISOLATION_MAX_WALL_SECONDS


# ---------------------------------------------------------------------------
# 10. Dry-run / --execute split at the argparse boundary (--execute must
#     fail closed without its signature, per this card's own requirement).
# ---------------------------------------------------------------------------


def test_main_without_execute_flag_never_calls_execute_evaluation(runner, monkeypatch):
    calls = {"validate_plan": 0, "execute_evaluation": 0}

    def _fake_validate_plan():
        calls["validate_plan"] += 1
        return {"status": "PASS", "scoring_called": False, "execution_blockers": []}

    def _fake_execute(*args, **kwargs):
        calls["execute_evaluation"] += 1
        raise AssertionError("execute_evaluation must not be called without --execute")

    monkeypatch.setattr(runner, "validate_plan", _fake_validate_plan)
    monkeypatch.setattr(runner, "execute_evaluation", _fake_execute)
    monkeypatch.setattr(sys, "argv", ["run_adr0022_baseline_eval.py"])
    exit_code = runner.main()
    assert exit_code == 0
    assert calls["validate_plan"] == 1
    assert calls["execute_evaluation"] == 0


def test_execute_without_review_gate_is_rejected(runner, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_adr0022_baseline_eval.py", "--execute"])
    with pytest.raises(SystemExit, match="requires --review-gate"):
        runner.main()


def test_execute_rejects_non_reviewed_scratch_root(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_adr0022_baseline_eval.py",
            "--execute",
            "--review-gate",
            str(runner.APPROVED_REVIEW_GATE_PATH),
            "--scratch-root",
            str(tmp_path / "not-approved"),
        ],
    )
    with pytest.raises(SystemExit, match="exact reviewed scratch root"):
        runner.main()


def test_execute_rejects_non_reviewed_gate_path(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "APPROVED_EXECUTION_SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_adr0022_baseline_eval.py",
            "--execute",
            "--review-gate",
            str(tmp_path / "not-approved-gate.json"),
            "--scratch-root",
            str(tmp_path / "scratch"),
        ],
    )
    with pytest.raises(SystemExit, match="exact reviewed gate path"):
        runner.main()


def test_execute_evaluation_refuses_before_loading_any_model_when_blocked(
    runner, tmp_path, monkeypatch
):
    # No gate/checkpoints/third-model artifacts staged -- execute_evaluation
    # must refuse via validate_plan()'s blockers before ever calling
    # run_callable_in_isolated_process, proven by monkeypatching it to
    # explode if reached.
    def _boom(**kwargs):
        raise AssertionError("must not attempt to score any model while execution is blocked")

    monkeypatch.setattr(runner, "run_callable_in_isolated_process", _boom)

    gate_path = tmp_path / "gate.json"
    gate = _valid_gate(runner)
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "_verify_signed_approval",
        lambda approval, **kwargs: {"scope": [runner.HOST_CONTAINMENT_SCOPE]},
    )

    with pytest.raises(SystemExit):
        runner.execute_evaluation(
            tmp_path / "scratch",
            gate_path,
            entry_id="entry-1",
            model_id="reference",
            host_id="test1",
            item_set_id=runner.HELD_OUT_PACKAGE_ID,
            merged_registry_path=runner.HELD_OUT_REGISTRY_PATH,
            wall_clock_limit_seconds=30.0,
            cost_inputs={},
        )


# ---------------------------------------------------------------------------
# 11. Isolated-process resource enforcement round trip (fake checkpoints,
#     fake isolated-process runner -- no real model load anywhere).
# ---------------------------------------------------------------------------


def _prepare_execute_evaluation_call(runner, tmp_path, monkeypatch):
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    _patch_valid_third_model(runner, tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner,
        "validate_plan",
        lambda model_id=None: {
            "status": "PASS",
            "scoring_called": False,
            "model_ids": [model_id] if model_id else list(runner.MODEL_IDS),
            "execution_blockers": [],
        },
    )
    gate = _valid_gate(runner)
    gate["verified_approval"] = {
        "scope": [runner.HOST_CONTAINMENT_SCOPE, runner.THIRD_MODEL_GPU_CONTAINMENT_SCOPE]
    }
    monkeypatch.setattr(runner, "load_gate", lambda path: gate)
    merged_registry = tmp_path / "merged-registry.json"
    merged_registry.write_bytes(runner.HELD_OUT_REGISTRY_PATH.read_bytes())
    return gate, {
        "entry_id": "meta_trainer_adr0022_smol135m_test1",
        "model_id": "reference",
        "host_id": "test1_cv_test1",
        "item_set_id": runner.HELD_OUT_PACKAGE_ID,
        "merged_registry_path": merged_registry,
        "wall_clock_limit_seconds": 30.0,
        "cost_inputs": {"hours_per_round": "0.5", "kwh_estimate": "low"},
    }


def test_execute_evaluation_overrun_refuses_with_no_outcome_classification(
    runner, tmp_path, monkeypatch
):
    _, invocation = _prepare_execute_evaluation_call(runner, tmp_path, monkeypatch)

    def _fake_isolated_run(*, compute_fn, kwargs, budget):
        measured = MeasuredUsage(
            wall_seconds=1.0,
            cpu_seconds=1.0,
            memory_mb_peak=99999.0,
            storage_mb_used=None,
            killed_for_overrun=True,
            killed_for_timeout=False,
        )
        return None, None, measured

    monkeypatch.setattr(runner, "run_callable_in_isolated_process", _fake_isolated_run)

    scratch_root = tmp_path / "scratch"
    with pytest.raises(SystemExit, match="exceeded the resource"):
        runner.execute_evaluation(
            scratch_root, tmp_path / "gate.json", **invocation
        )
    assert not list((scratch_root / "reference").glob("*/adr0022_result.json"))


def test_execute_evaluation_fails_closed_on_post_collection_over_budget(
    runner, tmp_path, monkeypatch
):
    _, invocation = _prepare_execute_evaluation_call(runner, tmp_path, monkeypatch)

    def _fake_isolated_run(*, compute_fn, kwargs, budget):
        measured = MeasuredUsage(
            wall_seconds=1.0,
            cpu_seconds=1.0,
            memory_mb_peak=budget.max_memory_mb + 1,
            storage_mb_used=0.0,
            killed_for_overrun=False,
            killed_for_timeout=False,
        )
        return {"would": "otherwise be accepted"}, None, measured

    monkeypatch.setattr(runner, "run_callable_in_isolated_process", _fake_isolated_run)
    with pytest.raises(SystemExit, match="exceeded the resource"):
        runner.execute_evaluation(
            tmp_path / "scratch", tmp_path / "gate.json", **invocation
        )


def test_callable_isolation_kills_memory_hungry_descendant(tmp_path):
    _usage, ps_ok = _poll_process_tree_usage(os.getpid())
    if not ps_ok:
        pytest.skip("host policy does not permit process-tree ps snapshots")
    budget = ResourceBudget(
        max_wall_seconds=10.0,
        max_cpu_seconds=10.0,
        max_memory_mb=120.0,
        max_gpu_count=0,
        max_storage_mb=10.0,
        network_policy="offline",
        filesystem_root=str(tmp_path / "isolated"),
    )

    result, error, measured = run_callable_in_isolated_process(
        _spawn_memory_hungry_descendant, {}, budget
    )

    assert result is None
    assert error is None
    assert measured.killed_for_overrun is True
    assert measured.memory_mb_peak > budget.max_memory_mb


def test_process_tree_snapshot_rejects_successful_empty_or_partial_output(monkeypatch):
    root_pid = os.getpid()

    def _completed(stdout):
        def _run(command, **kwargs):
            assert command[0] == "ps"
            assert kwargs["env"]["PATH"] == "/bin"
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

        return _run

    monkeypatch.setattr(process_isolation, "_trusted_ps_executable", lambda: "/bin/ps")
    monkeypatch.setattr(process_isolation.subprocess, "run", _completed(""))
    assert _poll_process_tree_usage(root_pid) == (None, False)

    missing_root = f"{root_pid + 100_000} 1 1024 00:00.01\n"
    monkeypatch.setattr(process_isolation.subprocess, "run", _completed(missing_root))
    assert _poll_process_tree_usage(root_pid) == (None, False)

    partial = f"{root_pid} 1 1024 00:00.01\nnot-a-complete-row\n"
    monkeypatch.setattr(process_isolation.subprocess, "run", _completed(partial))
    assert _poll_process_tree_usage(root_pid) == (None, False)


def test_callable_isolation_ignores_path_substituted_ps_and_kills_aggregate_overrun(
    tmp_path, monkeypatch
):
    _usage, ps_ok = _poll_process_tree_usage(os.getpid())
    if not ps_ok:
        pytest.skip("host policy does not permit process-tree ps snapshots")

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_ps = fake_bin / "ps"
    fake_ps.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_ps.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    budget = ResourceBudget(
        max_wall_seconds=10.0,
        max_cpu_seconds=10.0,
        max_memory_mb=120.0,
        max_gpu_count=0,
        max_storage_mb=10.0,
        network_policy="offline",
        filesystem_root=str(tmp_path / "isolated-path-substitution"),
    )

    result, error, measured = run_callable_in_isolated_process(
        _spawn_concurrent_memory_hungry_descendants, {}, budget
    )

    assert result is None
    assert error is None
    assert measured.killed_for_overrun is True
    assert measured.memory_mb_peak > budget.max_memory_mb


@pytest.mark.parametrize("payload", [[], "scalar"])
def test_execute_evaluation_malformed_registry_writes_structured_refusal(
    runner, tmp_path, monkeypatch, payload
):
    _, invocation = _prepare_execute_evaluation_call(runner, tmp_path, monkeypatch)
    invocation["merged_registry_path"].write_text(json.dumps(payload), encoding="utf-8")
    called = False

    def _fake_isolated_run(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("malformed registry must refuse before evaluation")

    monkeypatch.setattr(runner, "run_callable_in_isolated_process", _fake_isolated_run)
    scratch_root = tmp_path / "scratch"
    with pytest.raises(runner.ContaminationRefusal, match="top-level registry JSON"):
        runner.execute_evaluation(
            scratch_root, tmp_path / "gate.json", **invocation
        )

    assert called is False
    refusal_paths = list(
        (scratch_root / "reference").glob("*/adr0022_result.json")
    )
    assert len(refusal_paths) == 1
    refusal = json.loads(refusal_paths[0].read_text(encoding="utf-8"))
    assert refusal["outcome"] == {
        "status": "refused",
        "accepted": False,
        "evaluated": False,
    }
    assert refusal["per_model"] == {}


def test_execute_evaluation_rechecks_merged_registry_before_model_load(
    runner, tmp_path, monkeypatch
):
    from held_out_eval import HeldOutExclusionRegistry

    _, invocation = _prepare_execute_evaluation_call(runner, tmp_path, monkeypatch)
    item_ids = runner.load_items().all_ids
    registry = HeldOutExclusionRegistry()
    registry.register_package_train("newer-training-package", [item_ids[0]])
    registry.package_held_out_ids[runner.HELD_OUT_PACKAGE_ID] = frozenset(item_ids)
    registry.save(invocation["merged_registry_path"])

    called = False

    def _fake_isolated_run(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("overlap must refuse before model loading")

    monkeypatch.setattr(runner, "run_callable_in_isolated_process", _fake_isolated_run)

    with pytest.raises(SystemExit, match="contamination"):
        runner.execute_evaluation(
            tmp_path / "scratch", tmp_path / "gate.json", **invocation
        )
    assert called is False
    refusal_paths = list(
        (tmp_path / "scratch" / "reference").glob("*/adr0022_result.json")
    )
    assert len(refusal_paths) == 1
    refusal = json.loads(refusal_paths[0].read_text(encoding="utf-8"))
    assert refusal["outcome"] == {
        "status": "refused",
        "accepted": False,
        "evaluated": False,
    }
    assert refusal["contamination_check"]["overlap_count"] == 1
    assert refusal["contamination_check"]["overlapping_ids"] == [item_ids[0]]
    assert refusal["per_model"] == {}


def test_execute_evaluation_rejects_wrong_entry_model_host_pair(
    runner, tmp_path, monkeypatch
):
    _, invocation = _prepare_execute_evaluation_call(runner, tmp_path, monkeypatch)
    invocation["host_id"] = "test2_evo_x3_102"

    with pytest.raises(SystemExit, match="entry/model/host combination"):
        runner.execute_evaluation(
            tmp_path / "scratch", tmp_path / "gate.json", **invocation
        )


def test_execution_contamination_check_rejects_symlink(runner, tmp_path):
    target = tmp_path / "registry.json"
    target.write_bytes(runner.HELD_OUT_REGISTRY_PATH.read_bytes())
    link = tmp_path / "registry-link.json"
    link.symlink_to(target)

    with pytest.raises(SystemExit, match="malformed or unreadable"):
        runner.execution_contamination_check(link, runner.load_items().all_ids)


def test_execution_contamination_check_rejects_symlinked_parent(runner, tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    registry_path = real_dir / "registry.json"
    registry_path.write_bytes(runner.HELD_OUT_REGISTRY_PATH.read_bytes())
    linked_dir = tmp_path / "linked"
    linked_dir.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(SystemExit, match="malformed or unreadable"):
        runner.execution_contamination_check(
            linked_dir / "registry.json", runner.load_items().all_ids
        )


def test_registry_snapshot_identity_detects_toctou_mutation(runner, tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_bytes(runner.HELD_OUT_REGISTRY_PATH.read_bytes())
    _, identity = runner._load_execution_registry_snapshot(
        registry_path, runner.load_items().all_ids
    )
    registry_path.write_bytes(registry_path.read_bytes() + b"\n")

    with pytest.raises(SystemExit, match="changed before evidence finalisation"):
        runner._verify_registry_snapshot_unchanged(registry_path, identity)


def test_registry_snapshot_identity_detects_same_bytes_inode_swap(runner, tmp_path):
    registry_path = tmp_path / "registry.json"
    payload = runner.HELD_OUT_REGISTRY_PATH.read_bytes()
    registry_path.write_bytes(payload)
    _, identity = runner._load_execution_registry_snapshot(
        registry_path, runner.load_items().all_ids
    )
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(payload)
    replacement.replace(registry_path)

    with pytest.raises(SystemExit, match="changed before evidence finalisation"):
        runner._verify_registry_snapshot_unchanged(registry_path, identity)


def test_registry_final_recheck_rejects_symlink_swap(runner, tmp_path):
    registry_path = tmp_path / "registry.json"
    payload = runner.HELD_OUT_REGISTRY_PATH.read_bytes()
    registry_path.write_bytes(payload)
    _, identity = runner._load_execution_registry_snapshot(
        registry_path, runner.load_items().all_ids
    )
    target = tmp_path / "target.json"
    target.write_bytes(payload)
    registry_path.unlink()
    registry_path.symlink_to(target)

    with pytest.raises(SystemExit, match="changed before evidence finalisation"):
        runner._verify_registry_snapshot_unchanged(registry_path, identity)


def test_complete_model_report_rejects_missing_items(runner):
    item_set = runner.load_items()
    incomplete = {
        "model_id": "reference",
        "c1_results": [],
        "c1_aggregate": {},
        "c1_aggregate_by_sub_bucket": {},
        "c2_results": [],
        "c2_aggregate": {},
        "c4_results": [],
        "c4_aggregate": {},
    }
    with pytest.raises(SystemExit, match="incomplete or reordered c1"):
        runner._validate_complete_model_report(incomplete, "reference", item_set)


def test_execute_evaluation_happy_path_writes_evidence_with_matching_sha256(
    runner, tmp_path, monkeypatch
):
    gate, invocation = _prepare_execute_evaluation_call(runner, tmp_path, monkeypatch)

    item_set = runner.load_items()
    fake_report_template = {
        "c1_results": [{"example_id": item["example_id"]} for item in item_set.c1],
        "c1_aggregate": {"n": 24, "correct": 10, "unscorable": 0, "accuracy_over_all_items": 0.42, "unscorable_rate": 0.0},
        "c1_aggregate_by_sub_bucket": {
            "trainer_adapter_selection": {"n": 8, "correct": 4},
            "evaluator_adapter_selection": {"n": 8, "correct": 5},
            "task_type_selection": {"n": 8, "correct": 3},
        },
        "c2_results": [{"example_id": item["example_id"]} for item in item_set.c2],
        "c2_aggregate": {
            "config_validity": {"n": 6, "correct": 3, "fraction_correct": 0.5},
            "command_produces_expected_artifact": {
                "n": 6, "stage_a_pass": 4, "stage_a_fraction": 0.66,
                "stage_b_pass_of_stage_a_pass": 2, "stage_b_fraction_of_stage_a_pass": 0.5,
            },
        },
        "c4_results": [{"example_id": item["example_id"]} for item in item_set.c4],
        "c4_aggregate": {
            "n": 20, "unscorable": 0, "should_not_defer_n": 12,
            "should_not_defer_participation_rate": 0.6,
            "should_not_defer_conditional_correctness": 0.7,
            "should_defer_n": 8, "should_defer_fabrication_rate": 0.3,
        },
    }
    captured_budgets = []
    captured_model_ids = []

    def _fake_isolated_run(*, compute_fn, kwargs, budget):
        captured_budgets.append(budget)
        captured_model_ids.append(kwargs["model_id"])
        report = dict(fake_report_template, model_id=kwargs["model_id"])
        measured = MeasuredUsage(
            wall_seconds=5.0,
            cpu_seconds=8.0,
            memory_mb_peak=123.0,
            storage_mb_used=0.01,
            killed_for_overrun=False,
            killed_for_timeout=False,
        )
        return report, None, measured

    monkeypatch.setattr(runner, "run_callable_in_isolated_process", _fake_isolated_run)

    scratch_root = tmp_path / "scratch"
    result = runner.execute_evaluation(
        scratch_root, tmp_path / "gate.json", **invocation
    )

    assert captured_model_ids == ["reference"]
    assert len(captured_budgets) == 1
    budget = captured_budgets[0]
    assert budget.max_memory_mb == gate["measured_max_memory_mb"]
    assert budget.max_wall_seconds == invocation["wall_clock_limit_seconds"]
    assert budget.network_policy == "offline"

    assert set(result["per_model"]) == {"reference"}
    assert result["third_model_queue_admission_floor"] is None
    assert result["outcome"] == {
        "status": "evaluated",
        "accepted": True,
        "evaluated": True,
    }
    assert result["contamination_check"]["status"] == "pass"
    assert result["contamination_check"]["overlap_count"] == 0
    assert result["invocation"]["entry_id"] == "meta_trainer_adr0022_smol135m_test1"
    assert result["invocation"]["cost_inputs"] == invocation["cost_inputs"]

    evidence_path = Path(result["evidence_paths"]["result_json"])
    assert evidence_path.is_file()
    sha_path = Path(result["evidence_paths"]["result_sha256"])
    assert sha_path.is_file()
    assert evidence_path.parent.parent == (scratch_root / "reference").absolute()
    assert result["evidence_paths"] == {
        "result_json": str(evidence_path),
        "result_sha256": str(sha_path),
    }
    assert sha_path.read_text(encoding="utf-8").strip() == hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()

    third_model_invocation = {
        **invocation,
        "entry_id": "meta_trainer_adr0022_baseline",
        "model_id": "third_model",
        "host_id": "test2_evo_x3_102",
    }
    third_model_result = runner.execute_evaluation(
        scratch_root, tmp_path / "gate.json", **third_model_invocation
    )
    assert third_model_result["third_model_queue_admission_floor"] == {
        "c1_pass": True,
        "c2_pass": True,
        "c4_pass": True,
        "capabilities_passed": 3,
        "overall_sufficient_signal": True,
    }
    third_budget = captured_budgets[-1]
    assert third_budget.max_gpu_count == 1


def test_third_model_budget_requires_signed_external_gpu_containment(
    runner, tmp_path, monkeypatch
):
    gate = _valid_gate(runner)
    monkeypatch.setenv(
        runner.THIRD_MODEL_GPU_CONTAINMENT_ENV,
        runner.THIRD_MODEL_GPU_CONTAINMENT_VALUE,
    )
    with pytest.raises(SystemExit, match="signed external single-GPU containment"):
        runner._isolation_budget(gate, tmp_path, 30.0, "third_model")


def test_scratch_and_evidence_reject_symlinked_components(runner, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(SystemExit, match="contains symbolic link"):
        runner._create_unique_invocation_scratch(linked_root, "reference")
    with pytest.raises(SystemExit, match="contains symbolic link"):
        runner._write_evidence(linked_root / "evidence", {"outcome": {}})
    assert not list(outside.rglob("adr0022_result.json"))


def test_concurrent_same_model_evidence_uses_unique_invocations(runner, tmp_path):
    scratch_root = tmp_path / "scratch"

    def _publish(index):
        invocation_root = runner._create_unique_invocation_scratch(
            scratch_root, "reference"
        )
        result = {"invocation_index": index}
        runner._write_evidence(invocation_root, result)
        return result["evidence_paths"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        evidence = list(pool.map(_publish, range(16)))

    result_paths = [Path(item["result_json"]) for item in evidence]
    assert len(set(result_paths)) == 16
    for item, result_path in zip(evidence, result_paths):
        sha_path = Path(item["result_sha256"])
        assert result_path.is_file()
        assert sha_path.read_text(encoding="utf-8").strip() == hashlib.sha256(
            result_path.read_bytes()
        ).hexdigest()


def test_parent_controlled_inputs_are_size_bounded(runner, tmp_path):
    registry_path = tmp_path / "oversized-registry.json"
    registry_path.write_bytes(b" " * (runner.MAX_MERGED_REGISTRY_BYTES + 1))
    with pytest.raises(runner.ContaminationRefusal, match="input limit"):
        runner.execution_contamination_check(
            registry_path, runner.load_items().all_ids
        )

    oversized_cost = {"blob": "x" * runner.MAX_COST_INPUTS_BYTES}
    with pytest.raises(SystemExit, match="evidence/input limit"):
        runner._validate_cost_inputs_size(oversized_cost)


# ---------------------------------------------------------------------------
# 12. Structural: this script's own content hash helper is self-consistent.
# ---------------------------------------------------------------------------


def test_this_script_sha256_matches_a_direct_recompute(runner):
    digest = hashlib.sha256(RUNNER_PATH.read_bytes()).hexdigest()
    assert runner._this_script_sha256() == digest
