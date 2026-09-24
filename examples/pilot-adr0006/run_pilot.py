#!/usr/bin/env python3
"""ADR-0006 bounded real TRL pilot runner: trainer -> evaluator, wired.

This is the actual bounded pilot execution ADR-0006 describes, run only
after:

1. ADR-0006's locked configuration (model, dataset, resource budget,
   concurrency=1) -- see docs/decisions/0006-bounded-real-trl-pilot-plan.md.
2. The pilot-specific live-execution security review (issue #7 step 5,
   PR #18 review) -- CLEAR TO EXECUTE, residual LOW/non-blocking note only
   (macOS ps-timing behaviour), findings 1-3 fixed and merged (PR #21,
   c3cbc5513fff8fc78526403ea75513c43229c1b8).
3. Owner authorisation to execute this specific bounded run.

What this script does, in order, per ADR-0006's "Independent evaluation
wiring" section:

  a. Builds TrainingInputs/ResourceBudget exactly matching ADR-0006's
     locked values, constructs TRLTrainerAdapter, and calls
     run_trainer_contract (real train() call -- this is the actual
     pilot execution, not another dry validation).
  b. If, and only if, the trainer run's TrainingOutput.status is
     ACCEPTED, loads the pilot's registered HeldOutSet (package id
     adr0006-pilot-heldout-v1, dataset_hash
     c5ade195c2b4b8697b175b8c94e10add6206ad4d72137ea83d24c65dbf489e65)
     and the pilot's HeldOutExclusionRegistry, and scores the trained
     artifact through run_evaluator_contract using
     HFLocalCausalLMEvaluatorAdapter -- the real (non-fake) evaluator
     adapter, not FakeEvaluatorAdapter.
  c. Prints/records both TrainingOutput and EvaluationOutput. Never
     resolves them into an accept/reject/promotion decision -- that is
     a separate, later governed action neither contract performs (see
     docs/ARCHITECTURE.md core contract #6).

A non-ACCEPTED TrainingOutput status halts here and is reported as-is;
the evaluator is never invoked against an untrusted/incomplete artifact.

Everything --model, dataset, resource budget, filesystem_root-- is
exactly ADR-0006's locked spec. filesystem_root is a fresh scratch
directory under this pilot's own examples/ tree (not the repo checkout
root, not any other project's data), created here and left in place as
evidence (not auto-deleted) so TrainingOutput.evidence_locator and the
evaluator's evidence bundle remain inspectable after this script exits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from held_out_eval import HeldOutExclusionRegistry

from codevolt_mdf.evaluator_contract import (
    EvaluationOutput,
    HeldOutExample,
    HeldOutSet,
    run_evaluator_contract,
)
from codevolt_mdf.hf_local_evaluator_adapter import HFLocalCausalLMEvaluatorAdapter
from codevolt_mdf.trainer_contract import (
    ResourceBudget,
    TrainingInputs,
    TrainingOutput,
    run_trainer_contract,
)
from codevolt_mdf.trl_adapter import TRLTrainerAdapter, _hash_path_identity

# --------------------------------------------------------------------------
# ADR-0006 locked values (see docs/decisions/0006-bounded-real-trl-pilot-plan.md,
# "Locked pilot parameters" section). Re-derived by content hash below, not
# hardcoded as trusted strings, except where the ADR itself declares a fixed
# revision id.
# --------------------------------------------------------------------------

MODEL_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M"
    / "snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
)
MODEL_REVISION = "HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
DATASET_PATH = HERE / "dataset" / "train.jsonl"
DATASET_VERSION = "adr0006-pilot-train-v1"
HELD_OUT_PATH = HERE / "held_out.json"
HELD_OUT_PACKAGE_ID = "adr0006-pilot-heldout-v1"
EXPECTED_HELD_OUT_DATASET_HASH = (
    "c5ade195c2b4b8697b175b8c94e10add6206ad4d72137ea83d24c65dbf489e65"
)
REGISTRY_PATH = HERE / "held_out_exclusion_registry.json"
SEED = 20260917
RUN_ID_PREFIX = "adr0006-pilot"

MAX_STEPS = 50


def _load_held_out_set() -> HeldOutSet:
    payload = json.loads(HELD_OUT_PATH.read_text(encoding="utf-8"))
    examples = [
        HeldOutExample(
            example_id=item["example_id"],
            input=item["input"],
            expected=item["expected"],
        )
        for item in payload["examples"]
    ]
    held_out = HeldOutSet.create(payload["package_id"], examples)
    held_out.validate()  # tamper/shape check before anything else
    if held_out.dataset_hash != EXPECTED_HELD_OUT_DATASET_HASH:
        raise SystemExit(
            f"FATAL: held_out.json content hashes to {held_out.dataset_hash}, "
            f"expected the locked {EXPECTED_HELD_OUT_DATASET_HASH} -- refusing "
            "to evaluate against an unverified held-out set"
        )
    return held_out


def main() -> int:
    run_id = f"{RUN_ID_PREFIX}-{SEED}"
    scratch_root = HERE / "scratch" / run_id
    scratch_root.mkdir(parents=True, exist_ok=True)
    budget = ResourceBudget(
        max_wall_seconds=1800,
        max_cpu_seconds=3600,
        max_memory_mb=8192,
        max_gpu_count=0,
        max_storage_mb=2048,
        network_policy="offline",
        filesystem_root=str(scratch_root),
    )

    model_hash = _hash_path_identity(MODEL_SNAPSHOT)
    dataset_hash = _hash_path_identity(DATASET_PATH)

    inputs = TrainingInputs.create(
        model_revision=MODEL_REVISION,
        model_hash=model_hash,
        dataset_version=DATASET_VERSION,
        dataset_hash=dataset_hash,
        seed=SEED,
        training_params={
            "model_path": str(MODEL_SNAPSHOT),
            "dataset_path": str(DATASET_PATH),
            "max_steps": MAX_STEPS,
        },
        dataset_licence="synthetic-CodeVolt-owned",
        contamination_checked=True,
        run_id=run_id,
    )

    adapter = TRLTrainerAdapter(work_dir=scratch_root / "trainer_work")

    print("=" * 72)
    print("ADR-0006 bounded real TRL pilot -- TRAINER phase")
    print(f"run_id={run_id}")
    print(f"model_hash={model_hash}")
    print(f"dataset_hash={dataset_hash}")
    print(f"max_steps={MAX_STEPS}  filesystem_root={scratch_root}")
    print("=" * 72)

    training_output = run_trainer_contract(adapter, inputs, budget)

    print()
    print("TrainingOutput:")
    print(f"  status           = {training_output.status.value}")
    print(f"  reason           = {training_output.reason}")
    print(f"  artifact_id      = {training_output.artifact_id}")
    print(f"  evidence_locator = {training_output.evidence_locator}")
    print(f"  evidence_hash    = {training_output.evidence_hash}")
    if training_output.resource_usage is not None:
        ru = training_output.resource_usage
        print(
            f"  resource_usage   = wall={ru.wall_seconds}s cpu={ru.cpu_seconds}s "
            f"mem_peak={ru.memory_mb_peak}MB storage={ru.storage_mb_used}MB "
            f"gpu={ru.gpu_count_used}"
        )
    if training_output.error_class:
        print(f"  error_class      = {training_output.error_class}")

    result: dict[str, Any] = {"training_output": _training_output_to_dict(training_output)}

    if training_output.status.value != "accepted":
        print()
        print(
            "HALT: TrainingOutput.status != ACCEPTED -- per ADR-0006 kill "
            "criteria, this halts the pilot and is reported as evidence, "
            "not silently retried or reinterpreted as success. Evaluator "
            "is NOT invoked against a non-accepted trainer run."
        )
        _write_result(scratch_root, result)
        return 1

    final_dir = Path(training_output.evidence_locator).parent / "final"
    if not final_dir.exists():
        print(
            f"FATAL: TrainingOutput.status == accepted but expected final "
            f"artifact dir {final_dir} does not exist -- evidence integrity "
            "problem, halting before evaluation."
        )
        result["evaluation_output"] = None
        result["evaluation_error"] = f"final_dir missing: {final_dir}"
        _write_result(scratch_root, result)
        return 1

    print()
    print("=" * 72)
    print("ADR-0006 bounded real TRL pilot -- EVALUATOR phase")
    print("(scoring trained artifact through run_evaluator_contract against")
    print(" the registered held-out set, per ADR-0006's independent")
    print(" evaluation wiring requirement)")
    print("=" * 72)

    held_out = _load_held_out_set()
    registry = HeldOutExclusionRegistry.load(REGISTRY_PATH)
    contaminated = registry.check_held_out_not_trained(held_out.example_ids)
    if contaminated:
        print(f"FATAL: held-out contamination detected pre-flight: {sorted(contaminated)}")
        result["evaluation_output"] = None
        result["evaluation_error"] = f"contaminated_ids={sorted(contaminated)}"
        _write_result(scratch_root, result)
        return 1

    evaluator = HFLocalCausalLMEvaluatorAdapter()
    evaluation_output = run_evaluator_contract(
        adapter=evaluator,
        artifact_id=training_output.artifact_id or run_id,
        artifact_locator=str(final_dir),
        held_out=held_out,
        registry=registry,
        work_dir=scratch_root / "evaluation_work",
    )

    print()
    print("EvaluationOutput:")
    print(f"  status          = {evaluation_output.status.value}")
    print(f"  reason          = {evaluation_output.reason}")
    print(f"  package_id      = {evaluation_output.package_id}")
    print(f"  aggregate_score = {evaluation_output.aggregate_score}")
    print(f"  evidence_locator= {evaluation_output.evidence_locator}")
    print(f"  evidence_hash   = {evaluation_output.evidence_hash}")
    if evaluation_output.contaminated_ids:
        print(f"  contaminated_ids= {evaluation_output.contaminated_ids}")
    if evaluation_output.error_class:
        print(f"  error_class     = {evaluation_output.error_class}")
    print()
    print("  Per-example results:")
    for r in evaluation_output.results:
        print(
            f"    {r.example_id}: correct={r.correct} score={r.score} "
            f"raw_output={r.raw_output!r}"
        )

    result["evaluation_output"] = _evaluation_output_to_dict(evaluation_output)
    _write_result(scratch_root, result)

    print()
    print("=" * 72)
    print("This is a measurement, not a promotion decision. Aggregate score,")
    print("critical-failure review, and any accept/reject/promotion action")
    print("remain separate, later governed steps this script does not perform.")
    print("=" * 72)
    return 0


def _training_output_to_dict(output: TrainingOutput) -> dict[str, Any]:
    ru = output.resource_usage
    return {
        "status": output.status.value,
        "reason": output.reason,
        "artifact_id": output.artifact_id,
        "evidence_locator": output.evidence_locator,
        "evidence_hash": output.evidence_hash,
        "error_class": output.error_class,
        "resource_usage": None
        if ru is None
        else {
            "wall_seconds": ru.wall_seconds,
            "cpu_seconds": ru.cpu_seconds,
            "memory_mb_peak": ru.memory_mb_peak,
            "storage_mb_used": ru.storage_mb_used,
            "gpu_count_used": ru.gpu_count_used,
        },
    }


def _evaluation_output_to_dict(output: EvaluationOutput) -> dict[str, Any]:
    return {
        "status": output.status.value,
        "reason": output.reason,
        "artifact_id": output.artifact_id,
        "package_id": output.package_id,
        "aggregate_score": output.aggregate_score,
        "evidence_locator": output.evidence_locator,
        "evidence_hash": output.evidence_hash,
        "error_class": output.error_class,
        "contaminated_ids": list(output.contaminated_ids),
        "results": [
            {
                "example_id": r.example_id,
                "correct": r.correct,
                "score": r.score,
                "raw_output": r.raw_output,
            }
            for r in output.results
        ],
    }


def _write_result(scratch_root: Path, result: dict[str, Any]) -> None:
    out_path = scratch_root / "pilot_result.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nFull machine-readable result written to: {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
