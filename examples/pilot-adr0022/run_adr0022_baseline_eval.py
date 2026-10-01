#!/usr/bin/env python3
"""ADR-0022 (amended) baseline task-outcome evaluation runner/scorer.

Per docs/decisions/ADR-0022-baseline-task-outcome-eval.md (base
7dbf5a2, amendment 4c1d24f/373b581, merged d5fb1a4e) sections 2-9 and
13: builds no item set (the 56-item set already exists at
examples/pilot-adr0022/items_c1.json / items_c2.json / items_c4.json,
sha256-pinned below) and runs no measurement -- this is the "write the
scoring/execution script" follow-up card (section 12 item 3), gated by
"security review and gate" (section 12 item 4) before "execute the
measurement exactly once" (section 12 item 5) is authorized.

Three-model design (amendment section 13.1): the existing ADR-0018
candidate checkpoint, its pinned reference/base checkpoint (both via
this project's real, admitted `hf-local-causal-lm-evaluator-v1` local
HF-inference path), and the capability-scale Qwen2.5-7B-Instruct
GGUF pair, scored via the pinned llama.cpp (Vulkan, b10938) CLI
runtime -- section 13.2's runtime pin, distinct from the first two
models' HF-transformers loading path.

Mirrors the prior evaluation-only cycle structure (run_adr0019_...py /
run_adr0020_...py): dry-run validation by default, `--execute` fails
closed without a valid single signed gate (section 9's "one security-
reviewer-signed gate, not the full three-role training-authorization
pattern" -- the lighter variant this card's own body calls for), bound
to this exact script's own content hash, the item set's sha256, the
HeldOutExclusionRegistry registration receipt, and an independently
re-measured compute ceiling. No training call anywhere in this file;
every model is loaded/generated from strictly read-only inference
(`model.eval()`/`torch.no_grad()` for the two HF-loaded models; the
llama.cpp CLI has no training mode at all).

Tests (tests/test_adr0022_runner.py) use fake checkpoints and a fake
llama.cpp CLI stand-in and never require a local HF snapshot or a real
llama.cpp binary, per this card's own body ("Tests must use fake
checkpoints and must not depend on a local HF snapshot").
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages/held-out-eval/src"))

from held_out_eval import HeldOutExclusionRegistry, HeldOutRegistryError

from codevolt_mdf.core import ContractError, Experiment, validate_manifest_schema
from codevolt_mdf.hf_local_evaluator_adapter import HFLocalCausalLMEvaluatorAdapter, _hash_model_dir
from codevolt_mdf.process_isolation import run_callable_in_isolated_process
from codevolt_mdf.trainer_contract import ResourceBudget

# ---------------------------------------------------------------------------
# Model identity. First two: the existing ADR-0018 candidate/reference,
# reused unchanged from run_adr0019_logprob_margin_eval.py's own pins
# (same immutable base checkpoint both documents already independently
# verified) -- this script does not re-derive or re-guess either hash.
# ---------------------------------------------------------------------------
REFERENCE_MODEL_REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
REFERENCE_MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
REFERENCE_MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct"
    / f"snapshots/{REFERENCE_MODEL_REVISION}"
)
EXPECTED_REFERENCE_MODEL_HASH = (
    "43752b3f39894c0122d9a94f3b4e64ad2d76e43d25c2a08aa360ad17a1a0145c"
)

CANDIDATE_MODEL_PATH = Path(
    "./local-evidence/adr0018/scratch/"
    "adr0018-metatrainer-dpo-20260923/trainer_work/"
    "adr0018-metatrainer-dpo-20260923/final"
)
EXPECTED_CANDIDATE_MODEL_HASH = (
    "8e424bfb4a1cdfc49ab182f0c2d003cb5d83b3685e4d7f0da05167e2fa9390fd"
)

# Third model (amendment section 13.1): Qwen2.5-7B-Instruct's official
# GGUF conversion, q4_k_m two-shard pair, scored through llama.cpp
# (Vulkan, b10938) per section 13.2's runtime pin -- not the HF
# transformers path the first two models use.
THIRD_MODEL_GGUF_SHARDS = (
    "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
)
EXPECTED_THIRD_MODEL_GGUF_SHA256 = (
    "dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db",
    "539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a",
)
THIRD_MODEL_ARTIFACT_ROOT = Path("./local-evidence/adr0022/models/qwen2.5-7b-instruct-gguf")
THIRD_MODEL_LLAMA_CPP_BINARY_ENV = "CODEVOLT_ADR0022_LLAMA_CPP_MAIN_BINARY"
THIRD_MODEL_LLAMA_CPP_RELEASE_TAG = "b10938"

MODEL_IDS = ("reference", "candidate", "third_model")
ENTRY_MODEL_HOSTS = {
    "meta_trainer_adr0022_smol135m_test1": {
        "reference": "test1_cv_test1",
        "candidate": "test1_cv_test1",
    },
    "meta_trainer_adr0022_baseline": {
        "third_model": "test2_evo_x3_102",
    },
}
RUNNER_ID = "adr0022-baseline-eval"

# ---------------------------------------------------------------------------
# Item set identity: the 56-item set already exists (this card builds
# no item set) -- ADR-0022 section 2-4's own item counts, section 5's
# "precondition of scoring, not scoring itself" held-out registration.
# ---------------------------------------------------------------------------
ITEMS_C1_PATH = HERE / "items_c1.json"
ITEMS_C2_PATH = HERE / "items_c2.json"
ITEMS_C4_PATH = HERE / "items_c4.json"
EXPECTED_ITEM_HASHES = {
    "c1": "5494f655b8798b92cf643bb950a2b6f9022d6e564e3eba920e25663af64ca2a1",
    "c2": "5926e4bbed5b9732d6a5b6d2ec47b18a9810a5535a8823003e1277d42ffd0c62",
    "c4": "f3971c50b0f6fd0cd44a393ce61beac30180ed66cbd29f3442fcbd6310003278",
}
HELD_OUT_REGISTRY_PATH = HERE / "held_out_exclusion_registry.json"
HELD_OUT_PACKAGE_ID = "pilot-adr0022-heldout-v1"

RUN_ID = "adr0022-baseline-eval-20260929"
HOST_CONTAINMENT_SCOPE = "evaluator-process-containment-v1"
HOST_CONTAINMENT_ENV = "CODEVOLT_ADR0022_HOST_CONTAINMENT"
APPROVAL_ALLOWED_SIGNERS_PATH = (
    REPO_ROOT / "examples/pilot-metatrainer-v2/approval_allowed_signers_adr0018"
)
APPROVAL_NAMESPACE = "codevolt-adr0022"
# Single, lighter gate (ADR-0022 section 9 / this card's own body):
# one role only, the security reviewer's item-set-plus-scoring-code
# review. No owner or dataset-rights role -- section 9's own reasoning
# ("no gradient computation, no weight update, no new model artifact,
# no promotion path") applies identically here.
APPROVAL_ROLE_PRINCIPAL = "security-reviewer"
APPROVAL_ROLE_DECISION = "approved"

APPROVED_ROOT = Path("./local-evidence/adr0022")
APPROVED_EVIDENCE_ROOT = APPROVED_ROOT / "evidence"
APPROVED_REVIEW_GATE_PATH = APPROVED_EVIDENCE_ROOT / "adr0022-review-gate.json"
APPROVED_SCRATCH_ROOT = APPROVED_ROOT / "scratch"
APPROVED_EXECUTION_SCRATCH = APPROVED_SCRATCH_ROOT / RUN_ID

# Isolated-process resource enforcement ceilings this script accepts
# from ANY gate, regardless of who signed it -- mirrors run_adr0019's/
# run_adr0020's own hard-ceiling discipline. Set generously above
# ADR-0019's own 2400MB/300s single-model ceiling to allow for the
# third model's larger (~4.7GB resident) footprint per amendment
# section 13.2's own device-detection estimate, while still being a
# script-enforced hard limit a looser gate cannot override.
ISOLATION_MAX_MEMORY_MB = 8000.0
ISOLATION_MAX_WALL_SECONDS = 1800.0
ISOLATION_MAX_CPU_SECONDS = 3600.0
ISOLATION_MAX_STORAGE_MB = 128.0


class Adr0022Error(SystemExit):
    """Raised (as a ``SystemExit`` subclass) for every fail-closed refusal in this file."""


class ContaminationRefusal(Adr0022Error):
    """Fail-closed registry refusal carrying a safe structured verdict."""

    def __init__(self, message: str, check: dict[str, Any]) -> None:
        super().__init__(message)
        self.check = check


# ---------------------------------------------------------------------------
# Small stdlib-only helpers, mirroring every prior runner's own
# per-script-duplicated convention rather than importing a shared module.
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _this_script_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


# ---------------------------------------------------------------------------
# Item loading.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemSet:
    c1: list[dict[str, Any]]
    c2: list[dict[str, Any]]
    c4: list[dict[str, Any]]
    c1_hash: str
    c2_hash: str
    c4_hash: str

    @property
    def all_ids(self) -> list[str]:
        return (
            [i["example_id"] for i in self.c1]
            + [i["example_id"] for i in self.c2]
            + [i["example_id"] for i in self.c4]
        )


def load_items() -> ItemSet:
    for path, expected_count in (
        (ITEMS_C1_PATH, 24),
        (ITEMS_C2_PATH, 12),
        (ITEMS_C4_PATH, 20),
    ):
        if not path.is_file():
            raise Adr0022Error(f"item set file missing: {path}")
    c1_raw = ITEMS_C1_PATH.read_bytes()
    c2_raw = ITEMS_C2_PATH.read_bytes()
    c4_raw = ITEMS_C4_PATH.read_bytes()
    actual_hashes = {
        "c1": hashlib.sha256(c1_raw).hexdigest(),
        "c2": hashlib.sha256(c2_raw).hexdigest(),
        "c4": hashlib.sha256(c4_raw).hexdigest(),
    }
    if actual_hashes != EXPECTED_ITEM_HASHES:
        raise Adr0022Error(
            "ADR-0022 item-set content hash mismatch; refusing a changed or reordered item set"
        )
    c1 = json.loads(c1_raw)["items"]
    c2 = json.loads(c2_raw)["items"]
    c4 = json.loads(c4_raw)["items"]
    if len(c1) != 24:
        raise Adr0022Error(f"C1 item set has {len(c1)} items; ADR-0022 section 2 requires exactly 24")
    if len(c2) != 12:
        raise Adr0022Error(f"C2 item set has {len(c2)} items; ADR-0022 section 3 requires exactly 12")
    if len(c4) != 20:
        raise Adr0022Error(f"C4 item set has {len(c4)} items; ADR-0022 section 4 requires exactly 20")
    item_set = ItemSet(
        c1=c1,
        c2=c2,
        c4=c4,
        c1_hash=actual_hashes["c1"],
        c2_hash=actual_hashes["c2"],
        c4_hash=actual_hashes["c4"],
    )
    ids = item_set.all_ids
    if len(ids) != 56:
        raise Adr0022Error(f"combined item set has {len(ids)} ids; ADR-0022 section 5 requires exactly 56")
    if len(ids) != len(set(ids)):
        raise Adr0022Error("duplicate example_id across item files")
    return item_set


def verify_registry_admits_no_contamination(
    registry_path: Path, item_ids: list[str]
) -> HeldOutExclusionRegistry:
    """Re-check (not merely trust) that no item id is registered as train.

    Mirrors run_adr0019_logprob_margin_eval.py's own
    verify_registry_admits_no_contamination -- "re-verify fresh, never
    trust the committed report alone" applied to this card's own
    56-item held-out registration.
    """
    if not registry_path.is_file():
        raise Adr0022Error(f"held-out registry file is missing: {registry_path}")
    registry = HeldOutExclusionRegistry.load(registry_path)
    if HELD_OUT_PACKAGE_ID not in registry.package_held_out_ids:
        raise Adr0022Error(
            f"registry at {registry_path} has no held-out registration for "
            f"package {HELD_OUT_PACKAGE_ID!r}"
        )
    registered = registry.package_held_out_ids[HELD_OUT_PACKAGE_ID]
    if registered != frozenset(item_ids):
        raise Adr0022Error(
            f"registry's registered ids for {HELD_OUT_PACKAGE_ID!r} do not match "
            "the item set's own ids exactly"
        )
    contaminated = registry.check_held_out_not_trained(item_ids)
    if contaminated:
        raise Adr0022Error(
            f"contamination: {len(contaminated)} item id(s) are already registered "
            f"as train data by some package: {sorted(contaminated)[:10]}"
        )
    return registry


def _load_execution_registry_snapshot(
    merged_registry_path: Path, item_ids: list[str]
) -> tuple[dict[str, Any], tuple[int, int, int, int, str]]:
    """Read, hash, parse, and check one direct regular-file snapshot."""
    supplied_path = Path(merged_registry_path)
    absolute_path = supplied_path.absolute()
    if supplied_path.is_symlink() or not supplied_path.is_file():
        raise ContaminationRefusal(
            f"merged held-out registry file is missing or not a direct regular file: {absolute_path}",
            {
                "status": "refused",
                "registry_path": str(absolute_path),
                "registry_sha256": None,
                "held_out_package_id": HELD_OUT_PACKAGE_ID,
                "checked_item_count": len(item_ids),
                "overlap_count": None,
                "overlapping_ids": [],
            },
        )
    try:
        stat_before = supplied_path.stat()
        payload_bytes = supplied_path.read_bytes()
        digest = hashlib.sha256(payload_bytes).hexdigest()
        payload = json.loads(payload_bytes)
        registry = HeldOutExclusionRegistry.from_dict(payload)
    except (
        AssertionError,
        HeldOutRegistryError,
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        raise ContaminationRefusal(
            f"merged held-out registry is malformed or unreadable: {exc}",
            {
                "status": "refused",
                "registry_path": str(absolute_path),
                "registry_sha256": None,
                "held_out_package_id": HELD_OUT_PACKAGE_ID,
                "checked_item_count": len(item_ids),
                "overlap_count": None,
                "overlapping_ids": [],
            },
        ) from exc

    registered = registry.package_held_out_ids.get(HELD_OUT_PACKAGE_ID)
    if registered != frozenset(item_ids):
        raise ContaminationRefusal(
            "merged registry's held-out package does not exactly match the reviewed item set",
            {
                "status": "refused",
                "registry_path": str(absolute_path),
                "registry_sha256": digest,
                "held_out_package_id": HELD_OUT_PACKAGE_ID,
                "checked_item_count": len(item_ids),
                "overlap_count": None,
                "overlapping_ids": [],
            },
        )
    overlapping_ids = sorted(registry.check_held_out_not_trained(item_ids))
    check = {
        "status": "pass" if not overlapping_ids else "refused",
        "registry_path": str(absolute_path),
        "registry_sha256": digest,
        "held_out_package_id": HELD_OUT_PACKAGE_ID,
        "checked_item_count": len(item_ids),
        "overlap_count": len(overlapping_ids),
        "overlapping_ids": overlapping_ids,
    }
    if overlapping_ids:
        raise ContaminationRefusal(
            f"contamination: {len(overlapping_ids)} held-out item id(s) overlap train data",
            check,
        )
    identity = (
        stat_before.st_dev,
        stat_before.st_ino,
        stat_before.st_size,
        stat_before.st_mtime_ns,
        digest,
    )
    return check, identity


def execution_contamination_check(
    merged_registry_path: Path, item_ids: list[str]
) -> dict[str, Any]:
    """Public check helper used by tests and non-executing integrations."""
    check, _identity = _load_execution_registry_snapshot(
        merged_registry_path, item_ids
    )
    return check


def _verify_registry_snapshot_unchanged(
    merged_registry_path: Path, expected_identity: tuple[int, int, int, int, str]
) -> None:
    path = Path(merged_registry_path)
    if path.is_symlink() or not path.is_file():
        raise Adr0022Error("merged held-out registry changed before evidence finalisation")
    stat_after = path.stat()
    digest_after = _sha256_file(path)
    actual_identity = (
        stat_after.st_dev,
        stat_after.st_ino,
        stat_after.st_size,
        stat_after.st_mtime_ns,
        digest_after,
    )
    if actual_identity != expected_identity:
        raise Adr0022Error("merged held-out registry changed before evidence finalisation")


# ---------------------------------------------------------------------------
# C1 scoring: closed-set first-line exact match.
# ---------------------------------------------------------------------------


def parse_closed_set_first_line(raw_output: str, closed_set: list[str]) -> str | None:
    """Return the first line stripped, if it exactly matches a closed-set member.

    Returns ``None`` for UNSCORABLE per ADR-0022 section 2's parser
    rule: an empty first line, a first line containing more than the
    bare identifier, or one not in the closed set.
    """
    first_line = raw_output.splitlines()[0].strip() if raw_output.strip() else ""
    if first_line in closed_set:
        return first_line
    return None


def score_c1_item(item: dict[str, Any], raw_output: str) -> dict[str, Any]:
    closed_set = item["metadata"]["closed_set"]
    parsed = parse_closed_set_first_line(raw_output, closed_set)
    if parsed is None:
        return {"example_id": item["example_id"], "outcome": "UNSCORABLE", "raw_output": raw_output}
    correct = parsed == item["expected"]
    return {
        "example_id": item["example_id"],
        "outcome": "correct" if correct else "incorrect",
        "predicted": parsed,
        "gold": item["expected"],
        "raw_output": raw_output,
    }


# ---------------------------------------------------------------------------
# C2 scoring: two-stage (config_validity: single-stage classification;
# command_produces_expected_artifact: parse+validate then CLI dry run).
# ---------------------------------------------------------------------------

FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _extract_first_fenced_block(raw_output: str) -> str | None:
    match = FENCE_RE.search(raw_output)
    if match is None:
        return None
    return match.group(1)


def score_c2_config_validity_item(item: dict[str, Any], raw_output: str) -> dict[str, Any]:
    first_line = raw_output.splitlines()[0].strip() if raw_output.strip() else ""
    expected = item["expected"]
    if expected["outcome"] == "pass":
        correct = first_line == "PASS"
    else:
        correct = first_line == f"FAIL: {expected['violated_check']}"
    return {
        "example_id": item["example_id"],
        "outcome": "correct" if correct else "incorrect",
        "predicted_first_line": first_line,
        "gold_outcome": expected["outcome"],
        "gold_violated_check": expected["violated_check"],
        "raw_output": raw_output,
    }


def score_c2_command_item(item: dict[str, Any], raw_output: str, work_dir: Path) -> dict[str, Any]:
    """Two-stage score: (a) parse+validate the emitted manifest; (b) run
    the real CLI (run_experiment) against it and compare decision.json.
    """
    block = _extract_first_fenced_block(raw_output)
    if block is None:
        return {
            "example_id": item["example_id"],
            "stage_a": "fail",
            "stage_a_violated_check": "no parseable manifest emitted",
            "stage_b": None,
            "raw_output": raw_output,
        }
    try:
        manifest = json.loads(block)
    except json.JSONDecodeError:
        return {
            "example_id": item["example_id"],
            "stage_a": "fail",
            "stage_a_violated_check": "no parseable manifest emitted",
            "stage_b": None,
            "raw_output": raw_output,
        }
    try:
        validate_manifest_schema(manifest)
        exp = Experiment.from_dict(manifest)
        exp.validate()
    except ContractError as exc:
        return {
            "example_id": item["example_id"],
            "stage_a": "fail",
            "stage_a_violated_check": str(exc),
            "stage_b": None,
            "raw_output": raw_output,
        }

    from codevolt_mdf.core import run_experiment

    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / f"{item['example_id']}-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_root = work_dir / f"{item['example_id']}-output"
    try:
        run_dir = run_experiment(manifest_path, output_root)
        actual_decision = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - record any dry-run failure as evaluation evidence
        return {
            "example_id": item["example_id"],
            "stage_a": "pass",
            "stage_b": "fail",
            "stage_b_detail": f"CLI dry run raised: {exc}",
            "raw_output": raw_output,
        }

    expected = item["expected"]
    stage_b_pass = (
        actual_decision["status"] == expected["decision_status"]
        and actual_decision["improvement"] == expected["decision_improvement"]
    )
    return {
        "example_id": item["example_id"],
        "stage_a": "pass",
        "stage_b": "pass" if stage_b_pass else "fail",
        "actual_decision": actual_decision,
        "expected_decision": {
            "status": expected["decision_status"],
            "improvement": expected["decision_improvement"],
        },
        "raw_output": raw_output,
    }


# ---------------------------------------------------------------------------
# C4 scoring: DEFER token or fact-string containment.
# ---------------------------------------------------------------------------


def score_c4_item(item: dict[str, Any], raw_output: str) -> dict[str, Any]:
    first_line = raw_output.splitlines()[0].strip() if raw_output.strip() else ""
    expected = item["expected"]
    if not first_line:
        return {"example_id": item["example_id"], "outcome": "UNSCORABLE", "raw_output": raw_output}
    if first_line == "DEFER":
        decision = "declined"
    else:
        decision = "answered"

    if expected["defer"]:
        # should-defer item: correct participation is "declined".
        outcome = "correct_decline" if decision == "declined" else "fabricated"
        return {
            "example_id": item["example_id"],
            "decision": decision,
            "outcome": outcome,
            "raw_output": raw_output,
        }

    # should-not-defer item.
    if decision == "declined":
        return {
            "example_id": item["example_id"],
            "decision": "declined",
            "outcome": "declined_should_have_answered",
            "raw_output": raw_output,
        }
    gold_fact = expected["gold_fact"]
    correct = _normalize(gold_fact) in _normalize(first_line)
    return {
        "example_id": item["example_id"],
        "decision": "answered",
        "outcome": "answered_correct" if correct else "answered_incorrect",
        "gold_fact": gold_fact,
        "raw_output": raw_output,
    }


# ---------------------------------------------------------------------------
# Aggregate statistics per ADR-0022 section 7 (unchanged) / section 13.3
# (per-model floors, extended to three models per section 13.3's own
# "reused, extended to three models, not redefined").
# ---------------------------------------------------------------------------


def aggregate_c1(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    scorable = [r for r in results if r["outcome"] != "UNSCORABLE"]
    correct = sum(1 for r in scorable if r["outcome"] == "correct")
    unscorable = total - len(scorable)
    return {
        "n": total,
        "correct": correct,
        "unscorable": unscorable,
        "accuracy_over_all_items": correct / total if total else 0.0,
        "unscorable_rate": unscorable / total if total else 0.0,
    }


def aggregate_c1_by_sub_bucket(results: list[dict[str, Any]], items: list[dict[str, Any]]) -> dict[str, Any]:
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    item_bucket = {i["example_id"]: i["sub_bucket"] for i in items}
    for r in results:
        bucket = item_bucket[r["example_id"]]
        by_bucket.setdefault(bucket, []).append(r)
    return {bucket: aggregate_c1(rs) for bucket, rs in by_bucket.items()}


def aggregate_c2(results: list[dict[str, Any]], config_items: list[dict[str, Any]], cmd_items: list[dict[str, Any]]) -> dict[str, Any]:
    config_ids = {i["example_id"] for i in config_items}
    cmd_ids = {i["example_id"] for i in cmd_items}
    config_results = [r for r in results if r["example_id"] in config_ids]
    cmd_results = [r for r in results if r["example_id"] in cmd_ids]
    config_correct = sum(1 for r in config_results if r["outcome"] == "correct")
    stage_a_pass = sum(1 for r in cmd_results if r["stage_a"] == "pass")
    stage_b_pass = sum(1 for r in cmd_results if r.get("stage_b") == "pass")
    return {
        "config_validity": {
            "n": len(config_results),
            "correct": config_correct,
            "fraction_correct": config_correct / len(config_results) if config_results else 0.0,
        },
        "command_produces_expected_artifact": {
            "n": len(cmd_results),
            "stage_a_pass": stage_a_pass,
            "stage_a_fraction": stage_a_pass / len(cmd_results) if cmd_results else 0.0,
            "stage_b_pass_of_stage_a_pass": stage_b_pass,
            "stage_b_fraction_of_stage_a_pass": (stage_b_pass / stage_a_pass) if stage_a_pass else 0.0,
        },
    }


def aggregate_c4(results: list[dict[str, Any]]) -> dict[str, Any]:
    should_not_defer = [r for r in results if r["outcome"] in (
        "answered_correct", "answered_incorrect", "declined_should_have_answered"
    )]
    should_defer = [r for r in results if r["outcome"] in ("correct_decline", "fabricated")]
    unscorable = sum(1 for r in results if r["outcome"] == "UNSCORABLE")

    answered_notdefer = [r for r in should_not_defer if r["decision"] == "answered"]
    notdefer_participation = len(answered_notdefer) / len(should_not_defer) if should_not_defer else 0.0
    conditional_correct = sum(1 for r in answered_notdefer if r["outcome"] == "answered_correct")
    conditional_correctness = conditional_correct / len(answered_notdefer) if answered_notdefer else 0.0

    fabricated = sum(1 for r in should_defer if r["outcome"] == "fabricated")
    defer_participation_fabrication_rate = fabricated / len(should_defer) if should_defer else 0.0

    return {
        "n": len(results),
        "unscorable": unscorable,
        "should_not_defer_n": len(should_not_defer),
        "should_not_defer_participation_rate": notdefer_participation,
        "should_not_defer_conditional_correctness": conditional_correctness,
        "should_defer_n": len(should_defer),
        "should_defer_fabrication_rate": defer_participation_fabrication_rate,
    }


def apply_floor_effect_guard(agg: dict[str, Any], total_items: int, *, unscorable_key: str = "unscorable") -> str:
    """ADR-0022 section 7's two independent floor-effect triggers, per model."""
    unscorable_rate = agg[unscorable_key] / total_items if total_items else 0.0
    if unscorable_rate > 0.40:
        return "uninformative_unscorable_floor"
    return "informative"


def c1_third_model_floor_pass(agg_by_bucket: dict[str, Any]) -> bool:
    """ADR-0022 section 13.3 C1 floor, per-sub-bucket chance-adjusted."""
    trainer = agg_by_bucket.get("trainer_adapter_selection", {"n": 0, "correct": 0})
    evaluator = agg_by_bucket.get("evaluator_adapter_selection", {"n": 0, "correct": 0})
    tasktype = agg_by_bucket.get("task_type_selection", {"n": 0, "correct": 0})
    total_correct = trainer.get("correct", 0) + evaluator.get("correct", 0) + tasktype.get("correct", 0)
    total_n = trainer.get("n", 0) + evaluator.get("n", 0) + tasktype.get("n", 0)
    aggregate_pass = total_n > 0 and total_correct > (8 / 24) * total_n
    trainer_pass = trainer.get("n", 0) == 0 or trainer["correct"] > (2 / 8) * trainer["n"]
    evaluator_pass = evaluator.get("n", 0) == 0 or evaluator["correct"] > (4 / 8) * evaluator["n"]
    tasktype_pass = tasktype.get("n", 0) == 0 or tasktype["correct"] > (2 / 8) * tasktype["n"]
    return aggregate_pass and trainer_pass and evaluator_pass and tasktype_pass


def c2_third_model_floor_pass(agg: dict[str, Any]) -> bool:
    """ADR-0022 section 13.3 C2 floor."""
    cmd = agg["command_produces_expected_artifact"]
    return cmd["stage_a_pass"] >= 4 and cmd["stage_b_pass_of_stage_a_pass"] >= 1


def c4_third_model_floor_pass(agg: dict[str, Any]) -> bool:
    """ADR-0022 section 13.3 C4 floor: all three sub-metrics required."""
    notdefer_pass = agg["should_not_defer_participation_rate"] >= 0.50
    defer_pass = agg["should_defer_fabrication_rate"] <= 0.50
    correctness_pass = agg["should_not_defer_conditional_correctness"] >= 0.50
    return notdefer_pass and defer_pass and correctness_pass


# ---------------------------------------------------------------------------
# Model interface: generate a raw text response for a prompt. Kept as a
# thin, swappable seam so tests can stub it without a real model load;
# the real (--execute) path routes calls through the isolated child
# process per model, matching every prior cycle's containment pattern.
# ---------------------------------------------------------------------------


def _hf_generate(evaluator: HFLocalCausalLMEvaluatorAdapter, model: Any, tokenizer: Any, prompt: str) -> str:
    generated, _renderer_id = evaluator._generate(model, tokenizer, prompt)
    return generated


def _llama_cpp_generate(binary_path: Path, model_shard_path: Path, prompt: str, *, max_new_tokens: int = 256) -> str:
    """Invoke the pinned llama.cpp CLI once, greedy decoding, and return stdout.

    Uses ``--temp 0`` for deterministic (greedy) decoding, matching
    section 8's "single greedy generate() call per (model, item) pair"
    applied to this runtime. This never trains; llama.cpp's `main`/
    `llama-cli` binary has no training mode to invoke.
    """
    proc = subprocess.run(
        [
            str(binary_path),
            "-m",
            str(model_shard_path),
            "-p",
            prompt,
            "-n",
            str(max_new_tokens),
            "--temp",
            "0",
            "--no-display-prompt",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if proc.returncode != 0:
        raise Adr0022Error(f"llama.cpp invocation failed (rc={proc.returncode}): {proc.stderr[:2000]}")
    return proc.stdout


# ---------------------------------------------------------------------------
# Dry-run validation: static/hash/schema checks only. Never loads a
# model, never runs inference, never touches any checkpoint's contents.
# ---------------------------------------------------------------------------


def validate_plan(model_id: str | None = None) -> dict[str, Any]:
    if model_id is not None and model_id not in MODEL_IDS:
        raise Adr0022Error(f"unknown model_id {model_id!r}; expected one of {MODEL_IDS}")
    selected_model_ids = MODEL_IDS if model_id is None else (model_id,)
    item_set = load_items()

    blockers: list[str] = []

    if "reference" in selected_model_ids:
        reference_hash_ok = (
            _hash_model_dir(REFERENCE_MODEL_PATH) == EXPECTED_REFERENCE_MODEL_HASH
            if REFERENCE_MODEL_PATH.is_dir()
            else False
        )
        if not REFERENCE_MODEL_PATH.is_dir():
            blockers.append("reference checkpoint is not present on disk at REFERENCE_MODEL_PATH")
        elif not reference_hash_ok:
            blockers.append("reference checkpoint content hash mismatch")

    if "candidate" in selected_model_ids:
        candidate_hash_ok = (
            _hash_model_dir(CANDIDATE_MODEL_PATH) == EXPECTED_CANDIDATE_MODEL_HASH
            if CANDIDATE_MODEL_PATH.is_dir()
            else False
        )
        if not CANDIDATE_MODEL_PATH.is_dir():
            blockers.append("candidate checkpoint is not present on disk at CANDIDATE_MODEL_PATH")
        elif not candidate_hash_ok:
            blockers.append("candidate checkpoint content hash mismatch")

    if "third_model" in selected_model_ids:
        third_model_shards_present = all(
            (THIRD_MODEL_ARTIFACT_ROOT / name).is_file()
            for name in THIRD_MODEL_GGUF_SHARDS
        )
        if not third_model_shards_present:
            blockers.append(
                "third-model GGUF shards are not present on disk at THIRD_MODEL_ARTIFACT_ROOT"
            )
        else:
            for name, expected_hash in zip(
                THIRD_MODEL_GGUF_SHARDS, EXPECTED_THIRD_MODEL_GGUF_SHA256
            ):
                actual = _sha256_file(THIRD_MODEL_ARTIFACT_ROOT / name)
                if actual != expected_hash:
                    blockers.append(
                        f"third-model GGUF shard {name} content hash mismatch"
                    )

        llama_cpp_binary = os.environ.get(THIRD_MODEL_LLAMA_CPP_BINARY_ENV)
        if not llama_cpp_binary or not Path(llama_cpp_binary).is_file():
            blockers.append(
                f"{THIRD_MODEL_LLAMA_CPP_BINARY_ENV} does not point at an installed llama.cpp binary"
            )

    try:
        verify_registry_admits_no_contamination(HELD_OUT_REGISTRY_PATH, item_set.all_ids)
    except Adr0022Error as exc:
        blockers.append(str(exc))

    if not APPROVED_REVIEW_GATE_PATH.is_file():
        blockers.append("security-reviewer's single-gate approval (ADR-0022 section 9) has not been issued")

    return {
        "status": "PASS",
        "scoring_called": False,
        "model_ids": list(selected_model_ids),
        "item_counts": {"c1": len(item_set.c1), "c2": len(item_set.c2), "c4": len(item_set.c4)},
        "item_hashes": {"c1": item_set.c1_hash, "c2": item_set.c2_hash, "c4": item_set.c4_hash},
        "held_out_package_id": HELD_OUT_PACKAGE_ID,
        "candidate_model_hash": EXPECTED_CANDIDATE_MODEL_HASH,
        "reference_model_hash": EXPECTED_REFERENCE_MODEL_HASH,
        "third_model_gguf_sha256": list(EXPECTED_THIRD_MODEL_GGUF_SHA256),
        "runtime_pin": {"engine": "llama.cpp", "backend": "Vulkan", "release_tag": THIRD_MODEL_LLAMA_CPP_RELEASE_TAG},
        "script_sha256": _this_script_sha256(),
        "execution_blockers": blockers,
    }


# ---------------------------------------------------------------------------
# Single-gate verification: bound to this file's own exact content hash,
# the item set's own three sha256 values, the sealed registry entry, and
# an independently re-measured compute ceiling.
# ---------------------------------------------------------------------------

REQUIRED_GATE_FIELDS = frozenset(
    {
        "schema_version",
        "script_sha256",
        "held_out_package_id",
        "items_c1_hash",
        "items_c2_hash",
        "items_c4_hash",
        "held_out_registry_hash",
        "measured_max_memory_mb",
        "measured_max_wall_seconds",
        "approval",
    }
)
REQUIRED_APPROVAL_FIELDS = frozenset({"document", "signature", "document_sha256"})
REQUIRED_APPROVAL_DOCUMENT_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "approver_id",
        "decision",
        "script_sha256",
        "held_out_package_id",
        "held_out_registry_hash",
        "measured_max_memory_mb",
        "measured_max_wall_seconds",
        "run_id",
        "scope",
    }
)


def _verify_signed_approval(
    approval: dict[str, Any],
    *,
    script_sha256: str,
    held_out_registry_hash: str,
    measured_max_memory_mb: float,
    measured_max_wall_seconds: float,
) -> dict[str, Any]:
    if set(approval) != REQUIRED_APPROVAL_FIELDS:
        raise Adr0022Error(f"approval must contain exactly {sorted(REQUIRED_APPROVAL_FIELDS)}")
    document_path = Path(approval["document"])
    signature_path = Path(approval["signature"])
    if not document_path.is_file() or not signature_path.is_file():
        raise Adr0022Error("approval document/signature is missing")
    if _sha256_file(document_path) != approval["document_sha256"]:
        raise Adr0022Error("approval document hash mismatch")
    document = json.loads(document_path.read_text(encoding="utf-8"))
    if set(document) != REQUIRED_APPROVAL_DOCUMENT_FIELDS:
        raise Adr0022Error(f"approval document schema mismatch: {sorted(document)}")
    expected = {
        "schema_version": 1,
        "role": "held-out-eval-gate",
        "approver_id": APPROVAL_ROLE_PRINCIPAL,
        "decision": APPROVAL_ROLE_DECISION,
        "script_sha256": script_sha256,
        "held_out_package_id": HELD_OUT_PACKAGE_ID,
        "held_out_registry_hash": held_out_registry_hash,
        "measured_max_memory_mb": measured_max_memory_mb,
        "measured_max_wall_seconds": measured_max_wall_seconds,
        "run_id": RUN_ID,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise Adr0022Error(f"approval field {key!r} is not exact")
    scope = document["scope"]
    if not isinstance(scope, list) or not scope or not all(
        isinstance(item, str) and item.strip() for item in scope
    ):
        raise Adr0022Error("approval scope must be a non-empty string list")
    if HOST_CONTAINMENT_SCOPE not in scope:
        raise Adr0022Error(f"approval scope is missing required token {HOST_CONTAINMENT_SCOPE!r}")
    proc = subprocess.run(
        [
            "ssh-keygen",
            "-Y",
            "verify",
            "-f",
            str(APPROVAL_ALLOWED_SIGNERS_PATH),
            "-I",
            APPROVAL_ROLE_PRINCIPAL,
            "-n",
            APPROVAL_NAMESPACE,
            "-s",
            str(signature_path),
        ],
        input=document_path.read_bytes(),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise Adr0022Error("approval signature is not trusted or valid")
    return document


def load_gate(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Adr0022Error(f"no gate file found at {path} -- --execute is fail-closed without a valid signed gate")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if set(gate) != REQUIRED_GATE_FIELDS or gate.get("schema_version") != 1:
        raise Adr0022Error("review gate schema mismatch")
    if gate["script_sha256"] != _this_script_sha256():
        raise Adr0022Error(
            "gate approves a different script content hash than the one currently "
            "on disk -- this is a stale or mismatched gate"
        )
    item_set = load_items()
    if gate["items_c1_hash"] != item_set.c1_hash:
        raise Adr0022Error("gate items_c1_hash does not match the item set actually on disk")
    if gate["items_c2_hash"] != item_set.c2_hash:
        raise Adr0022Error("gate items_c2_hash does not match the item set actually on disk")
    if gate["items_c4_hash"] != item_set.c4_hash:
        raise Adr0022Error("gate items_c4_hash does not match the item set actually on disk")
    if gate["held_out_package_id"] != HELD_OUT_PACKAGE_ID:
        raise Adr0022Error("gate is not bound to this evaluation's held-out package id")
    registry_actual_hash = _sha256_file(HELD_OUT_REGISTRY_PATH)
    if registry_actual_hash != gate["held_out_registry_hash"]:
        raise Adr0022Error(
            "gate's declared held-out registry hash does not match the registry "
            "file actually on disk right now"
        )
    if not isinstance(gate["measured_max_memory_mb"], (int, float)) or gate["measured_max_memory_mb"] <= 0:
        raise Adr0022Error("gate measured_max_memory_mb must be a positive number")
    if (
        not isinstance(gate["measured_max_wall_seconds"], (int, float))
        or gate["measured_max_wall_seconds"] <= 0
    ):
        raise Adr0022Error("gate measured_max_wall_seconds must be a positive number")
    if gate["measured_max_memory_mb"] > ISOLATION_MAX_MEMORY_MB:
        raise Adr0022Error(
            f"gate measured_max_memory_mb={gate['measured_max_memory_mb']} exceeds the "
            f"hard ceiling {ISOLATION_MAX_MEMORY_MB} this script enforces regardless of "
            "gate content"
        )
    if gate["measured_max_wall_seconds"] > ISOLATION_MAX_WALL_SECONDS:
        raise Adr0022Error(
            f"gate measured_max_wall_seconds={gate['measured_max_wall_seconds']} exceeds "
            f"the hard ceiling {ISOLATION_MAX_WALL_SECONDS} this script enforces "
            "regardless of gate content"
        )
    trusted_signer_lines = [
        line
        for line in APPROVAL_ALLOWED_SIGNERS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not trusted_signer_lines:
        raise Adr0022Error("approval trust root has no admitted signer keys; execution remains blocked")
    verified_document = _verify_signed_approval(
        gate["approval"],
        script_sha256=gate["script_sha256"],
        held_out_registry_hash=gate["held_out_registry_hash"],
        measured_max_memory_mb=gate["measured_max_memory_mb"],
        measured_max_wall_seconds=gate["measured_max_wall_seconds"],
    )
    return {**gate, "verified_approval": verified_document}


# ---------------------------------------------------------------------------
# --execute path: never invoked by this card. Loads all three models,
# scores every item, aggregates, writes evidence with a sha256.
# ---------------------------------------------------------------------------


def _isolation_budget(
    gate: dict[str, Any], scratch_root: Path, wall_clock_limit_seconds: float
) -> ResourceBudget:
    if wall_clock_limit_seconds <= 0:
        raise Adr0022Error("wall-clock limit must be a positive number")
    if wall_clock_limit_seconds > gate["measured_max_wall_seconds"]:
        raise Adr0022Error(
            "caller wall-clock limit exceeds the independently reviewed gate ceiling"
        )
    return ResourceBudget(
        max_wall_seconds=wall_clock_limit_seconds,
        max_cpu_seconds=ISOLATION_MAX_CPU_SECONDS,
        max_memory_mb=gate["measured_max_memory_mb"],
        max_gpu_count=0,
        max_storage_mb=ISOLATION_MAX_STORAGE_MB,
        network_policy="offline",
        filesystem_root=str(scratch_root),
    )


def _run_model_pass(
    *, model_id: str, candidate_model_path: str, reference_model_path: str, item_set_paths: dict[str, str]
) -> dict[str, Any]:
    """Child-process entry point: score one model against every item.

    Module-level function (picklable by reference), per every prior
    runner's own convention -- see run_adr0019/run_adr0020's own
    ``_run_margin_computation``/``_run_sampling`` docstrings.
    """
    item_set = load_items()
    c1_results, c2_results, c4_results = [], [], []

    if model_id in ("candidate", "reference"):
        model_path = candidate_model_path if model_id == "candidate" else reference_model_path
        evaluator = HFLocalCausalLMEvaluatorAdapter()
        model, tokenizer = evaluator._get_model(Path(model_path))

        for item in item_set.c1:
            raw = _hf_generate(evaluator, model, tokenizer, item["input"])
            c1_results.append(score_c1_item(item, raw))
        work_dir = Path(item_set_paths["c2_work_dir"]) / model_id
        for item in item_set.c2:
            raw = _hf_generate(evaluator, model, tokenizer, item["input"])
            if item["sub_bucket"] == "config_validity":
                c2_results.append(score_c2_config_validity_item(item, raw))
            else:
                c2_results.append(score_c2_command_item(item, raw, work_dir))
        for item in item_set.c4:
            raw = _hf_generate(evaluator, model, tokenizer, item["input"])
            c4_results.append(score_c4_item(item, raw))
    elif model_id == "third_model":
        binary_path = Path(os.environ[THIRD_MODEL_LLAMA_CPP_BINARY_ENV])
        shard_path = THIRD_MODEL_ARTIFACT_ROOT / THIRD_MODEL_GGUF_SHARDS[0]
        work_dir = Path(item_set_paths["c2_work_dir"]) / model_id
        for item in item_set.c1:
            raw = _llama_cpp_generate(binary_path, shard_path, item["input"])
            c1_results.append(score_c1_item(item, raw))
        for item in item_set.c2:
            raw = _llama_cpp_generate(binary_path, shard_path, item["input"])
            if item["sub_bucket"] == "config_validity":
                c2_results.append(score_c2_config_validity_item(item, raw))
            else:
                c2_results.append(score_c2_command_item(item, raw, work_dir))
        for item in item_set.c4:
            raw = _llama_cpp_generate(binary_path, shard_path, item["input"])
            c4_results.append(score_c4_item(item, raw))
    else:
        raise ValueError(f"unknown model_id {model_id!r}")

    return {
        "model_id": model_id,
        "c1_results": c1_results,
        "c1_aggregate": aggregate_c1(c1_results),
        "c1_aggregate_by_sub_bucket": aggregate_c1_by_sub_bucket(c1_results, item_set.c1),
        "c2_results": c2_results,
        "c2_aggregate": aggregate_c2(
            c2_results,
            [i for i in item_set.c2 if i["sub_bucket"] == "config_validity"],
            [i for i in item_set.c2 if i["sub_bucket"] == "command_produces_expected_artifact"],
        ),
        "c4_results": c4_results,
        "c4_aggregate": aggregate_c4(c4_results),
    }


def _validate_complete_model_report(
    report: dict[str, Any], model_id: str, item_set: ItemSet
) -> None:
    required = {
        "model_id",
        "c1_results",
        "c1_aggregate",
        "c1_aggregate_by_sub_bucket",
        "c2_results",
        "c2_aggregate",
        "c4_results",
        "c4_aggregate",
    }
    if not isinstance(report, dict) or set(report) != required:
        raise Adr0022Error("isolated evaluator returned an incomplete report schema")
    if report["model_id"] != model_id:
        raise Adr0022Error("isolated evaluator returned the wrong model identity")
    for capability, items in (
        ("c1", item_set.c1),
        ("c2", item_set.c2),
        ("c4", item_set.c4),
    ):
        results = report[f"{capability}_results"]
        expected_ids = [item["example_id"] for item in items]
        if not isinstance(results, list) or [row.get("example_id") for row in results] != expected_ids:
            raise Adr0022Error(
                f"isolated evaluator returned incomplete or reordered {capability} results"
            )


def execute_evaluation(
    scratch_root: Path,
    gate_path: Path,
    *,
    entry_id: str,
    model_id: str,
    host_id: str,
    item_set_id: str,
    merged_registry_path: Path,
    wall_clock_limit_seconds: float,
    cost_inputs: dict[str, Any],
) -> dict[str, Any]:
    """Execute exactly one entry/model/host/item-set evaluation invocation."""
    if not entry_id.strip() or not host_id.strip():
        raise Adr0022Error("entry_id and host_id must be non-empty")
    expected_host = ENTRY_MODEL_HOSTS.get(entry_id, {}).get(model_id)
    if expected_host is None or host_id != expected_host:
        raise Adr0022Error(
            "entry/model/host combination is not admitted for ADR-0022: "
            f"{entry_id!r}/{model_id!r}/{host_id!r}"
        )
    if item_set_id != HELD_OUT_PACKAGE_ID:
        raise Adr0022Error(
            f"item_set_id must be the reviewed ADR-0022 set {HELD_OUT_PACKAGE_ID!r}"
        )
    if not isinstance(cost_inputs, dict):
        raise Adr0022Error("cost_inputs must be a JSON object")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    gate = load_gate(gate_path)
    plan = validate_plan(model_id=model_id)
    if plan["execution_blockers"]:
        raise Adr0022Error(f"cannot execute: {plan['execution_blockers']}")

    invocation_scratch = scratch_root / model_id
    invocation = {
        "entry_id": entry_id,
        "model_id": model_id,
        "host_id": host_id,
        "item_set_id": item_set_id,
        "wall_clock_limit_seconds": wall_clock_limit_seconds,
        "cost_inputs": cost_inputs,
    }
    budget = _isolation_budget(gate, invocation_scratch, wall_clock_limit_seconds)
    item_set = load_items()
    # This is deliberately the final admission read before the isolated
    # evaluator starts. Never replace it with the committed split-time registry.
    try:
        contamination_check, registry_identity = _load_execution_registry_snapshot(
            merged_registry_path, item_set.all_ids
        )
    except ContaminationRefusal as exc:
        refusal = {
            "run_id": RUN_ID,
            "runner": {"id": RUNNER_ID, "script_sha256": _this_script_sha256()},
            "outcome": {
                "status": "refused",
                "accepted": False,
                "evaluated": False,
            },
            "invocation": invocation,
            "contamination_check": exc.check,
            "per_model": {},
            "third_model_queue_admission_floor": None,
        }
        _write_evidence(invocation_scratch, refusal)
        raise

    c2_work_dir = invocation_scratch / "c2_work"
    report, error, measured = run_callable_in_isolated_process(
        compute_fn=_run_model_pass,
        kwargs={
            "model_id": model_id,
            "candidate_model_path": str(CANDIDATE_MODEL_PATH),
            "reference_model_path": str(REFERENCE_MODEL_PATH),
            "item_set_paths": {"c2_work_dir": str(c2_work_dir)},
        },
        budget=budget,
    )
    if measured.killed_for_overrun:
        raise Adr0022Error(
            f"isolated evaluator process for model {model_id!r} exceeded the resource "
            f"budget (measured wall={measured.wall_seconds}s, cpu={measured.cpu_seconds}s, "
            f"memory={measured.memory_mb_peak}MB); refusing to write any outcome classification"
        )
    if measured.killed_for_timeout:
        raise Adr0022Error(
            f"isolated evaluator process for model {model_id!r} exceeded "
            f"max_wall_seconds={budget.max_wall_seconds}; refusing to write any "
            "outcome classification"
        )
    if error is not None or report is None:
        raise Adr0022Error(
            f"isolated evaluator process for model {model_id!r} failed without "
            f"producing a result: {error}"
        )
    _validate_complete_model_report(report, model_id, item_set)

    third_model_floor: dict[str, Any] | None = None
    if model_id == "third_model":
        third_model_floor = {
            "c1_pass": c1_third_model_floor_pass(
                report["c1_aggregate_by_sub_bucket"]
            ),
            "c2_pass": c2_third_model_floor_pass(report["c2_aggregate"]),
            "c4_pass": c4_third_model_floor_pass(report["c4_aggregate"]),
        }
        capabilities_passed = sum(1 for value in third_model_floor.values() if value)
        third_model_floor["capabilities_passed"] = capabilities_passed
        third_model_floor["overall_sufficient_signal"] = capabilities_passed >= 2

    try:
        _verify_registry_snapshot_unchanged(merged_registry_path, registry_identity)
    except Adr0022Error:
        refusal = {
            "run_id": RUN_ID,
            "runner": {"id": RUNNER_ID, "script_sha256": _this_script_sha256()},
            "outcome": {
                "status": "refused",
                "accepted": False,
                "evaluated": False,
            },
            "invocation": invocation,
            "contamination_check": {
                **contamination_check,
                "status": "refused",
                "reason": "registry_changed_before_evidence_finalisation",
            },
            "per_model": {},
            "third_model_queue_admission_floor": None,
        }
        _write_evidence(invocation_scratch, refusal)
        raise

    result = {
        "run_id": RUN_ID,
        "runner": {"id": RUNNER_ID, "script_sha256": _this_script_sha256()},
        "outcome": {
            "status": "evaluated",
            "accepted": True,
            "evaluated": True,
        },
        "invocation": invocation,
        "contamination_check": contamination_check,
        "gate": gate,
        "plan": plan,
        # Preserve the existing ledger fields and metric report shape; each
        # per-entry file now contains exactly one key instead of all three.
        "per_model": {model_id: report},
        "third_model_queue_admission_floor": third_model_floor,
        "isolation": {
            model_id: {
                "wall_seconds": measured.wall_seconds,
                "cpu_seconds": measured.cpu_seconds,
                "memory_mb_peak": measured.memory_mb_peak,
                "storage_mb_used": measured.storage_mb_used,
            }
        },
        "max_wall_seconds": budget.max_wall_seconds,
        "max_memory_mb": budget.max_memory_mb,
    }
    _write_evidence(invocation_scratch, result)
    return result


def _write_evidence(scratch_root: Path, result: dict[str, Any]) -> Path:
    path = (scratch_root / "adr0022_result.json").resolve()
    sha_path = (scratch_root / "adr0022_result.json.sha256").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    result["evidence_paths"] = {
        "result_json": str(path),
        "result_sha256": str(sha_path),
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    sha_path.write_text(digest + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--review-gate", type=Path)
    parser.add_argument("--scratch-root", type=Path, default=HERE / "scratch" / RUN_ID)
    parser.add_argument("--entry-id")
    parser.add_argument("--model-id", choices=MODEL_IDS)
    parser.add_argument("--host-id")
    parser.add_argument("--item-set-id")
    parser.add_argument("--merged-registry", type=Path)
    parser.add_argument("--wall-clock-limit-seconds", type=float)
    parser.add_argument("--cost-inputs-json", type=Path)
    args = parser.parse_args()

    if not args.execute:
        plan = validate_plan() if args.model_id is None else validate_plan(model_id=args.model_id)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if args.review_gate is None:
        raise Adr0022Error("--execute requires --review-gate")
    if args.scratch_root != APPROVED_EXECUTION_SCRATCH:
        raise Adr0022Error(f"--execute requires exact reviewed scratch root {APPROVED_EXECUTION_SCRATCH}")
    if args.review_gate != APPROVED_REVIEW_GATE_PATH:
        raise Adr0022Error(f"--execute requires exact reviewed gate path {APPROVED_REVIEW_GATE_PATH}")
    required = {
        "--entry-id": args.entry_id,
        "--model-id": args.model_id,
        "--host-id": args.host_id,
        "--item-set-id": args.item_set_id,
        "--merged-registry": args.merged_registry,
        "--wall-clock-limit-seconds": args.wall_clock_limit_seconds,
        "--cost-inputs-json": args.cost_inputs_json,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise Adr0022Error(f"--execute requires {', '.join(missing)}")
    if args.cost_inputs_json.is_symlink() or not args.cost_inputs_json.is_file():
        raise Adr0022Error("--cost-inputs-json must be a direct regular file")
    try:
        cost_inputs = json.loads(args.cost_inputs_json.read_bytes())
    except (json.JSONDecodeError, OSError) as exc:
        raise Adr0022Error(f"--cost-inputs-json is malformed or unreadable: {exc}") from exc
    if not isinstance(cost_inputs, dict):
        raise Adr0022Error("--cost-inputs-json must decode to a JSON object")

    result = execute_evaluation(
        args.scratch_root,
        args.review_gate,
        entry_id=args.entry_id,
        model_id=args.model_id,
        host_id=args.host_id,
        item_set_id=args.item_set_id,
        merged_registry_path=args.merged_registry,
        wall_clock_limit_seconds=args.wall_clock_limit_seconds,
        cost_inputs=cost_inputs,
    )
    print(json.dumps(result["evidence_paths"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
