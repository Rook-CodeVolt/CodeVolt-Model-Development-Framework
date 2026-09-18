"""Static, registry-level verification of ADR-0011's pilot dataset.

Confirms the newly-built ``examples/pilot-adr0011/`` train/held-out
split satisfies the same disjointness and bidirectional-contamination
guarantees ``examples/pilot-adr0006``'s pilot dataset already
established, per ``docs/decisions/0011-minimind-bounded-pilot-plan.md``'s
"Dataset" section.

Scope, explicit: this module only exercises ``held_out_eval`` and
``evaluator_contract``'s pure data-shape/registry code. It never
imports or invokes ``minimind_adapter.py``, never calls
``MiniMindTrainerAdapter.train()``, and never calls a real evaluator
adapter's ``score_example`` -- no training and no model inference is
executed by this file, matching this task's explicit scope limit.
"""
from __future__ import annotations

import json
from pathlib import Path

from held_out_eval import ContaminationError, HeldOutExclusionRegistry

from codevolt_mdf.evaluator_contract import HeldOutExample, HeldOutSet

PILOT_DIR = Path(__file__).resolve().parent.parent / "examples" / "pilot-adr0011"

TRAIN_PACKAGE_ID = "adr0011-pilot-train-v1"
HELDOUT_PACKAGE_ID = "adr0011-pilot-heldout-v1"


def _load_held_out_set() -> HeldOutSet:
    payload = json.loads((PILOT_DIR / "held_out.json").read_text(encoding="utf-8"))
    examples = [
        HeldOutExample(example_id=ex["example_id"], input=ex["input"], expected=ex["expected"])
        for ex in payload["examples"]
    ]
    held_out = HeldOutSet.create(payload["package_id"], examples)
    assert held_out.package_id == payload["package_id"]
    return held_out


def _load_registry() -> HeldOutExclusionRegistry:
    payload = json.loads(
        (PILOT_DIR / "held_out_exclusion_registry.json").read_text(encoding="utf-8")
    )
    return HeldOutExclusionRegistry.from_dict(payload)


def _load_train_ids() -> list[str]:
    payload = json.loads(
        (PILOT_DIR / "held_out_exclusion_registry.json").read_text(encoding="utf-8")
    )
    return list(payload["package_train_ids"][TRAIN_PACKAGE_ID])


def test_dataset_files_exist():
    assert (PILOT_DIR / "train.jsonl").exists()
    assert (PILOT_DIR / "held_out.json").exists()
    assert (PILOT_DIR / "held_out_exclusion_registry.json").exists()
    assert (PILOT_DIR / "DATASET_CARD.md").exists()


def test_train_split_uses_minimind_chat_jsonl_shape():
    """ADR-0011 requires MiniMind's own multi-turn chat schema, not TRL's flat text."""
    lines = (PILOT_DIR / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 40
    for line in lines:
        record = json.loads(line)
        assert set(record.keys()) == {"conversations"}
        turns = record["conversations"]
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"
        assert isinstance(turns[0]["content"], str) and turns[0]["content"]
        assert isinstance(turns[1]["content"], str) and turns[1]["content"]


def test_held_out_set_package_id_matches_adr0011():
    held_out = _load_held_out_set()
    assert held_out.package_id == HELDOUT_PACKAGE_ID
    assert len(held_out.examples) == 10


def test_held_out_set_validates_cleanly_no_tamper_no_duplicates():
    held_out = _load_held_out_set()
    held_out.validate()  # raises on tamper/duplicate ids; must not raise here.


def test_train_and_held_out_package_ids_are_distinct_from_adr0006_and_wpb():
    assert TRAIN_PACKAGE_ID not in {"adr0006-pilot-train-v1"}
    assert HELDOUT_PACKAGE_ID not in {"adr0006-pilot-heldout-v1", "wpb-safety-probes-v1"}


def test_train_and_held_out_content_is_disjoint_not_merely_ids():
    """Held-out prompts must not appear verbatim among train prompts."""
    train_prompts = set()
    for line in (PILOT_DIR / "train.jsonl").read_text(encoding="utf-8").strip().splitlines():
        record = json.loads(line)
        train_prompts.add(record["conversations"][0]["content"])

    held_out = _load_held_out_set()
    held_out_prompts = {example.input for example in held_out.examples}

    overlap = train_prompts & held_out_prompts
    assert overlap == set(), f"train/held-out prompt overlap detected: {overlap}"


def test_registry_registers_both_packages_without_contamination():
    """Bidirectional registration succeeds in either order with no ContaminationError."""
    train_ids = _load_train_ids()
    held_out = _load_held_out_set()

    registry_a = HeldOutExclusionRegistry()
    registry_a.register_package_train(TRAIN_PACKAGE_ID, train_ids)
    registry_a.register_package_held_out(HELDOUT_PACKAGE_ID, held_out.example_ids)

    registry_b = HeldOutExclusionRegistry()
    registry_b.register_package_held_out(HELDOUT_PACKAGE_ID, held_out.example_ids)
    registry_b.register_package_train(TRAIN_PACKAGE_ID, train_ids)

    for registry in (registry_a, registry_b):
        assert registry.package_train_ids[TRAIN_PACKAGE_ID] == frozenset(train_ids)
        assert registry.package_held_out_ids[HELDOUT_PACKAGE_ID] == held_out.example_ids


def test_check_held_out_not_trained_returns_no_violations():
    """The exact precondition run_evaluator_contract re-checks at evaluation time.

    This is the core contamination-check assertion this task requires:
    ``HeldOutExclusionRegistry.check_held_out_not_trained`` must return
    an empty set for ADR-0011's newly-built, disjoint train/held-out
    split, both against the committed on-disk registry snapshot and
    against a freshly-built registry from this pilot's own files.
    """
    held_out = _load_held_out_set()

    committed_registry = _load_registry()
    violations = committed_registry.check_held_out_not_trained(held_out.example_ids)
    assert violations == frozenset()

    train_ids = _load_train_ids()
    fresh_registry = HeldOutExclusionRegistry()
    fresh_registry.register_package_train(TRAIN_PACKAGE_ID, train_ids)
    fresh_registry.register_package_held_out(HELDOUT_PACKAGE_ID, held_out.example_ids)
    violations = fresh_registry.check_held_out_not_trained(held_out.example_ids)
    assert violations == frozenset()


def test_registering_held_out_ids_as_train_data_elsewhere_is_rejected():
    """Sanity check the registry actually enforces contamination, not a no-op.

    Registers this pilot's held-out ids as *train* data under a
    different package id and confirms ContaminationError is raised --
    proves the clean pass above is a real pass, not an artifact of an
    empty/broken registry.
    """
    held_out = _load_held_out_set()
    registry = HeldOutExclusionRegistry()
    registry.register_package_held_out(HELDOUT_PACKAGE_ID, held_out.example_ids)

    try:
        registry.register_package_train("some-other-package-v1", held_out.example_ids)
    except ContaminationError as exc:
        assert exc.direction == "train"
    else:
        raise AssertionError("expected ContaminationError, registry accepted contaminated ids")
