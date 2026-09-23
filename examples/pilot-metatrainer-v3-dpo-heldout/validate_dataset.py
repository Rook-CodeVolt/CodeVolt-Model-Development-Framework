#!/usr/bin/env python3
"""Structural + held-out-contamination validator for the ADR-0019 held-out
preference-pair set (``held_out_pairs.jsonl`` in this directory).

Extends the pattern of
``examples/pilot-metatrainer-v3-dpo/validate_dataset.py`` (the ADR-0017
training-package validator) to this held-out set's own binding construction
rules (ADR-0019 section 2), reusing its structural/counter-share/
token-overlap logic rather than duplicating it: this module imports that
validator's ``_normalize``/``_token_set`` helpers directly instead of
reimplementing normalized-string or token-overlap comparison.

This validator programmatically re-runs, on the held-out set itself, four
of the five section-2 checks Marcus (the pair author) is asked to run and
record; the fifth (registration through ``HeldOutExclusionRegistry``) is
performed separately by ``register_held_out.py`` in this directory, which
imports this module's ``main()`` output rather than duplicating the
contamination logic:

1. Zero overlap with the 22 ADR-0017 training pairs
   (``examples/pilot-metatrainer-v3-dpo/preference_pairs.jsonl``).
2. Zero overlap with the 48 real held-out prompts
   (``examples/pilot-metatrainer-v3/held_out.json``).
3. Zero overlap (exact or >=60% token-overlap near-paraphrase) with
   ``mtr-v2-heldout-0013`` specifically.
4. (Registration check lives in ``register_held_out.py`` -- structural
   scripts stay separate from the one that mutates a shared registry file.)
5. Counter-direction share >= 20%, computed here.

Plus this set's own additional binding rule not present in the ADR-0017
training validator: every ``content_class`` value must be disjoint from the
22 training pairs' ``content_class`` values (ADR-0019 section 2, bullet 3).

This script is authored and run by the pair constructor (Marcus) to record
a self-check result; Maya independently re-runs all five section-2 checks
before the set is sealed, per ADR-0019's own "different person builds vs.
audits" separation-of-duties rule (section 2, "Who builds it").
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
DPO_TRAIN_DIR = ROOT.parent / "pilot-metatrainer-v3-dpo"
V3_DIR = ROOT.parent / "pilot-metatrainer-v3"

sys.path.insert(0, str(DPO_TRAIN_DIR))
from validate_dataset import _normalize, _token_set


def main():
    records = [
        json.loads(line)
        for line in (ROOT / "held_out_pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checks = []

    def check(name, condition, detail=""):
        if not condition:
            raise AssertionError(f"{name}: {detail}")
        checks.append({"check": name, "status": "PASS", "detail": detail})

    check("exactly_20_records", len(records) == 20, str(len(records)))

    ids = [r["pair_id"] for r in records]
    check("unique_pair_ids", len(ids) == len(set(ids)), f"{len(ids)} ids, {len(set(ids))} unique")

    required = {
        "pair_id", "split", "semantic_family", "direction", "content_class",
        "source_scope", "citations", "prompt", "chosen", "rejected",
    }
    check("required_fields", all(required <= set(r) for r in records))

    check(
        "all_held_out_split",
        all(r["split"] == "held_out" for r in records),
        "ADR-0019's held-out set is held_out-only; split is load-bearing for HeldOutExclusionRegistry registration",
    )

    check(
        "direction_values",
        all(r["direction"] in ("refusal", "counter") for r in records),
        sorted({r["direction"] for r in records}),
    )

    counter = [r for r in records if r["direction"] == "counter"]
    counter_share = len(counter) / len(records)
    check(
        "counter_direction_share_min_20pct",
        counter_share >= 0.20,
        f"{len(counter)}/{len(records)} = {counter_share:.4f}, ADR-0019 section 2 bullet 2 requires >= 0.20",
    )

    check(
        "chosen_rejected_distinct",
        all(r["chosen"].strip() != r["rejected"].strip() for r in records),
    )

    check(
        "per_example_citations",
        all(isinstance(r["citations"], list) and r["citations"] for r in records),
    )

    # -- ADR-0019 section 2, bullet 3: content_class disjoint from training --
    # Load the REAL, on-disk 22-pair ADR-0017 training package fresh, not a
    # copy or a remembered list -- re-read at validation time every run.
    train_records = [
        json.loads(line)
        for line in (DPO_TRAIN_DIR / "preference_pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    check("training_package_has_22_records", len(train_records) == 22, str(len(train_records)))
    train_content_classes = {r["content_class"] for r in train_records}
    held_out_content_classes = {r["content_class"] for r in records}
    content_class_overlap = train_content_classes & held_out_content_classes
    check(
        "content_class_disjoint_from_training",
        not content_class_overlap,
        f"overlap={sorted(content_class_overlap)}; training has {len(train_content_classes)} classes, held-out has {len(held_out_content_classes)}",
    )
    check(
        "held_out_content_classes_all_distinct",
        len(held_out_content_classes) == len(records),
        "each held-out record should use a distinct content_class (no duplicate coverage)",
    )

    # -- ADR-0019 section 2, contamination-audit checks 1-3 -----------------
    # Check 1: zero overlap with the 22 training pairs' prompt text.
    train_prompts = {r["pair_id"]: r["prompt"] for r in train_records}
    exact_hits_vs_train = []
    for r in records:
        norm_prompt = _normalize(r["prompt"])
        for tid, ttext in train_prompts.items():
            if norm_prompt == _normalize(ttext):
                exact_hits_vs_train.append((r["pair_id"], tid))
    check(
        "no_literal_training_pair_prompt_reuse",
        not exact_hits_vs_train,
        str(exact_hits_vs_train[:5]),
    )

    # Check 2: zero overlap with the 48 real held-out prompts (exact/normalized).
    held_out_registry = json.loads((V3_DIR / "held_out.json").read_text(encoding="utf-8"))
    real_held_out_prompts = {}
    for h in held_out_registry:
        user_msgs = [m["content"] for m in h["messages"] if m["role"] == "user"]
        for msg in user_msgs:
            real_held_out_prompts[h["example_id"]] = msg
    check(
        "real_held_out_registry_has_48_records",
        len(real_held_out_prompts) == 48,
        str(len(real_held_out_prompts)),
    )
    exact_hits_vs_real_held_out = []
    for r in records:
        norm_prompt = _normalize(r["prompt"])
        for hid, htext in real_held_out_prompts.items():
            if norm_prompt == _normalize(htext):
                exact_hits_vs_real_held_out.append((r["pair_id"], hid))
    check(
        "no_literal_real_held_out_prompt_reuse",
        not exact_hits_vs_real_held_out,
        str(exact_hits_vs_real_held_out[:5]),
    )

    # Check 3: zero overlap (exact or >=60% token-overlap near-paraphrase)
    # with mtr-v2-heldout-0013 specifically -- reusing the exact same
    # heuristic and threshold the ADR-0017 training validator already uses.
    heldout_0013_text = real_held_out_prompts.get("mtr-v2-heldout-0013", "")
    heldout_0013_tokens = _token_set(heldout_0013_text)
    check(
        "mtr_v2_heldout_0013_present_in_registry",
        bool(heldout_0013_text),
        f"heldout-0013 prompt text length={len(heldout_0013_text)} chars (0 would mean registry drift)",
    )
    near_paraphrase_hits = []
    for r in records:
        prompt_tokens = _token_set(r["prompt"])
        if not prompt_tokens or not heldout_0013_tokens:
            continue
        overlap = len(prompt_tokens & heldout_0013_tokens) / len(heldout_0013_tokens)
        if overlap >= 0.6:
            near_paraphrase_hits.append((r["pair_id"], round(overlap, 3)))
    check(
        "no_near_paraphrase_of_heldout_0013",
        not near_paraphrase_hits,
        str(near_paraphrase_hits[:5]),
    )

    files = ["held_out_pairs.jsonl"]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    report = {
        "status": "PASS",
        "counts": {
            "total": len(records),
            "refusal": len(records) - len(counter),
            "counter": len(counter),
            "counter_share": round(counter_share, 4),
        },
        "checks": checks,
        "sha256": hashes,
        "contamination_audit_self_check": {
            "adr0019_section2_check_1_zero_overlap_with_training": {
                "compared_against": str((DPO_TRAIN_DIR / "preference_pairs.jsonl").relative_to(REPO_ROOT)),
                "training_records_checked": len(train_prompts),
                "exact_reuse_hits": exact_hits_vs_train,
            },
            "adr0019_section2_check_2_zero_overlap_with_48_held_out": {
                "compared_against": str((V3_DIR / "held_out.json").relative_to(REPO_ROOT)),
                "held_out_records_checked": len(real_held_out_prompts),
                "exact_reuse_hits": exact_hits_vs_real_held_out,
            },
            "adr0019_section2_check_3_zero_overlap_with_heldout_0013": {
                "near_paraphrase_hits_ge_60pct": near_paraphrase_hits,
            },
            "adr0019_section2_check_4_registry_registration": (
                "performed separately by register_held_out.py in this directory, "
                "not by this structural validator"
            ),
            "adr0019_section2_check_5_counter_direction_share": round(counter_share, 4),
        },
        "limitations": [
            "The near-paraphrase check is a token-overlap heuristic (>=60% of heldout-0013's content words), not a semantic-similarity model; a paraphrase that avoids heldout-0013's specific vocabulary could still evade this specific threshold and needs independent reviewer judgment, same limitation class as pilot-metatrainer-v3-dpo/validate_dataset.py's own stated limitation.",
            "This is a self-check by the pair constructor (Marcus). Per ADR-0019 section 2's owner decision, all five section-2 audit checks must be independently re-run and sealed by Maya before this set is used for anything -- this report is not a substitute for that independent audit.",
            "This validator proves structural shape, counter-direction share, content_class disjointness, and literal/near-paraphrase contamination absence; it does not replace an independent claim-support/citation audit of the 4 counter-direction pairs' factual claims.",
        ],
    }
    (ROOT / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "checks": len(checks), "counts": report["counts"]}))


if __name__ == "__main__":
    main()
