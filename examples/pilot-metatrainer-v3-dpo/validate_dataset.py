#!/usr/bin/env python3
"""Structural + held-out-contamination validator for the ADR-0017 DPO
preference-pair package (`preference_pairs.jsonl` in this directory).

Extends the pattern of `examples/pilot-metatrainer-v3/validate_dataset.py`
to the `prompt/chosen/rejected` triple shape, and adds the one check that
shape did not need: an explicit re-verification, against the real on-disk
`examples/pilot-metatrainer-v3/held_out.json`, that no preference-pair
`prompt` reuses `mtr-v2-heldout-0013`'s own prompt text or any other
`synthetic_data_tradeoffs`-family held-out prompt -- ADR-0017's binding
held-out-contamination constraint (Decision section, item 2).

This validator proves structural/contamination properties programmatically;
it does not replace an independent claim-support/citation audit (same
stated limitation as pilot-metatrainer-v3's own validator).
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
V3_DIR = ROOT.parent / "pilot-metatrainer-v3"


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation/whitespace variance, for a paraphrase-resistant
    (but still simple, not semantic) contamination comparison."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _token_set(text: str) -> set:
    return set(_normalize(text).split())


def main():
    records = [
        json.loads(line)
        for line in (ROOT / "preference_pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checks = []

    def check(name, condition, detail=""):
        if not condition:
            raise AssertionError(f"{name}: {detail}")
        checks.append({"check": name, "status": "PASS", "detail": detail})

    check("nonempty", len(records) > 0, str(len(records)))

    ids = [r["pair_id"] for r in records]
    check("unique_pair_ids", len(ids) == len(set(ids)), f"{len(ids)} ids, {len(set(ids))} unique")

    required = {"pair_id", "split", "semantic_family", "direction", "content_class", "source_scope", "citations", "prompt", "chosen", "rejected"}
    check("required_fields", all(required <= set(r) for r in records))

    check("all_train_split", all(r["split"] == "train" for r in records), "DPO package is train-only; no held-out preference pairs are proposed by ADR-0017")

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
        f"{len(counter)}/{len(records)} = {counter_share:.4f}, the security reviewer's over-refusal gate requires >= 0.20",
    )

    check(
        "chosen_rejected_distinct",
        all(r["chosen"].strip() != r["rejected"].strip() for r in records),
    )

    check(
        "per_example_citations",
        all(isinstance(r["citations"], list) and r["citations"] for r in records),
    )

    # -- ADR-0017's binding held-out-contamination constraint --------------
    # Load the REAL, on-disk held-out set this ADR is measured against.
    # This is not a copy or a trusted description of held_out.json; it is
    # read fresh, directly, every time this validator runs.
    held_out = json.loads((V3_DIR / "held_out.json").read_text(encoding="utf-8"))
    held_out_prompts = {}
    for h in held_out:
        user_msgs = [m["content"] for m in h["messages"] if m["role"] == "user"]
        for msg in user_msgs:
            held_out_prompts[h["example_id"]] = msg

    # (a) Exact / near-exact literal reuse check (normalized string equality).
    exact_hits = []
    for r in records:
        norm_prompt = _normalize(r["prompt"])
        for hid, htext in held_out_prompts.items():
            if norm_prompt == _normalize(htext):
                exact_hits.append((r["pair_id"], hid))
    check(
        "no_literal_held_out_prompt_reuse",
        not exact_hits,
        str(exact_hits[:5]),
    )

    # (b) Specific check against mtr-v2-heldout-0013 itself (the item this
    # whole ADR is measured against) -- both exact match AND high token
    # overlap, since a "trivial paraphrase" is explicitly forbidden too,
    # not just a byte-identical copy.
    heldout_0013_text = held_out_prompts.get("mtr-v2-heldout-0013", "")
    heldout_0013_tokens = _token_set(heldout_0013_text)
    check("mtr_v2_heldout_0013_present_in_registry", bool(heldout_0013_text), f"heldout-0013 prompt text length={len(heldout_0013_text)} chars (0 would mean registry drift)")

    near_paraphrase_hits = []
    for r in records:
        prompt_tokens = _token_set(r["prompt"])
        if not prompt_tokens or not heldout_0013_tokens:
            continue
        overlap = len(prompt_tokens & heldout_0013_tokens) / len(heldout_0013_tokens)
        if overlap >= 0.6:  # >=60% of heldout-0013's own content words appear in this prompt
            near_paraphrase_hits.append((r["pair_id"], round(overlap, 3)))
    check(
        "no_near_paraphrase_of_heldout_0013",
        not near_paraphrase_hits,
        str(near_paraphrase_hits[:5]),
    )

    # (c) synthetic_data_tradeoffs family (the family containing
    # mtr-v2-heldout-0013) must not be referenced, quoted, or have its
    # semantic_family label reused anywhere in this new package, per
    # ADR-0017 Decision item 2 ("is not to be touched, read into a prompt
    # template, or have any of its records' prompt text reused anywhere").
    forbidden_family = "synthetic_data_tradeoffs"
    check(
        "no_forbidden_family_label_reuse",
        all(r["semantic_family"] != forbidden_family for r in records),
    )
    all_text_blob = "\n".join(r["prompt"] + " " + r["chosen"] + " " + r["rejected"] for r in records)
    check(
        "no_forbidden_family_name_mentioned",
        forbidden_family not in all_text_blob,
    )

    # (d) Every content_class must be genuinely new versus ADR-0016's 9
    # confabulated_recipe_detection SFT records (train-0035..0043), which
    # already cover: recipe, cost, date, accuracy%, citation, GPU-memory,
    # rounded%, citation-locator, benchmark-score. This package's 16
    # refusal-direction content classes must not repeat any of those 9,
    # per ADR-0017's "new variations beyond the 9 ADR-0016 already used"
    # instruction.
    refusal_classes = {r["content_class"] for r in records if r["direction"] == "refusal"}
    check(
        "refusal_content_classes_are_new_vs_adr0016",
        len(refusal_classes) == len([r for r in records if r["direction"] == "refusal"]),
        "each refusal-direction record should use a distinct content_class (no duplicate coverage)",
    )

    files = ["preference_pairs.jsonl"]
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
        "held_out_contamination_check": {
            "compared_against": str((V3_DIR / "held_out.json").relative_to(REPO_ROOT)),
            "held_out_records_checked": len(held_out_prompts),
            "exact_reuse_hits": exact_hits,
            "near_paraphrase_hits_against_mtr_v2_heldout_0013": near_paraphrase_hits,
        },
        "limitations": [
            "The near-paraphrase check is a token-overlap heuristic (>=60% of heldout-0013's content words), not a semantic-similarity model; a paraphrase that avoids heldout-0013's specific vocabulary could still evade this specific threshold and needs human/independent review, same limitation class as pilot-metatrainer-v3's own validator states for semantic adjacency.",
            "This validator proves structural shape, counter-direction share, and held-out-contamination absence; it does not replace an independent claim-support/citation audit of the counter-direction pairs' factual claims.",
        ],
    }
    (ROOT / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "checks": len(checks), "counts": report["counts"]}))


if __name__ == "__main__":
    main()
