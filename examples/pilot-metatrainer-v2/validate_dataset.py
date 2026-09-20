#!/usr/bin/env python3
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    train = [json.loads(line) for line in (ROOT / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    held = json.loads((ROOT / "held_out.json").read_text(encoding="utf-8"))
    records = train + held
    checks = []

    def check(name, condition, detail=""):
        if not condition:
            raise AssertionError(f"{name}: {detail}")
        checks.append({"check": name, "status": "PASS", "detail": detail})

    check("count_total", len(records) == 60, str(len(records)))
    check("count_train", len(train) == 40, str(len(train)))
    check("count_held_out", len(held) == 20, str(len(held)))
    ids = [r["example_id"] for r in records]
    check("unique_ids", len(ids) == len(set(ids)))
    required = {"example_id", "split", "semantic_family", "source_scope", "citations", "messages"}
    check("required_fields", all(required <= set(r) for r in records))
    check("per_example_citations", all(isinstance(r["citations"], list) and r["citations"] for r in records))

    locator = re.compile(r"^(\[(?:[1-9]|1\d|2[0-3])\]|Clara synthesis, SS(?:1\.1|1\.2|1\.3|2|3\.2|3\.3|4\.1|4\.2|5))")
    check("citation_locator_format", all(all(locator.match(c) for c in r["citations"]) for r in records))
    check("authorized_sections_only", all("SS3.1" not in r["source_scope"] and "SS3.1" not in " ".join(r["citations"]) for r in records))

    allowed_direct_sources = {
        "zero_score_diagnostic_ladder": {"1", "2"},
        "sft_objective_masking_format": {"1", "2", "5"},
        "qualified_hyperparameter_sweeps": {"1", "2", "3", "4", "11", "23"},
        "peft_lora_qlora_decisions": {"6", "7"},
        "contamination_split_controls": {"16", "17"},
        "limited_data_evaluation": {"1", "20", "21", "22"},
        "synthetic_data_tradeoffs": {"18", "19"},
        "catastrophic_forgetting_retention": {"6"},
    }
    citation_family_ok = True
    for record in records:
        direct_ids = {match.group(1) for citation in record["citations"] if (match := re.match(r"^\[(\d+)\]", citation))}
        if not direct_ids <= allowed_direct_sources[record["semantic_family"]]:
            citation_family_ok = False
            break
    check("family_source_allowlist", citation_family_ok)

    train_families = {r["semantic_family"] for r in train}
    held_families = {r["semantic_family"] for r in held}
    check("semantic_family_disjoint", train_families.isdisjoint(held_families), str(train_families & held_families))

    manifest = json.loads((ROOT / "semantic_family_manifest.json").read_text(encoding="utf-8"))
    manifest_ids = [rid for family in manifest for rid in family["example_ids"]]
    check("manifest_exact_ids", set(manifest_ids) == set(ids) and len(manifest_ids) == len(ids))
    check("manifest_family_single_split", all(len({r["split"] for r in records if r["semantic_family"] == family["semantic_family"]}) == 1 for family in manifest))

    registry = json.loads((ROOT / "held_out_exclusion_registry.json").read_text(encoding="utf-8"))
    check("registry_heldout_ids", registry["held_out_example_ids"] == [r["example_id"] for r in held])

    representatives = json.loads((ROOT / "REPRESENTATIVE_EXAMPLES.json").read_text(encoding="utf-8"))
    check("representatives_exact_match", all(next(r for r in records if r["example_id"] == x["example_id"]) == x for x in representatives))

    banned = [r"67 ids", r"issue\s*#", r"40 examples (?:are|is) enough", r"no licensing question", r"license(?:d|s)? (?:is|are) clear"]
    content = "\n".join(m["content"] for r in records for m in r["messages"])
    check(
        "forbidden_content_absent",
        not any(re.search(pattern, content, re.IGNORECASE) for pattern in banned),
    )
    check("message_schema", all(len(r["messages"]) == 2 and r["messages"][0]["role"] == "user" and r["messages"][1]["role"] == "assistant" for r in records))

    direct_source_usage = {}
    for r in records:
        for citation in r["citations"]:
            match = re.match(r"^\[(\d+)\]", citation)
            if match:
                direct_source_usage[match.group(1)] = direct_source_usage.get(match.group(1), 0) + 1

    files = ["train.jsonl", "held_out.json", "held_out_exclusion_registry.json", "semantic_family_manifest.json", "REPRESENTATIVE_EXAMPLES.json"]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    report = {
        "status": "PASS",
        "validated_at": "2026-09-20",
        "counts": {"total": len(records), "train": len(train), "held_out": len(held)},
        "checks": checks,
        "train_families": sorted(train_families),
        "held_out_families": sorted(held_families),
        "direct_source_usage": dict(sorted(direct_source_usage.items(), key=lambda item: int(item[0]))),
        "sha256": hashes,
        "limitations": [
            "This validator proves exact family-key disjointness and structural citation presence; it does not replace an independent claim-support audit.",
            "Subtle semantic adjacency across differently named families still requires human review."
        ]
    }
    (ROOT / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "checks": len(checks), "counts": report["counts"]}))


if __name__ == "__main__":
    main()
