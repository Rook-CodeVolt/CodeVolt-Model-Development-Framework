#!/usr/bin/env python3
"""Structural + policy validator for meta-trainer corpus v3 (ADR-0015 proposal).

Extends corpus v2's validator with: a larger corpus size window (100-120 train
/ 40-50 held-out instead of a fixed 40/20), 23 semantic families instead of 8,
a broader citation-locator format (numbered carried-forward sources [1]-[23], numbered
new sources [24]-[30], "research synthesis, MSx" labels, and free-text
repository-evidence/file-path/internal-task locators for the history and
confabulation-evidence families), and the same disjoint-split, unique-id,
manifest-consistency, and forbidden-content checks as v2.
"""
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

    check("count_total", 140 <= len(records) <= 170, str(len(records)))
    check("count_train_in_range", 100 <= len(train) <= 120, str(len(train)))
    check("count_held_out_in_range", 40 <= len(held) <= 50, str(len(held)))

    ids = [r["example_id"] for r in records]
    check("unique_ids", len(ids) == len(set(ids)), f"{len(ids)} ids, {len(set(ids))} unique")

    required = {"example_id", "split", "semantic_family", "source_scope", "citations", "messages"}
    check("required_fields", all(required <= set(r) for r in records))
    check("per_example_citations", all(isinstance(r["citations"], list) and r["citations"] for r in records))
    check("message_schema", all(len(r["messages"]) == 2 and r["messages"][0]["role"] == "user" and r["messages"][1]["role"] == "assistant" for r in records))

    # Citation-locator format: allow (a) numbered [1]-[30] sources, (b) "research
    # synthesis, MSx" labels, (c) "research synthesis, SSx.x" labels carried
    # forward unchanged from v2 records, (d) free-text locators that begin
    # with one of the recognized repository-evidence prefixes used by the
    # history-addition and confabulation-evidence families.
    numbered = re.compile(r"^\[(?:[1-9]|[12]\d|30)\]")
    marcus_synth = re.compile(r"^research synthesis, MS\d+$")
    clara_synth = re.compile(r"^research synthesis, SS(?:1\.1|1\.2|1\.3|2|3\.2|3\.3|4\.1|4\.2|5)\b")
    repo_evidence_prefixes = (
        "docs/decisions/", "examples/pilot-adr", "internal tracking item ",
        "local evidence file ", "Repository evidence:", "src/codevolt_mdf/",
        "https://github.com/Rook-CodeVolt/", "commit ",
    )

    def locator_ok(c):
        if numbered.match(c):
            return True
        if marcus_synth.match(c):
            return True
        if clara_synth.match(c):
            return True
        return any(c.startswith(p) for p in repo_evidence_prefixes)

    bad_locators = []
    for r in records:
        for c in r["citations"]:
            if not locator_ok(c):
                bad_locators.append((r["example_id"], c))
    check("citation_locator_format", not bad_locators, str(bad_locators[:5]))

    check("authorized_sections_only", all("SS3.1" not in r["source_scope"] and "SS3.1" not in " ".join(r["citations"]) for r in records))

    # Every semantic_family must be wholly on one split (structural leakage guard).
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

    # No accidental verbatim reproduction of a full source passage: spot-check
    # that no assistant answer is implausibly long (a crude verbatim-copy
    # smell test; the corpus is synthetic Q&A, not source reproduction).
    long_answers = [r["example_id"] for r in records if len(r["messages"][1]["content"]) > 1600]
    check("no_oversized_answers", not long_answers, str(long_answers))

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
        "validated_at": "2026-09-22",
        "counts": {"total": len(records), "train": len(train), "held_out": len(held)},
        "checks": checks,
        "train_families": sorted(train_families),
        "held_out_families": sorted(held_families),
        "direct_source_usage": dict(sorted(direct_source_usage.items(), key=lambda item: int(item[0]))),
        "sha256": hashes,
        "limitations": [
            "This validator proves exact family-key disjointness and structural citation-format presence; it does not replace an independent claim-support audit.",
            "Subtle semantic adjacency across differently named families still requires human/independent review, same as corpus v2's own stated limitation.",
            "The 1600-character answer-length check is a crude verbatim-copy smell test, not a citation-support or quote-length policy check.",
        ]
    }
    (ROOT / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "checks": len(checks), "counts": report["counts"]}))


if __name__ == "__main__":
    main()
