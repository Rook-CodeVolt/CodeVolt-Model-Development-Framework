#!/usr/bin/env python3
"""Register the ADR-0019 held-out preference-pair set with
``HeldOutExclusionRegistry`` (``packages/held-out-eval``) -- ADR-0019
section 2, bullet 4 (the fourth of the five section-2 audit checks).

This is check 4 specifically: the belt to check 1-3's token-overlap-
heuristic braces above them (``validate_dataset.py`` in this directory) --
a structural, bidirectional check against *every* other package's
registered ``train`` ids in this repository, not just the two files
``validate_dataset.py`` already compares against directly.

Follows the same registration pattern every existing package in this
repository uses (see ``examples/pilot-adr0011/generate_dataset.py``,
``examples/pilot-metatrainer-v2/run_bounded_cycle_adr0018.py``): build a
fresh, empty ``HeldOutExclusionRegistry``, register every other package's
*train* ids first (so this new held-out set is checked against the real,
current state of every other package's train contribution, not just the
two files this set was built to avoid), then register this package's own
20 ids as ``held_out`` under a fresh package id
(``pilot-metatrainer-v3-dpo-heldout-adr0019``). If ``register_package_held_out``
raises ``ContaminationError``, this script exits non-zero and writes
nothing -- the same fail-closed behavior the CLI's own ``register-held-out``
subcommand documents.

Writes ``held_out_exclusion_registry.json`` in this directory: this
package's own contribution record (schema matches
``examples/pilot-adr0011/held_out_exclusion_registry.json`` and every other
existing package's own registry-output file), not a merged/aggregate
registry snapshot -- consistent with every existing package's own registry
file being scoped to its own contribution.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages/held-out-eval/src"))

from held_out_eval import ContaminationError, HeldOutExclusionRegistry

PACKAGE_ID = "pilot-metatrainer-v3-dpo-heldout-adr0019"

# Every OTHER package's own registry-output file already committed to this
# repository, re-read fresh at registration time -- this is what makes the
# bidirectional check real ("against every other package's registered train
# ids, not just the two files named [in validate_dataset.py]", ADR-0019
# section 2 bullet 4).
OTHER_PACKAGE_REGISTRIES = [
    REPO_ROOT / "examples/pilot-adr0006/held_out_exclusion_registry.json",
    REPO_ROOT / "examples/pilot-adr0011/held_out_exclusion_registry.json",
    REPO_ROOT / "examples/pilot-metatrainer-v2/held_out_exclusion_registry.json",
    REPO_ROOT / "examples/pilot-metatrainer-v3/held_out_exclusion_registry.json",
    REPO_ROOT / "examples/safety-probes-wpb/held_out_exclusion_registry.json",
]

# The ADR-0017 training package (examples/pilot-metatrainer-v3-dpo/) has no
# committed held_out_exclusion_registry.json of its own -- its 22 pair_ids
# are registered as train data only inline, at run time, by
# examples/pilot-metatrainer-v2/run_bounded_cycle_adr0018.py (package id
# "pilot-metatrainer-v3-dpo-train"). Since this is the single package
# ADR-0019 section 2 names explicitly as the one this held-out set must
# never overlap, its 22 pair_ids are read directly from the real, on-disk
# preference_pairs.jsonl here (not from a registry file that does not
# exist) and registered under the exact same package id that runner uses,
# so this script's bidirectional structural check covers it too, not just
# the token-overlap heuristic in validate_dataset.py.
ADR0017_TRAIN_PACKAGE_ID = "pilot-metatrainer-v3-dpo-train"
ADR0017_TRAIN_PAIRS_PATH = REPO_ROOT / "examples/pilot-metatrainer-v3-dpo/preference_pairs.jsonl"


def _train_ids_from_legacy_payload(payload: dict) -> list[str]:
    """Extract a flat list of train ids from either registry file shape.

    Two on-disk shapes exist in this repository: the plain
    ``HeldOutExclusionRegistry.to_dict()`` shape
    (``package_train_ids``/``package_held_out_ids``, e.g.
    ``pilot-adr0011``/``pilot-adr0006``) and the older, richer
    ``pilot-metatrainer-v2``/``v3`` "family manifest" shape
    (``train_families`` list of ``{semantic_family, example_ids, ...}``
    plus no flat ``package_train_ids`` key). Both are read here so this
    registration checks against every package's real train contribution
    regardless of which generation of the schema wrote its file.
    """
    if "package_train_ids" in payload:
        ids: list[str] = []
        for pkg_ids in payload["package_train_ids"].values():
            ids.extend(pkg_ids)
        return ids
    if "train_families" in payload:
        ids = []
        for family in payload["train_families"]:
            ids.extend(family.get("example_ids", []))
        return ids
    return []


def _train_package_id_from_legacy_payload(payload: dict, path: Path) -> str:
    if "package_train_ids" in payload:
        keys = list(payload["package_train_ids"].keys())
        if len(keys) == 1:
            return keys[0]
        return f"{path.parent.name}-train (multi-key)"
    if "train_families" in payload:
        return f"{path.parent.name}-train"
    return f"{path.parent.name}-train (empty)"


def build_registry() -> HeldOutExclusionRegistry:
    registry = HeldOutExclusionRegistry()
    for path in OTHER_PACKAGE_REGISTRIES:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        train_ids = _train_ids_from_legacy_payload(payload)
        if not train_ids:
            continue
        pkg_id = _train_package_id_from_legacy_payload(payload, path)
        registry.register_package_train(pkg_id, train_ids)

    # Register the ADR-0017 training package's real, on-disk pair_ids
    # directly (see comment above OTHER_PACKAGE_REGISTRIES) -- this is the
    # one package ADR-0019 section 2 names explicitly as forbidden overlap.
    adr0017_train_records = [
        json.loads(line)
        for line in ADR0017_TRAIN_PAIRS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    adr0017_train_ids = [r["pair_id"] for r in adr0017_train_records]
    registry.register_package_train(ADR0017_TRAIN_PACKAGE_ID, adr0017_train_ids)

    return registry


def main() -> int:
    records = [
        json.loads(line)
        for line in (ROOT / "held_out_pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    held_out_ids = [r["pair_id"] for r in records]

    registry = build_registry()
    try:
        registry.register_package_held_out(PACKAGE_ID, held_out_ids)
    except ContaminationError as exc:
        print(json.dumps({"status": "CONTAMINATION_ERROR", "detail": str(exc)}))
        return 1

    # Re-run the read-only precondition check as an extra, explicit
    # confirmation (same "re-verify at evaluation time, don't just trust the
    # split-time registration" discipline registry.py's own docstring
    # describes for run_evaluator_contract's own use of this method).
    leaked = registry.check_held_out_not_trained(held_out_ids)
    if leaked:
        print(json.dumps({"status": "LEAK_DETECTED", "detail": sorted(leaked)}))
        return 1

    # Persist this package's OWN contribution only (package_held_out_ids
    # scoped to PACKAGE_ID), matching every existing package's own
    # registry-output file convention -- not the full merged registry this
    # script built in-memory to run the check against every other package.
    own_registry = HeldOutExclusionRegistry()
    own_registry.register_package_held_out(PACKAGE_ID, held_out_ids)
    out_path = ROOT / "held_out_exclusion_registry.json"
    own_registry.save(out_path)

    print(
        json.dumps(
            {
                "status": "REGISTERED",
                "package_id": PACKAGE_ID,
                "held_out_ids_count": len(held_out_ids),
                "checked_against_other_packages": sorted(registry.package_train_ids.keys()),
                "contamination_hits": [],
                "leak_check_hits": [],
                "output_path": str(out_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
