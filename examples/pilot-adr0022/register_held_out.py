#!/usr/bin/env python3
"""Register the ADR-0022 item set with HeldOutExclusionRegistry
(packages/held-out-eval) -- ADR-0022 section 5 / section 12 item 2.

Precondition of scoring, not scoring itself: builds a fresh registry
containing every OTHER existing package's real, on-disk train
contribution (re-read fresh, never trusted from a remembered list),
then registers this package's own 56 item ids as held-out under a
fresh package id distinct from every existing registered package. If
register_package_held_out raises ContaminationError, this script exits
non-zero and writes nothing -- the same fail-closed behaviour every
prior register_held_out.py in this repository uses.

Mirrors examples/pilot-metatrainer-v3-dpo-heldout/register_held_out.py's
structure exactly, extended to the additional registry files that now
exist in this repository.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages/held-out-eval/src"))

from held_out_eval import ContaminationError, HeldOutExclusionRegistry

PACKAGE_ID = "pilot-adr0022-heldout-v1"

# Every OTHER package's own registry-output file already committed to this
# repository, re-read fresh at registration time.
OTHER_PACKAGE_REGISTRIES = [
    REPO_ROOT / "examples/pilot-adr0006/held_out_exclusion_registry.json",
    REPO_ROOT / "examples/pilot-adr0011/held_out_exclusion_registry.json",
    REPO_ROOT / "examples/pilot-metatrainer-v2/held_out_exclusion_registry.json",
    REPO_ROOT / "examples/pilot-metatrainer-v3/held_out_exclusion_registry.json",
    REPO_ROOT / "examples/pilot-metatrainer-v3-dpo-heldout/held_out_exclusion_registry.json",
    REPO_ROOT / "examples/safety-probes-wpb/held_out_exclusion_registry.json",
]

# The ADR-0017 training package (examples/pilot-metatrainer-v3-dpo/) has no
# committed held_out_exclusion_registry.json of its own -- its 22 pair_ids
# are registered as train data only inline, at run time, by
# run_bounded_cycle_adr0018.py (package id "pilot-metatrainer-v3-dpo-train").
# Read directly from the real, on-disk preference_pairs.jsonl here, same
# pattern the ADR-0019 held-out registration script established.
ADR0017_TRAIN_PACKAGE_ID = "pilot-metatrainer-v3-dpo-train"
ADR0017_TRAIN_PAIRS_PATH = REPO_ROOT / "examples/pilot-metatrainer-v3-dpo/preference_pairs.jsonl"

ITEM_FILES = [
    ROOT / "items_c1.json",
    ROOT / "items_c2.json",
    ROOT / "items_c4.json",
]


def _train_ids_from_legacy_payload(payload: dict) -> list[str]:
    """Extract a flat list of train ids from either registry file shape.

    Two on-disk shapes exist in this repository: the plain
    HeldOutExclusionRegistry.to_dict() shape (package_train_ids/
    package_held_out_ids) and the older, richer pilot-metatrainer-v2/v3
    "family manifest" shape (train_families list). Both are read here.
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

    if ADR0017_TRAIN_PAIRS_PATH.exists():
        adr0017_train_records = [
            json.loads(line)
            for line in ADR0017_TRAIN_PAIRS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        adr0017_train_ids = [r["pair_id"] for r in adr0017_train_records]
        registry.register_package_train(ADR0017_TRAIN_PACKAGE_ID, adr0017_train_ids)

    return registry


def load_item_ids() -> list[str]:
    ids: list[str] = []
    for path in ITEM_FILES:
        payload = json.loads(path.read_text(encoding="utf-8"))
        ids.extend(item["example_id"] for item in payload["items"])
    return ids


def main() -> int:
    item_ids = load_item_ids()
    if len(item_ids) != 56:
        print(json.dumps({"status": "SCHEMA_ERROR", "detail": f"expected 56 item ids, got {len(item_ids)}"}))
        return 1
    if len(item_ids) != len(set(item_ids)):
        print(json.dumps({"status": "SCHEMA_ERROR", "detail": "duplicate example_id across item files"}))
        return 1

    registry = build_registry()
    try:
        registry.register_package_held_out(PACKAGE_ID, item_ids)
    except ContaminationError as exc:
        print(json.dumps({"status": "CONTAMINATION_ERROR", "detail": str(exc)}))
        return 1

    leaked = registry.check_held_out_not_trained(item_ids)
    if leaked:
        print(json.dumps({"status": "LEAK_DETECTED", "detail": sorted(leaked)}))
        return 1

    own_registry = HeldOutExclusionRegistry()
    own_registry.register_package_held_out(PACKAGE_ID, item_ids)
    out_path = ROOT / "held_out_exclusion_registry.json"
    own_registry.save(out_path)

    print(
        json.dumps(
            {
                "status": "REGISTERED",
                "package_id": PACKAGE_ID,
                "held_out_ids_count": len(item_ids),
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
