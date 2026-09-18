#!/usr/bin/env python3
"""Deterministic synthetic dataset generator for ADR-0011's bounded
MiniMind pilot -- single-turn SI-prefix unit-conversion task, MiniMind
chat-JSONL format, disjoint train/held-out id namespaces AND disjoint
numeric/unit content so the held-out set is not merely an unseen id
over already-trained content.

Per ADR-0011's "Dataset" section: distinct in both *format* (MiniMind's
own ``{"conversations": [...]}`` chat schema, not TRL's flat
prompt/completion text) and *content* (unit conversion, not ADR-0006's
arithmetic and not ADR-0008's safety probes) from every existing
package registered in this repository. Reusing any existing held-out
set as either split is explicitly disallowed by ADR-0011 and would be
caught by ``HeldOutExclusionRegistry.check_held_out_not_trained`` if
attempted after registration.

Deterministic: fixed seed, reproducible byte-for-byte on every run.
Not executed as part of any test; run directly to (re)generate
train.jsonl / held_out.json / held_out_exclusion_registry.json.

This script builds dataset *artifacts only*. It does not call
``MiniMindTrainerAdapter``, does not invoke any training engine, and
does not import ``minimind_adapter.py``.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20260918
TRAIN_PACKAGE_ID = "adr0011-pilot-train-v1"
HELDOUT_PACKAGE_ID = "adr0011-pilot-heldout-v1"
N_TRAIN = 40
N_HELDOUT = 10

# (unit_from, unit_to, factor, value_min, value_max)
# factor: multiply a value in unit_from by factor to get the value in unit_to.
# All conversions are exact under this factor for the integer value ranges
# chosen below, so "expected" is always an exact integer, not a rounded
# approximation -- avoids any floating-point ambiguity in exact-match scoring.
CONVERSIONS: list[tuple[str, str, int, int, int]] = [
    ("kilometres", "metres", 1000, 1, 40),
    ("metres", "centimetres", 100, 1, 90),
    ("kilograms", "grams", 1000, 1, 40),
    ("grams", "milligrams", 1000, 1, 90),
    ("litres", "millilitres", 1000, 1, 40),
    ("centilitres", "millilitres", 10, 1, 90),
    ("metres", "millimetres", 1000, 1, 40),
    ("kilometres", "centimetres", 100000, 1, 9),
]


def render_prompt(value: int, unit_from: str, unit_to: str) -> str:
    return f"Convert {value} {unit_from} to {unit_to}."


def render_answer(value: int, unit_from: str, unit_to: str, factor: int) -> str:
    converted = value * factor
    return f"{value} {unit_from} = {converted} {unit_to}."


def render_expected(value: int, factor: int) -> str:
    return str(value * factor)


def make_items(
    rng: random.Random, count: int, exclude: set[tuple[str, str, int]]
) -> list[tuple[str, str, int, int]]:
    """Sample ``count`` distinct (unit_from, unit_to, factor, value) tuples.

    Excludes any (unit_from, unit_to, value) triple already present in
    ``exclude`` -- disjointness is enforced on the *content* triple, not
    merely on a generated id, mirroring
    ``examples/pilot-adr0006/generate_dataset.py``'s own disjointness
    discipline.
    """
    items: list[tuple[str, str, int, int]] = []
    seen = set(exclude)
    attempts = 0
    max_attempts = count * 200
    while len(items) < count and attempts < max_attempts:
        attempts += 1
        unit_from, unit_to, factor, lo, hi = rng.choice(CONVERSIONS)
        value = rng.randint(lo, hi)
        key = (unit_from, unit_to, value)
        if key in seen:
            continue
        seen.add(key)
        items.append((unit_from, unit_to, factor, value))
    if len(items) < count:
        raise RuntimeError(
            f"could only sample {len(items)}/{count} distinct conversion items "
            "-- widen CONVERSIONS' value ranges or reduce N_TRAIN/N_HELDOUT"
        )
    return items


def main() -> int:
    rng = random.Random(SEED)

    train_items = make_items(rng, N_TRAIN, exclude=set())
    train_keys = {(u_from, u_to, value) for (u_from, u_to, _factor, value) in train_items}
    # Held-out items must be disjoint from train items by *content*
    # (unit_from, unit_to, value), not only by id -- otherwise a
    # held-out id could point at a conversion the model already saw
    # verbatim during training.
    held_out_items = make_items(rng, N_HELDOUT, exclude=train_keys)
    held_out_keys = {(u_from, u_to, value) for (u_from, u_to, _factor, value) in held_out_items}

    # Train split: MiniMind SFTDataset chat-JSONL schema --
    # {"conversations": [{"role": "user", ...}, {"role": "assistant", ...}]}
    # per dataset/lm_dataset.py, NOT TRL's flat prompt/completion text
    # ADR-0006's examples/pilot-adr0006/dataset/train.jsonl used.
    train_dir = HERE
    train_path = train_dir / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for unit_from, unit_to, factor, value in train_items:
            record = {
                "conversations": [
                    {"role": "user", "content": render_prompt(value, unit_from, unit_to)},
                    {
                        "role": "assistant",
                        "content": render_answer(value, unit_from, unit_to, factor),
                    },
                ]
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    held_out_examples = [
        {
            "example_id": f"adr0011-heldout-{idx:04d}",
            "input": render_prompt(value, unit_from, unit_to),
            "expected": render_expected(value, factor),
        }
        for idx, (unit_from, unit_to, factor, value) in enumerate(held_out_items, start=1)
    ]
    held_out_path = HERE / "held_out.json"
    held_out_path.write_text(
        json.dumps(
            {
                "package_id": HELDOUT_PACKAGE_ID,
                "examples": held_out_examples,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    train_ids = [f"adr0011-train-{idx:04d}" for idx in range(1, len(train_items) + 1)]
    held_out_ids = [ex["example_id"] for ex in held_out_examples]
    registry_path = HERE / "held_out_exclusion_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "package_train_ids": {TRAIN_PACKAGE_ID: sorted(train_ids)},
                "package_held_out_ids": {HELDOUT_PACKAGE_ID: sorted(held_out_ids)},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"train items: {len(train_items)} -> {train_path}")
    print(f"held-out items: {len(held_out_items)} -> {held_out_path}")
    overlap = train_keys & held_out_keys
    print(f"train/held-out content-triple overlap: {len(overlap)} (must be 0)")
    if overlap:
        print("FATAL: overlap detected", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
