#!/usr/bin/env python3
"""Deterministic synthetic dataset generator for ADR-0006's bounded real
TRL pilot -- 2-digit addition, fixed format, disjoint train/held-out id
namespaces and disjoint numeric pairs so the held-out set is not merely
an unseen id over seen content.

Per ADR-0006's "Dataset" section: "The pilot's specific dataset must be
constructed new for this pilot ... reusing any held-out set from
evaluator_contract.py's test fixtures or from any other package's
registered held-out ids is explicitly disallowed." This generator
produces exactly that -- new content, new package ids, checked against
the live HeldOutExclusionRegistry before being accepted as this pilot's
locked dataset.

Deterministic: fixed seed, reproducible byte-for-byte on every run.
Not executed as part of any test; run directly to (re)generate
train.jsonl / held_out.json / held_out_exclusion_registry.json.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20260917
TRAIN_PACKAGE_ID = "adr0006-pilot-train-v1"
HELDOUT_PACKAGE_ID = "adr0006-pilot-heldout-v1"
N_TRAIN = 40
N_HELDOUT = 10


def make_pairs(rng: random.Random, count: int, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    seen = set(exclude)
    while len(pairs) < count:
        a = rng.randint(10, 79)
        b = rng.randint(10, 79)
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def render_text(a: int, b: int) -> str:
    return f"Add: {a} + {b} = {a + b}"


def render_prompt(a: int, b: int) -> str:
    return f"Add: {a} + {b} ="


def render_expected(a: int, b: int) -> str:
    return str(a + b)


def main() -> int:
    rng = random.Random(SEED)
    train_pairs = make_pairs(rng, N_TRAIN, exclude=set())
    # Held-out pairs must be disjoint from train pairs by *content*, not
    # only by id -- otherwise a held-out id could point at a numeric
    # problem the model already saw verbatim during training.
    held_out_pairs = make_pairs(rng, N_HELDOUT, exclude=set(train_pairs))

    train_dir = HERE / "dataset"
    train_dir.mkdir(parents=True, exist_ok=True)
    train_path = train_dir / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for idx, (a, b) in enumerate(train_pairs, start=1):
            record = {"text": render_text(a, b)}
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    held_out_examples = [
        {
            "example_id": f"heldout-{idx:04d}",
            "input": render_prompt(a, b),
            "expected": render_expected(a, b),
        }
        for idx, (a, b) in enumerate(held_out_pairs, start=1)
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

    print(f"train pairs: {len(train_pairs)} -> {train_path}")
    print(f"held-out pairs: {len(held_out_pairs)} -> {held_out_path}")
    overlap = set(train_pairs) & set(held_out_pairs)
    print(f"train/held-out numeric-pair overlap: {len(overlap)} (must be 0)")
    if overlap:
        print("FATAL: overlap detected", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
