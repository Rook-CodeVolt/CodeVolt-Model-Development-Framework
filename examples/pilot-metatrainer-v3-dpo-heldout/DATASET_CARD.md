# ADR-0019 held-out preference-pair set — dataset card

Status: held-out asset proposal only. This corpus, its validator, and this
card are the 20-pair held-out preference-pair set named in ADR-0019 section
2 (`docs/decisions/ADR-0019-held-out-logprob-margin-eval.md`, PR #102, base
commit `ad610be8789035e0602f01802cf71e5bb8fd894b`). Repository admission of
this directory does not by itself authorize any scoring/evaluation run —
ADR-0019 section 5's proportionate gate (Maya's dataset-admissibility and
evaluation-scope review) must clear first, and the evaluation script itself
is separate follow-up work (ADR-0019 section 6, card 2) that this task does
not write.

## Scope and provenance

20 new preference-pair records (16 refusal-direction, 4 counter-direction —
20.0% counter share, meeting ADR-0019 section 2 bullet 2's `>= 20%` minimum
exactly), all newly authored for this proposal. Nothing here is copied
from, or modifies, `examples/pilot-metatrainer-v3-dpo/`'s own 22-record
training package or its locked hashes, and nothing here touches
`examples/pilot-metatrainer-v3/`'s existing 48-item held-out set — both
remain untouched, exactly as ADR-0019 section 4/"Explicitly out of scope"
requires.

Every record contains: a stable `pair_id`; `split` (always `"held_out"` —
this field is load-bearing for `HeldOutExclusionRegistry` registration,
never `"train"`); `semantic_family` (`confabulated_recipe_refusal_dpo` for
refusal-direction pairs, `calibrated_confidence_dpo` for counter-direction
pairs — the same two families the ADR-0017 training package uses, since
this set targets the same behaviour axis); `direction` (`"refusal"` or
`"counter"`); `content_class` (a short label naming the specific
invented-content category the pair targets, used to prove non-duplicate
coverage against the training package); `source_scope`; a non-empty
`citations` array; and the `prompt`/`chosen`/`rejected` triple itself.

## Construction rules and how each is met (ADR-0019 section 2)

1. **Same triple shape as the ADR-0017 training package.** Every record has
   the identical field set (`pair_id`, `prompt`, `chosen`, `rejected`,
   `direction`, `semantic_family`, `content_class`, `source_scope`,
   `citations`), with `"split": "held_out"` instead of `"train"`.
2. **Counter-direction share >= 20%, computed and reported.** 4 of 20
   records are counter-direction: `4/20 = 0.2000` (see
   `VALIDATION_REPORT.json`, `counts.counter_share`), computed
   programmatically by `validate_dataset.py`, not merely asserted.
3. **`content_class` disjoint from the 22 training pairs.** Re-read fresh
   from `examples/pilot-metatrainer-v3-dpo/preference_pairs.jsonl` at
   validation time (not assumed): the training package's 22 `content_class`
   values and this set's 20 `content_class` values have zero overlap (see
   `VALIDATION_REPORT.json`, `content_class_disjoint_from_training` check).
   This set targets the same `confabulated_recipe_refusal_dpo`/
   `calibrated_confidence_dpo` behaviour axis via 20 new invented-content
   categories (e.g. unpublished benchmark names, unmeasured latency
   figures, unconfirmed acceptance rates — refusal side; and real facts
   about `packages/held-out-eval` specifically, a different file set than
   the training package's 6 `trl_adapter.py`/`hf_local_evaluator_adapter.py`
   counter facts — counter side), never rephrasing one of the 22 already
   used.

## Held-out-contamination discipline (ADR-0019 section 2's binding audit)

`validate_dataset.py` in this directory (extending, not duplicating,
`examples/pilot-metatrainer-v3-dpo/validate_dataset.py`'s own
`_normalize`/`_token_set` helpers) and `register_held_out.py` together
implement all five of ADR-0019 section 2's audit checks, self-run by Marcus
(the pair constructor) as recorded below. **Per ADR-0019 section 2's "Who
builds it" decision, this is a self-check only — all five checks must be
independently re-run and sealed by Maya, not by the pair constructor,
before this set is used for anything.**

| # | Check | Mechanism | Self-check result |
|---|---|---|---|
| 1 | Zero overlap with the 22 ADR-0017 training pairs | Exact/normalized-string match on `prompt` text against the real, on-disk `examples/pilot-metatrainer-v3-dpo/preference_pairs.jsonl` | PASS, 0 hits |
| 2 | Zero overlap with the 48 real held-out prompts | Exact/normalized-string match against the real, on-disk `examples/pilot-metatrainer-v3/held_out.json` | PASS, 0 hits |
| 3 | Zero overlap with `mtr-v2-heldout-0013` specifically | Exact match plus the same >=60%-token-overlap near-paraphrase heuristic `pilot-metatrainer-v3-dpo/validate_dataset.py` already applies | PASS, 0 hits (registry item present, 76 chars, confirming no registry drift) |
| 4 | Registration through `HeldOutExclusionRegistry` | `register_held_out.py`: registers this set's 20 `pair_id`s as `held_out` under package id `pilot-metatrainer-v3-dpo-heldout-adr0019`, after registering every other package's real train ids (including the ADR-0017 training package's 22 `pair_id`s under `pilot-metatrainer-v3-dpo-train`) into the same in-memory registry first | PASS — `register_package_held_out` raised nothing; `check_held_out_not_trained` re-confirms 0 leaked ids |
| 5 | Counter-direction share | `validate_dataset.py`'s `counter_direction_share_min_20pct` check | PASS, `0.2000 >= 0.20` |

This is a heuristic-plus-structural audit, not a semantic-similarity model —
see `VALIDATION_REPORT.json`'s own stated limitations. It structurally
proves the absence of literal/near-literal/exact-id reuse; it does not
replace an independent reviewer's own judgment on subtler semantic
adjacency, the same discipline every prior validator in this project states
for its own contamination check.

## Refusal-direction pairs (16 records, `confabulated_recipe_refusal_dpo`)

`chosen` = a corpus-consistent hedged/cited refusal; `rejected` = a
plausible, confidently-worded fabrication in the same register the ADR-0017
training package's own 16 refusal pairs use. Each of the 16 records targets
a distinct `content_class`: an unpublished benchmark dataset name, an
unmeasured model latency figure, an unconfirmed conference acceptance rate,
a hypothetical ablation result percentage, an unlisted repository star
count, an undisclosed salary range, an unverified patent application
number, an unrecorded test-coverage percentage, a speculative bug-fix
release version, an unconfirmed dataset license type, an unmeasured carbon
footprint figure, an invented survey respondent count, an unverified
conference venue location, a hypothetical accuracy figure on an unreleased
benchmark, an unrecorded build duration, and a fabricated API rate-limit
value. None of these 16 classes duplicates any of the ADR-0017 training
package's 16 refusal-direction classes, nor ADR-0016's 9
`confabulated_recipe_detection` SFT classes the training package itself
already avoided.

## Counter-direction pairs (4 records, `calibrated_confidence_dpo`)

Per ADR-0019 section 2 bullet 2's over-refusal-guard rationale (adopting
Maya's ADR-0017 gate (b)(2) identically here): `chosen` = a confident,
correct, directly-answered response to a genuinely answerable question;
`rejected` = an unwarranted refusal/hedge on that same answerable question.

Every `chosen` answer here states a real, independently-checkable fact
about `packages/held-out-eval` and this repository's own CI/governance
files specifically — a deliberately different file set than the training
package's 6 counter facts (which are all about `trl_adapter.py`/
`hf_local_evaluator_adapter.py`), each directly re-verified during this
drafting session:

1. `packages/held-out-eval` has zero runtime dependencies — confirmed
   directly from its `pyproject.toml`'s `dependencies = []` and the
   accompanying comment.
2. `HeldOutExclusionRegistry`'s current on-disk schema version is `1` —
   quoted directly from `registry.py`'s `CURRENT_SCHEMA_VERSION = 1`.
3. This repository's CI tests against Python 3.9, 3.11, and 3.13 — quoted
   directly from `.github/workflows/ci.yml`'s `matrix.python-version`.
4. `packages/held-out-eval` ships under Apache-2.0 — quoted directly from
   its `pyproject.toml`'s `license` field and its README's own "License"
   section.

Each `rejected` side is an unwarranted hedge/non-answer to that same,
genuinely answerable question, chosen from real, independently checkable
project facts specifically so the "correct" side is verifiable by any
reviewer re-reading the cited file, not merely plausible-sounding.

## Explicit exclusions and cautions

This package does not assert:

- that 20 pairs is a sufficient sample size to detect a small-to-medium
  held-out log-prob margin shift — ADR-0019 section 3 states this sample's
  power limits honestly (roughly 80% power to detect only a large paired
  effect at this `n`); this dataset-shape proposal makes no claim beyond
  meeting ADR-0019's own pre-specified construction rules;
- a specific evaluation outcome or decision-rule result — this document
  proposes and validates a *set*, it does not run any scoring pass (ADR-0019
  section 6, card 2/4 are separate, not performed here);
- that this self-check audit (above) is a substitute for Maya's independent
  re-run of all five section-2 checks — per ADR-0019 section 2's own "Who
  builds it" decision, it is explicitly not;
- a promotion, deployment, or production-readiness claim of any kind;
- a licensing or legal conclusion of any kind. All content in this package
  is originally authored for this proposal, not sourced from a third party
  requiring a licensing decision.

## Intended use

Per ADR-0019 section 2's "Does it become a permanent held-out asset?"
answer: **yes, if it passes Maya's audit.** Once independently sealed, this
set is added to this project's standing held-out inventory (alongside the
existing 48-item `mtr-v2-heldout-*`/`mtr-v3n-heldout-*` set) for reuse by
future preference-training evaluations on this same axis. It is explicitly
**not** added to `examples/pilot-metatrainer-v3-dpo/`'s own training
content or hash-locked files, and its future reuse as held-out data does
not itself authorize any new training run. Repository admission of this
directory does not by itself authorize the ADR-0019 scoring pass — that
requires ADR-0019 section 5's gate to clear first, against a script this
task explicitly does not write (ADR-0019 section 6, card 2).

## Files

- `held_out_pairs.jsonl`: 20 records, generated deterministically by
  `generate_pairs.py`.
- `generate_pairs.py`: deterministic generator; the source of truth for
  every record's content (edit here, then re-run, rather than hand-editing
  the `.jsonl` directly).
- `validate_dataset.py`: structural, counter-share, content_class-
  disjointness, and held-out-contamination validator; reuses
  `examples/pilot-metatrainer-v3-dpo/validate_dataset.py`'s
  `_normalize`/`_token_set` helpers rather than duplicating them. Re-run
  after any content change.
- `VALIDATION_REPORT.json`: validator output, written after validation.
- `register_held_out.py`: registers this set's 20 ids as `held_out` via
  `HeldOutExclusionRegistry`, checked against every other package's real
  train ids in this repository (ADR-0019 section 2 bullet 4). Writes
  `held_out_exclusion_registry.json`.
- `held_out_exclusion_registry.json`: this package's own registry
  contribution (`package_held_out_ids` keyed by
  `pilot-metatrainer-v3-dpo-heldout-adr0019`), written by `register_held_out.py`.

## Next steps (not performed by this task)

1. **Maya's independent re-run of all five section-2 audit checks**,
   distinct from this self-check, per ADR-0019 section 2's "Who builds it"
   decision — sealing this set in `HeldOutExclusionRegistry` is her call,
   not this task's.
2. **Write the held-out log-prob margin evaluation script** (ADR-0019
   section 3/6 card 2) — separate follow-up work, not performed here per
   this task's explicit scope (no evaluation script, no scoring).
3. **Maya's gate review** of ADR-0019 section 5's proposed proportionate
   gate (ADR-0019 section 6, card 3), against the real script and this
   sealed dataset.
4. **Execute the evaluation exactly once** (ADR-0019 section 6, card 4), by
   a different specialist than this set's author, once card 3's gate
   clears.
