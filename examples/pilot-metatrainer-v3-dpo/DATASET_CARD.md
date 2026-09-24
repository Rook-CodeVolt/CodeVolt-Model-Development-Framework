# ADR-0017 DPO preference-pair package — dataset card (design/dataset-shape proposal)

Status: proposal only. This corpus, its validator, and this card are the
new, standalone `prompt/chosen/rejected` triple package named in ADR-0017's
Decision section item 3 (`docs/decisions/ADR-0017-dpo-preference-refusal-axis.md`).
Repository admission of this directory does not by itself authorize any
training run — a fresh execution-config ADR with its own three-gate signing
(security, dataset-rights, owner) is required first, exactly as every prior
real cycle on this project (ADR-0013 through ADR-0016) has required, per
ADR-0017's own "What this ADR does and does not authorize" section.

## Scope and provenance

22 new preference-pair records (16 refusal-direction, 6 counter-direction —
27.3% counter share, above the security reviewer's >=20% over-refusal-guard minimum), all
newly authored for this proposal. Nothing here is copied from, or modifies,
`examples/pilot-metatrainer-v3/` — that corpus's `train.jsonl`, `held_out.json`,
and all of its hashes remain untouched, exactly as ADR-0017 requires
("`examples/pilot-metatrainer-v3/` is untouched by this proposal").

Every record contains: a stable `pair_id`; `split` (always `"train"` — this
package proposes no held-out preference pairs); `semantic_family`
(`confabulated_recipe_refusal_dpo` for refusal-direction pairs,
`calibrated_confidence_dpo` for counter-direction pairs); `direction`
(`"refusal"` or `"counter"`); `content_class` (a short label naming the
specific invented-content category the pair targets, used to prove
non-duplicate coverage); `source_scope`; a non-empty `citations` array; and
the `prompt`/`chosen`/`rejected` triple itself.

## Held-out-contamination discipline (ADR-0017's binding constraint)

ADR-0017's Decision section requires: "Every `prompt` is a **new**,
train-split record... never the literal `mtr-v2-heldout-0013` prompt text or
a trivial paraphrase of it," and the `synthetic_data_tradeoffs` family
(which contains `mtr-v2-heldout-0013`) "is **not** to be touched, read into
a prompt template, or have any of its records' prompt text reused anywhere
in the new package."

`validate_dataset.py` in this directory enforces this programmatically, not
just by author intent, by loading the *real*, on-disk
`examples/pilot-metatrainer-v3/held_out.json` fresh on every run and
checking every one of this package's 22 prompts against all 48 real
held-out prompts for: (a) exact/normalized-string reuse, (b) a token-overlap
near-paraphrase heuristic specifically against `mtr-v2-heldout-0013`'s own
prompt text (>=60% of its content words), and (c) any mention of the
forbidden `synthetic_data_tradeoffs` family label or name anywhere in the
package's text. All three checks pass with zero hits as of this proposal
(see `VALIDATION_REPORT.json`, `held_out_contamination_check`).

This is a heuristic, not a semantic-similarity model — see
`VALIDATION_REPORT.json`'s own stated limitation. It structurally proves the
absence of literal/near-literal reuse; it does not replace an independent
reviewer's own judgment on subtler semantic adjacency, same discipline as
`pilot-metatrainer-v3/validate_dataset.py`'s own stated limitation for its
`semantic_family_disjoint` check.

## Refusal-direction pairs (16 records, `confabulated_recipe_refusal_dpo`)

`chosen` = a corpus-consistent hedged/cited refusal; `rejected` = a
plausible, confidently-worded fabrication in the same register the real
ADR-0015/ADR-0016 outputs demonstrated (word-for-word "Yes, a synthetic
answer can invent a learning-rate recipe if it sounds plausible" —
`docs/decisions/ADR-0017-dpo-preference-refusal-axis.md`, "Context" section
— is the real failure this register is drawn from; no record here reuses
that exact sentence or the heldout-0013 prompt it answers).

Each of the 16 records targets a distinct `content_class` — the specific
kind of invented content the fabrication produces (an unreleased version
number, a hardware GPU count, a throughput figure, an attributed quote, a
funding amount, a leaderboard rank, a citation line number, a job title, a
compliance percentage, a dataset row count, a meeting date, a config
default, quoted error text, a contact email, a commit SHA, a model
parameter count). None of these 16 classes duplicates ADR-0016's own 9
`confabulated_recipe_detection` SFT records
(`mtr-v3n-train-0035` through `mtr-v3n-train-0043`), which already cover:
numeric training recipe, cost figure, calendar date, accuracy percentage,
citation, GPU-memory figure, rounded percentage, citation locator, and
benchmark score — this package deliberately extends into new invented-content
categories rather than re-teaching the same 9 shapes as preference pairs,
per ADR-0017's explicit "new variations beyond the 9 ADR-0016 already used"
instruction.

## Counter-direction pairs (6 records, `calibrated_confidence_dpo`)

Per the security reviewer's over-refusal gate (ADR-0017 safety finding (b)(2)): `chosen` = a
confident, correct, directly-answered response to a genuinely answerable
question; `rejected` = an unwarranted refusal/hedge on that same answerable
question.

Every `chosen` answer here states a real, independently-checkable fact
about *this project's own codebase*, each one directly re-verified during
this drafting session (not merely asserted):

1. `trl_adapter.py`'s current scope is SFT-only, no RL trainers — quoted
   directly from the module's own docstring line 11.
2. `trl_adapter.py` pins `trl==0.24.0` exactly (`TRL_MIN_VERSION`==
   `TRL_MAX_VERSION`=="0.24.0", lines 90-91).
3. The real installed `trl==0.24.0`'s `DPOConfig.beta` default is `0.1` —
   confirmed by direct `inspect.signature()` introspection of the installed
   package in `.venv-adr0014-test`, not from documentation alone (see
   the ADR-0017 review's independent verification, which confirmed the same fact
   independently).
4. LoRA's core mechanism (freeze base, learn low-rank `BA` update) per the
   LoRA paper already cited as source `[6]` in this project's own
   `SOURCE_MAP.md`.
5. `hf_local_evaluator_adapter.py` has no PEFT/LoRA adapter-merge or
   adapter-scoring logic anywhere — directly re-confirmed by a fresh module
   search during this session (grep for `PeftModel`/`merge_and_unload`/
   `import peft`/`from peft`: zero matches), matching ADR-0017's own cited
   evaluator-gap finding.
6. `trl_adapter.py`'s `model_hash` verification is content-hash-based
   (`_hash_path_identity`), not name-trusted — quoted directly from
   `prepare()`, lines 265-271.

Each `rejected` side is an unwarranted hedge/non-answer to that same,
genuinely answerable question — the calibration-preserving control this
gate requires, chosen from real project facts specifically so the "correct"
side is independently checkable by any reviewer re-reading the cited file,
not merely plausible-sounding.

## Explicit exclusions and cautions

This package does not assert:

- that 22 pairs (or any specific pair count) is sufficient DPO training
  volume — no claim about sample-count sufficiency is made here; that is
  an execution-config-stage question, not a dataset-shape question;
- a specific `beta`/KL-strength value for the eventual DPO run — the security
  reviewer's gate
  4 (explicit beta/KL review) is intentionally left to the execution-config
  document, not decided here;
- that this package alone resolves the refusal-collapse risk the security
  reviewer's gate 2
  names — the 27.3% counter-direction share is this package's concrete
  mitigation, but its *effectiveness* is an empirical question for the
  eventual training cycle's own held-out measurement (per a non-blocking
  recommendation on the PR #97 review: the execution-config
  document should make the counter-direction share's effect an explicit
  measured post-training success/failure criterion against the
  `uncertainty_refusal_boundary` rubric axis);
- a promotion, deployment, or production-readiness claim of any kind;
- a licensing or legal conclusion of any kind. No licensing determination
  was made for any content in this package (all content is originally
  authored for this proposal, not sourced from a third party requiring a
  licensing decision).

## Intended use

This corpus is a proposal for repository admission and independent review
only, consistent with `examples/pilot-metatrainer-v3/DATASET_CARD.md`'s own
"Intended use" section and this project's standing governance pattern.
Repository admission does not by itself authorize any training run; a
separate, explicitly new execution-config ADR (three-gate signing: security,
dataset-rights, owner) is required before this package, or any subset of
it, is used in a training run.

## Files

- `preference_pairs.jsonl`: 22 records, generated deterministically by
  `generate_pairs.py`.
- `generate_pairs.py`: deterministic generator; the source of truth for
  every record's content (edit here, then re-run, rather than hand-editing
  the `.jsonl` directly).
- `validate_dataset.py`: structural, counter-share, and held-out-
  contamination validator. Re-run after any content change.
- `VALIDATION_REPORT.json`: validator output, written after validation.

## Next steps (not performed by this task)

1. Independent audit of this package with the same rigor as
   `pilot-metatrainer-v3`'s own corpus audits — full per-pair citation
   re-verification (especially the 6 counter-direction pairs' codebase-fact
   claims, each independently re-checked against the live file), and
   independent semantic-adjacency review of the held-out-contamination
   heuristic's negative result (not just trusting the token-overlap
   threshold).
2. Security-reviewer/dataset-rights review of this package and the accompanying
   adapter code PR (see `src/codevolt_mdf/dpo_adapter.py`), per this
   project's standing governance pattern.
3. A fresh execution-config ADR (three-gate signing: security,
   dataset-rights, owner) is required before this package is used in any
   training run — this proposal authorizes drafting it, not skipping it.
