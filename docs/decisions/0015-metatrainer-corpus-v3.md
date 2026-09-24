# ADR-0015: Meta-trainer corpus v3 — grow corpus to ~160 examples with calibrated-refusal emphasis

- Status: proposed — repository-admission proposal only; does not authorize any
  training run
- Date: 2026-09-22
- Tracking: internal execution records (see linked PR numbers above where applicable)
- Supersedes: does not repeal ADR-0013 or ADR-0014. Both remain the final,
  evidence-preserved, honest record of their own runs. This ADR proposes new
  training material for a possible future bounded cycle; it is not itself a
  training-execution ADR and grants no training authority.

## Context

ADR-0014's corrected hyperparameter configuration (1 epoch, LR 5e-6) fixed the
catastrophic-forgetting regression seen in ADR-0013 (safety and arithmetic
retention held or nearly held), but the run still missed its own promotion
rubric threshold (a real 0.45/4.0 against the required >=3.0/4.0, per
ADR-0013's inherited "Pass, continue, and stop thresholds" section) and the
`meta_trainer` target-task exact-match score remained 0.0%, as it had across
all three real runs to date (untrained baseline, ADR-0013 candidate, ADR-0014
candidate).

An independent root-cause diagnosis established that the flat
0.0% reading is explained by the exact-match string-containment scorer having
no dynamic range for this free-form, sentence-length-answer task — the
untrained baseline scores identically to both trained candidates, which rules
out training mechanics as the explanation for the literal zero. That
diagnosis explicitly left corpus size (the current 40-example train split) as
a real but currently unfalsifiable compounding factor, separate from the
scorer-mismatch question, and recommended (among other options) growing the
corpus as one part of addressing it — while being clear that a corpus change
alone does not resolve the scorer-mismatch question.

The same evaluation run also captured a real, safety-relevant confabulation:
on held-out item `mtr-v2-heldout-0013`, the ADR-0014 candidate invented a
specific, uncited numeric learning-rate recipe ("The learning rate is set to
0.01...") with no supporting source and no correspondence to this project's
own reviewed hyperparameter guidance. This is a materially worse failure mode
than a merely wrong or degenerate answer, because a confident, precise-looking
fabrication is easy to mistake for a validated recommendation.

The owner's decision (recorded in this ADR's commissioning task):
grow the training corpus, keeping the existing full-SFT pipeline and
exact-match scorer unchanged for now. Architecture change and eval-methodology
change are explicitly out of scope for this ADR and are separate future
design work. This ADR is the corpus-growth response; it does not itself
change the trainer, the evaluator, or any threshold.

## Decision

Propose `examples/pilot-metatrainer-v3/`, a new, standalone dataset directory
(160 records: 112 train / 48 held-out) for repository admission, built from
three sources:

1. **meta-trainer-corpus-v2** (60 records: 40 train / 20 held-out), copied
   byte-identical from `examples/pilot-metatrainer-v2/{train.jsonl,held_out.json}`
   — independently audited to 60/60 in task. No content, citation,
   or split assignment is changed.
2. **The training-history case-study addition** (12 records: 8 train / 4
   held-out) from `examples/metatrainer-corpus-addition-training-history/`,
   independently audited in task (11/12 clean; the one required
   fix — rewriting `mtr-hist-case-heldout-0001` to test ADR-0011's explicit
   non-authorization-of-further-pilot scope statement instead of restating a
   train-side point — applied here exactly as the audit recommended).
3. **New material** (88 records: 64 train / 24 held-out) added by this
   proposal, spanning 11 new semantic families, weighted toward calibrated
   uncertainty and refusal per the real confabulation finding above. Sources
   [24]-[30] (Kadavath et al.; TruthfulQA; Kalai et al.; SelfCheckGPT; OpenAI
   Model Spec's "Express uncertainty" guideline; the Hugging Face exact_match
   metric card; and the real pilot Anthropic-OpenAI alignment evaluation
   comparing refusal and hallucination rates) were fetched and read directly
   for this proposal, in the same public/synthetic-Q&A citation pattern as
   corpus v2's existing 23 sources — no source is reproduced verbatim beyond
   short, attributed, directly-quoted phrases.

Full detail (scope, provenance, split construction, citation policy, explicit
exclusions) is in `examples/pilot-metatrainer-v3/DATASET_CARD.md`. Source
verification detail is in `examples/pilot-metatrainer-v3/SOURCE_MAP.md` and
`SYNTHESIS_NOTES.md`. A self-audit against the same C1-C6-equivalent
checklist used for corpus v2 is in
`examples/pilot-metatrainer-v3/SELF_AUDIT_REPORT.md`, including its own
explicitly stated limitation (self-review only, not yet an independent
second-reviewer audit of the 88 new records) and two specific citation
sub-claims flagged for independent re-confirmation.

This ADR does not touch, replace, or invalidate:

- `examples/pilot-metatrainer-v2/train.jsonl` / `held_out.json`, or their
  recorded SHA-256 hashes (`6fa2d114a822ee081ba09d5d4f31ef4dd107fe92c43d6037abbd9c98492f17da`
  / `d99b76b3d24298fa7660b1335835bf37a1e16e7bca858c9f3f980f21e70017a4`), which
  ADR-0013 and ADR-0014 gate on and already recorded honest (negative) results
  against;
- `examples/metatrainer-corpus-addition-training-history/`;
- any ADR-0013/ADR-0014 dataset_hash, review-gate, or approval-namespace
  value.

`examples/pilot-metatrainer-v3/generate_corpus.py` is a deterministic
generator that reads the two carried-forward sources unchanged (plus the one
documented, audit-recommended fix) and appends only new material; it does not
modify any file under `pilot-metatrainer-v2/` or
`metatrainer-corpus-addition-training-history/`. `validate_dataset.py` is
extended from corpus v2's validator for the larger size window (100-120
train / 40-50 held-out), the larger family count (32 families total: 8
v2-train + 3 v2-held-out unchanged, 4 history-train + 4 history-held-out, 8
new-train + 3 new-held-out), and a broader recognized citation-locator format
(numbered `[1]`-`[30]` sources, `research synthesis, MSx` labels, carried-forward
`research synthesis, SSx.x` labels, and free-text repository-evidence locators).
A fresh run of `validate_dataset.py` in this proposal's own session reports
`{"status": "PASS", "checks": 16, "counts": {"total": 160, "train": 112,
"held_out": 48}}`; this record is written to
`examples/pilot-metatrainer-v3/VALIDATION_REPORT.json`.

## What this ADR does and does not authorize

This ADR proposes repository admission of a dataset only. Consistent with
corpus v2's own "Intended use" section and ADR-0013/ADR-0014's governance
pattern:

- Repository admission (merging this PR) does not by itself authorize any
  training run.
- A future training run using this corpus (in whole or in part) requires a
  separate, explicitly new governance decision (its own ADR, its own fresh
  three-gate signing — security, dataset-rights, owner — per ADR-0014's own
  precedent of not reusing ADR-0013's already-consumed approvals) before
  `--execute` on any runner.
- No licensing or legal determination is made for any source cited in this
  corpus, new or carried-forward.
- No promotion, deployment, or production-readiness claim is made about any
  model.

## Review path

Per this project's standing governance pattern for anything proposed for the
training corpus:

1. Independent audit of the 88 new records with the same rigor as corpus v2's
   own audits and the training-history addition's audit (full per-example
   citation-locator re-verification against the actual source text,
   independent semantic-split content review — not a self-report). The 72
   carried-forward records already have this independent audit history
    and are not re-audited here.
2. Security/dataset-rights review of this PR, per this repository's
   existing governance pattern for training-corpus proposals.
3. Repository admission, if both above clear. This does not by itself
   authorize training (see above).

## Consequences

- ADR-0013's and ADR-0014's rejected/negative-rubric results and their
  evidence remain the authoritative, unchanged record of those two runs; this
  ADR does not retroactively reinterpret either result.
- The exact-match scorer's structural mismatch with this task's free-form
  answer format is unresolved by this ADR and remains a
  separate, undecided design question — this proposal is explicitly scoped to
  corpus growth, not scorer redesign.
- If a future training run using this corpus still floors at 0.0% on the
  `meta_trainer` suite, that would not by itself indicate the corpus failed —
  the scorer-mismatch diagnosis predicts a possible floor regardless of
  corpus quality — and should not be read as evidence against this corpus
  without also accounting for that known confound.
- Growing the corpus does not by itself guarantee improved calibration or
  reduced confabulation in any future trained candidate; it is one
  evidence-based corpus-design correction addressing a specific real finding,
  not a guaranteed fix.

## Documentation impact and maintenance triggers

This decision adds `examples/pilot-metatrainer-v3/` (DATASET_CARD.md,
SOURCE_MAP.md, SYNTHESIS_NOTES.md, SELF_AUDIT_REPORT.md, generate_corpus.py,
validate_dataset.py, VALIDATION_REPORT.json, train.jsonl, held_out.json,
semantic_family_manifest.json, held_out_exclusion_registry.json,
REPRESENTATIVE_EXAMPLES.json). No change to `run_bounded_cycle.py`,
`run_bounded_cycle_adr0014.py`, `approval_allowed_signers*`,
`src/codevolt_mdf/trl_adapter.py`, `src/codevolt_mdf/hf_local_evaluator_adapter.py`,
or any resource-budget/threshold/renderer identity. Any future training-run
proposal against this corpus is a new ADR, not an edit to this one.
