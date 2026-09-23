# Meta-trainer corpus addition proposal: the ADR-0018 "training-internal metrics moved, decode-level generations barely did" lesson

Status: DRAFT PROPOSAL, NOT YET REVIEWED. This is a separate addition
proposal, in the same pattern as
`examples/metatrainer-corpus-addition-training-history/`,
`examples/metatrainer-corpus-addition-four-cycle-lesson/`, and
`examples/metatrainer-corpus-addition-dpo-method-switch-lesson/` (its
nearest sibling; this addition is a fresh follow-on case study about the
*result* of the DPO cycle that sibling's own held-out item anticipated risks
for, not a revision of it). It does not modify
`examples/pilot-metatrainer-v2/train.jsonl` / `held_out.json`, does not
modify `examples/pilot-metatrainer-v3/`, `examples/pilot-metatrainer-v3-dpo/`,
`examples/metatrainer-corpus-addition-dpo-method-switch-lesson/`, or any of
their locked hashes, and is not committed as part of any training-eligible
corpus by this change. Per this project's standing pattern, admission into a
training-eligible corpus requires a separate PR, an independent per-example
citation-locator and semantic-split audit with the same rigor as the prior
corpus-v2, history-addition, four-cycle-lesson, and dpo-method-switch-lesson
audits, and Maya's security/dataset-rights review — none of which is
performed by drafting this file.

## Scope

3 candidate examples (2 train-candidate, 1 held-out-candidate), one semantic
family, grounded entirely in this repository's own real, already-completed
ADR-0018 execution (`docs/decisions/ADR-0018-outcome.md`): the first real governed DPO cycle on this
project completed successfully (`training.status=accepted`, real loss/
reward-margin curve, real checkpoint, resource use well under ceiling), and
its own internal preference-training metrics moved in the intended
direction overall by the end of the run, while the resulting model's held-out
generations were byte-identical to the pre-training baseline on 42 of 48
`meta_trainer` items (87.5%), 10 of 10 `capability_retention` items, and 13
of 15 `safety` items — 65 of 73 items (89.0%) across all three suites
combined — with zero score-boundary crossings on any of the 8 items whose
text did differ. Source: `docs/decisions/ADR-0018-outcome.md`
("Near-zero behavioural change: the central observation of this cycle").

This addition is intentionally narrow and single-purpose: it teaches the
reasoning pattern that a training run reaching `TrainingStatus.ACCEPTED`
with real, moving internal optimizer/loss metrics is a distinct claim from
"the trained model's generations changed," and that both must be checked
independently before drawing any conclusion about whether a training cycle
"worked." It is a distinct lesson from
`metatrainer-corpus-addition-dpo-method-switch-lesson/`'s
`two_convergent_failures_license_mechanism_switch` family (which teaches how
convergent *failures* across two differently-designed interventions license
a mechanism switch) — this family teaches how to read a cycle that
technically succeeded end to end but produced almost no measurable
downstream effect, which is a different diagnostic situation from either a
clean success or a clean (wrong-direction) failure.

File: `CANDIDATE_CASE_STUDIES.jsonl` (3 JSON-Lines records, one per line).

## Semantic family: `training_internal_metrics_vs_decode_level_effect_divergence`

- 2 train-candidate records:
  1. Establishes the core finding: a real, gate-accepted training run
     (loss/reward-margin metrics moving in the intended direction, exit
     code 0, no resource-ceiling breach) is not the same claim as "the
     trained model's outputs changed," and both must be measured and
     reported, not inferred from one another.
  2. Establishes the correct next step when this divergence is observed:
     report the near-zero decode-level effect as the primary factual
     finding, offer hyperparameter/step-count/dataset-size explanations
     explicitly labeled as untested hypotheses rather than findings, and do
     not treat the technical-success signal (accepted training status) as
     evidence that the behavioural objective was achieved.
- 1 held-out-candidate record: tests a distinct, harder angle -- that
  applying a run's own pre-stated binding failure criteria
  mechanically, criterion by criterion, can produce a negative verdict even when the
  specific failure pattern the run was designed to prevent did not recur,
  and that this is not a contradiction requiring the criteria to be
  reinterpreted after the fact -- the criteria are evaluated as stated in
  advance, and any criterion whose literal text is met must be reported as
  firing even when the broader qualitative picture is more mixed than a
  single fired criterion alone would suggest.

## Citation policy

Every record's `source_locators` field names a specific section of
`docs/decisions/ADR-0018-outcome.md` or `docs/decisions/ADR-0018-dpo-execution-config.md`.
No record cites a source that was not directly read and checked against the
claim in that record during drafting (see "Self-audit" below). This follows
the same citation-locator discipline as the three prior corpus-addition
proposals' own citation policies.

## Explicit exclusions and cautions

This proposal does not assert:

- that DPO, or preference-training generally, structurally cannot move
  decode-level generations -- this addition is scoped to the specific real
  evidence in this repository's own ADR-0018 cycle (one hyperparameter
  combination, one 22-record dataset, 22 steps), not a general law about the
  method;
- any causal explanation for *why* the training-internal metric and the
  decode-level effect diverged in this run -- `docs/decisions/ADR-0018-outcome.md`
  itself labels its own candidate explanations as untested hypotheses, and
  this corpus addition preserves that distinction rather than picking one;
- that the `synthetic_data_tradeoffs` family's specific held-out content
  (the confabulated learning-rate-recipe question and its correct "No"
  answer) is reproduced anywhere in this file -- no record quotes
  `mtr-v2-heldout-0013`'s own prompt or correct target answer verbatim; only
  the fact pattern of the ADR-0018 cycle's own aggregate result and verdict
  process is taught, per the same held-out-registry discipline
  `examples/pilot-metatrainer-v3-dpo/validate_dataset.py`'s own
  contamination check enforces;
- a promotion, deployment, or production-readiness claim about any model
  discussed;
- a recommendation on whether or how a future DPO cycle should be attempted
  -- `docs/decisions/ADR-0018-outcome.md` names possible hyperparameter
  changes as an unauthorized, unscoped recommendation for whoever holds that
  decision next, and this addition does not go further than that;
- a licensing or legal conclusion of any kind.

## Self-audit (same C1-C6-equivalent rigor as the prior three corpus-addition audits)

This is a self-review by the drafting task, not a substitute for an
independent audit, exactly as the prior additions' own self-audits state
about themselves. It is included so a reviewer has a starting checklist, not
as a claim of independent acceptance.

| Check | Result | Notes |
|---|---|---|
| Every `example_id` unique | PASS | 3/3 unique, distinct from every existing corpus-v2, history-addition, four-cycle-lesson, corpus-v3, and dpo-method-switch-lesson id (new `mtr-dpo-nearzero-case-*` prefix, checked against `examples/pilot-metatrainer-v3/held_out_exclusion_registry.json`'s full id list and `examples/metatrainer-corpus-addition-dpo-method-switch-lesson/CANDIDATE_CASE_STUDIES.jsonl`'s `mtr-dpo-switch-case-*` ids). |
| Every record has >=1 source locator | PASS | 3/3. |
| Every record has exactly one user + one assistant message | PASS | 3/3. |
| Source-boundary: every factual claim traces to a real, directly-read artifact at or before drafting time | PASS (self-review) | Every claim (the 42/48, 10/10, 13/15 identical-output counts; the accepted training status and resource usage; the two-of-four-criteria-fire verdict) was copied from `docs/decisions/ADR-0018-outcome.md`, itself independently re-derived from `cycle_result.json` (sha256 `e09f75c46c567c718b633e078685f49b7666a09db6765f4fbaeb9f18bb70e634`) during this same task's drafting, not paraphrased from memory. |
| Citation-claim support: locator text actually supports the specific claim made | PASS (self-review) | Spot-checked during drafting: the "42/48 byte-identical" claim in `mtr-dpo-nearzero-case-train-0001` is directly supported by `docs/decisions/ADR-0018-outcome.md`'s "Near-zero behavioural change" section, itself independently re-derived by diffing `cycle_result.json`'s baseline/candidate `raw_output` fields programmatically (not eyeballed). |
| No fabricated framing / no invented numbers | PASS (self-review) | No numeric figures beyond what is stated in `docs/decisions/ADR-0018-outcome.md` and independently re-verified against `cycle_result.json` during this same task. |
| Train/held-out semantic-family split: family assigned before content was finalized, held-out record tests a related-but-distinct angle | PASS (self-review), NEEDS INDEPENDENT VERIFICATION | Structural check (single family, wholly train/held-out disjoint per record) passed by construction (2 train ids, 1 held-out id, no overlap). Content-level non-overlap is asserted by the drafting task (the held-out record tests "apply pre-stated criteria literally even when the qualitative picture is mixed," a distinct question from either train record's "training-internal vs decode-level divergence exists and must be checked separately") but was not independently re-verified by a second reviewer. |
| No held-out contamination: does not reproduce or paraphrase `mtr-v2-heldout-0013`'s own prompt or correct answer text, or any other real held-out item's prompt/answer text | PASS (self-review), NEEDS INDEPENDENT VERIFICATION | No record quotes any real held-out item's own prompt text or correct answer. A self-check token-overlap review (same method as `examples/pilot-metatrainer-v3-dpo/validate_dataset.py`) against `mtr-v2-heldout-0013`'s real prompt text found zero overlap hits, but this is a self-check by the drafting task, not an independent audit. |
| No licensing/legal conclusion asserted | PASS | See "Explicit exclusions." |
| Distinct provenance from prior additions (no accidental reuse of Clara-sourced content or citation ids, no duplicate coverage of an existing family) | PASS | Confirmed by construction: no record references any Clara section or numbered Clara source `[N]`; all locators cite `docs/decisions/ADR-0018-outcome.md` and `docs/decisions/ADR-0018-dpo-execution-config.md` directly, not restating any of the three prior additions' own family content. |

### Known limitation this self-audit cannot close

The drafting task is the same task proposing the content, so the
citation-locator, semantic-split, and held-out-contamination checks above
are self-review, not independent audit -- exactly the distinction the prior
precedents established matters. Per that precedent, this file should not be
merged into any training-eligible corpus until an independent reviewer
re-checks every locator against the cited source directly and independently
re-examines the train/held-out pair for content-level leakage, and
independently re-runs the held-out-contamination token-overlap check
against every real held-out item's prompt/answer text.

## Files

- `CANDIDATE_CASE_STUDIES.jsonl`: 3 candidate records (2 train-candidate, 1
  held-out-candidate), one JSON object per line.
- `DATASET_CARD.md`: this file.

## Next steps (not performed by this task)

1. Independent audit of this file with the same rigor as the three prior
   corpus-addition audits -- full per-example citation-locator verification
   against the actual source text, independent semantic-split content
   review, and an independent re-run of the held-out-contamination
   token-overlap check against every real held-out item's prompt/answer
   text, not a self-report.
2. If accepted, a small separate PR adding this file (or its
   post-correction revision) under `examples/`, explicitly not touching
   `pilot-metatrainer-v2/`, `pilot-metatrainer-v3/`,
   `pilot-metatrainer-v3-dpo/`, or any of the three prior
   `metatrainer-corpus-addition-*` directories, and not changing any frozen
   ADR's `dataset_hash` value those ADRs gate on.
3. Maya security/dataset-rights review of that PR per this repository's
   existing governance pattern for anything proposed for the training
   corpus, before any merge.
4. Repository admission (if it happens) does not by itself authorize any
   training run; a separate governance decision would be required for that,
   identical in kind to every other corpus addition's own "Intended use"
   section.
