# Meta-trainer corpus addition proposal: the ADR-0016 -> ADR-0017 "two failed rewrites, switch mechanism" lesson

Status: DRAFT PROPOSAL, NOT YET REVIEWED. This is a separate addition
proposal, in the same pattern as
`examples/metatrainer-corpus-addition-training-history/` and
`examples/metatrainer-corpus-addition-four-cycle-lesson/`. It does not
modify `examples/pilot-metatrainer-v2/train.jsonl` / `held_out.json`, does
not modify `examples/pilot-metatrainer-v3/`, `examples/pilot-metatrainer-v3-dpo/`,
or any of their locked hashes, and is not committed as part of any
training-eligible corpus by this change. Per this project's standing pattern
(ADR-0013/0014/0015's own "Review path" sections), admission into a
training-eligible corpus requires a separate PR, an independent per-example
citation-locator and semantic-split audit with the same rigor as the prior
corpus-v2, history-addition, and four-cycle-lesson audits, and Maya's
security/dataset-rights review -- none of which is performed by drafting
this file.

## Scope

3 candidate examples (2 train-candidate, 1 held-out-candidate), one semantic
family, grounded entirely in this repository's own real, already-completed
ADR-0015 -> ADR-0016 -> ADR-0017 arc: two independently constructed
corpus-rewrite attempts at fixing one specific held-out item's calibrated
refusal (`mtr-v2-heldout-0013`), both converging on the identical wrong,
confidently-confabulating output, followed by a documented decision to
switch the training *mechanism* (SFT corpus rewrite -> DPO/preference
training) rather than attempt a third rewrite. Source: `docs/decisions/ADR-0016-outcome.md`
("The pattern across ADR-0015 and ADR-0016 on this one item") and
`docs/decisions/ADR-0017-dpo-preference-refusal-axis.md` ("Context" and
Marcus's/Maya's assessments).

This addition is intentionally narrow and single-purpose: it teaches the
reasoning pattern that when two independently constructed, differently
designed interventions aimed at the same specific behavior both converge on
the *same* wrong answer, that convergence is itself evidence about the
mechanism's insensitivity to further attempts of the same kind -- not proof
that a third differently-worded attempt is guaranteed to fail, but a real
signal that should raise, not lower, the bar for trying the same mechanism a
third time before considering a mechanism change. It is a distinct lesson
from `metatrainer-corpus-addition-four-cycle-lesson/`'s
`mixed_signal_method_choice_stays_open` family (which teaches how to read
two *disagreeing* signals about general capability) -- this family teaches
how to read two *agreeing* (converging) signals about one specific narrow
failure, and what that convergence licenses concluding about mechanism
choice specifically, not general capability.

File: `CANDIDATE_CASE_STUDIES.jsonl` (3 JSON-Lines records, one per line).

## Semantic family: `two_convergent_failures_license_mechanism_switch`

- 2 train-candidate records:
  1. Establishes the core finding: two independently constructed
     interventions (ADR-0015's 8 long third-person case-study examples,
     ADR-0016's 9 short first-person closed-question examples purpose-built
     after diagnosing the first attempt's format mismatch) both produced
     byte-identical wrong output on the same held-out item
     (`mtr-v2-heldout-0013`) -- and that this specific kind of
     convergence (not mere repetition of failure, but two *differently
     designed* attempts landing on the identical answer) is stronger
     evidence of a decision-boundary insensitivity than either failure
     alone, without over-claiming it proves the mechanism can *never* work
     at any scale or data volume.
  2. Establishes the actual decision made and its reasoning: rather than
     attempting a third differently-shaped SFT corpus rewrite, the SME
     recommendation and the project's decision was to switch the training
     *mechanism* (SFT -> DPO/preference training) for this one axis
     specifically, while explicitly not treating a rising, separately
     measured general-capability rubric trend (from the same period) as
     evidence against the mechanism switch for this one narrow, already-twice-
     failed item -- distinguishing a general capability signal from a
     narrow, specific-item signal, and correctly not letting one substitute
     for the other in a mechanism-choice decision.
- 1 held-out-candidate record: tests a distinct, harder angle -- that a
  mechanism switch was accompanied by newly identified mechanism-specific
  risks (the receiving mechanism's own new failure modes, e.g.
  over-refusal/calibration-collapse if the new mechanism's training data is
  built one-directionally), and that adopting a new mechanism to fix one
  known failure requires proactively naming and gating the new mechanism's
  own distinct risks up front, not just carrying forward the old mechanism's
  gates unchanged.

## Citation policy

Every record's `source_locators` field names one or more of: a specific
section of `docs/decisions/ADR-0016-outcome.md`, a specific section of
`docs/decisions/ADR-0017-dpo-preference-refusal-axis.md`, or a specific
internal tracking item id. No record cites a source that was not directly read and
checked against the claim in that record during drafting (see "Self-audit"
below). This follows the same citation-locator discipline as
`examples/metatrainer-corpus-addition-training-history/`'s and
`examples/metatrainer-corpus-addition-four-cycle-lesson/`'s own citation
policies.

## Explicit exclusions and cautions

This proposal does not assert:

- that any two convergent SFT failures on any project always license a
  mechanism switch -- this addition is scoped to the specific real evidence
  in this repository's own ADR-0015/ADR-0016 pair, not a general law;
- that DPO (the mechanism actually chosen here) is guaranteed to succeed
  where SFT rewrites did not -- ADR-0017 itself states training remains
  blocked pending gates, and no DPO cycle has been executed as of this
  file's drafting; this addition teaches the reasoning that licensed
  *trying* a different mechanism, not a claim about that mechanism's actual
  outcome;
- that the `synthetic_data_tradeoffs` family's specific held-out content
  (the confabulated learning-rate-recipe question and its correct "No"
  answer) is reproduced anywhere in this file -- no record quotes the
  held-out item's own prompt or the correct target answer verbatim; only
  the *fact pattern* (two convergent wrong outputs, a mechanism-switch
  decision) is taught, per the same held-out-registry discipline
  `examples/pilot-metatrainer-v3-dpo/validate_dataset.py`'s own
  contamination check enforces on the DPO preference-pair package itself;
- a promotion, deployment, or production-readiness claim about any model
  discussed;
- a licensing or legal conclusion of any kind.

## Self-audit (same C1-C6-equivalent rigor as the corpus v2, history-addition, and four-cycle-lesson audits)

This is a self-review by the drafting task, not a substitute for an
independent audit, exactly as the prior additions' own self-audits state
about themselves. It is included so a reviewer has a starting checklist,
not as a claim of independent acceptance.

| Check | Result | Notes |
|---|---|---|
| Every `example_id` unique | PASS | 3/3 unique, distinct from every existing corpus-v2, history-addition, four-cycle-lesson, and corpus-v3 id (new `mtr-dpo-switch-case-*` prefix). |
| Every record has >=1 source locator | PASS | 3/3. |
| Every record has exactly one user + one assistant message | PASS | 3/3. |
| Source-boundary: every factual claim traces to a real, directly-read artifact at or before drafting time | PASS (self-review) | Every claim (byte-identical convergence on `mtr-v2-heldout-0013`; the ADR-0015 8-example / ADR-0016 9-example corpus-construction facts; the mechanism-switch decision and its stated rationale; the over-refusal/calibration-collapse risk Maya named for the new mechanism) was copied from `docs/decisions/ADR-0016-outcome.md` and `docs/decisions/ADR-0017-dpo-preference-refusal-axis.md`, both re-read in full during this file's drafting, not paraphrased from memory. |
| Citation-claim support: locator text actually supports the specific claim made | PASS (self-review) | Spot-checked during drafting: the "byte-identical convergence" claim in `mtr-dpo-switch-case-train-0001` is directly supported by ADR-0016-outcome.md's "The pattern across ADR-0015 and ADR-0016 on this one item" table, which records the same quoted candidate output for both cycles. |
| No fabricated framing / no invented numbers | PASS (self-review) | No numeric figures beyond the record counts (8/9 examples) already stated in ADR-0016-outcome.md's own "Why this cycle exists" section; no interpolated figures. |
| Train/held-out semantic-family split: family assigned before content was finalized, held-out record tests a related-but-distinct angle | PASS (self-review), NEEDS INDEPENDENT VERIFICATION | Structural check (single family, wholly train/held-out disjoint per record) passed by construction (2 train ids, 1 held-out id, no overlap). Content-level non-overlap is asserted by the drafting task (the held-out record tests "a mechanism switch requires naming the new mechanism's own distinct risks up front," a distinct question from either train record's "convergent failure is evidence" or "a general-capability trend doesn't override a narrow-item mechanism decision") but was not independently re-verified by a second reviewer. |
| No held-out contamination: does not reproduce or paraphrase `mtr-v2-heldout-0013`'s own prompt or correct answer text | PASS (self-review), NEEDS INDEPENDENT VERIFICATION | No record quotes the held-out item's own prompt text or its correct "No" answer; only the fact pattern of the two training cycles' own process and outcomes is described. A fresh token-overlap check (same method as `examples/pilot-metatrainer-v3-dpo/validate_dataset.py`) against the real `mtr-v2-heldout-0013` prompt text found zero overlap hits, but this is a self-check by the drafting task, not an independent audit. |
| No licensing/legal conclusion asserted | PASS | See "Explicit exclusions." |
| Distinct provenance from prior additions (no accidental reuse of Clara-sourced content or citation ids, no duplicate coverage of an existing family) | PASS | Confirmed by construction: no record references any Clara section or numbered Clara source `[N]`; all locators are internal-repository/internal-only, citing `docs/decisions/ADR-0016-outcome.md` and `docs/decisions/ADR-0017-dpo-preference-refusal-axis.md` directly, not restating any of `metatrainer-corpus-addition-training-history/`'s or `metatrainer-corpus-addition-four-cycle-lesson/`'s own family content. |

### Known limitation this self-audit cannot close

The drafting task is the same task proposing the content, so the
citation-locator, semantic-split, and held-out-contamination checks above
are self-review, not independent audit -- exactly the distinction the prior
precedents (`examples/metatrainer-corpus-addition-four-cycle-lesson/`'s
own revision history) established matters. Per that precedent, this file
should not be merged into any training-eligible corpus until an independent
reviewer re-checks every locator against the cited source directly and
independently re-examines the train/held-out pair for content-level leakage,
and independently re-runs the held-out-contamination token-overlap check
against `mtr-v2-heldout-0013`'s real prompt text.

## Files

- `CANDIDATE_CASE_STUDIES.jsonl`: 3 candidate records (2 train-candidate, 1
  held-out-candidate), one JSON object per line.
- `DATASET_CARD.md`: this file.

## Next steps (not performed by this task)

1. Independent audit of this file with the same rigor as the prior
   corpus-v2, training-history, and four-cycle-lesson audits -- full
   per-example citation-locator verification against the actual source
   text, independent semantic-split content review, and an independent
   re-run of the held-out-contamination check against `mtr-v2-heldout-0013`'s
   real prompt/answer text, not a self-report.
2. If accepted, a small separate PR adding this file (or its post-correction
   revision) under `examples/`, explicitly not touching
   `pilot-metatrainer-v2/`, `pilot-metatrainer-v3/`,
   `pilot-metatrainer-v3-dpo/`, `metatrainer-corpus-addition-training-history/`,
   or `metatrainer-corpus-addition-four-cycle-lesson/`, and not changing any
   frozen ADR's `dataset_hash` value those ADRs gate on.
3. Maya security/dataset-rights review of that PR per this repository's
   existing governance pattern for anything proposed for the training
   corpus, before any merge.
4. Repository admission (if it happens) does not by itself authorize any
   training run; a separate governance decision would be required for that,
   identical in kind to every other corpus addition's own "Intended use"
   section.
