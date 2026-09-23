# Meta-trainer corpus addition proposal: the ADR-0019 "decode-level null plus probability-space positive" lesson

Status: DRAFT PROPOSAL, NOT YET REVIEWED. This is a separate addition
proposal, sibling to
`examples/metatrainer-corpus-addition-adr0018-dpo-near-zero-effect-lesson/`
(its nearest sibling; this addition is a fresh follow-on case study about
the *result* of the held-out probability-space evaluation that record's
own text named as the next step, not a revision of it). It does not
modify `examples/pilot-metatrainer-v2/train.jsonl` / `held_out.json`, does
not modify `examples/pilot-metatrainer-v3/`, `examples/pilot-metatrainer-v3-dpo/`,
`examples/pilot-metatrainer-v3-dpo-heldout/`, or any of their locked
hashes, and is not committed as part of any training-eligible corpus by
this change. Per this project's standing pattern, admission into a
training-eligible corpus requires a separate PR, an independent
per-example citation-locator and semantic-split audit with the same rigor
as the prior corpus-v2, history-addition, four-cycle-lesson, and
dpo-method-switch-lesson audits, and Maya's security/dataset-rights
review — none of which is performed by drafting this file.

## Scope

3 candidate examples (2 train-candidate, 1 held-out-candidate), one
semantic family, grounded entirely in this repository's own real,
already-completed ADR-0019 execution
(`docs/decisions/ADR-0019-outcome.md`, run
`adr0019-logprob-margin-eval-20260923`): a preregistered held-out
log-probability margin evaluation of the ADR-0018 DPO candidate found
that, on the authoritative full 20-pair held-out set, every pair's
candidate-minus-reference margin moved in the trained direction
(Wilcoxon `p = 9.6e-5`, statistic at its maximum possible value, 20/20
positive), with a Hodges–Lehmann point estimate at roughly 81% of the
preregistered training-set-magnitude threshold `θ` — landing the
preregistered classification at outcome 3, "no shift of training-set
magnitude," while leaving a smaller, real, sub-θ shift explicitly
undecided by that outcome's own definition. This is the same trained
checkpoint ADR-0018's own decode-level evaluation had already found to
produce byte-identical generations on 65 of 73 (89.0%) held-out items.
Source: `docs/decisions/ADR-0019-outcome.md` ("The result, applying
section 3's preregistered decision rule exactly as fixed in advance" and
"Mapping to ADR-0019 section 4's outcome-3 row" sections).

This addition is intentionally narrow and single-purpose: it teaches the
reasoning pattern that a decode-level (greedy-argmax) null result and a
separately measured, real, statistically significant, sub-threshold
probability-space shift are answers to two different, differently
sensitive questions about the same trained model, and neither overturns
the other — the correct response is to report both honestly, side by
side, and to reason about competing untested hypotheses only in terms of
which direction each moves, not which is "proven." It is a distinct
lesson from
`metatrainer-corpus-addition-adr0018-dpo-near-zero-effect-lesson/`'s
`training_internal_metrics_vs_decode_level_effect_divergence` family
(which teaches that an accepted training run's internal optimizer metrics
moving is a separate claim from the model's generations changing) — this
family teaches the next, narrower step in the same investigative chain:
once a decode-level null is already established, how to read a
follow-up, more sensitive measurement that finds a real-but-sub-threshold
positive result on the very same checkpoint, and how that finding should
and should not move a small set of previously-named, explicitly
provisional hypotheses.

File: `CANDIDATE_CASE_STUDIES.jsonl` (3 JSON-Lines records, one per line).

## Semantic family: `decode_level_null_vs_probability_space_positive_shift`

- 2 train-candidate records:
  1. Establishes the core finding: a decode-level null result and a later,
     separately preregistered probability-space measurement finding a
     real, statistically significant, uniformly-signed, sub-threshold
     shift on the same trained model are not in contradiction — they
     measure different things at different sensitivities, and both must
     be reported honestly rather than one being read as overturning the
     other.
  2. Establishes the correct next step when a sub-threshold-but-real
     result arrives after several untested candidate explanations were
     already on the table: state precisely which of those hypotheses the
     new result strengthens or weakens, in which direction, without
     declaring any of them confirmed, ruled out, or proven by one
     measurement that was not designed to settle the question outright.
- 1 held-out-candidate record: tests a distinct, harder angle — that a
  secondary, non-authoritative subset breakdown landing on the knife-edge
  of a fixed threshold, well inside that method's own resampling noise
  floor, must be reported honestly exactly as computed, but explicitly
  flagged as not calling a separately authoritative, adequately-powered
  full-set result into question.

## Citation policy

Every record's `source_locators` field names a specific section of
`docs/decisions/ADR-0019-outcome.md`. No record cites a source that was
not directly read and checked against the claim in that record during
drafting (see "Self-audit" below). This follows the same
citation-locator discipline as the four prior corpus-addition proposals'
own citation policies.

## Explicit exclusions and cautions

This proposal does not assert:

- that the held-out probability-space shift found by ADR-0019
  overturns, softens, or reverses ADR-0018's own decode-level **NEGATIVE**
  verdict — `docs/decisions/ADR-0019-outcome.md` itself states plainly
  that both records remain independently authoritative and that this
  evaluation asked a narrower, different question, and this corpus
  addition preserves that distinction rather than blurring it;
- any causal explanation for *why* the probability-space shift falls
  short of the preregistered threshold, or for which of the three named
  hypotheses (conservative beta/lr, log-prob-vs-greedy-decode mismatch,
  dataset too small) is the actual cause — `docs/decisions/ADR-0019-outcome.md`
  itself labels its own directional movements of each hypothesis as
  hypothesis-level reasoning, not findings, and this corpus addition
  preserves that distinction rather than picking one;
- that the refusal-direction held-out subset's `ambiguous_underpowered`
  classification indicates a materially different underlying effect from
  the full set — the source record explicitly cautions against
  over-reading that classification, and this corpus addition preserves
  that caution rather than treating the subset breakdown as a competing
  verdict;
- that any specific held-out pair's own prompt or answer text (e.g. any
  `dpo-heldout-refusal-*`/`dpo-heldout-counter-*` item, or
  `mtr-v2-heldout-0013`) is reproduced anywhere in this file — no record
  quotes any real held-out item's own prompt or answer text; only the
  fact pattern of the ADR-0019 cycle's own aggregate result and reasoning
  process is taught, per the same held-out-registry discipline
  `examples/pilot-metatrainer-v3-dpo-heldout/`'s own contamination audit
  enforces (self-check performed below; independent re-verification is
  the required next step, per this family's own precedent);
- a promotion, deployment, or production-readiness claim about any model
  discussed;
- a recommendation on whether or how a future training or scoring-method
  change should be attempted — `docs/decisions/ADR-0019-outcome.md` names
  a possible decode-sensitive scoring method as an explicitly
  non-binding recommendation for whoever holds that decision next, and
  this addition does not go further than that;
- a licensing or legal conclusion of any kind.

## Self-audit (same rigor as the prior four corpus-addition audits)

This is a self-review by the drafting task, not a substitute for an
independent audit, exactly as the prior additions' own self-audits state
about themselves. It is included so a reviewer has a starting checklist,
not as a claim of independent acceptance.

| Check | Result | Notes |
|---|---|---|
| Every `example_id` unique | PASS | 3/3 unique, distinct from every existing corpus-v2, history-addition, four-cycle-lesson, dpo-method-switch-lesson, and adr0018-dpo-near-zero-effect-lesson id (new `mtr-dpo-subtheta-case-*` prefix, checked against every sibling `CANDIDATE_CASE_STUDIES.jsonl`'s own id list and against `examples/pilot-metatrainer-v3/held_out_exclusion_registry.json`'s full id list). |
| Every record has >=1 source locator | PASS | 3/3. |
| Every record has exactly one user + one assistant message | PASS | 3/3. |
| Source-boundary: every factual claim traces to a real, directly-read artifact at or before drafting time | PASS (self-review) | Every claim (the full-set outcome-3 classification and its CI bounds, the 20/20-positive Wilcoxon result, the ~81%-of-theta point estimate, the refusal-subset ambiguous-underpowered classification and its knife-edge distance from theta, the three-hypothesis directional-movement language) was copied from `docs/decisions/ADR-0019-outcome.md`, itself independently re-derived from the result file's SHA-256-verified statistics during this same review's drafting, not paraphrased from memory. |
| Citation-claim support: locator text actually supports the specific claim made | PASS (self-review) | Spot-checked during drafting: each record's locator section was re-read against the exact sentence(s) the record's assistant response restates. |
| No fabricated framing / no invented numbers | PASS (self-review) | No numeric figures beyond what is stated in `docs/decisions/ADR-0019-outcome.md` and independently re-verified against the underlying result file and its SHA-256 during this same review. |
| Train/held-out semantic-family split: family assigned before content was finalized, held-out record tests a related-but-distinct angle | PASS (self-review), NEEDS INDEPENDENT VERIFICATION | Structural check (single family, wholly train/held-out disjoint per record) passed by construction (2 train ids, 1 held-out id, no overlap). Content-level non-overlap is asserted by the drafting task (the held-out record tests "a knife-edge secondary-subset classification must not be over-read against a separately authoritative full-set result," a distinct question from either train record's "decode-level null and probability-space positive coexist without contradiction" / "state hypothesis movement without declaring resolution") but was not independently re-verified by a second reviewer. |
| No held-out contamination: does not reproduce or paraphrase any real held-out item's own prompt or answer text (`mtr-v2-heldout-*`, `mtr-v3n-heldout-*`, `dpo-heldout-refusal-*`, `dpo-heldout-counter-*`) | PASS (self-review), NEEDS INDEPENDENT VERIFICATION | No record quotes any real held-out item's own prompt text or correct answer, and no record references any specific held-out `pair_id`. A self-check text-search for `dpo-heldout-refusal`/`dpo-heldout-counter`/`mtr-v2-heldout-0013`/the specific confabulated-recipe wording against this file found zero hits, but this is a self-check by the drafting task, not an independent audit, and must respect the full held-out registry (including the new `pilot-metatrainer-v3-dpo-heldout-adr0019` package) per this project's registry discipline. |
| No licensing/legal conclusion asserted | PASS | See "Explicit exclusions." |
| Distinct provenance from prior additions (no accidental reuse of Clara-sourced content or citation ids, no duplicate coverage of an existing family) | PASS | Confirmed by construction: no record references any Clara section or numbered Clara source `[N]`; all locators cite `docs/decisions/ADR-0019-outcome.md` directly, not restating any of the four prior additions' own family content. |

### Known limitation this self-audit cannot close

The drafting task is the same task proposing the content, so the
citation-locator, semantic-split, and held-out-contamination checks above
are self-review, not independent audit — exactly the distinction the
prior precedents established matters. Per that precedent, this file
should not be merged into any training-eligible corpus until an
independent reviewer re-checks every locator against the cited source
directly, independently re-examines the train/held-out pair for
content-level leakage, and independently re-runs the held-out-contamination
text-search check against every real held-out item's prompt/answer text,
including the new `pilot-metatrainer-v3-dpo-heldout-adr0019` package this
ADR added to the standing held-out inventory.

## Files

- `CANDIDATE_CASE_STUDIES.jsonl`: 3 candidate records (2 train-candidate,
  1 held-out-candidate), one JSON object per line.
- `DATASET_CARD.md`: this file.

## Next steps (not performed by this task)

1. Independent audit of this file with the same rigor as the four prior
   corpus-addition audits — full per-example citation-locator
   verification against the actual source text, independent
   semantic-split content review, and an independent re-run of the
   held-out-contamination text-search check against every real held-out
   item's prompt/answer text (including the ADR-0019 held-out package),
   not a self-report.
2. If accepted, a small separate PR adding this file (or its
   post-correction revision) under `examples/`, explicitly not touching
   `pilot-metatrainer-v2/`, `pilot-metatrainer-v3/`,
   `pilot-metatrainer-v3-dpo/`, `pilot-metatrainer-v3-dpo-heldout/`, or
   any of the four prior `metatrainer-corpus-addition-*` directories, and
   not changing any frozen ADR's `dataset_hash` value those ADRs gate on.
3. Maya security/dataset-rights review of that PR per this repository's
   existing governance pattern for anything proposed for the training
   corpus, before any merge.
4. Repository admission (if it happens) does not by itself authorize any
   training run; a separate governance decision would be required for
   that, identical in kind to every other corpus addition's own "Intended
   use" section.
