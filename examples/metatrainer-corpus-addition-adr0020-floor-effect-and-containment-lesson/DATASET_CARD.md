# Meta-trainer corpus addition proposal: the ADR-0020 "floor effect and execution-path containment" lesson

Status: DRAFT PROPOSAL, NOT YET REVIEWED. This is a separate addition
proposal, sibling to
`examples/metatrainer-corpus-addition-adr0019-subtheta-heldout-shift-lesson/`
(its nearest sibling; this addition is a fresh follow-on case study about
the *result* of the decode-sensitive sampling evaluation that record's
own text named as the next diagnostic step, not a revision of it). It
does not modify `examples/pilot-metatrainer-v2/train.jsonl` /
`held_out.json`, does not modify `examples/pilot-metatrainer-v3/`,
`examples/pilot-metatrainer-v3-dpo/`,
`examples/pilot-metatrainer-v3-dpo-heldout/`, or any of their locked
hashes, and is not committed as part of any training-eligible corpus by
this change. Per this project's standing pattern, admission into a
training-eligible corpus requires a separate PR, an independent
per-example citation-locator and semantic-split audit with the same rigor
as the prior corpus-v2, history-addition, four-cycle-lesson,
dpo-method-switch-lesson, and adr0019-subtheta-heldout-shift-lesson
audits, and the security-reviewer/dataset-rights review — none of which is
performed by drafting this file.

## Scope

5 candidate examples (4 train-candidate, 1 held-out-candidate), one
semantic family, grounded entirely in this repository's own real,
already-completed ADR-0020 execution and its outcome record
(`docs/decisions/ADR-0020-outcome.md`): a preregistered decode-sensitive
sampling evaluation of the ADR-0018 DPO candidate required nine execution
attempts before any result file was produced, surfacing three real
platform defects (an admission-check failure on a relative script path,
the platform's terminal layer wrapping the exact command in a multi-line
shell script before an already-landed admitted-runner's exact-command
matcher saw it, and that same admitted-runner path binding the whole
evaluation root instead of the per-run scratch directory, which made its
write scope broader than intended and caused the evaluation script's own
host-containment check to refuse to run) and a real multiprocessing queue
deadlock (misattributed at first to host load, then correctly diagnosed
as a liveness-check-before-drain race triggered by
large result payloads, and fixed upstream). The eventual real result then
applied its own preregistered decision rule correctly and honestly to
fire a "no shift of meaningful magnitude" outcome — but the record shows
that outcome fires on a floor effect: 16 of the 20 primary held-out pairs
had a calibration-correct rate of exactly zero for both models compared,
roughly three-quarters of all generated samples were labelled
unclassifiable by the scoring method, and the resulting degenerate
zero-width confidence interval reflects that near-total floor, not a
precisely measured absence of any real effect.
Source: `docs/decisions/ADR-0020-outcome.md` ("Execution history,
recorded factually", "The floor-effect structure this result rests on",
and "Mapping to ADR-0020 section 6's outcome-3 row" sections).

This addition is intentionally two-part, reflecting the two genuinely
distinct lessons the same real cycle produced: (1) a set of concrete,
transferable lessons about validating test containment and execution
paths through their real invocation route rather than through shortcuts
that only prove correctness in isolation, and about recognising a hang
rather than assuming slowness when overruns track a changing limit; and
(2) a harder, statistics-adjacent lesson about recognising when a
preregistered "no shift" label is mechanically true but uninformative
because of a floor effect in the underlying measurement instrument, and
reporting that structure honestly alongside the label rather than in
place of it. It is a distinct lesson from
`metatrainer-corpus-addition-adr0019-subtheta-heldout-shift-lesson/`'s
`decode_level_null_vs_probability_space_positive_shift` family (which
teaches that a decode-level null and a separately measured probability-
space positive result on the same model are not in contradiction) — this
family teaches, instead, how to evaluate whether a given "no shift" or
"no effect" result was actually capable of detecting an effect had one
existed, and how to validate the execution machinery that produced any
such result in the first place.

File: `CANDIDATE_CASE_STUDIES.jsonl` (5 JSON-Lines records, one per
line).

## Semantic family: `test_execution_and_measurement_validity_reasoning`

- 4 train-candidate records:
  1. Establishes that testing containment/admission logic through a
     direct, in-process function call does not validate the real
     execution path, and that dry-run-style proofs can hide platform
     defects that only appear on the real invocation route.
  2. Establishes that a job repeatedly killed just past whatever wall
     budget is currently configured, across multiple different budget
     values, should be treated as evidence of a hang or deadlock, not of
     genuine slowness needing a larger number.
  3. Establishes that a short timing probe returning a small payload
     cannot expose a payload-size-dependent deadlock, so a passing short
     probe does not clear an execution path for a much larger real
     payload shape.
  4. Establishes that exact-command admission systems should be given
     exact, literal commands (absolute paths, unwrapped single lines),
     and that a semantically-equivalent but differently-formatted command
     failing admission is the containment model working as intended, not
     a bug to route around.
- 1 held-out-candidate record: tests a distinct, harder angle — that a
  preregistered statistical decision rule firing a clean "no shift"
  result with a degenerate, zero-width confidence interval should
  prompt a check for a floor effect in the underlying measurement before
  that result is read as a precisely-powered null, and that the correct
  practice is to report the preregistered label exactly as it fired while
  also stating the floor-effect structure honestly alongside it, not
  instead of it.

## Citation policy

Every record's `source_locators` field names a specific section of
`docs/decisions/ADR-0020-outcome.md`. No record cites a source that was
not directly read and checked against the claim in that record during
drafting (see "Self-audit" below). This follows the same
citation-locator discipline as the five prior corpus-addition proposals'
own citation policies.

## Explicit exclusions and cautions

This proposal does not assert:

- that the floor effect documented in `docs/decisions/ADR-0020-outcome.md`
  proves the ADR-0018 candidate and its reference checkpoint behave
  identically, or that no real decode-level effect exists — the source
  record explicitly states that this measurement's power to distinguish
  between the competing hypotheses is lower than its own preregistered
  outcome-3 row assumed, precisely because of the floor effect, and this
  corpus addition preserves that uncertainty rather than resolving it;
- that the preregistered "no shift of meaningful magnitude" outcome
  should have been relabeled — `docs/decisions/ADR-0020-outcome.md`
  itself states plainly that the outcome fires exactly as defined and is
  reported unrelabeled, with the floor-effect structure reported
  alongside it as required context, and this corpus addition preserves
  that distinction rather than blurring it;
- any causal claim about why the queue-deadlock defect existed beyond
  what `docs/decisions/ADR-0020-outcome.md` itself states (a liveness
  check performed before a result queue was fully drained) — this corpus
  addition does not speculate about implementation history or intent
  beyond that record's own text;
- that the classifier used in this evaluation is defective or should be
  redesigned — the source record states only that its coverage was a
  limiting factor for this specific measurement's power, not that the
  classifier was built or applied incorrectly;
- that any specific held-out pair's own prompt or answer text (e.g. any
  `dpo-heldout-refusal-*`/`dpo-heldout-counter-*` item, or
  `mtr-v2-heldout-0013`) is reproduced anywhere in this file — no record
  quotes any real held-out item's own prompt or answer text, or any
  actual model-generated sample text; only the fact pattern of the
  ADR-0020 cycle's own aggregate execution history and result structure
  is taught, per the same held-out-registry discipline
  `examples/pilot-metatrainer-v3-dpo-heldout/`'s own contamination audit
  enforces (self-check performed below; independent re-verification is
  the required next step, per this family's own precedent);
- a promotion, deployment, or production-readiness claim about any model
  discussed;
- a recommendation on whether or how a future scoring-method,
  classifier-coverage, or sampling-protocol change should be attempted —
  `docs/decisions/ADR-0020-outcome.md` takes no such position, and this
  addition does not go further than that;
- a licensing or legal conclusion of any kind.

## Self-audit (same rigor as the prior five corpus-addition audits)

This is a self-review by the drafting task, not a substitute for an
independent audit, exactly as the prior additions' own self-audits state
about themselves. It is included so a reviewer has a starting checklist,
not as a claim of independent acceptance.

| Check | Result | Notes |
|---|---|---|
| Every `example_id` unique | PASS | 5/5 unique, distinct from every existing corpus-v2, history-addition, four-cycle-lesson, dpo-method-switch-lesson, adr0018-dpo-near-zero-effect-lesson, and adr0019-subtheta-heldout-shift-lesson id (new `mtr-adr0020-case-*` prefix, checked against every sibling `CANDIDATE_CASE_STUDIES.jsonl`'s own id list). |
| Every record has >=1 source locator | PASS | 5/5. |
| Every record has exactly one user + one assistant message | PASS | 5/5. |
| Source-boundary: every factual claim traces to a real, directly-read artifact at or before drafting time | PASS (self-review) | Every claim (the nine-attempt execution history and its specific failure modes, the queue-deadlock diagnosis and its payload-size dependence, the exact-command admission failures, the floor-effect statistics: 18/20 zero-delta ties (16 tied at exactly zero, 2 tied at a nonzero rate), ~74% AMBIGUOUS rate, the degenerate [0,0] CI, and the outcome-3/hypothesis-mapping reasoning) was copied from `docs/decisions/ADR-0020-outcome.md`, itself independently re-verified against the underlying scoring-results file and its sha256 during this same review's drafting, not paraphrased from memory. |
| Citation-claim support: locator text actually supports the specific claim made | PASS (self-review) | Spot-checked during drafting: each record's locator section was re-read against the exact sentence(s) the record's assistant response restates. |
| No fabricated framing / no invented numbers | PASS (self-review) | No numeric figures beyond what is stated in `docs/decisions/ADR-0020-outcome.md` and independently re-verified against the underlying scoring-results file and its sha256 during this same review. |
| Train/held-out semantic-family split: family assigned before content was finalized, held-out record tests a related-but-distinct angle | PASS (self-review), NEEDS INDEPENDENT VERIFICATION | Structural check (single family, wholly train/held-out disjoint per record) passed by construction (4 train ids, 1 held-out id, no overlap). Content-level non-overlap is asserted by the drafting task (the held-out record tests "recognising a floor effect behind a degenerate confidence interval and reporting it honestly alongside a preregistered label," a distinct question from any of the four train records' own execution-path/containment-validation angles) but was not independently re-verified by a second reviewer. |
| No held-out contamination: does not reproduce or paraphrase any real held-out item's own prompt or answer text, or any real generated sample text (`mtr-v2-heldout-*`, `mtr-v3n-heldout-*`, `dpo-heldout-refusal-*`, `dpo-heldout-counter-*`) | PASS (self-review), NEEDS INDEPENDENT VERIFICATION | No record quotes any real held-out item's own prompt text, correct answer, or any actual candidate/reference generated sample text; no record references any specific held-out `pair_id`. A self-check text-search for `dpo-heldout-refusal`/`dpo-heldout-counter`/`mtr-v2-heldout-0013` against this file found zero hits, but this is a self-check by the drafting task, not an independent audit, and must respect the full held-out registry per this project's registry discipline. |
| No licensing/legal conclusion asserted | PASS | See "Explicit exclusions." |
| Distinct provenance from prior additions (no accidental reuse of prior sourced content or citation ids, no duplicate coverage of an existing family) | PASS | Confirmed by construction: all locators cite `docs/decisions/ADR-0020-outcome.md` directly, not restating any of the five prior additions' own family content. |

### Known limitation this self-audit cannot close

The drafting task is the same task proposing the content, so the
citation-locator, semantic-split, and held-out-contamination checks above
are self-review, not independent audit — exactly the distinction the
prior precedents established matters. Per that precedent, this file
should not be merged into any training-eligible corpus until an
independent reviewer re-checks every locator against the cited source
directly, independently re-examines the train/held-out pair for
content-level leakage, and independently re-runs the held-out-
contamination text-search check against every real held-out item's
prompt/answer text and every real generated sample text this cycle
produced.

## Files

- `CANDIDATE_CASE_STUDIES.jsonl`: 5 candidate records (4 train-candidate,
  1 held-out-candidate), one JSON object per line.
- `DATASET_CARD.md`: this file.

## Next steps (not performed by this task)

1. Independent audit of this file with the same rigor as the five prior
   corpus-addition audits — full per-example citation-locator
   verification against the actual source text, independent
   semantic-split content review, and an independent re-run of the
   held-out-contamination text-search check against every real held-out
   item's prompt/answer text and every real generated sample text this
   cycle produced, not a self-report.
2. If accepted, a small separate PR adding this file (or its
   post-correction revision) under `examples/`, explicitly not touching
   `pilot-metatrainer-v2/`, `pilot-metatrainer-v3/`,
   `pilot-metatrainer-v3-dpo/`, `pilot-metatrainer-v3-dpo-heldout/`, or
   any of the five prior `metatrainer-corpus-addition-*` directories, and
   not changing any frozen ADR's `dataset_hash` value those ADRs gate on.
3. Security-reviewer/dataset-rights review of that PR per this repository's
   existing governance pattern for anything proposed for the training
   corpus, before any merge.
4. Repository admission (if it happens) does not by itself authorize any
   training run; a separate governance decision would be required for
   that, identical in kind to every other corpus addition's own "Intended
   use" section.
