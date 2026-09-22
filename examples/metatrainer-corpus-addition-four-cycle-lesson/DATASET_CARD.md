# Meta-trainer corpus addition proposal: the four-cycle "mixed real signal, method-choice is open" lesson

Status: DRAFT PROPOSAL, NOT YET REVIEWED. This is a separate addition
proposal, in the same pattern as
`examples/metatrainer-corpus-addition-training-history/`. It does not modify
`examples/pilot-metatrainer-v2/train.jsonl` / `held_out.json`, does not modify
`examples/pilot-metatrainer-v3/` or any of its locked hashes, and is not
committed as part of any training-eligible corpus by this change. Per this
project's standing pattern (ADR-0013/0014/0015's own "Review path" sections),
admission into a training-eligible corpus requires a separate PR, an
independent per-example citation-locator and semantic-split audit with the
same rigor as the prior corpus-v2 and history-addition audits, and Maya's
security/dataset-rights review — none of which is performed by drafting this
file.

## Revision note (2026-09-22)

This file's first draft taught a single "four iterations, no target-task
movement, change method" conclusion, based only on the automated
`exact_match` scorer reading `0.0%` across all four cycles. That conclusion
was incomplete: Maya's review of the parent PR (#91) identified that an
independent manual rubric review — which existed before this
file was first drafted but had not yet been incorporated — found the
ADR-0015 candidate's rubric score is `0.65/4.0`, continuing a strictly
monotonic improving trend across all four runs (`0.15 -> 0.35 -> 0.45 ->
0.65`), while also finding a new, arguably sharper safety-relevant critical
error in the same run. Teaching the model the original single-conclusion
framing would have been teaching it a wrong (or at least materially
incomplete) lesson from real data. This revision replaces the original
content with a lesson that teaches the *actually supported* pattern: reading
two real, partially-conflicting quantitative signals about the same system
honestly, distinguishing what each one separately supports, and correctly
recognizing when the evidence does not resolve to one clean recommendation
rather than picking a side. See `docs/decisions/ADR-0015-outcome.md`'s own
"Revision history" section for the parallel correction on the outcome-record
side.

## Scope

3 candidate examples (2 train-candidate, 1 held-out-candidate), one semantic
family, grounded entirely in this repository's own real, already-completed
four-cycle training arc (ADR-0011, ADR-0013, ADR-0014, ADR-0015), the
independent manual rubric review of that arc, and the outcome
record that synthesizes both (`docs/decisions/ADR-0015-outcome.md`, as
revised). This addition is intentionally narrow and single-purpose: it
teaches exactly one reasoning pattern — that when two real signals about the
same system move in different directions (a flat/structurally-limited
automated score and a rising-but-still-far-from-threshold manual rubric
score, alongside a newly-appeared regression in one specific failure mode),
the correct response is to report both honestly, state what each one does
and does not support, and treat the higher-level method-choice question as
open rather than manufacturing a single directional conclusion from a
partial reading — rather than re-teaching the individual per-cycle facts the
existing `metatrainer-corpus-addition-training-history/` family (4 families,
12 records) already covers for ADR-0011 and ADR-0013 individually. This
addition is the direct feedback-loop this project's own framework is
designed for: a real, mixed-signal finding — including the real correction
this file itself required — fed back as training material for the exact
model the finding is about.

File: `CANDIDATE_CASE_STUDIES.jsonl` (3 JSON-Lines records, one per line).

## Semantic family: `mixed_signal_method_choice_stays_open`

- 2 train-candidate records:
  1. Establishes the core finding: the automated exact-match scorer reads a
     flat `0.0%` across all four cycles while an independent manual rubric
     review of the same candidates shows a strictly monotonic improving
     trend (`0.15 -> 0.35 -> 0.45 -> 0.65`) — and that when two real
     measurements of the same underlying question disagree, the correct
     response is to report both with their respective caveats, not to treat
     the more dramatic-sounding one (flat zero) as the whole picture.
  2. Establishes why a real, still-far-below-threshold improving trend
     (`0.65` vs a `3.0` bar) combined with a newly-appeared, arguably
     worsened critical safety error in the exact family a targeted
     intervention was aimed at fixing, is not resolvable into one clean
     "continue" or "stop" recommendation from the data alone — and that
     manufacturing one anyway (in either direction) is a more serious error
     than reporting the mixed picture and naming it as a human decision
     point.
- 1 held-out-candidate record: tests a distinct, harder angle than either
  train record — recognizing that a document's own first-draft conclusion
  can itself be a source of error requiring correction once a real, then-
  unincorporated piece of evidence becomes available, and that acknowledging
  and revising a prior conclusion under new evidence (rather than defending
  it, or silently rewriting history) is the correct response, not a
  reflection of the earlier draft being illegitimate.

## Citation policy

Every record's `source_locators` field names one or more of: the revised
outcome record `docs/decisions/ADR-0015-outcome.md` (specifically its "ADR-
0015's manual rubric result and the four-run trend" and "Revision history"
sections) and the specific section relied on, a specific existing ADR's
markdown section, a specific on-disk evidence JSON file plus its recorded
field/hash, or a specific completed internal tracking item id. No record cites a source
that was not directly read and checked against the claim in that record
during drafting (see "Self-audit" below). This follows exactly the same
citation-locator discipline as
`examples/metatrainer-corpus-addition-training-history/`'s own citation
policy, applied to the corrected four-cycle synthesis rather than any single
cycle or the original, incomplete single-conclusion framing.

## Explicit exclusions and cautions

This proposal does not assert:

- that a flat automated-scorer result combined with a rising manual-rubric
  result always means "keep going" or always means "the automated scorer is
  wrong to weight" in any other project — this addition is scoped to the
  specific real evidence in this repository's own four cycles and their
  independent rubric review, not a general law about all such disagreements;
- that continuing corpus growth is confirmed to work, or that a method
  change (scorer redesign, different training method/base model) is
  confirmed necessary — both readings remain live, and the outcome record
  and this addition explicitly leave that choice to Rook rather than
  asserting either as a pre-validated conclusion;
- that the item-level critical-error regression on `mtr-v2-heldout-0013`
  (endorsing invented numeric recipes) is representative of all safety
  behavior, only that it is a real, concrete, unresolved regression in the
  exact family a targeted intervention aimed at fixing;
- a promotion, deployment, or production-readiness claim about any model
  discussed;
- a licensing or legal conclusion of any kind.

## Self-audit (same C1-C6-equivalent rigor as the corpus v2 and history-addition audits)

This is a self-review by the drafting task, not a substitute for an
independent audit, exactly as the prior history-addition's own self-audit
states about itself. It is included so a reviewer has a starting checklist,
not as a claim of independent acceptance.

| Check | Result | Notes |
|---|---|---|
| Every `example_id` unique | PASS | 3/3 unique, distinct from every existing corpus-v2, history-addition, and corpus-v3 `mtr-*`/`mtr-v3n-*`/`mtr-hist-case-*` id, and from this file's own withdrawn first-draft `mtr-4cycle-case-*` ids (new `mtr-mixedsignal-case-*` prefix, to avoid any ambiguity with the superseded draft). |
| Every record has >=1 source locator | PASS | 3/3. |
| Every record has exactly one user + one assistant message | PASS | 3/3. |
| Source-boundary: every factual claim traces to a real, directly-read artifact at or before drafting time | PASS (self-review) | Every numeric value (0.0% exact-match figures; 0.15/0.35/0.45/0.65 rubric means; the `3.0/4.0` bar; the `0.65/4.0` vs `3.0/4.0` gap; item `mtr-v2-heldout-0013`/`0019` outcomes) was copied from the revised `docs/decisions/ADR-0015-outcome.md` (itself sourced from 's `adr0015_rubric_and_4run_trend.json` and the real `cycle_result.json`), not paraphrased from memory. |
| Citation-claim support: locator text actually supports the specific claim made | PASS (self-review) | Spot-checked during drafting: the "0.15/0.35/0.45/0.65 strictly monotonic" claim in `mtr-mixedsignal-case-train-0001` is directly supported by the outcome record's "ADR-0015's manual rubric result and the four-run trend" section, itself sourced from 's artifact's `part3_four_point_trend_directional_movement.rubric_mean_sequence` field. |
| No fabricated framing / no invented numbers | PASS (self-review) | All numeric figures are copied exactly from the revised `docs/decisions/ADR-0015-outcome.md`; no interpolated or rounded-for-effect figures. |
| Train/held-out semantic-family split: family assigned before content was finalized, held-out record tests a related-but-distinct angle | PASS (self-review), NEEDS INDEPENDENT VERIFICATION | Structural check (single family, wholly train/held-out disjoint per record) passed by construction (2 train ids, 1 held-out id, no overlap). Content-level non-overlap is asserted by the drafting task (the held-out record tests "revising a document's own prior conclusion under new evidence," a distinct question from either train record's "report two disagreeing signals honestly" and "a mixed signal doesn't resolve to one recommendation") but was not independently re-verified by a second reviewer — the same open item the prior drafts' own self-audits flagged for their family splits, per that precedent for what a reviewer should scrutinize most closely. |
| No licensing/legal conclusion asserted | PASS | See "Explicit exclusions." |
| Distinct provenance from `meta-trainer-corpus-v2` and the training-history addition (no accidental reuse of Clara-sourced content or citation ids, no duplicate coverage of an existing family) | PASS | Confirmed by construction: no record references any Clara section or numbered Clara source `[N]`; all locators are internal-repository/internal-only, citing the revised outcome record and prior ADR/rubric-review text, not restating any of the four existing `metatrainer-corpus-addition-training-history` families' own content. |
| Superseded-draft handling: original single-conclusion version fully replaced, not left alongside a conflicting version | PASS | This revision replaces the entire prior `CANDIDATE_CASE_STUDIES.jsonl` content (all 3 `mtr-4cycle-case-*` records) rather than appending alongside it; no record from the withdrawn draft remains in this file. |

### Known limitation this self-audit cannot close

The drafting task is the same task proposing the content, so the
citation-locator and semantic-split checks above are self-review, not
independent audit — exactly the distinction the corpus-v2 and
training-history precedents (tasks) established matters. Per that precedent, this
file should not be merged into any training-eligible corpus until an
independent reviewer re-checks every locator against the cited source
directly and independently re-examines the train/held-out pair for
content-level leakage. Given that this exact file already required one
correction after independent review caught an incomplete first draft, a
reviewer should treat that history as reason for, not against, thorough
re-verification here specifically.

## Files

- `CANDIDATE_CASE_STUDIES.jsonl`: 3 candidate records (2 train-candidate, 1
  held-out-candidate), one JSON object per line.
- `DATASET_CARD.md`: this file.

## Next steps (not performed by this task)

1. Independent audit of this file with the same rigor as the prior
   corpus-v2 and training-history audits — full per-example citation-locator
   verification against the actual source text/JSON (in this case, primarily
   the revised `docs/decisions/ADR-0015-outcome.md` and the underlying
    rubric-review artifact and `cycle_result.json` evidence it
   cites), and independent semantic-split content review, not a self-report.
2. If accepted, a small separate PR adding this file (or its post-correction
   revision) under `examples/`, explicitly not touching
   `pilot-metatrainer-v2/`, `pilot-metatrainer-v3/`, or
   `metatrainer-corpus-addition-training-history/`, and not changing any
   frozen ADR-0013/0014/0015 `dataset_hash` value those ADRs gate on.
3. Maya security/dataset-rights review of that PR per this repository's
   existing governance pattern for anything proposed for the training
   corpus, before any merge.
4. Repository admission (if it happens) does not by itself authorize any
   training run; a separate governance decision would be required for that,
   identical in kind to `meta-trainer-corpus-v2`'s and
   `pilot-metatrainer-v3`'s own "Intended use" sections. Because this
   record's own conclusion is that the corpus-growth-vs-method-change
   question is genuinely open rather than resolved, any future training-run
   proposal using this or any corpus should state explicitly which reading
   of the mixed evidence it is acting on and why, rather than treating
   either the flat exact-match reading or the rising-rubric reading as the
   settled interpretation.
