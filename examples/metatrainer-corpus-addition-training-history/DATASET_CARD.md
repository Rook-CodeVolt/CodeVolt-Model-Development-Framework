# Meta-trainer corpus addition proposal: real training-run history as case studies

Status: DRAFT PROPOSAL, NOT YET REVIEWED. This is a separate addition proposal,
not part of the merged `./local-evidence/meta-trainer-corpus-v2/` corpus
(`train.jsonl` SHA-256 `6fa2d114a822ee081ba09d5d4f31ef4dd107fe92c43d6037abbd9c98492f17da`,
`held_out.json` SHA-256 `d99b76b3d24298fa7660b1335835bf37a1e16e7bca858c9f3f980f21e70017a4`).
Neither of those two files, nor any other file already admitted to
`examples/pilot-metatrainer-v2/`, is modified by this proposal. Nothing here is
committed to git or opened as a PR by this task; that is explicitly left for a
separate review/PR cycle per the task that requested this document.

## Scope

12 new candidate examples (8 train-candidate, 4 held-out-candidate) grounded
only in this repository's own real, already-happened training-run history and
its own internal audit record -- not in the external research corpus, and not
overlapping in content or source with the existing `meta-trainer-corpus-v2`
records. This is intentionally a different provenance class from that corpus:
every claim here traces to this repository's own git history, its own
committed ADR text, or a real evidence JSON file already on this host, rather
than to a third-party research bibliography.

File: `CANDIDATE_CASE_STUDIES.jsonl` (12 JSON-Lines records, one per line).

## Case-study families (4 families, 3 records each: 2 train-candidate + 1 held-out-candidate)

1. `adr0011_minimind_mechanical_success_vs_task_failure` -- the ADR-0011
   MiniMind from-scratch pilot: training accepted within budget
   (final loss 2.0751, 1 epoch / 40 examples) but scored 0.0/10 on held-out
   evaluation, with a confirmed root cause (prompt-dependent immediate-EOS
   generation).
2. `corpus_audit_catches_leakage_before_training` -- the real two-corpus
   rejection plus third-corpus targeted-fix-and-reaccept history: two
   independently built ~75/80-example corpora both failed an independent
   citation-locator and semantic-split audit despite passing their own
   generator's structural validator; a third corpus was audited, found
   close-to-acceptable with 4 citation and 7 semantic-leakage defects, fixed,
   and re-audited clean.
3. `adr0013_full_sft_overfitting_regression` -- the real ADR-0013 bounded
   full-SFT cycle: training accepted, but paired baseline/candidate evaluation
   showed no target-task gain (0.0%->0.0%) and regression on two unrelated
   held-out suites (capability_retention 30%->0%, safety 46.7%->33.3%), with
   degenerate repetition-loop generations as corroborating evidence.
4. `governance_stop_and_reject_trigger_fires_correctly` -- the real governance
   response: ADR-0013's own predeclared stop-and-reject trigger fired on this
   exact result, a same-script hyperparameter-edit request was declined
   pending explicit owner direction, and the correct response (a new ADR-0014
   with its own fresh three-gate approval) was taken instead of an informal
   retry under the old approval.

## Citation policy

Every record's `source_locators` field names one or more of: a specific
markdown file path plus section heading and line range at a specific commit
SHA, a specific JSON evidence file path plus its recorded `evidence_hash`, a
specific field path inside a real evidence JSON already on this host
(`./local-evidence/adr0013/scratch/adr0013-metatrainer-sft-20260920/cycle_result.json`),
or a specific completed internal tracking item id plus the exact metadata field(s) relied
on. No record cites a source that was not directly read and checked against
the claim in that record during drafting (see "Self-audit" below). No
`research synthesis`-style locator is used anywhere in this file: none of this
material is drawn from the external research corpus, and it must not be confused
with or merged into `meta-trainer-corpus-v2`'s existing citation namespace.

## Explicit exclusions and cautions

This proposal does not assert:

- that these four incidents generalize to a rule about all training runs of
  any size or architecture -- each record is scoped to the specific real run
  it cites, with model size / example count / epoch count stated inline
  where load-bearing;
- a promotion, deployment, or production-readiness claim about any model
  discussed;
- that ADR-0013's regression was caused solely by learning rate, epoch count,
  or any single hyperparameter -- the records describe the observed
  before/after evidence and the ADR's own predeclared response, not a novel
  causal theory beyond what the cited ADR text itself states;
- a licensing or legal conclusion of any kind.

## Self-audit (same C1-C6-equivalent rigor as the corpus v2 audits)

This is a self-review by the drafting task, not a substitute for an
independent audit; the task that requested this material explicitly asks for
it to be "reviewed separately... same rigor as the original corpus v2 audit"
before merge, which is out of scope for the drafting worker to certify
alone. It is included so a reviewer has a starting checklist, not as a
claim of independent acceptance.

| Check | Result | Notes |
|---|---|---|
| Every `example_id` unique | PASS | 12/12 unique, verified programmatically. |
| Every record has >=1 source locator | PASS | 12/12, verified programmatically. |
| Every record has exactly one user + one assistant message | PASS | 12/12, verified programmatically. |
| Source-boundary: every factual claim traces to a real, directly-read artifact (git commit, JSON evidence file, or completed internal tracking item metadata) at or before drafting time | PASS (self-review) | Every numeric value (loss=2.0751, aggregate_score values, check_totals, verdicts) was copied from a file this task directly read in this same session -- ADR-0011/0013/0014 markdown at their committed SHAs, the sanitised ADR-0011 evidence JSON, the real `cycle_result.json` on disk, and the four corpus-audit internal tracking items' own `metadata` fields -- not paraphrased from memory or another report. |
| Citation-claim support: locator text actually supports the specific claim made, not just topical relevance | PASS (self-review) | Spot-checked during drafting: e.g. the "58-46-46-46..." and "PEPFET, a variant of PEPFET..." quotes were copied verbatim from `cycle_result.json`'s `candidate.capability_retention.results[2].raw_output` and `candidate.meta_trainer.results[0].raw_output` fields respectively, not reconstructed from the ADR's own paraphrase of them. |
| No fabricated framing / no invented numbers | PASS (self-review) | All numeric evaluation scores, loss values, and pass/fail counts are copied exactly from source; no interpolated or rounded-for-effect figures. (One user-turn prompt originally said "around 2.08" for the ADR-0011 loss and was corrected to "around 2.1" with the exact 2.0751 value preserved in the assistant answer, to avoid a numeric claim in a user turn that doesn't exactly match the cited source.) |
| Train/held-out semantic-family split: family assigned before content was finalized, held-out record tests a related-but-distinct angle of the family rather than a rephrasing of a train record | PASS (self-review), NEEDS INDEPENDENT VERIFICATION | Structural check (family-level split, no id overlap) passed programmatically. Content-level non-overlap is asserted by the drafting task (e.g. family 1's two train records cover "accepted status + loss don't imply capability" and "check EOS-emission root cause," while its held-out record covers "reconcile the pipeline-worked claim with the zero score" -- a distinct question from either train record) but was not independently re-verified by a second reviewer, which is exactly the check that caught real leakage in `meta-trainer-corpus-v2`'s first audit. This is the single item on this table a reviewer should scrutinize most closely, per that corpus's own precedent. |
| No licensing/legal conclusion asserted | PASS | See "Explicit exclusions." |
| Distinct provenance from `meta-trainer-corpus-v2` (no accidental reuse of externally-sourced content or citation ids) | PASS | Confirmed by construction: no record references any external research-synthesis section, numbered external source `[N]`, or `Seed-rich A/B` locator; all locators are internal-repository/internal-only. |

### Known limitation this self-audit cannot close

The drafting task is the same task proposing the content, so the
citation-locator and semantic-split checks above are self-review, not
independent audit, exactly the distinction the corpus-v2 precedent (tasks) established matters: a
generator or drafter's own self-report is not equivalent to an independent
re-verification from scratch. Per that precedent and per this task's own
instructions, this file should not be merged into any training corpus or
proposed for training use until an independent reviewer re-checks every
locator against the cited source directly (not against this card's
paraphrase of it) and independently re-examines the four train/held-out
pairs for content-level leakage, the same way re-verified
`meta-trainer-corpus-v2` from scratch rather than trusting its generator's
own `VALIDATION_REPORT.json`.

## Files

- `CANDIDATE_CASE_STUDIES.jsonl`: 12 candidate records (8 train-candidate,
  4 held-out-candidate), one JSON object per line.
- `DATASET_CARD.md`: this file.

## Next steps (not performed by this task)

1. Independent audit of this file with the same rigor as
   `meta-trainer-corpus-v2`'s audits (tasks) -- full per-example citation-locator
   verification against the actual source text/JSON, and independent
   semantic-split content review, not a self-report.
2. If accepted, a small separate PR adding this file (or its
   post-correction revision) under `examples/`, explicitly not touching
   `meta-trainer-corpus-v2/train.jsonl` or `held_out.json` and not changing
   the frozen ADR-0013/0014 `dataset_hash` values those ADRs gate on.
3. Security-reviewer/dataset-rights review of that PR per this repository's
   existing governance pattern for anything proposed for the training
   corpus, before any merge.
4. Repository admission (if it happens) does not by itself authorize any
   training run; a separate governance decision would be required for that,
   identical in kind to `meta-trainer-corpus-v2`'s own "Intended use"
   section.
