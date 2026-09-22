# Self-audit report — meta-trainer corpus v3 (ADR-0015 proposal)

Author: marcus (the same task that drafted this corpus). Per this project's own
established precedent (`examples/metatrainer-corpus-addition-training-history/DATASET_CARD.md`'s
own self-audit section, and the corpus v2 audit history — tasks), **this is a self-review, not a
substitute for an independent audit.** A drafting task auditing its own output
is not equivalent to an independent, no-sampling re-verification from a fresh
session with no stake in the content being accepted. This report exists so a
reviewer (independent audit task, then Maya) has a concrete starting checklist
and a record of what was actually checked and how, not as a claim that this
corpus is already independently accepted.

Scope: all 160 records (112 train / 48 held-out) — 60 carried forward from
corpus v2 (already independently audited to 60/60 in), 12 carried
forward from the training-history addition (already independently audited in
, with the one required fix applied exactly as recommended), and
88 new records (this proposal's own new content, not yet independently
audited by anyone other than this same task).

## C1-C6-equivalent checklist (same categories corpus v2's audits used)

| Check | Result | Basis |
|---|---|---|
| C1 (reasoning-seed-only: claims trace to the actual source's own claim, not a broader unsupported generalization) | Self-review PASS, spot-verified | See "New-source verification" below — 6 of 7 new sources' key quoted claims were independently re-fetched and checked against the live source text during this session (not merely trusted from the prior drafting pass). |
| C2 (no unsupported categorical claim) | Self-review PASS | `DATASET_CARD.md`'s "Explicit exclusions and cautions" section states what this corpus does not assert; spot-read of new-family records (`accuracy_vs_calibration_tradeoff`, `calibrated_refusal_generalization`) confirms each numeric or comparative claim is scoped to the specific benchmark/model it was reported for, not generalized. |
| C3 (numeric-recipe qualification: no invented specific number presented as fact) | Self-review PASS | All numeric figures in new records (52%/22%/26%, 1%/24%/75%, 58%/94%, 12%, 70%, the confabulated 0.01 LR) are either directly cited real reported figures or the real captured confabulated text quoted as an example of what NOT to do. No new record states an invented number as if verified. |
| C4 (synthetic/tinyBenchmarks-style caveats present where relevant) | Self-review PASS | `exact_match_scorer_limits_and_proxies` family explicitly scopes tinyBenchmarks' claim to item-count sufficiency for benchmark-score *estimation*, distinct from scorer-task fit — checked directly against this session's own re-fetch of source [21]'s abstract (already carried forward unchanged from v2; not re-verified again here since content citing it is new but the source itself was v2-audited). |
| C5 (citation-locator support: locator actually supports the specific claim) | Self-review PASS for carried-forward 72/160 (independently re-verified in); Self-review PASS, partially independently spot-checked for new 88/160 — see below. NOT yet independently audited for the new 88. | See "New-source verification" and "Repository-evidence verification" below. |
| C6 (semantic split: no train/held-out content leakage) | Self-review PASS, structurally proven + manually reviewed | `validate_dataset.py`'s `semantic_family_disjoint` check proves no family spans both splits (structural, not just self-reported). Manual content read of all 3 new held-out families against their corresponding new train families (below) found no restated-verbatim leakage. |

## Structural checks (programmatic, this session)

Re-ran `validate_dataset.py` fresh in this session: `{"status": "PASS", "checks": 16, "counts": {"total": 160, "train": 112, "held_out": 48}}`.

Independently re-verified (this session, via direct JSON diff, not by trusting
`generate_corpus.py`'s own docstring claims) that the 60 corpus-v2 records and
12 history-addition records embedded in `train.jsonl`/`held_out.json` are
byte-for-byte identical in every field to their source files in
`examples/pilot-metatrainer-v2/` and
`examples/metatrainer-corpus-addition-training-history/` respectively
(field-by-field comparison script, 0 mismatches across all 72 carried-forward
records, with `mtr-hist-case-heldout-0001` correctly excluded from the
identity check as the one intentional, audit-recommended rewrite). This
directly confirms the DATASET_CARD.md claim that this proposal does not alter
already-audited content, rather than merely trusting the generator script's
own comment.

## New-source verification (this session)

Of the 7 newly-added sources ([24]-[30]), the following specific quoted claims
were independently re-fetched from the live source (not merely re-read from
the prior drafting pass's own SOURCE_MAP.md text) and confirmed to match:

- [24] Kadavath et al. — confirmed verbatim: "Models perform well at
  predicting P(IK) and partially generalize across tasks, though they
  struggle with calibration of P(IK) on new tasks." Matches the corpus's
  `accuracy_vs_calibration_tradeoff` record's quoted claim exactly.
- [25] Lin, Hilton, Evans (TruthfulQA) — confirmed: 817 questions, 38
  categories, best model truthful on 58% vs. 94% human performance; on the
  multiple-choice task, the 6B-parameter GPT-J model was 12% less truthful
  than its 125M-parameter counterpart (a separate generation-task finding,
  not the same comparison, is that the largest GPT-Neo/J model was 17% less
  truthful than a model 60x smaller, with no specific model size named).
  Corpus record and SOURCE_MAP.md corrected to the specific 12% multiple-
  choice figure per the independent audit finding that the
  prior drafting pass had conflated these two distinct figures.
- [26] Kalai et al. — confirmed the real reported SimpleQA table
  (gpt-5-thinking-mini: 52% abstention / 22% accuracy / 26% error;
  OpenAI o4-mini: 1% abstention / 24% accuracy / 75% error), the "accurate
  responses, errors, and abstentions" framing with errors ranked worse than
  abstentions, and the "accuracy will never reach 100%... hallucinations are
  not inevitable... because language models can abstain" framing. All match.
  (The specific Māori-language / "being calibrated requires much less
  computation than being accurate" worked example was not independently
  re-confirmed against the primary PDF in this session — carried over from
  the prior drafting pass's own read. Flagged for the independent auditor to
  check directly against the paper's "Conclusion" section.)
- [27] Manakul, Liusie, Gales (SelfCheckGPT) — confirmed: zero-resource,
  black-box, sampling-based; the consistency-vs-hallucination-divergence core
  premise; validated on GPT-3-generated WikiBio passages with human
  annotation; reports higher AUC-PR than grey-box baselines for sentence-level
  detection and best correlation for passage-level ranking. All match.
- [28] OpenAI Model Spec, 2025-02-12 revision, "Express uncertainty" — the
  outcome ranking "confident right answer > hedged right answer > no answer >
  hedged wrong answer > confident wrong answer" and the example phrasings ("I
  don't know", "I'm not sure", "I was unable to solve...", "I think", "I
  believe", "It might be") were independently confirmed against a live search
  snippet of `model-spec.openai.com/2025-02-12.html#express_uncertainty` this
  session. The five-named-causes list, the "If I understand what you mean" /
  "If my calculations are correct" / "If my sources are correct" / "If my
  information is up to date" phrasings, and the "impact of incorrect
  information" governing-factor language were NOT independently re-confirmed
  against that exact page in this session (the page could not be fetched in
  full via this session's available extraction tool, which is failing
  open-ended URL extraction; only search-snippet coverage was available, and
  it did not surface those specific substrings for the 2025-02-12 revision
  specifically, though closely matching language appears in adjacent Model
  Spec revisions). This is a genuine, stated gap: the corpus's own
  `uncertainty_language_calibration` family relies on these specific
  sub-claims, and they should be treated as carried over from the prior
  drafting pass's own direct read rather than independently re-verified in
  this session. **Flagged explicitly for the independent auditor to
  re-confirm directly against the live page (or an archived/cached copy of
  the exact 2025-02-12 revision) before this corpus is accepted.**
- [29] Hugging Face exact_match metric card — confirmed: binary per-example
  score, aggregate is the mean, the "Happy Birthday!"/"Happy New Year!" (score
  0) worked example, and the four optional normalization parameters
  (`ignore_case`, `ignore_punctuation`, `ignore_numbers`, `regexes_to_ignore`)
  that do not add paraphrase tolerance. All match exactly.
- [30] OpenAI, pilot Anthropic-OpenAI alignment evaluation post — confirmed
  verbatim: "Claude models had an extremely high rate of refusals—as much as
  70%," and "the high refusal rate limits utility." The specific claims about
  o3/o4-mini's contrasted lower-refusal/higher-hallucination pattern and the
  "Person Hallucination Test (v4)" / "SimpleQA No Browse (v1)" benchmark
  descriptions were carried over from the prior drafting pass's own direct
  read of the full post and were not independently re-confirmed against the
  full post text in this session (only the two verbatim quotes above were
  reconfirmed via search snippet). **Flagged for the independent auditor to
  re-confirm the benchmark-name and o3/o4-mini claims directly against the
  full post.**

## Repository-evidence verification (this session)

The `confabulated_recipe_detection` family's central factual claim — that the
ADR-0014 candidate model fabricated a specific numeric learning-rate recipe on
held-out item `mtr-v2-heldout-0013` — was independently re-verified this
session by directly reading the live, real, on-disk evidence file
`./local-evidence/adr0014/scratch/adr0014-metatrainer-sft-20260922/cycle_result.json`
(not by trusting the prior drafting pass's paraphrase). The exact
`raw_output` text quoted in the corpus's records
("Sure, I can help with that. Here's a possible synthetic answer: ... The
learning rate is set to 0.01. The model is trained for a certain number of
epochs...") was confirmed present verbatim at `candidate.meta_trainer.results`
(the `example_id="mtr-v2-heldout-0013"` entry with `score: 0.0`). This is the
real, safety-relevant confabulation that motivates this corpus's entire
calibrated-refusal emphasis, and its exact wording is independently confirmed
correct.

The `confabulated_recipe_detection` family's reference (`mtr-v3n-train-0034`)
to a `candidate.capability_retention.results` degeneration-loop text ("The
58-46-46-46..." and "PEPFET, a variant of PEPFET...") was checked against
this same ADR-0014 file and NOT found there — those strings do not appear in
`adr0014-metatrainer-sft-20260922/cycle_result.json`. They were instead
independently confirmed present at that field path in the earlier
`./local-evidence/adr0013/scratch/adr0013-metatrainer-sft-20260920/cycle_result.json`
(line ~363 for the "58-46-46..." text, line ~430 for the PEPFET text), matching
the record's own answer prose, which correctly attributes this evidence to
"this project's own ADR-0013 evidence." The record's `citations` field has
been corrected to point at the ADR-0013 file accordingly.

## Semantic train/held-out split — manual content review (this session)

For each of the 3 new held-out-only families, the actual content of every
held-out record was read and compared against its corresponding new
train-only family's content to check for restated-rather-than-distinct
reasoning, the exact defect class this project's audit history has twice
caught (corpus v2's first audit; the history addition's one required fix):

- `calibrated_refusal_generalization` (held-out) vs.
  `calibrated_refusal_when_evidence_absent` / `bounded_reasoning_under_missing_citation`
  (train): the held-out records apply the calibrated-refusal principle to
  genuinely new scenarios (a hypothetical future ADR-0016 threshold, an
  unnamed model's quantization support, a "rough estimate" framing pressure
  test, extrapolating a wall-clock duration, a changed-key-variable
  historical-recurrence question, a security-review-odds question, an
  unverifiable-replication question, and reconciling two apparently
  conflicting calibration/accuracy findings) not restated verbatim from any
  train record. No leakage found.
- `hallucination_incentive_diagnosis` (held-out) vs.
  `confabulated_recipe_detection` / `accuracy_vs_calibration_tradeoff` (train):
  the held-out records apply Kalai et al.'s incentive framework to new
  applied questions (fabricated-citation mechanism, whether the fix is purely
  socio-technical or also project-local, whether this project's own
  exact-match scorer specifically instantiates the described incentive
  problem, whether corpus-only changes fix a scorer-level problem, and
  P(IK)'s generalization limits applied to this corpus's own transfer claims)
  rather than restating the train family's confabulation-detection content.
  No leakage found.
- `sampling_consistency_application` (held-out) vs.
  `self_consistency_and_sampling_checks` (train): the held-out records apply
  SelfCheckGPT's method to new scenarios (a fresh 3-sample disagreement case,
  a per-example-id exclusion-registry design question, a resource-bounded
  minimum-sample-count question, a perfectly-consistent-but-wrong scenario,
  whether resampling would have caught the real confabulated-recipe finding,
  contrasting this method against OpenAI's evaluation-best-practices
  guidance, and whether adopting it would resolve the exact-match/free-form
  mismatch) rather than restating the train family's definitional content. No
  leakage found.

This structural pattern (train teaches definitions/scope/reported results;
held-out tests genuinely new application) mirrors corpus v2's own established
train/held-out design discipline, per DATASET_CARD.md.

## Forbidden-content / structural checks

`validate_dataset.py`'s `forbidden_content_absent` and `no_oversized_answers`
checks both passed; independently spot-read a sample of the longest new
answers (all under the 1600-character crude verbatim-copy threshold) and
confirmed none is a reproduction of a source passage rather than a synthetic
answer citing it.

## Known limitations this self-audit cannot close

1. This is a self-review by the drafting task; per this project's own
   established precedent, it does not substitute for an independent,
   no-sampling re-audit from a separate task/session with no stake in the
   content being accepted. All 88 new records specifically (not the 72
   carried-forward records, which already have independent audit history)
   need that independent pass before this corpus should be considered ready
   for repository commit in the same sense corpus v2 was after.
2. Two specific sub-claims — the Model Spec's "five listed causes of
   uncertainty" / example phrasings naming the uncertainty source, and the
   pilot Anthropic-OpenAI evaluation post's o3/o4-mini contrast and named
   benchmark descriptions — were not independently re-confirmed against the
   live primary source in this session (tool limitation, not a judgment
   call); they are carried over from the prior drafting pass's own claimed
   direct read. The independent auditor should re-confirm these two items
   specifically before accepting the `uncertainty_language_calibration` and
   `accuracy_vs_calibration_tradeoff`/`refusal_vs_overrefusal_balance`
   families' citation support as fully verified.
3. As stated throughout `DATASET_CARD.md`, this self-audit and the underlying
   corpus assert no licensing/legal conclusion, no promotion/deployment
   claim, and no claim that adding these examples resolves the separately
   diagnosed exact-match scorer/free-form-task mismatch.

## Recommendation

Proceed to: (1) open the PR proposing this corpus for repository admission,
documenting this self-audit's exact scope and the two flagged
not-yet-independently-confirmed citation items; (2) request an independent
audit of the 88 new records (same rigor as) as a
follow-up task before any merge decision; (3) Maya's security/dataset-rights
review, per this project's standing governance pattern, gates merge
regardless of the independent-audit outcome. This self-audit alone does not
constitute acceptance.
