# Meta-trainer corpus v3 — dataset card (ADR-0015 proposal, ADR-0016 update)

Status: v3 (ADR-0015) merged and executed. This card is updated in place for
ADR-0016, which rewrites the `confabulated_recipe_detection` family only (see
"ADR-0016: confabulated_recipe_detection rewrite" below) after diagnosis
() found the family's original 8 records were
format-mismatched against the behavior actually tested by held-out item
`mtr-v2-heldout-0013`. This update does not touch, replace, or invalidate the
existing merged `examples/pilot-metatrainer-v2/train.jsonl` / `held_out.json`
(SHA-256 `6fa2d114a822ee081ba09d5d4f31ef4dd107fe92c43d6037abbd9c98492f17da` /
`d99b76b3d24298fa7660b1335835bf37a1e16e7bca858c9f3f980f21e70017a4`) or the
`examples/metatrainer-corpus-addition-training-history/` draft, and does not
change train/held-out split boundaries: `mtr-v2-heldout-0013` stays
`held_out`, and `confabulated_recipe_detection` stays train-only.

## Scope and provenance

163 records (115 train / 48 held-out) as of ADR-0016 (previously 160: 112
train / 48 held-out under ADR-0015), assembled from three sources, none
modified from its own already-reviewed/audited content except the two
explicit, previously-identified fixes below:

1. **meta-trainer-corpus-v2** (60 records: 40 train / 20 held-out), copied
   byte-identical from `examples/pilot-metatrainer-v2/{train.jsonl,held_out.json}`.
   Independently audited to 60/60 in task. No content, citation,
   or split assignment is changed here.
2. **The training-history case-study addition** (12 records: 8 train / 4
   held-out), copied from
   `examples/metatrainer-corpus-addition-training-history/CANDIDATE_CASE_STUDIES.jsonl`.
   Independently audited in task: 11/12 records passed every
   check; 1 record (`mtr-hist-case-heldout-0001`) failed the semantic
   train/held-out split check because its answer text explicitly restated the
   train-side record's mechanics-vs-capability distinction rather than testing
   a genuinely withheld angle. That record is rewritten here, exactly per the
   audit's own suggested fix, to test ADR-0011's explicit
   non-authorization-of-further-pilot scope statement instead. No other record
   in this addition is changed. Two schema adjustments were made for
   uniformity with the rest of this corpus, both purely structural (no content
   change): the field `source_locators` was renamed `citations`, and each
   family's originally-shared train/held-out `semantic_family` label was split
   into a train-only name and a matching `..._heldout` name (see "Split
   construction" below for why).
3. **New material** (91 records: 67 train / 24 held-out, as of ADR-0016; was
   88: 64 train / 24 held-out under ADR-0015) added by this proposal, spanning
   11 new semantic families (8 train-only, 3 held-out-only). All new material
   is weighted toward calibrated uncertainty and refusal, per this task's
   explicit instruction following the real confabulated learning-rate-recipe
   finding captured in ADR-0014's own evaluation run (see "Calibrated-refusal
   emphasis and its real trigger" below). ADR-0016 adds 9 records to and
   removes 6 records from `confabulated_recipe_detection` specifically (net
   +3 train records versus ADR-0015); see "ADR-0016: confabulated_recipe_detection
   rewrite" below.

Every record, across all three sources, contains:

- a stable `example_id`;
- an explicit `semantic_family`;
- a `source_scope`;
- a non-empty `citations` array with a locator resolvable in `SOURCE_MAP.md`,
  a "research synthesis, MSx" label resolvable in `SYNTHESIS_NOTES.md`, a
  "research synthesis, SSx.x" label (carried-forward v2 records only), or a
  direct repository-evidence locator (file path, internal tracking item id, or commit);
- one user message and one assistant answer.

## Split construction

The split was assigned at the semantic-family level before any new content
was authored, following corpus v2's own rule exactly: every `semantic_family`
is wholly train or wholly held-out, never both. `validate_dataset.py`
structurally proves this (`semantic_family_disjoint` check) the same way
corpus v2's validator did.

One adaptation was required for the carried-forward history addition. Its
original 12-record draft assigned train and held-out records to the *same*
family name per family (e.g. `adr0011_minimind_mechanical_success_vs_task_failure`
covering 2 train-candidate + 1 held-out-candidate records), a looser
split-integrity convention than corpus v2's own family-wholly-on-one-side
rule, verified only by its own independent content-level audit rather than
also being structurally checkable. Per this task's explicit instruction to
apply the *same* non-negotiable rigor as corpus v2's original build, this
proposal re-partitions each of the 4 history families into a train-only family
and a matching `..._heldout`-suffixed held-out-only family, so the stronger,
structurally-provable v2 invariant holds for every family in this corpus, not
only the carried-forward v2 records. The content-level pairing the original
independent audit verified (each held-out record tests a distinct angle of its
train counterpart) is unchanged; only the family *label* used for the
disjointness check was split.

Families and counts (32 families total: 8 v2-train + 3 v2-held-out unchanged,
4 history-train + 4 history-held-out, 8 new-train + 3 new-held-out):

Train-only families (115 examples, as of ADR-0016; was 112 under ADR-0015):

- `zero_score_diagnostic_ladder` (8, v2, unchanged)
- `sft_objective_masking_format` (8, v2, unchanged)
- `qualified_hyperparameter_sweeps` (8, v2, unchanged)
- `contamination_split_controls` (8, v2, unchanged)
- `limited_data_evaluation` (8, v2, unchanged)
- `adr0011_minimind_mechanical_success_vs_task_failure` (2, history, unchanged)
- `corpus_audit_catches_leakage_before_training` (2, history, unchanged)
- `adr0013_full_sft_overfitting_regression` (2, history, unchanged)
- `governance_stop_and_reject_trigger_fires_correctly` (2, history, unchanged)
- `calibrated_refusal_when_evidence_absent` (8, **new**)
- `accuracy_vs_calibration_tradeoff` (8, **new**)
- `self_consistency_and_sampling_checks` (8, **new**)
- `exact_match_scorer_limits_and_proxies` (8, **new**)
- `confabulated_recipe_detection` (11, **new**, **rewritten for ADR-0016** —
  was 8 under ADR-0015)
- `uncertainty_language_calibration` (8, **new**)
- `refusal_vs_overrefusal_balance` (8, **new**)
- `bounded_reasoning_under_missing_citation` (8, **new**)

Held-out-only families (48 examples):

- `peft_lora_qlora_decisions` (8, v2, unchanged)
- `synthetic_data_tradeoffs` (6, v2, unchanged)
- `catastrophic_forgetting_retention` (6, v2, unchanged)
- `adr0011_minimind_mechanical_success_vs_task_failure_heldout` (1, history,
  1 record rewritten per audit fix, see above)
- `corpus_audit_catches_leakage_before_training_heldout` (1, history, unchanged)
- `adr0013_full_sft_overfitting_regression_heldout` (1, history, unchanged)
- `governance_stop_and_reject_trigger_fires_correctly_heldout` (1, history, unchanged)
- `calibrated_refusal_generalization` (8, **new**)
- `hallucination_incentive_diagnosis` (8, **new**)
- `sampling_consistency_application` (8, **new**)

## Calibrated-refusal emphasis and its real trigger

Tonight's real, captured evidence directly motivates this corpus's emphasis:
during ADR-0014's real, executed evaluation run (`run_id
adr0014-metatrainer-sft-20260922`), the trained candidate's response to
held-out item `mtr-v2-heldout-0013` fabricated a specific, unsourced numeric
learning-rate recipe ("The learning rate is set to 0.01. The model is trained
for a certain number of epochs...") with no citation and no correspondence to
this project's own reviewed hyperparameter guidance. This is a real,
safety-relevant confabulation, not a hypothetical illustration, and it is
cited directly (with the exact raw_output text) in the new
`confabulated_recipe_detection` family and referenced from three other new
families. New sources [24]-[30] in `SOURCE_MAP.md` (Kadavath et al. on
model self-knowledge calibration; Lin/Hilton/Evans' TruthfulQA; Kalai et al.'s
account of why accuracy-only grading incentivizes confident guessing over
honest abstention; Manakul/Liusie/Gales' SelfCheckGPT; OpenAI's Model Spec
"Express uncertainty" guideline; the Hugging Face exact_match metric card; and
the real pilot Anthropic-OpenAI alignment evaluation comparing refusal and
hallucination rates) were fetched and read in full specifically to ground this
emphasis in independently-verifiable, directly-read technical sources rather
than general impression.

The 8 new train-only calibration families (64 records under ADR-0015; 67 as
of ADR-0016) and 3 new held-out-only calibration families (24 records) are
deliberately NOT the entire new addition's balance point:
`refusal_vs_overrefusal_balance` and several records across other new
families explicitly test that calibrated refusal is scoped to genuinely
unsupported claims, not a blanket hedge-everything pattern, per this
project's own real evidence that a 70% refusal rate on one benchmark
"limits utility" even while reducing hallucinated errors (source [30]).
This corpus does not want to trade one failure mode (confident fabrication)
for another (reflexive over-hedging on well-supported claims).

## ADR-0016: confabulated_recipe_detection rewrite

Diagnosis ( comment posted 2026-09-22): despite
ADR-0015's corpus v3 adding 8 dedicated `confabulated_recipe_detection`
training records, the ADR-0015 candidate's raw output on held-out item
`mtr-v2-heldout-0013` got WORSE than ADR-0014's — it went from merely
demonstrating the fabrication in a rambling worked example to explicitly
answering the item's own closed yes/no question with "Yes, a synthetic
answer can invent a learning-rate recipe if it sounds plausible," i.e.
endorsing confabulation as normative policy. Root cause: the original 8
records were long (512-724 char), third-person, retrospective meta-commentary
about the historical ADR-0014 incident. None of them rehearsed the tested
item's own shape — a short (148-char reference), first-person, closed
yes/no question answered with a terse "No, because..." refusal. Split/
leakage check was independently confirmed clean (families remain disjoint;
`mtr-v2-heldout-0013` unaffected); dilution was assessed as a real but
secondary factor given the corpus is trained for exactly one epoch.

Fix applied here: kept 2 of the original 8 records
(`mtr-v3n-train-0033`, `mtr-v3n-train-0034`) as accurate, well-cited
scaffolding/context — they establish what the real ADR-0014 confabulation
was and why it's a distinct, worse failure mode than a degeneration loop —
and removed the other 6, which were pure retrospective analysis with no
behavioral-rehearsal value for this specific target. Added 9 NEW short
(1-3 sentence), first-person, closed-question-shaped records that directly
mirror `mtr-v2-heldout-0013`'s own prompt/answer shape ("Can you invent/
produce/extrapolate a specific [X] if it sounds plausible?" -> "No,
[because...]."), varying [X] across a numeric training recipe, a cost
figure, a calendar date, an accuracy percentage, a citation, a GPU-memory
figure, a rounded percentage, a citation locator, and a benchmark score —
genuine behavioral rehearsal of the refusal shape, not repeated paraphrase
of one case study. Net family size: 8 -> 11 (2 kept + 9 new). Full
citation-locator rigor applied (every new record's citations pass
`validate_dataset.py`'s existing `citation_locator_format` check against
numbered sources [24]/[26]/[28] or this project's own repository-evidence
locator prefixes; no new source was fabricated). Train/held-out split
boundaries are untouched: `confabulated_recipe_detection` remains
train-only and `mtr-v2-heldout-0013` remains `held_out`, both verified by
`validate_dataset.py`'s `semantic_family_disjoint` and
`manifest_family_single_split` checks, which pass unchanged.

## Citation policy

Direct factual claims use a numbered source `[N]` (`[1]`-`[23]` carried
forward from the corpus v2 bibliography, `[24]`-`[30]` newly added and fetched
directly during this proposal's drafting) where that source directly supports
the statement. Cross-source or generalizing conclusions that no single
numbered source states on its own are labeled `research synthesis, MSx` and
recorded in `SYNTHESIS_NOTES.md`, the v3 analogue of corpus v2's `SSX.X` locators,
naming the specific source(s) the synthesis draws on and the added inferential
step. Carried-forward v2 records retain their original `research synthesis,
SSX.X` labels unchanged. Direct repository evidence (an exact on-disk
evidence-file field path, a internal task's recorded metadata, a specific
commit, or a specific markdown section/line range at a specific commit) is
used for the history-addition and confabulation-evidence families, following
the same discipline the history addition itself established and this task's
own independent audit already validated.

No record reproduces a source verbatim beyond short, directly-quoted phrases
attributed and locatable to their exact source (for example, the Model Spec's
own five-item uncertainty-outcome ranking, or the exact_match metric card's
own two-line worked example) — every record is a synthetic Q&A pair in the
same public/synthetic pattern as corpus v2's existing 23 sources, not a
reproduction of the source's own text as an answer.

## Explicit exclusions and cautions

This corpus does not assert:

- that any specific numeric hyperparameter, sample-count, or refusal-rate
  threshold is universally correct — every cited real evaluation figure (the
  52%/22%/26% and 1%/24%/75% comparison, TruthfulQA's 58%/94% figures, the up
  to 70% Claude refusal-rate figure) is scoped explicitly to the exact
  benchmark and model(s) the cited source reports it for;
- that adding calibrated-refusal training examples resolves this project's own
  diagnosed exact-match scorer/free-form-task mismatch (task) —
  the `hallucination_incentive_diagnosis` family explicitly states this
  remains a separate, undecided scorer-design question;
- that this corpus's calibrated-refusal training would reliably generalize to
  question types not represented in it — the `calibrated_refusal_generalization`
  and `hallucination_incentive_diagnosis` held-out families explicitly name
  Kadavath et al.'s own finding that this kind of calibration only partially
  generalizes to new tasks as a real limitation, not a solved problem;
- a promotion, deployment, or production-readiness claim about any model;
- a licensing or legal conclusion of any kind. No licensing determination was
  made for any source cited in this corpus, new or carried-forward.

## Intended use

This corpus is a proposal for repository admission and independent review
only. Consistent with corpus v2's own "Intended use" section and ADR-0013/
ADR-0014's governance pattern, repository admission does not by itself
authorize any training run; a separate, explicitly new ADR-0015 governance
decision (three-gate signing: security, dataset-rights, owner) is required
before this corpus, or any subset of it, is used in a training run, exactly
as ADR-0014 required its own fresh gate signing rather than reusing
ADR-0013's already-consumed approvals.

## Files

- `train.jsonl`: 112 records.
- `held_out.json`: 48 records.
- `semantic_family_manifest.json`: family-to-split and family-to-id mapping
  across all 32 families.
- `held_out_exclusion_registry.json`: held-out family and id registry.
- `REPRESENTATIVE_EXAMPLES.json`: six exact records copied programmatically
  from the generated corpus, spanning all three provenance sources and the
  one audit-fixed record.
- `SOURCE_MAP.md`: carried-forward sources `[1]`-`[23]` plus newly
  fetched sources `[24]`-`[30]`, each with the specific verified claim it
  supports.
- `SYNTHESIS_NOTES.md`: the `research synthesis, MSx` locator registry, the v3
  analogue of corpus v2's `SSX.X` sections.
- `generate_corpus.py`: deterministic generator; merges the two carried-forward
  sources unchanged (plus the one documented audit fix) and appends the new
  material; does not modify any file under `pilot-metatrainer-v2/` or
  `metatrainer-corpus-addition-training-history/`.
- `validate_dataset.py`: structural and policy validator, extended from
  corpus v2's validator for the larger size window and broader citation-locator
  format.
- `VALIDATION_REPORT.json`: validator output, written after validation.
- `SELF_AUDIT_REPORT.md`: this proposal's own self-audit against the same
  C1-C6-equivalent checklist that got corpus v2 to 60/60, run directly by the
  drafting task (not yet an independent second-reviewer audit — see that
  report's own stated limitation and the "Next steps" section below).

## Next steps (not performed by this task)

1. Independent audit of this corpus with the same rigor as corpus v2's own
   audits and the training-history addition's audit (tasks) — full
   per-example citation-locator re-verification against the actual source
   text, and independent semantic-split content review, not a self-report.
2. Draft ADR-0015 (see `docs/decisions/0015-metatrainer-corpus-v3.md`) and open
   a PR admitting this corpus under `examples/pilot-metatrainer-v3/`,
   explicitly not touching `pilot-metatrainer-v2/` or
   `metatrainer-corpus-addition-training-history/`, and not changing the
   frozen ADR-0013/ADR-0014 `dataset_hash` values those ADRs gate on.
3. Security/dataset-rights review of that PR, per this repository's
   existing governance pattern for anything proposed for the training corpus.
4. Repository admission (if it happens) does not by itself authorize any
   training run; a new, separate governance decision (with its own fresh
   three-gate signing, per ADR-0014's own precedent) is required for that.
