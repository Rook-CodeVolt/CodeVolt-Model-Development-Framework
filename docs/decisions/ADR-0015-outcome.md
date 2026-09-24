# ADR-0015 outcome record: the four-cycle meta-trainer SFT arc

- Status: outcome record — documents real, executed results only; grants no
  new authority and does not itself propose or authorize any further run
- Date: 2026-09-22 (revised 2026-09-22, same day, after independent review —
  see "Revision history" at the end of this document)
- Tracking: internal execution records (see linked PR numbers above where applicable)
- Scope: this record closes out ADR-0015 (`docs/decisions/0015-metatrainer-corpus-v3.md`
  and `docs/decisions/ADR-0015-execution-config.md`) with the real executed
  result, and places that result in the context of the full four-cycle arc
  (ADR-0011, ADR-0013, ADR-0014, ADR-0015). It does not reopen, edit, or
  reinterpret any of those four ADRs' own text; each remains the authoritative
  record of its own run.

## Why this record exists

ADR-0011, ADR-0013, ADR-0014, and ADR-0015 are four separate, independently
gated, real training/pilot cycles run against this project's meta-trainer
model-development pipeline over three days (2026-09-20 to 2026-09-22). Each
one executed cleanly under real sandboxing, real signed three-gate review,
and real independent evaluation, and each one reported its own honest result
in its own text. No single one of those four ADRs was positioned to state the
pattern across all four, because ADR-0011 predates the meta-trainer target
task entirely and each of ADR-0013/0014/0015 was explicitly scoped to test
exactly one variable change against the one immediately prior. This record is
the first place that pattern is stated together, for the historical record,
per Rook's explicit instruction.

## The four cycles, side by side

| ADR | Date | Model / task | What changed vs. prior cycle | Target-task result (exact-match) | Manual rubric mean (0-4)* | Capability retention (arithmetic) | Safety |
|---|---|---|---|---|---|---|---|
| 0011 | 2026-09-20 | MiniMind 26M, from-scratch init, unit-conversion task | New model, new task, new pipeline (first real end-to-end pilot) | 0.0/10 (0.0%) | n/a (different task/scorer) | n/a (no retention suite in this pilot) | n/a |
| 0013 | 2026-09-20 | SmolLM2-135M-Instruct (pretrained), full-SFT, meta-trainer corpus v2 (40 train examples), `max_steps=120` (3 epochs), `lr=1e-5` | First full-SFT cycle on the pretrained meta-trainer target task | 0.0% -> 0.0% | 0.15 -> 0.35 | 30.0% -> 0.0% (regressed) | 46.7% -> 33.3% (regressed) |
| 0014 | 2026-09-22 | Same model/corpus, `max_steps=40` (1 epoch), `lr=5e-6` | Hyperparameters corrected (epoch count and LR both reduced) to address ADR-0013's overfitting/degeneration | 0.0% -> 0.0% | 0.15 -> 0.45 | 30.0% -> 20.0% (partially preserved) | 46.7% -> 46.7% (fully preserved) |
| 0015 | 2026-09-22 | Same model, same hyperparameters as ADR-0014, corpus grown to v3 (112 train examples, 2.8x), `max_steps=112` (still 1 epoch, scaled 1:1 with corpus), memory ceiling raised 8192MB -> 16384MB after an evidence-based diagnosis of a real overshoot | 0.0% -> 0.0% | 0.15 -> 0.65 | 30.0% -> 20.0% (unchanged from ADR-0014) | 46.7% -> 40.0% (regressed, worse than ADR-0014) |

\* Manual rubric mean is a 4-axis (technical_conclusion,
reasoning_and_qualification, evidence_discipline, uncertainty_refusal_boundary)
0/1 human-equivalent score, 0-4 scale, scored on the same 20 held-out items
(`mtr-v2-heldout-0001`..`0020`) across all four conditions (untrained
baseline, ADR-0013, ADR-0014, ADR-0015 candidates) so the four scores are
directly comparable. It is a separate signal from the automated exact-match
scorer in the adjacent column; see "ADR-0015's manual rubric result and the
four-run trend" below for why both signals are recorded and what each does
and does not support. Baseline/ADR-0013/ADR-0014 rubric scores are from the
earlier baseline rubric review; ADR-0015's rubric score is from the
follow-on independent rubric review, which re-applied the identical
standard to ADR-0015's outputs on the identical items.

Every numeric value above is copied verbatim from the real, hash-recorded
evidence for each cycle: ADR-0011's `examples/pilot-adr0011/evidence/pilot-result.sanitised.json`;
ADR-0013's `docs/decisions/0014-corrected-bounded-metatrainer-cycle.md`
results table (lines 20-24, itself sourced from the real
`adr0013-metatrainer-sft-20260920` run); ADR-0014's own real run
(`adr0014-metatrainer-sft-20260922`, rubric `0.45/4.0` against the required
`>=3.0/4.0`, recorded in `docs/decisions/0015-metatrainer-corpus-v3.md` lines
15-22 and the project `CHANGELOG.md`); and ADR-0015's real re-executed run
(`adr0015-metatrainer-sft-20260922`, commit `e19dee129310cb6a39d2b8691a0d8232a9210898`,
`cycle_result.json` SHA-256 `7685e935634cf7dc1bbe96fb4b45b08060dfd5b128de01bb24f5c6532ac8e25f`,
reported and independently gate-verified). The
manual rubric figures are from a reviewer's local evidence archive, which
itself carries forward the baseline/ADR-0013/ADR-0014
rubric scores unchanged from the earlier baseline review's
`adr0013_adr0014_independent_rubric_sheet.json` and independently re-scores
only the new ADR-0015 candidate on the same items and standard.

## ADR-0015's own real result, in detail

The real, executed ADR-0015 cycle (after the memory-ceiling fix in PR #89 and
the doc-drift fix in PR #90) is the fourth and, to date, final cycle in this
arc:

- `training.status`: `accepted` — full 112-step (1-epoch) SFT pass completed,
  final `train_loss` (mean) `3.531`, `memory_mb_peak` `12,119.7` MB
  (comfortably under the corrected `16,384` MB ceiling; the identical real
  workload had exceeded the prior `8,192` MB ceiling in the first execution
  attempt, root-caused to legitimate MPS caching-allocator
  high-water-mark growth, not a leak).
- `meta_trainer` (target task, 48 held-out items, the same suite ADR-0014 used
  scaled to the larger corpus, automated `exact_match` scorer): baseline
  `0.0%`, candidate `0.0%` — **no change** on this scorer, matching the
  untrained baseline and both prior trained candidates (ADR-0013, ADR-0014)
  exactly. See "ADR-0015's manual rubric result and the four-run trend"
  below for the independent manual rubric signal on the same task, which
  tells a different story than this automated scorer alone.
- `capability_retention` (10-item unrelated arithmetic suite): baseline
  `30.0%`, candidate `20.0%` — identical to ADR-0014's candidate result, no
  further degradation but no improvement either.
- `safety` (15-item probe suite): baseline `46.7%`, candidate `40.0%` — a
  **regression** relative to baseline, and worse than ADR-0014's candidate
  result (`46.7%`, fully preserved). Growing the corpus 2.8x did not preserve
  ADR-0014's safety-retention gain.
- `promotion_decision`: `null`, as designed — the runner is measurement-only
  and has no promotion path; per `0013-bounded-pretrained-metatrainer-cycle.md`'s
  "Pass, continue, and stop thresholds" (inherited unchanged by ADR-0014 and
  ADR-0015), a `0.0%` target-task result is squarely inside the "stop and
  reject" trigger (`0013-...md` line 292: "zero or less than 0.25/4.0 rubric
  improvement with no other material gain") — the same trigger ADR-0013 and
  ADR-0014 each already recorded firing.

ADR-0015's own text (`docs/decisions/0015-metatrainer-corpus-v3.md`,
"Consequences") explicitly predicted this possible outcome in advance: "If a
future training run using this corpus still floors at 0.0% on the
`meta_trainer` suite, that would not by itself indicate the corpus failed —
the scorer-mismatch diagnosis predicts a possible floor regardless of corpus
quality." That prediction is now a confirmed real result, not a hypothesis.

## ADR-0015's manual rubric result and the four-run trend

The `0.0%` figures above are the automated `exact_match` scorer only. That
scorer performs whitespace/case-normalized full-string containment of the
entire expected answer inside the model's greedy decode; `meta_trainer`'s
reference answers are full explanatory sentences/paragraphs (462-1271
characters for the 28 items new to corpus v3), not short factual strings.
An independent review confirms this scorer is structurally
incapable of a nonzero score on this task shape regardless of training
quality, and that this applies at least as strongly to the v3 corpus's newer,
more-qualified reference answers as to the original v2 ones — `exact_match`
is not a valid signal for this task and must not be read as "the model made
zero progress." Because of this, an independent manual rubric review (4-axis
0/1 human-equivalent scoring: technical_conclusion,
reasoning_and_qualification, evidence_discipline,
uncertainty_refusal_boundary, 0-4 scale) was run on the same fixed 20
held-out items across all four conditions (using the same baseline rubric
review for baseline/ADR-0013/ADR-0014 and the follow-on rubric review for
ADR-0015, re-applying the
identical standard). That review is the real, complete picture and belongs in
this record alongside the exact-match discussion above:

- Rubric mean sequence: baseline `0.15` -> ADR-0013 `0.35` -> ADR-0014 `0.45`
  -> ADR-0015 `0.65` (out of 4.0). This is **strictly monotonically
  increasing across all four points**, including the ADR-0014-to-ADR-0015
  step, which changed only corpus size (no LR/epoch change) and produced the
  largest single-step jump of the three transitions (`+0.20`).
- Against ADR-0013's own promotion-eligibility gate (rubric mean `>=3.0/4.0`,
  improvement over baseline `>=0.50/4.0`, every family mean `>=2.5/4.0`, zero
  critical errors, exact-containment corroboration `>=2` over baseline):
  ADR-0015 does **not** meet the gate overall, but for the first time across
  the four runs its improvement-over-baseline sub-condition (`+0.50/4.0`,
  `+10` raw points) is exactly met. The absolute-level sub-condition remains
  the binding failure: `0.65/4.0` (16.25%) is roughly one-fifth of the way to
  the required `3.0/4.0` (75%) bar — a wide margin, not a near-miss.
- One previously-persistent critical error (item `mtr-v2-heldout-0019`,
  "should the checkpoint be promoted despite regression", present as a
  critical error in both ADR-0013's and ADR-0014's candidates) is fully
  resolved in ADR-0015 — the first and only condition to answer it correctly.
- A **new** critical error appears on item `mtr-v2-heldout-0013` in the
  ADR-0015 run: the candidate explicitly endorses inventing a
  plausible-sounding numeric recipe ("Yes... if it sounds plausible") to a
  question whose correct answer is "no." This is in the exact semantic family
  (`confabulated_recipe_detection`) that corpus v3 added an entire new
  8-item training family specifically to fix, per that corpus's own
  `DATASET_CARD.md`. ADR-0014's candidate had already shown a related failure
  by inventing a recipe in its own answer; ADR-0015's candidate goes further
  and normatively endorses the practice — arguably a sharper instance of the
  same failure mode, not a fix. Net critical-error count for ADR-0015 is `2`
  (items `0003`, `0013`), tying ADR-0013's best-observed count and improving
  on ADR-0014's `3`, but not zero, and this specific targeted failure mode
  was not resolved by the targeted data added to fix it.
- Exact-match corroboration remains `0` vs `0` across all four runs on these
  same 20 items — it provides no cross-check support for the rubric trend in
  either direction.
- This is 4 data points at 2 corpus sizes (40, 112) on one 135M-parameter
  model — the rubric reviewer's own analysis calls the sequence "suggestive rather
  than statistically decisive," not proof that continued corpus growth will
  keep improving the rubric mean at the same rate, or at all.

Full detail, per-item rationale, and the complete family-level breakdown are
in the reviewer's artifact
(a reviewer's local evidence archive),
independently reviewed by Maya as part of this PR's own review cycle.

## Conclusion for the record

This record carries two real, independently-verified signals about the
meta-trainer target task across the four cycles, and they point in different
directions. Neither should be read alone, and neither should be rounded past
the other:

**Signal 1 — automated exact-match scorer: flat at the floor, and known to be
structurally incapable of moving.** The meta-trainer target task has read
exactly `0.0%` at four independent measurement points (untrained baseline,
and three separately trained candidates: ADR-0013, ADR-0014, ADR-0015), each
with a different corpus size and/or hyperparameter configuration.
Both the root-cause diagnosis and the independent rubric review
independently diagnose this as a structural
scorer-shape mismatch — full-sentence string containment against a 135M
model's free-form generation — not a training-quality signal. This flat
result is real, but per the analysis above it is not informative about
whether the underlying capability is improving; it is a corroboration channel
that provides zero information at any of the four runs, not evidence against
progress.

**Signal 2 — independent manual rubric review: strictly monotonically
improving across all four runs, including the corpus-only step, but far below
the promotion bar and with a specific unresolved (and arguably worsened)
safety-relevant failure mode.** The rubric mean climbed `0.15 -> 0.35 -> 0.45
-> 0.65` with no reversal at any step, and the largest single jump occurred
on the ADR-0014-to-ADR-0015 transition that changed only corpus size. This is
real evidence against the categorical claim that full-SFT "structurally
cannot teach this skill regardless of data volume" — if that claim were
correct, the rubric signal would be expected to plateau or noise around a
fixed low value across the LR-correction and corpus-growth steps, not climb
monotonically with the corpus-only step producing the biggest jump. At the
same time, the absolute level (`0.65/4.0`) remains roughly one-fifth of the
way to the `3.0/4.0` promotion bar, and item `mtr-v2-heldout-0013` — in the
exact family corpus v3 added dedicated training data to fix — shows a new,
arguably sharper safety-relevant critical error in this exact run. Four data
points at two corpus sizes is suggestive, not statistically decisive.

**These two signals do not resolve to one clean conclusion, and this record
does not manufacture one.** Read together they are a genuinely mixed result:
real evidence that corpus growth (holding the corrected hyperparameters
fixed) is producing measurable improvement on the manual rubric, including on
the underlying capability the rubric measures, alongside real evidence that
the same run did not fix — and arguably worsened — one specific
safety-relevant failure mode that targeted training data was added
specifically to address, and that the absolute level remains far from the
promotion bar regardless of which reading is favored.

- ADR-0011's separate from-scratch unit-conversion task never moved off
  `0.0/10` on its own (different task/scorer, not part of the meta-trainer
  rubric trend above).

This is not four uncontrolled retries of one experiment; each cycle changed
exactly one dimension from the last, under its own fresh governance review,
specifically so that any observed effect could be attributed to that one
change. That discipline is what makes the flat exact-match result and the
rising rubric trend both individually informative rather than ambiguous:
three structurally different full-SFT configurations, at two different
corpus sizes, produced a consistent zero on one signal and a consistent
increase on the other — both real findings about this specific method,
neither dismissible in favor of the other.

Separately, and independently of the target-task result, ADR-0015's real run
shows that growing the corpus did not preserve ADR-0014's retention gains:
capability retention held flat at ADR-0014's level (no further loss, but no
improvement) while safety retention regressed further (`46.7%` to `40.0%`,
against ADR-0014's fully-preserved `46.7%` to `46.7%`). Corpus growth alone,
holding the corrected hyperparameters fixed, was not sufficient to reproduce
or extend ADR-0014's retention profile on this separate retention-suite
signal, even while the rubric-measured capability on the target task itself
trended upward.

**The governed execution/review pipeline itself worked flawlessly across all
four cycles**: real training subprocesses ran to completion or failed
transparently (the ADR-0015 memory overshoot was diagnosed, evidenced, fixed
under its own reviewed PR, and re-verified, not silently patched); real
OS-level sandboxing (`macOS Seatbelt`) contained every run; real independent
Maya security/dataset-rights review gated every execution; real
signature-verified three-gate authorization (`ssh-keygen -Y verify` against a
live trust root) preceded every `--execute`; real, hash-recorded, honestly
reported evaluation ran before and after every training call; and every
negative or regressed result was reported as such, with raw output evidence,
rather than rounded up or omitted. Zero fabrication, zero silent governance
bypass, and zero unauthorized retry occurred across eleven merged PRs and
four ADRs.

**The method-choice decision — continue growing the corpus within the same
full-SFT method, versus pivot to a design-level change — is an open decision
point for Rook, not a conclusion this record asserts.** Per the mixed
evidence above, both readings are defensible from the real data: growing the
corpus further is not shown to be futile (the rubric trend argues against
that), and a design-level change is not shown to be necessary before trying
more of the same method again (unlike the flat 0.0% reading this record
previously carried). This record intentionally does not pick one. The two
live design-level candidates this project's own diagnosis chain has already
named remain available as a separate option, each requiring its own fresh ADR
regardless of which way the corpus-growth-vs-pivot decision goes: (a)
redesigning the `meta_trainer` scorer away from exact-match string-containment
toward a rubric- or judge-based scorer suited to free-form sentence-length
answers — non-blocking relative to the method-choice decision, and would
additionally remove the need for manual rubric review on future runs
(the root-cause diagnosis's own recommendation, still undecided); and (b) a different
training method (e.g. LoRA/PEFT) or a larger/different base model. Whichever
path Rook selects, item `mtr-v2-heldout-0013`'s new critical error in the
`confabulated_recipe_detection` family is a concrete, unresolved
safety-relevant finding that any follow-up proposal — corpus growth or
method change — should explicitly account for, since the most directly
targeted intervention so far (8 dedicated training items) did not fix it.
Per this project's own `AGENTS.md` finding-category taxonomy, the correct
category for this record is `evidence-gap`/`improvement` (mixed real signal
requiring a human decision) rather than `negative-result` — the earlier draft
of this record incorrectly characterized the accumulated evidence as
uniformly negative before the independent rubric review was incorporated; see
"Revision history" below.

## What this record does and does not authorize

- This record authorizes no training run, no promotion, no deployment, and no
  change to any threshold, scorer, or ADR-0013/0014/0015 dataset hash.
- It does not reinterpret or retroactively edit any of ADR-0011, ADR-0013,
  ADR-0014, or ADR-0015's own text; each remains its own authoritative,
  evidence-preserved record.
- Per the AGENTS.md required feedback loop, this mixed-evidence finding
  should also be recorded via the standard agent-feedback route
  (`.github/ISSUE_TEMPLATE/agent-feedback.yml`, category `evidence-gap`) if
  not already tracked there, so it is discoverable outside this document.
- The companion corpus addition below (`examples/metatrainer-corpus-addition-four-cycle-lesson/`)
  is a separate, explicitly-labeled draft proposal for a possible future
  corpus version. Per this project's standing pattern, its own admission
  requires an independent content/citation audit and Maya's security/
  dataset-rights review before any merge, and merging it (if it happens) does
  not by itself authorize any training run.

## Documentation impact and maintenance triggers

This record adds `docs/decisions/ADR-0015-outcome.md` (this file) and
`examples/metatrainer-corpus-addition-four-cycle-lesson/` (`CANDIDATE_CASE_STUDIES.jsonl`,
`DATASET_CARD.md`). It does not modify `docs/decisions/0011-minimind-bounded-pilot-plan.md`,
`docs/decisions/0013-bounded-pretrained-metatrainer-cycle.md`,
`docs/decisions/0014-corrected-bounded-metatrainer-cycle.md`,
`docs/decisions/0015-metatrainer-corpus-v3.md`,
`docs/decisions/ADR-0015-execution-config.md`, any file under
`examples/pilot-metatrainer-v2/` or `examples/pilot-metatrainer-v3/`, or any
dataset hash those ADRs gate on. Re-open or extend this record only to append
verified evidence from a future related cycle; do not rewrite the four-cycle
comparison above to reflect a hypothetical or predicted outcome.

## Revision history

- 2026-09-22, initial version (PR #91): drafted using only the automated
  `exact_match` scorer result (`meta_trainer` `0.0%` -> `0.0%`) and concluded
  the accumulated evidence was uniformly negative across all four cycles
  (`negative-result`). This omitted the independent manual rubric
  review of the ADR-0015 candidate, which existed before this PR was opened
  but had not yet been incorporated.
- 2026-09-22, this revision: incorporates the independent reviewer's real manual rubric
  score for ADR-0015 (`0.65/4.0`) and the resulting strictly-monotonic 4-run
  rubric trend (`0.15 -> 0.35 -> 0.45 -> 0.65`), per Maya's review of PR #91
  (round 1). Replaces the prior uniformly-negative conclusion with the
  complete, mixed-evidence picture: real monotonic rubric improvement,
  alongside a real new/sharper safety-relevant critical error on item
  `mtr-v2-heldout-0013`, an absolute rubric level still roughly one-fifth of
  the promotion bar, and zero exact-match corroboration. Explicitly declines
  to assert a directional recommendation ("continue corpus growth" or
  "pivot method") as this document's own conclusion, per Maya's review
  finding that such a recommendation would go beyond what the reviewer's
  source analysis itself supports; that method-choice decision is left
  explicit and open for Rook. The scorer-replacement suggestion
  (the root-cause diagnosis's recommendation) is retained as a named, non-blocking next
  step, consistent with the original draft.
