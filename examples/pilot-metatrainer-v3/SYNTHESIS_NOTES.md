# Synthesis notes — meta-trainer corpus v3 (`research synthesis, MSx`)

This is the v3 analogue of corpus v2's `SSX.X` synthesis-locator registry from
`examples/pilot-metatrainer-v2/SOURCE_MAP.md`. Each entry below names the
exact record(s) in `train.jsonl` / `held_out.json` that cite it, the specific
numbered source(s) in `SOURCE_MAP.md` (or repository evidence) the synthesis
draws on, and the added inferential step that no single cited source states
on its own. Per this corpus's citation policy, a `research synthesis, MSx` label
is only used where the underlying claim genuinely requires combining sources
or drawing a conclusion beyond what any one source states outright; where a
single source directly supports the claim, the record cites that source
number instead.

## MS1
Used by: `refusal_vs_overrefusal_balance` record on the practical difference
between a calibrated refusal and an over-refusal.
Draws on: [28] (Model Spec's uncertainty-expression guidance) and [30]
(the pilot Anthropic-OpenAI evaluation's refusal/utility tradeoff finding).
Added inferential step: neither source itself defines "over-refusal" as a
named failure mode or contrasts it against "calibrated refusal" as a
paired concept — [28] describes when hedging is appropriate and [30]
reports refusal-rate/utility tradeoffs empirically, but combining them to
state the practical, behavioral distinction between a scoped refusal and a
blanket-declining one is this corpus's own synthesis, not a claim either
source makes directly.

## MS2
Used by: `refusal_vs_overrefusal_balance` record on whether a
calibrated-refusal-emphasizing corpus should include non-refusal examples.
Draws on: [30] (utility cost of high refusal rate) and repository evidence
(this corpus's own family composition: `qualified_hyperparameter_sweeps`,
`sft_objective_masking_format`, and the rest of corpus v2/history-addition
content already trains direct, well-cited answers).
Added inferential step: [30] shows a *real model's* high refusal rate
correlates with lower utility; it says nothing about *corpus design* for
training a different model. The inference that a corpus should therefore
deliberately balance refusal examples against existing non-refusal
examples, and a description of which existing families provide that
balance, is this corpus's own design reasoning.

## MS3
Used by: `refusal_vs_overrefusal_balance` record on whether a well-cited
answer in SOURCE_MAP.md should be hedged just because the corpus now
emphasizes uncertainty.
Draws on: [28] (hedging should track actual evidentiary support) and this
corpus's own `uncertainty_language_calibration` family (specifically the
MS-free record on both under- and over-expressed confidence being
miscalibrated).
Added inferential step: [28] does not discuss corpus-emphasis effects at
all; connecting "a corpus emphasizes calibrated refusal" to "well-cited
claims should still not be hedged" as a non-consequence is an inference
this corpus draws to head off a specific plausible misapplication of its
own design intent, not a statement either source makes.

## MS4
Used by: `refusal_vs_overrefusal_balance` record on whether teaching
refusal risks reduced usefulness on well-supported questions.
Draws on: [30] (utility cost of unconditional high refusal) and
repository evidence (this corpus's own composition: calibrated-refusal
families are a minority addition, not the corpus's entirety — see
DATASET_CARD.md "Split construction" family counts).
Added inferential step: [30] is about one already-deployed model's
observed behavior; the claim that *this specific corpus's* proportion of
refusal-vs-non-refusal examples avoids the same risk is a design-level
inference about corpus balance, not a fact stated in the cited source.

## MS5
Used by: `refusal_vs_overrefusal_balance` record on what would count as
evidence of refusal-emphasis overshoot if this corpus were used in
training.
Draws on: repository evidence only (this project's own paired
before/after evaluation methodology, as used across ADR-0011/0013/0014's
capability-retention checks on arithmetic/safety suites).
Added inferential step: no external source addresses this project's
specific evaluation methodology; describing how that same existing
paired-comparison pattern would apply to a hypothetical future
refusal-overshoot check is a synthesis of this project's own established
practice applied to a new, not-yet-run scenario, explicitly flagged as not
itself run or predicted by this corpus.

## MS6
Used by: `bounded_reasoning_under_missing_citation` record on whether
picking one of two disagreeing sources without noting the disagreement is
calibrated.
Draws on: repository evidence (corpus v2's `qualified_hyperparameter_sweeps`
family, which models stating LR-range disagreement across TRL/Axolotl/
Unsloth explicitly rather than picking one).
Added inferential step: corpus v2's family demonstrates the pattern for a
specific hyperparameter-range case; generalizing that demonstrated pattern
into an explicit general rule ("state disagreement rather than silently
picking one source") is this corpus's own stated inference, not a rule any
one source articulates directly.

## MS7
Used by: `bounded_reasoning_under_missing_citation` record on describing a
topically-related-but-non-supporting claim.
Draws on: repository evidence (this project's own citation policy in
`pilot-metatrainer-v2/SOURCE_MAP.md` requiring a distinct synthesis label
for cross-source conclusions, and the independent audits — e.g. task
 — that specifically checked for and flagged citation-locator
support defects).
Added inferential step: the citation policy and audit practice exist to
prevent misattributing a claim to a non-supporting source in *this
corpus's own records*; extending that same discipline as the correct
behavior for a *model's own real-time answers* to a user is this corpus's
inferential step, not a rule the citation policy itself was written to
govern (it governs corpus authoring, not model output).

## MS8
Used by: `bounded_reasoning_under_missing_citation` record on whether a
bare refusal is sufficient or should be paired with something else.
Draws on: [28] (Model Spec's guidance that expressing uncertainty does not
mean avoiding an answer altogether, and its example phrasings naming what
the uncertainty is about) applied by inference beyond its literal scope.
Added inferential step: [28] addresses *how* to phrase uncertainty, not
whether a refusal should be paired with a path-forward or a bounded range;
the claim that a calibrated refusal is "more useful" when paired with
either of those is this corpus's own practical extension of the Model
Spec's general precision-over-blanket-hedging principle.

## MS9
Used by: `bounded_reasoning_under_missing_citation` record on whether a
scenario superficially similar to a real historical incident implies the
same outcome.
Draws on: repository evidence (ADR-0013's own exact predeclared trigger
text — "zero or less than 0.25/4.0 rubric improvement" — as a real example
of an outcome tied to exact conditions rather than loose similarity).
Added inferential step: ADR-0013's trigger text is a governance rule for
one specific decision; generalizing "outcomes are evaluated against exact
predeclared conditions, not loose scenario similarity" into a general
reasoning norm for answering hypothetical questions is this corpus's own
synthesis, illustrated by but not stated as a general principle in the
ADR itself.

## MS10
Used by: `bounded_reasoning_under_missing_citation` record on why this
family exists separately from `qualified_hyperparameter_sweeps` or
`contamination_split_controls`.
Draws on: repository evidence only (this corpus's own family design, per
DATASET_CARD.md).
Added inferential step: this is a corpus-design rationale, not a claim
grounded in any external source; recorded here rather than cited to an
external source because it explains an authorial choice, not a technical
fact.

## MS11
Used by: `calibrated_refusal_generalization` (held-out) record on
extrapolating a wall-clock duration for a hypothetical future training run.
Draws on: repository evidence (ADR-0013/0014's real resource-usage
evidence and recorded run durations) and [28] (Model Spec's precision on
hedging framed to actual evidentiary support).
Added inferential step: neither the ADR evidence nor the Model Spec states
a rule for "when is an extrapolated estimate acceptable"; combining the
existence of real anchor figures with the Model Spec's evidentiary-support
principle to conclude that a labeled, source-anchored estimate is
acceptable while an unlabeled fabricated-sounding figure is not, is this
corpus's own synthesis.

## MS12
Used by: `calibrated_refusal_generalization` (held-out) record on whether
ADR-0013's exact 0.0%-to-0.0% outcome would recur under a changed key
variable.
Draws on: repository evidence (ADR-0013's exact recorded configuration and
outcome) and this corpus's own `bounded_reasoning_under_missing_citation`
family's MS9 reasoning about exact-condition-scoped outcomes.
Added inferential step: applying the "outcomes are scoped to exact
conditions" principle (already synthesized as MS9) to a different concrete
scenario (a changed corpus size rather than a governance trigger) is a
further applied inference, not a restatement of a single source.

## MS13
Used by: `calibrated_refusal_generalization` (held-out) record on
verifying whether a cited arXiv paper's result still replicates.
Draws on: no external source directly addresses this; drawn from the
general evidentiary-boundary principle common to [24] (a model's P(IK)
should reflect what it actually has access to) and [28] (hedging should
track actual evidentiary support).
Added inferential step: neither source discusses replication-verification
specifically; applying the general "state only what your actual access
supports" principle to the specific scenario of an inability to
independently verify a claim's current replication status is this
corpus's own applied inference.

## MS14
Used by: `hallucination_incentive_diagnosis` (held-out) record on why this
family tests the underlying incentive mechanism rather than only a surface
phrase.
Draws on: repository evidence only (this corpus's own family design intent,
cross-referenced against [24]'s own stated generalization limitation, used
by the immediately preceding record in the same family).
Added inferential step: this is a corpus-design rationale explaining a
held-out/train split choice, not an external technical claim; recorded as
synthesis because it draws the connection between "testing surface-phrase
recall is weaker" and "the previous record's named generalization limit"
rather than restating either on its own.

## MS15
Used by: `sampling_consistency_application` (held-out) record on why this
family is held out rather than trained on, given
`self_consistency_and_sampling_checks` already covers SelfCheckGPT's core
mechanics in training.
Draws on: repository evidence only (this corpus's own train/held-out
design pattern, and corpus v2's precedent of teaching a concept in train
and testing genuinely novel application in held-out).
Added inferential step: this is a corpus-design rationale, not an external
technical claim; it states why the split boundary was drawn where it was,
by analogy to corpus v2's own established pattern rather than a claim
either the SelfCheckGPT paper or corpus v2 states about this project's own
family design.
