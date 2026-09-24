# ADR-0021: Meta-trainer mission, capability map, and evaluation strategy

- Status: proposed — decision and design record only; authorises no run,
  no training, no promotion, no deployment, and no new gate signature
- Date: 2026-09-24
- Tracking: this record supersedes part of ADR-0013's Context section (see
  "Supersession note" below); it does not reopen ADR-0013's own text, its
  gates, or its already-recorded outcome, and it authorises no execution
  under ADR-0013, ADR-0014, or any later cycle. No GitHub issue is opened by
  this record; it is a standalone decision document.
- Scope: docs only. Nothing in this record starts, schedules, or pre-clears
  a training run. Every capability named below still goes through this
  project's existing ADR, signed-gate, independent-review, and fail-closed
  containment process exactly as ADR-0013 through ADR-0020 already
  required.

## 1. Mission statement

The meta-trainer's job is to do the heavy lifting of **continual model
training**, inside this project's existing governed pipeline, for the
copy of the model that keeps training. Concretely, it:

- knows which framework adapter, training method, and evaluation to use for
  a given step, and can say why;
- produces correct framework configuration and commands, and carries out
  the governed pipeline steps those configs describe;
- diagnoses a run's outcome once it finishes, correctly, using the same
  evidence a human reviewer would use;
- knows the boundary of its own knowledge: when it does not know, it
  defers to a frontier model or goes to fetch the specific data it needs,
  rather than inventing an answer; and
- feeds what it learns from doing this work — diagnoses, gaps, retrieved
  data — back to the copy of the model that continues training, as
  structured, provenance-tagged material for human/gate review, not as
  data it admits itself.

It does **not** replace internet-based frontier models. Those remain the
source of up-to-date information and general-purpose context; the
meta-trainer's job is the bounded, repetitive operational work around
training this project's own models, plus knowing when a question is
outside that bounded scope and needs a frontier model or a fetch instead
of a guess.

### Supersession note on ADR-0013's Context paragraph

ADR-0013's Context section described this project's goal as: "a small
internal reasoning assistant for work around training other models:
corpus design, bounded hyperparameter recommendations, run diagnosis,
evaluation interpretation, evidence qualification, uncertainty, and
refusal. It is not a numerical training engine and it receives no
authority to schedule, promote, publish, or deploy models."

That framing is now superseded as a statement of mission scope by the
mission statement above. Two things carry forward from it unchanged and
are restated, not weakened, in section 3 below: the meta-trainer still
receives **no authority** to schedule, promote, publish, or deploy models,
and every action it takes remains a proposal or a bounded execution of an
already-admitted step, never a self-authorised one. What changes is the
"not a numerical training engine" framing: the mission now explicitly
includes carrying out the governed pipeline's procedural steps (C2 below)
and going to get data when the model's own knowledge is insufficient (C4,
C5 below) — work ADR-0013's narrower framing did not name.

This note supersedes only that one Context paragraph's framing of mission
scope. **ADR-0013's own record — its decision, gates, thresholds,
resource budget, and its recorded rejected outcome — is not edited,
reopened, or reinterpreted by this record.** ADR-0013 remains the
authoritative record of its own run.

## 2. Capability map

Each capability below is stated with a measurable definition: what
"working" looks like in a way that can be scored mechanically, not
narratively. Section 4 gives the evaluation strategy these definitions are
designed for.

### C1 — Method and tool selection

**Definition:** given a stated training or evaluation goal and the current
project state (existing adapters, existing corpus, existing evidence),
correctly select which framework adapter, method, or evaluation applies,
from the actually-available set — not a hypothetical superset.

**Measurable form:** a structured choice (adapter id, method id, or
evaluation id) checked against a labelled gold choice for a held-out set
of scenarios. Correctness is exact-match against the gold selection, not a
free-text judgement of "reasonableness."

**What the evidence says is achievable (hypothesis, not established for
this project):** small models (roughly 1B–8B) can reach competent
tool/method-selection accuracy when trained on execution-verified data —
Salesforce's APIGen pipeline, which verifies each function-calling example
through format checking, real execution, and semantic verification before
admitting it to training data, produced a 1B model ("xLAM-1b-fc-r") that
surpasses GPT-3.5-Turbo and Claude-3-Haiku on the Berkeley Function-Calling
Leaderboard (BFCL) [Salesforce AI Research, "APIGen," arXiv:2406.18518;
"APIGen-MT," arXiv:2504.03601]. This is a hypothesis about transfer to
this project's specific adapter set, not a claim this project has verified
directly — no source found evaluates a model on selecting among *this
project's own* adapters.

### C2 — Procedural execution

**Definition:** given an admitted method/tool choice, produce a valid
framework configuration and command, and carry out the already-admitted
governed pipeline steps (baseline evaluation, training call, post-training
evaluation) exactly as configured.

**Measurable form:** does the emitted configuration parse and validate
against the framework's own contract (`TrainingInputs.validate()` or
equivalent)? Does the emitted command, when run, produce the expected
evidence artifacts with the expected hashes? This is mechanically
checkable pass/fail, not a judgement call.

**What the evidence says:** the field's evidence on procedural competence
at small scale is thin and mostly indirect. The most relevant available
signal is that on SWE-bench (real GitHub issues, success defined as making
the project's own hidden test suite pass — fully mechanical, no judge
model), the overwhelming majority of the field's progress since 2023 is
attributed by independent survey analysis to agent-scaffolding and
tooling-interface design, not underlying model scale [Jimenez et al.,
SWE-bench, ICLR 2024, characterized via secondary analysis]. That is a
hypothesis worth taking seriously for C2's design: a well-built config/
command interface likely matters more than base-model size for this
capability, but this project has not measured it directly.

### C3 — Run diagnosis and interpretation

**Definition:** given a completed run's raw evidence (metrics, evaluator
output, evidence-file contents), correctly classify the outcome against
the run's own preregistered decision rule, and correctly identify the
structural cause when a result is ambiguous (e.g. a floor effect, not a
true null result).

**Measurable form:** classification accuracy against a labelled set of
past runs' actual outcomes (this project already has ADR-0013 through
ADR-0020's real outcome records as a natural evaluation set — see section
6, roadmap item 1). A diagnosis is scored correct only if it reaches the
same classification a human reviewer reached from the same raw evidence,
not merely a plausible-sounding narrative.

**What the evidence says:** this project's own ADR-0020-outcome.md is the
most directly relevant evidence available, and it is evidence about
*evaluation design*, not about C3 in isolation: it shows that even a
correctly-applied, preregistered decision rule can produce a technically
correct but substantively misleading label ("no shift of meaningful
magnitude") when the underlying instrument has a floor effect (16 of 20
held-out pairs scored a calibration-correct rate of exactly zero for both
models under test) [`docs/decisions/ADR-0020-outcome.md`]. A meta-trainer
performing C3 needs to be able to surface that kind of structural caveat,
not just apply the label mechanically — which is why C3's evaluation
(section 4) must check for the caveat, not just the label.

### C4 — Knowing the limits of its own knowledge

**Definition:** given a query or a step in the pipeline where the
meta-trainer's own trained knowledge is insufficient, correctly decide to
escalate to a frontier model or retrieve external data, rather than
fabricating an answer — and correctly *not* defer on ordinary in-scope
questions it does know.

**Measurable form:** classification against a labelled should-defer /
should-not-defer held-out set, scored as participation rate plus
conditional correctness when it does answer (not a free-text judgement of
whether an answer "sounds confident").

This is where the ADR-0016 through ADR-0020 refusal-calibration line
belongs, reframed: those ADRs trained and measured whether the model
correctly refuses to invent a plausible-sounding fabricated detail
(`mtr-v2-heldout-0013`, a fixed numeric-recipe confabulation item tracked
across ADR-0015 through ADR-0020). That work is the direct ancestor of
C4 — "refuse to fabricate" generalises here to "decide to defer or fetch"
across the whole space of things the meta-trainer might not know, not just
one held-out item's numeric-recipe question.

**What the evidence says:** "decide to defer" is a trained, evaluable
capability with several distinct mechanisms, not a single solved
technique:

- Inference-time, no training required: a confidence threshold on the
  model's own token probabilities can trigger retrieval — FLARE
  generates a look-ahead sentence and retrieves when any token falls
  below a confidence threshold, with retrieval triggered on 40-80% of
  sentences found optimal depending on task [Jiang, Xu, Gao, Sun et al.,
  "Active Retrieval Augmented Generation" (FLARE), arXiv:2305.06983,
  EMNLP 2023].
- Trained directly into the model's output: Self-RAG fine-tunes a single
  LM to emit "reflection tokens" — a Retrieve token and separate
  relevance/support/utility critique tokens — letting the model decide
  on-demand whether to retrieve and self-filter what comes back; Self-RAG
  7B beat ChatGPT and a retrieval-augmented 13B baseline on several
  QA/reasoning/fact-verification tasks [Asai, Wu, Wang, Sil, Hajishirzi,
  "Self-RAG," arXiv:2310.11511].
- A cheap external classifier gating escalation to a larger model:
  FrugalGPT's LLM-cascade approach matched the best individual LLM's
  accuracy at up to 98% cost reduction, or improved accuracy up to 4% at
  equal cost, across three real datasets [Chen, Zaharia, Zou, "FrugalGPT,"
  arXiv:2305.05176, Stanford].
- Preference-based routing: RouteLLM trains a binary router on human
  preference data to predict whether a strong or weak model should handle
  a query, reporting over 2x cost reduction without compromising quality
  and generalising to different strong/weak model pairs without
  retraining [Ong, Almahairi, Wu, Chiang, Wu, Gonzalez, Kadous, Stoica,
  "RouteLLM," arXiv:2406.18665, ICLR 2025].

The FrugalGPT/RouteLLM pattern (a cheap decision layer gating escalation to
a frontier model) is the most directly reusable evidence for this
project's own "escalate to a frontier model" half of C4; FLARE and Self-RAG
are the most directly reusable evidence for the "fetch data" half.

**A load-bearing caveat, evidenced, not hypothetical:** calibration is
known not to transfer reliably under distribution shift — a model that
appears well-calibrated in-distribution can fail to move its own
confidence even when the evidence quality genuinely degrades. Any
abstention/defer gate this project trains and validates against one
distribution (the current framework version, the current tool schema)
should not be assumed to generalise its defer behaviour to a materially
different distribution without re-validation. This is a hypothesis carried
from adjacent-domain evidence (a VLM confidence-under-degradation study),
not a claim independently confirmed in this project's own framework —
flagged as such deliberately, per this project's evidence discipline.

### C5 — Data feedback

**Definition:** when the meta-trainer identifies a knowledge or data gap
during C4, it emits a structured, provenance-tagged data request or
candidate training example for the next training copy — never data it
admits to a training split itself.

**Measurable form:** does the emitted request/example validate against a
fixed schema (source, retrieval method, retrieval date, license claim,
intended use)? A request that lacks a required provenance field, or that
attempts to bypass the schema, fails mechanically, independent of the
content's plausibility.

**What the evidence says, and why this capability is the most
security-sensitive one in this map (see section 3):**

- Self-improvement/data-flywheel loops (the STaR/ReST family: generate,
  verify against ground truth, fine-tune only on verified-correct
  trajectories) are evidence-based *only* when paired with an independent,
  automatic correctness check on each self-generated example — not when
  "correct" is the model's own unverified judgement of its own operational
  experience. This is the single most load-bearing caveat for C5's design:
  the meta-trainer's own sense that an example is good is not itself a
  verifier.
- Data poisoning and indirect prompt injection via retrieved content are
  demonstrated, current threats, not hypothetical ones. A 2026 paper shows
  a black-box attack that guarantees near-100% retrieval of a poisoned
  document across 11 benchmarks and 8 embedding models, at roughly $0.21
  per target query using only API access to an embedding model, and
  demonstrates an end-to-end exploit where a single poisoned email coerced
  GPT-4o into exfiltrating SSH keys with over 80% success in a multi-agent
  workflow [Chang, Bao, Luo, Yu, "Overcoming the Retrieval Barrier,"
  arXiv:2601.07072]. The paper's own defense evaluation found the tested
  defenses insufficient to prevent retrieval of the malicious text.
- License and provenance metadata found "in the wild" cannot be trusted at
  face value: a large-scale audit of 1,800+ text datasets found license
  omission above 70% and, where a license was specified, a substantial
  mismatch rate against the original author's intended license category
  [Longpre, Mahari, Chen et al., "The Data Provenance Initiative,"
  arXiv:2310.16787]. Any provenance record C5 emits must require
  independent confirmation against the origin, not just carry whatever
  license label the source page claims.

### C6 — Retention across continual rounds (forgetting control)

**Definition:** after any admitted training round, the model's performance
on all prior rounds' task-outcome evaluations must not regress beyond a
predeclared threshold. This capability is measured as a delta, not an
absolute score: re-run round N's evaluation suite after round N+1's
training and report the change.

**Measurable form:** per-suite accuracy delta (round N+1 candidate vs.
round N's own accepted result), against a fixed non-degradation threshold
— directly reusing the pattern this project's own cumulative-training
engine work already validated internally (see section 5) and the same
shape ADR-0013 through ADR-0020's own retention-suite gates already use.

**What the evidence says:** LoRA/PEFT materially reduces catastrophic
forgetting relative to full fine-tuning — a controlled study on BERT-base
across a four-task sequence found full fine-tuning caused 19.9%±4.8%
average forgetting versus 0.6%±1.4% for standard LoRA (r=8,
query/value modules), a statistically significant 97% relative reduction
(paired t-test p=0.002) [Pandey, "Low-Rank Adaptation Reduces Catastrophic
Forgetting," arXiv:2603.27707]. This is evidence for the direction, not a
guarantee at this project's own model scale or task type — the cited study
is encoder classification tasks, not the autoregressive, tool-use,
continual-round setting this project actually runs. No LoRA-continual-
learning method eliminates forgetting entirely; the most robust fallback
across the literature reviewed remains experience replay (mixing old-round
data back in), which is exactly what an internal evidence source already
in use for this project (see section 5) had already set at 10-15% of
each round's training batches — a figure the research pass commissioned
for this ADR did not find any source contradicting; every source bracketed
a 1-20% adequate range for that use.

## 3. Authority model

This is the most important section in this record. It states, without
qualification, the boundary the meta-trainer operates inside.

**The meta-trainer acts only through this project's existing governed
gates.** Every mechanism ADR-0013 already established — the ADR process
itself, signed review gates (independent role-specific SSH signatures),
independent review before execution, and fail-closed sandboxed containment
— remains exactly as binding on any meta-trainer-proposed or
meta-trainer-executed step as it is on a human-proposed one. Nothing in
this mission statement or capability map creates a new path around any of
those gates, and nothing in it authorises loosening one.

**The meta-trainer proposes and executes bounded, admitted steps. It never
self-authorises.** Concretely, and without exception:

- It never schedules its own next training round.
- It never promotes a candidate.
- It never signs, or causes to be signed, a review gate — including its
  own security or dataset-rights sign-off.
- It never admits fetched data into a training split. C5's data-feedback
  output is a *request*, always reviewed by an independent gate before any
  part of it can reach a training split — the same discipline ADR-0013's
  dataset-rights admission gate already established for human-sourced
  data, applied without exception to meta-trainer-sourced data too.
- It never expands its own authority, credentials, or network egress.

These are not new restrictions invented for this ADR; they are the direct,
unweakened continuation of ADR-0013's own authority language ("This ADR
grants no publication, deployment, promotion, repeated scheduling,
continuous unattended training, new credentials, new egress, or new
authority") and ADR-0013's Continuous-training semantics section ("No
cycle schedules the next one, changes its own thresholds, selects its own
training data, grants its own review, or promotes its own artifact").
Nothing in this record's expanded mission statement changes that; it
expands *what kind of bounded step* the meta-trainer may be asked to
execute, not *who decides* whether a step happens.

### Fetched data is untrusted input by default

Every piece of data C4/C5 causes to be fetched — a web page, a
documentation source, an issue tracker entry — is treated as adversarial
input from the moment it is retrieved until it clears an independent
review, never as instructions and never as directly training-admissible.
This follows directly from the evidence in section 2's C5 discussion:

- **Data poisoning of training material is a demonstrated threat, not a
  hypothetical one**, and no data-level filtering defense evaluated in the
  literature reviewed for this ADR is sufficient alone. A poisoning
  technique that survives every tested data-level defense, including
  independent paraphrasing, has been demonstrated to transfer a covert
  behavior through an ordinary-looking instruction-tuning dataset across
  different teacher/student model families [characterized in the evidence
  brief commissioned for this ADR; the ADR's own primary-source citations
  above (arXiv:2601.07072, arXiv:2310.16787) establish the same
  insufficiency-of-filtering-alone conclusion independently]. The working
  recommendation from this literature is layered: provenance review plus a
  post-training behavioral audit of any model version trained on newly
  admitted data, not training-time filtering by itself.
- **Prompt injection via retrieved content is a demonstrated, not
  hypothetical, threat specifically relevant to a model that fetches data
  autonomously.** The near-100%-retrieval, $0.21-per-query attack cited in
  section 2's C5 discussion [arXiv:2601.07072] demonstrates that the
  retrieval step itself, not just the payload content, is the attacker's
  point of leverage — meaning C4/C5's fetch mechanism needs review as a
  security surface in its own right, not just the data it returns.

Every fetched item that C5 proposes for possible training use must carry a
provenance record — source, retrieval date, retrieval method, and a
license claim requiring independent confirmation against the origin, not
just the source page's own label — before an independent reviewer can
consider it for admission. This mirrors, and does not relax, the dataset-
rights admission gate ADR-0013 already established for human-sourced
corpora.

## 4. Evaluation strategy: from text-pattern classifiers to task-outcome evals

ADR-0020's outcome record is the direct evidence for why this project
moves away from text-pattern classification as its primary scoring
mechanism. That record's deterministic classifier — designed to label
generated text REFUSE / ANSWER / FABRICATE / AMBIGUOUS — returned AMBIGUOUS
on roughly 74% of all 420 samples generated, for both the candidate and
reference model alike, and 16 of 20 primary held-out pairs showed a
calibration-correct rate of exactly zero for *both* models
[`docs/decisions/ADR-0020-outcome.md`]. The record's own honest conclusion
was that the preregistered outcome fired *mechanically, on a floor
effect*: the classifier had too little coverage of what these two models
actually generated to resolve the underlying question either way, not that
no real effect existed.

The field's established alternative is scoring based on **task outcomes**
— whether an executable or structurally checkable artifact is correct —
not a judgement of free-form text:

- **BFCL and tau-bench** score AST-match against a gold function call and
  actual database end-state comparison against an annotated goal,
  respectively — neither relies on a judge classifying free text
  [gorilla.cs.berkeley.edu for BFCL; Yao, Shinn, Razavi, Narasimhan,
  "tau-bench," arXiv:2406.12045].
- **SWE-bench** scores whether a generated patch makes a real project's
  own hidden test suite pass — fully mechanical, zero judge-model
  involvement [Jimenez et al., ICLR 2024].
- **MLAgentBench** scores whether an agent improved a stated performance
  metric over a starter baseline by a predeclared margin, in a real
  workspace with file read/write and code-execution actions — the closest
  existing analogue to this project's own mission (an agent that must
  select a method, execute a training/eval pipeline, and produce a
  measurable improvement). Its best-reported agent (Claude v3 Opus +
  ReAct-style scaffold) reached 37.5% average success across 13 tasks,
  ranging from 100% on well-established datasets to 0% on genuinely novel
  ones [Huang, Vora, Liang, Leskovec, "MLAgentBench," arXiv:2310.03302].
- **MLE-bench** scores agents against real Kaggle competition medal
  thresholds using actual human-competitor leaderboards. Its
  best-performing configuration (o1-preview + AIDE scaffolding) reached a
  medal in 16.9% of 75 competitions at a single attempt, rising to 34.1%
  across 8 attempts [Chowdhury et al. (OpenAI), "MLE-bench,"
  arXiv:2410.07095].

These ceilings matter for calibrating expectations, not just for citing
method: even frontier-scale agents with elaborate scaffolding solve a
minority of realistic ML-engineering tasks. No source found in the
research pass commissioned for this ADR evaluates a small (under 8B
parameter) model directly on MLE-bench- or MLAgentBench-style full
ML-engineering tasks — this is a genuine evidence gap, not a settled
expectation, and is named as a hypothesis in the roadmap below (item 2)
rather than assumed.

**Applying this to this project's own capability map, C1/C2/C4/C5's
evaluations should be scored the same way**, per capability:

- C1: exact-match of the selected adapter/method/eval id against a gold
  label.
- C2: does the emitted config parse and validate; does the emitted command
  produce the expected evidence artifact with the expected hash.
- C3: does the diagnosis classification match the human-reviewer
  classification on a held-out set of this project's own past runs (see
  roadmap item 1) — and does it surface a known structural caveat (like
  ADR-0020's floor effect) when one is present, not just apply the label.
- C4: participation rate plus conditional correctness against a labelled
  should-defer/should-not-defer set (per section 2).
- C5: schema-conformance of the emitted data request/example against the
  fixed provenance schema.
- C6: task-outcome accuracy delta across rounds, per section 2.

None of these six scoring mechanisms is a free-text classifier judging
whether output "sounds" correct. This is the structural fix for the
74%-ambiguous floor ADR-0020 recorded, not a claim that the underlying
model or task got easier.

## 5. How ADR-0015 through ADR-0020 map onto this mission

**What carries forward unchanged:**

- **Governance and containment.** Every gate ADR-0013 established —
  signed review gates, independent review, fail-closed macOS Seatbelt
  containment, the exact-SHA re-execution discipline — applies to every
  capability in this map exactly as it applied to ADR-0013's single SFT
  cycle. Nothing about the expanded mission loosens this.
- **Evidence discipline.** ADR-0015 through ADR-0020's practice of
  reporting real, executed results honestly — including negative and
  ambiguous ones, and including same-cycle self-corrections (e.g.
  ADR-0015's rubric-review gap, caught and fixed within the same record;
  ADR-0020's floor-effect disclosure alongside its preregistered label) —
  is the standard this record expects every future capability evaluation
  to meet. A capability is never claimed "working" from a narrative
  argument; it needs a held-out evaluation passing a predeclared
  threshold, exactly as this project's existing packages already require.
- **The refusal-calibration line (ADR-0016 through ADR-0020).** This work
  — the SFT corpus rewrite that failed twice (ADR-0015, ADR-0016), the
  SME decision to switch to DPO (ADR-0017), the DPO run that produced
  near-zero decode-visible change despite a real sub-threshold
  log-probability shift (ADR-0018, ADR-0019), and the decode-sensitive
  sampling evaluation that hit the floor effect (ADR-0020) — is not
  discarded. It is C4's direct ancestor (section 2), and its outcome
  records are exactly the kind of labelled evidence C3's evaluation should
  be built from (roadmap item 1).
- **The internal continual-learning rules already in force for this
  project's related model-training programme**: a 10-15% replay buffer of
  real examples from every prior round, a hard ceiling on teacher- or
  self-generated synthetic data per training batch, never training a batch
  at 0% real data, and an explicit prohibition on pure self-training/
  self-critique loops. These rules are reused here, not reinvented — see
  the contested-ratio note below for one place they need a documented
  refinement, and the cumulative-training-engine control pattern
  (lineage tagging, dual non-degradation gate against both the immediate
  parent and the original base, rollback, and a kill condition on
  repeated non-improving rounds) already prototyped and independently
  design-reviewed for a related internal training programme, which C6's
  retention gate should follow rather than reinvent.

**What is genuinely new here, not previously in this repository's ADR
lineage:** C1, C2, C4's fetch/escalate half, and C5 did not exist as named
capabilities before this record — ADR-0013's original framing explicitly
excluded "a numerical training engine" and named no data-feedback or
tool-selection capability at all. Section 4's task-outcome evaluation
strategy is also new to this repository's own evaluation practice, though
it is the field's established direction, not a novel invention.

**What gets deprioritised:** continued refinement of the free-text
rule-based classifier approach ADR-0020's outcome record showed hits a
structural floor. This does not mean the underlying refusal-calibration
question (C4) is deprioritised — only the specific scoring mechanism that
was shown not to resolve it.

**One place the evidence is genuinely contested, flagged rather than
resolved by this record:** the internal guidance capping teacher- or
self-generated synthetic data at 30% of a training batch has real
empirical support specifically for *rephrased* synthetic data — a
large-scale study (>1,000 LLMs, >100k GPU hours) found roughly 30%
synthetic converges as a "good ratio" with no collapse for rephrased text,
though the same study found textbook-style, purely-generated synthetic
data shows degradation patterns consistent with model collapse
[Kang et al., "Demystifying Synthetic Data in LLM Pre-training,"
arXiv:2510.01631, EMNLP 2025]. A separate, more theoretical result argues
that even a small fixed nonzero synthetic fraction can cause a performance
plateau that does not improve with more data, in a simplified supervised-
regression regime empirically verified on language-model and image
experiments [Dohmatob, Feng, Yin, Bartlett, "Strong Model Collapse,"
arXiv:2410.04840, NeurIPS 2024]. These two results are not a clean
contradiction — they characterise different synthetic-data generation
methods — but C5's data-feedback design should distinguish the two cases
explicitly: rephrased or lightly-transformed operational-log data has real
support for a 30% cap; data the meta-trainer generates from scratch (closer
to a STaR/ReST-style generated trajectory) does not have a proven-safe
ratio in the evidence reviewed for this record, and should be treated more
conservatively pending this project's own measurement.

## 6. Sequenced roadmap: the next 3-5 ADRs

Listed smallest and most foundational first. Each later item depends on
the one before it clearing; none of these are authorised by this record,
which is docs-only per its own status line above.

1. **ADR-0022 (proposed next): baseline task-outcome measurement on
   C1/C2/C4, no training.** Before any further training round, measure
   the *current* candidate and the *current* reference checkpoint (the
   same two models ADR-0018 through ADR-0020 already evaluate) against
   fresh, mechanically-scored task-outcome evaluations for C1 (method/tool
   selection, exact-match against gold), C2 (does an emitted config
   parse/validate and does an emitted command produce the expected
   artifact), and C4 (participation rate and conditional correctness on a
   labelled should-defer/should-not-defer set, directly extending the
   existing `mtr-v2-heldout-0013`-style refusal-calibration material).
   This is measurement only — no training call, matching the shape of
   ADR-0013's own validation-only first step. This is deliberately the
   smallest ADR in this roadmap, and comes first because every later item
   needs this baseline to measure against.
2. **ADR-0023: C3 evaluation harness built from this project's own
   outcome-record history.** ADR-0013 through ADR-0020's real outcome
   records are a natural labelled evaluation set for run diagnosis — build
   a held-out set of (raw evidence, correct diagnosis) pairs from these
   already-recorded, independently-reviewed real results, including at
   least one case requiring the model to surface a structural caveat (the
   ADR-0020 floor effect) rather than just apply a label. Addresses the
   evidence gap named in section 4: no small-model evaluation of this kind
   exists in the literature reviewed for this ADR, so this project's own
   history is the most direct evidence source available.
3. **ADR-0024: C5 data-request schema and quarantine-review path,
   design only.** Specifies the structured provenance schema C5 must
   emit, and the independent-review quarantine path any fetched item must
   clear before any part of it can be considered for training admission —
   reusing ADR-0013's existing dataset-rights admission-gate pattern.
   Does not authorise any actual fetch.
4. **ADR-0025: C6 retention-gate design, reusing the cumulative-training-
   engine control pattern.** Specifies the non-degradation check (against
   both the immediate prior round and the original base, per the pattern
   already prototyped and independently design-reviewed for a related
   internal training programme) that every future admitted round must
   pass before being treated as an improvement, plus the rollback and
   kill-condition mechanics. Design only; no training authorised.
5. **ADR-0026 (contingent): first bounded training round explicitly
   targeting a measured C1/C2/C4 gap from ADR-0022's baseline.** Only
   proposed once ADR-0022 through ADR-0025 are in place and ADR-0022's
   baseline has identified a specific, evidenced gap worth training
   against — not a default next step taken on schedule. Subject to the
   full existing gate stack (signed review, independent review, fail-
   closed containment) exactly as every prior training ADR in this
   lineage.

## Consequences

- ADR-0013's Context paragraph's framing of mission scope is superseded by
  the mission statement in section 1; ADR-0013's own decision, gates, and
  recorded outcome are untouched.
- No training, promotion, publication, or deployment is authorised by this
  record.
- The refusal-calibration work in ADR-0016 through ADR-0020 is reframed as
  part of C4, not discarded.
- Future evaluation work moves toward task-outcome scoring (section 4);
  the existing free-text classifier approach is deprioritised for further
  refinement, not for the underlying question it was trying to answer.
- The next real step (ADR-0022, per the roadmap) is a baseline measurement
  with no training call — consistent with this project's standing
  practice of measuring before proposing a change.

## Documentation impact and maintenance triggers

This record adds `docs/decisions/ADR-0021-metatrainer-mission-and-capability-map.md`
(this file) and `ROADMAP.md`'s pointer to it (see repository root). It does
not modify `docs/decisions/ADR-0013-bounded-pretrained-metatrainer-cycle.md`
or any other prior ADR's own text, gates, or recorded outcome.

Re-open or supersede this record if: the owner's stated mission changes
again; any capability's measurable definition in section 2 is found not to
be mechanically checkable in practice; the authority model in section 3 is
found to leave a gap a real incident exploited; or a cited primary source
is retracted, corrected, or superseded by later work materially changing
one of the claims above. This record's roadmap (section 6) is itself
subject to revision as ADR-0022's baseline produces real evidence — later
items in the sequence may be reordered or dropped based on what that
baseline actually shows, per this project's standing "evidence over
schedule" discipline.
