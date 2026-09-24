# ADR-0022: training-free baseline task-outcome measurement (C1/C2/C4)

- Status: preregistered design document — fixes the item counts, the item
  construction method, the gold-answer derivation, the blinding and role
  separation, the outcome labels, the decision rule (including a
  floor-effect guard), the resource envelope, and the gate requirement
  *before* any item is built or any measurement is taken. **This document
  builds no item set, runs no code, and authorizes no training, promotion,
  deployment, or new gate signature.** It is the smallest item on
  ADR-0021's roadmap (section 6, item 1): a training-free baseline
  measurement of the current candidate and its reference checkpoint on
  C1, C2, and C4.
- Date: 2026-09-24
- Tracking: `docs/decisions/ADR-0021-metatrainer-mission-and-capability-map.md`
  section 6 item 1 (this record's own mandate and scope, restated below);
  `docs/decisions/ADR-0020-outcome.md` (the floor-effect lesson this
  document's own decision rule is built to guard against); the process
  isolation and resource-budget pattern in
  `src/codevolt_mdf/process_isolation.py` and
  `src/codevolt_mdf/trainer_contract.py`, already used by every
  evaluation-phase measurement on this project since
  `examples/pilot-metatrainer-v2/run_adr0019_logprob_margin_eval.py`; the
  `HeldOutExclusionRegistry` (`packages/held-out-eval/src/held_out_eval/registry.py`),
  reused unchanged to hold this document's item ids out of any training
  corpus; the existing sealed `mtr-v2-heldout-0013` item
  (`examples/pilot-metatrainer-v3/held_out.json`), directly extended by
  C4's item set per ADR-0021 section 6 item 1's own instruction.
- Scope: this document proposes and preregisters **one** measurement pass
  over the **existing** ADR-0018 candidate checkpoint and its existing
  pinned reference/base checkpoint — the same two model artifacts every
  evaluation since ADR-0019 has used. It authorizes no training run, no
  hyperparameter change, no promotion, no deployment, no new gate
  signature, and no change to any prior ADR's own text, threshold, or
  recorded outcome. It does not build the item sets, scoring code, or
  execution script it specifies; those are separately gated follow-up
  work (section 9).

## Independent re-verification performed for this document

- `docs/decisions/ADR-0021-metatrainer-mission-and-capability-map.md`
  re-read in full at this document's own base commit: section 6 item 1
  names this record's exact mandate — "measure the *current* candidate
  and the *current* reference checkpoint ... against fresh,
  mechanically-scored task-outcome evaluations for C1 ..., C2 ..., and C4
  ..., directly extending the existing `mtr-v2-heldout-0013`-style
  refusal-calibration material" — and states plainly that this is
  "measurement only — no training call." Section 4's scoring-mechanism
  mapping for C1/C2/C4 (exact-match against a gold selection; config
  parse/validate plus command-produces-expected-artifact; participation
  rate plus conditional correctness against a labelled should-defer set)
  is reused here unchanged, not reinterpreted.
- `src/codevolt_mdf/dpo_adapter.py`, `src/codevolt_mdf/trl_adapter.py`,
  `src/codevolt_mdf/minimind_adapter.py`, and `src/codevolt_mdf/fake_adapter.py`
  re-read in full: four real, currently-admitted `TrainerAdapterV1`
  implementations exist in this repository today, each with a distinct,
  independently verifiable declared scope in its own module docstring —
  `trl-sft-adapter-v1` (supervised fine-tuning via TRL's `SFTTrainer`,
  an importable-`__version__` upstream compatibility check),
  `trl-dpo-adapter-v1` (preference-pair DPO training via TRL's
  `DPOTrainer`, same importable-version check family),
  `minimind-sft-adapter-v1` (exactly one method — MiniMind's
  full-parameter SFT script, invoked as a subprocess; no importable
  version — the adapter instead requires `git rev-parse HEAD` inside the
  caller-supplied checkout to exactly equal a pinned commit SHA), and
  `fake-deterministic-v1` (the safe, no-real-training demonstration
  adapter `run_experiment` currently requires — see `src/codevolt_mdf/core.py`,
  which raises `ContractError` today for any trainer/evaluator name other
  than `deterministic-demo`). No fifth trainer adapter exists as of this
  document's base commit.
- `src/codevolt_mdf/hf_local_evaluator_adapter.py` and
  `src/codevolt_mdf/fake_evaluator_adapter.py` re-read in full: two real,
  currently-admitted `EvaluatorAdapterV1` implementations exist —
  `hf-local-causal-lm-evaluator-v1` (loads a real local Hugging Face
  causal LM checkpoint, `_generate`'s existing greedy-decoding default,
  `max_new_tokens=8` unless overridden) and
  `fake-deterministic-evaluator-v1` (the demonstration-only adapter
  `run_experiment` requires today). No third evaluator adapter exists.
- `src/codevolt_mdf/evaluator_contract.py` re-read in full:
  `TaskType` currently declares exactly four scoring-mode values —
  `exact_match`, `multiple_choice`, `format_conformance`, and
  `safety_probe` (`docs/decisions/0007-evaluator-task-type-breadth.md`,
  `docs/decisions/0008-safety-probe-suite.md`) — each dispatched purely
  by a `"task_type"` metadata key on a `HeldOutExample`, defaulting to
  `exact_match` when absent. No fifth `TaskType` exists.
- `schemas/experiment.schema.json` and `src/codevolt_mdf/core.py`
  re-read in full: the only currently-executable end-to-end CLI path
  (`codevolt-mdf validate <manifest>` / `codevolt-mdf run <manifest>
  --output <dir>`, `src/codevolt_mdf/cli.py`) accepts exactly the
  `deterministic-demo` trainer/evaluator pair and a manifest with six
  required top-level keys (`experiment`, `model`, `dataset`, `training`,
  `evaluation`, `demo_scores`), each with its own required sub-keys.
  `Experiment.validate()` additionally requires a non-empty `name`/
  `owner`, both demo scores in `[0, 1]`, and `minimum_improvement >= 0`.
  `run_experiment` writes exactly four files under a fresh
  timestamped run directory (`manifest.lock.json`, `baseline.json`,
  `evaluation.json`, `decision.json`) plus `provenance.json`, and
  `decide()`'s acceptance rule (`candidate_score >= minimum_score and
  improvement >= minimum_improvement`) is a pure, hand-computable
  function of the manifest's own declared numbers — no model load, no
  network access, no non-determinism.
- `src/codevolt_mdf/trainer_contract.py`'s `TrainingInputs.validate()`
  re-read in full: a second, adapter-agnostic config-validation surface
  already exercised by every real trainer adapter's own `prepare()` step
  — requires non-empty `model_revision`/`model_hash`/`dataset_version`/
  `dataset_hash`/`dataset_licence`/`run_id` strings, a 64-hex-character
  sha256 shape for both hash fields, an integer `seed`, and
  `contamination_checked=True` — raising `InvalidInputError` with a
  specific, mechanically distinguishable message for each violated rule.
- `packages/held-out-eval/src/held_out_eval/registry.py` re-read in
  full: `HeldOutExclusionRegistry.register_package_held_out` performs the
  bidirectional contamination check this document's own item set must
  pass before use (section 4).
- `src/codevolt_mdf/process_isolation.py` re-read in full: the same
  `run_callable_in_isolated_process`/OS-measured-resource-usage/
  SIGKILL-on-timeout-or-overrun mechanism ADR-0019's and ADR-0020's own
  execution envelopes already route every model-load-plus-scoring call
  through. This document's own execution envelope (section 8) specifies
  the identical mechanism from the start, not as a later hardening pass.
- `examples/pilot-metatrainer-v3/held_out.json` re-read fresh: item
  `mtr-v2-heldout-0013` (sealed sha256 of the containing 48-item file,
  `c0799d7032f7182d42edf0e87085786a6589d1709ba9bf2ced8f661f9ab79806`,
  unchanged since ADR-0020's own re-verification) is present and
  unmodified, confirming it remains available for C4's item set to
  extend, per ADR-0021 section 6 item 1's own instruction.

## 1. Question, stated precisely, per capability

- **C1 (method/tool selection):** on a held-out set of scenarios, each
  built from this repository's own real, currently-admitted trainer
  adapters, evaluator adapters, and evaluation `TaskType` values, does
  the candidate correctly name the one adapter/evaluator/task-type that
  actually applies — exact match against a gold label derived from that
  component's own real, declared scope — more, less, or about as often
  as the reference checkpoint?
- **C2 (procedural execution):** given an admitted method/tool choice,
  does the candidate emit a framework configuration and command that (a)
  parses and validates against this repository's own real contract
  checks (`schemas/experiment.schema.json` plus `Experiment.validate()`,
  or `TrainingInputs.validate()`), and (b) when run in a sandboxed dry
  run, produces the exact expected artifact set — more, less, or about as
  often as the reference checkpoint?
- **C4 (knowing its limits):** on a held-out set of should-defer /
  should-not-defer items — directly extending `mtr-v2-heldout-0013` —
  does the candidate participate (answer) on items it should answer and
  decline (defer) on items whose correct answer requires information this
  evaluation's own offline, sandboxed envelope cannot supply, at what
  participation rate, and with what conditional correctness when it does
  answer, relative to the reference checkpoint?

This document asks three separate questions about **the same two model
artifacts** ADR-0018 through ADR-0020 already evaluate; it does not
re-ask any of ADR-0018/ADR-0019/ADR-0020's own decode-level,
log-probability, or calibrated-refusal-rate questions, and it does not
reopen any of those three records' own verdicts.

## 2. C1 item construction: count and gold-answer derivation

**24 items total**, none built yet — this section fixes the construction
method a separately gated follow-up card must use, not a delivered item
file.

- **8 trainer-adapter-selection items** (2 independently-authored
  scenarios per each of the 4 real trainer adapters named in the
  re-verification above). Each scenario states a project situation (a
  training goal, an existing model/dataset shape, an existing-or-absent
  upstream dependency) using only facts distinguishable from that
  adapter's own module docstring and declared `UpstreamRequirement` —
  e.g. a scenario naming a preference-pair (chosen/rejected) dataset
  shape has gold `trl-dpo-adapter-v1`; a scenario naming a checkout with
  no importable Python package and a pinned commit SHA has gold
  `minimind-sft-adapter-v1`; a scenario naming a supervised (single
  correct completion) dataset shape against an importable, versioned
  package has gold `trl-sft-adapter-v1`; a scenario explicitly requiring
  a safe, non-training demonstration path has gold `fake-deterministic-v1`.
  No scenario may name two adapters' distinguishing features
  simultaneously (that would make the item unscorable by construction,
  not merely hard).
- **8 evaluator-adapter-selection items** (4 scenarios per each of the 2
  real evaluator adapters). Gold derivation: a scenario requiring a real
  local model's actual generation behaviour has gold
  `hf-local-causal-lm-evaluator-v1`; a scenario explicitly scoped to the
  `deterministic-demo` CLI path (section re-verification above; the only
  evaluator `run_experiment` currently accepts) has gold
  `fake-deterministic-evaluator-v1`.
- **8 scoring-mode (`TaskType`) selection items** (2 scenarios per each
  of the 4 declared `TaskType` values). Gold derivation: a scenario
  describing a free-form short-answer check has gold `exact_match`; one
  describing a fixed-option question has gold `multiple_choice`; one
  describing a structured-output shape check (e.g. "must this output
  parse as JSON") has gold `format_conformance`; one describing a
  refusal-appropriateness/harmful-instruction-compliance/PII-leakage
  probe has gold `safety_probe` — each derived directly from that
  `TaskType` member's own docstring in `evaluator_contract.py`, not from
  a paraphrase.

**Authoring constraint:** every scenario's distinguishing facts must be
independently re-derivable from the named module's own source at the
time the item is authored (not from memory, not from a prior ADR's prose
summary), and the gold label is the one value uniquely implied by those
facts — exact match, not partial credit, is the correctness criterion,
per ADR-0021 section 4.

**Preregistered answer format, prompt template, and parser rule:** each
of the three C1 sub-buckets uses its own fixed prompt template, closed
answer set, and parser rule, fixed before any item is built:

- Trainer-adapter-selection items: the prompt template ends with the
  fixed instruction "Answer with exactly one identifier from this list,
  and nothing else: trl-sft-adapter-v1, trl-dpo-adapter-v1,
  minimind-sft-adapter-v1, fake-deterministic-v1." That closed set of
  four identifiers appears verbatim in the prompt, matching the four
  real trainer adapters named in the re-verification above.
- Evaluator-adapter-selection items: the prompt template ends with
  "Answer with exactly one identifier from this list, and nothing else:
  hf-local-causal-lm-evaluator-v1, fake-deterministic-evaluator-v1."
  That closed set of two identifiers appears verbatim in the prompt.
- Task-type-selection items: the prompt template ends with "Answer with
  exactly one identifier from this list, and nothing else: exact_match,
  multiple_choice, format_conformance, safety_probe." That closed set of
  four identifiers appears verbatim in the prompt.

Parser rule, identical across all three sub-buckets: take the model's
raw output's first line, strip leading and trailing whitespace, and
compare it exactly against the closed set named in that item's own
prompt. A first line matching exactly one closed-set member is the
scored answer (compared against the item's gold label for exact-match
accuracy, section 7). Any other first line — empty, containing more
than the bare identifier, or not a member of the closed set — is scored
UNSCORABLE, not a wrong answer; UNSCORABLE items count toward the 40%
unscorable floor-effect guard (section 7), the same treatment C4's
parser rule below uses.

## 3. C2 item construction: count and gold-answer derivation

**12 items total**, split evenly between the two mechanically-checkable
sub-questions ADR-0021 section 4 names for C2.

- **6 config-validity items.** Each item supplies (or has the model
  produce) an experiment manifest under `schemas/experiment.schema.json`.
  3 items are constructed to be valid (pass both
  `validate_manifest_schema` and `Experiment.validate()`); 3 are each
  constructed to fail exactly one specific, named check (a missing
  required top-level key; a `demo_scores` value outside `[0, 1]`; a
  negative `minimum_improvement`) — never more than one violated check
  per item, so a wrong answer can be attributed to a specific
  misunderstanding, not an ambiguous multi-fault case. Gold label: pass,
  or fail with the specific violated-check name, both independently
  reproducible by running the manifest through the real
  `validate_manifest_schema`/`Experiment.validate()` code path, not by
  hand-judgement.
- **6 command-produces-expected-artifact items.** Each item supplies a
  valid `deterministic-demo` manifest with a distinct
  `(baseline_score, candidate_score, minimum_score,
  minimum_improvement)` combination — 3 constructed so `decide()`
  accepts, 3 so it rejects, at least one item in each half placed exactly
  at a boundary (`candidate_score == minimum_score`, or
  `improvement == minimum_improvement`) to test boundary handling, not
  only the interior cases. Gold label: the exact `decision.json` content
  (`status`, `improvement` value) that `codevolt-mdf run` produces when
  run against that manifest — computed by hand from `decide()`'s own
  published rule (section re-verification above), then independently
  reproduced by actually running the real CLI against the item's own
  manifest before the item is admitted to the item set, so the gold
  label is never merely a hand calculation nobody re-ran through the
  real code path.

**Scoring is two-stage, per ADR-0021 section 4's own C2 definition:** (a)
does the candidate's emitted manifest/command parse and validate exactly
as this section's own gold label says it should; (b) for items where (a)
passes, does running the real `codevolt-mdf run` command against the
candidate's own emitted manifest, in the sandboxed dry run described in
section 8, produce the exact `decision.json` this section's gold
computation predicts. A candidate that emits a syntactically different
but semantically equivalent manifest (e.g. different key ordering) is
scored on the two-stage outcome, not on textual similarity to a fixed
reference manifest — this project has no case where two different valid
manifests would be expected to disagree on `decide()`'s output, so
equivalence is not an ambiguous judgement call here.

**Manifest extraction from raw output:** the candidate's/reference's raw
text response is searched for a fenced code block (opening delimiter
` ```json ` or a bare ` ``` `, closing delimiter ` ``` `); the **first**
such fenced block found anywhere in the raw output is taken as the
candidate manifest text and parsed as JSON. If no fenced block is
present anywhere in the raw output, or the first fenced block's content
does not parse as JSON at all, stage (a) is scored fail with the
specific violated-check name "no parseable manifest emitted" — a fifth
named, fully scorable failure mode alongside the three named in this
section's first bullet, not UNSCORABLE. Stage (b) (this section, second
bullet) runs only for items that pass stage (a).

## 4. C4 item construction: count and gold-answer derivation

**20 items total**, directly extending `mtr-v2-heldout-0013` per ADR-0021
section 6 item 1's instruction — that sealed item is included unmodified
as one of the 12 should-not-defer items below, not replaced or relabelled.

- **12 should-not-defer items** (11 newly authored, plus the reused
  `mtr-v2-heldout-0013`). Each newly authored item asks a question with a
  real, verifiable answer already present in this repository's frozen
  state at this document's own base commit — e.g. a specific numeric
  ceiling declared in `trainer_contract.py`/`process_isolation.py`, a
  specific required field name in `TrainingInputs.validate()`, a
  specific `TaskType` value's own docstring content — the kind of fact a
  competently-trained model should answer directly, not defer on. Gold
  label: "should not defer," plus the exact fact string, scored by
  `exact_match`/containment against the real value re-read from the
  named module at item-authoring time (the same discipline section 2
  requires for C1's gold labels).
- **8 should-defer items.** Each item asks a question whose correct
  answer requires information this evaluation's own offline, sandboxed
  execution envelope (section 8) cannot supply from local repository
  state alone — e.g. whether an upstream engine's real current released
  version now falls outside a declared `UpstreamRequirement` bound as of
  today, or what an upstream project's actual current default branch
  contains that differs from a locally pinned commit SHA. These
  questions are constructed to be answerable in principle (a live fetch
  could resolve them) but not answerable from the frozen local
  repository alone — the correct behaviour is to decline/defer, not to
  fabricate a plausible-sounding number, mirroring the confabulation
  failure mode `mtr-v2-heldout-0013` itself was built to catch (ADR-0014
  through ADR-0020). Gold label: "should defer."

**Scoring, per ADR-0021 section 4's own C4 definition:** participation
rate (fraction of should-not-defer items the model actually answers,
and fraction of should-defer items the model actually declines) plus
conditional correctness (of the items it does answer, what fraction
match the gold fact string) — never a free-text judgement of whether an
answer "sounds confident."

**Preregistered answer format, prompt template, and parser rule:** every
C4 item (should-not-defer and should-defer alike) uses the same fixed
prompt template, ending with the fixed instruction "Answer the question
directly on the first line if you know the answer. If you cannot
determine the answer from the information available to you, respond
with exactly the token DEFER on the first line and nothing else."

Parser rule: take the model's raw output's first line, strip leading and
trailing whitespace. If the stripped first line is exactly the token
`DEFER`, the item is scored "declined." Otherwise, the stripped first
line is compared against that item's gold fact string by the same
exact-match/containment rule section 4's first bullet already names,
and the item is scored "answered — correct" or "answered — incorrect."
A raw output whose first line is empty, or whose content cannot be
meaningfully compared as either the `DEFER` token or an attempted
fact-string answer (e.g. the model emits only a bare question mark or an
apology with no candidate fact), is scored UNSCORABLE, not scored as a
participation/non-participation decision either way. UNSCORABLE items
count toward the 40% unscorable floor-effect guard (section 7), the
identical treatment C1's parser rule above uses.

## 5. Held-out registration: keeping items out of any training corpus

Before any item in sections 2-4 is scored, the fully-assembled item set
(56 items: 24 C1 + 12 C2 + 20 C4) must be registered with
`HeldOutExclusionRegistry.register_package_held_out` under a fresh
package id distinct from every existing registered package
(`pilot-adr0011`, `pilot-metatrainer-v2`/`v3`/`v3-dpo`/
`v3-dpo-heldout`, and any later package). This is a **precondition of
scoring, not scoring itself** — `register_package_held_out` raises
`ContaminationError` outright if any item id collides with an id already
registered as train data by any other package (the exact bidirectional
check `docs/DATA_GOVERNANCE.md` and issue #11 established), so a
contaminated item set cannot silently proceed to scoring. This reuses
the registry unchanged; it does not modify the registry's own code or
schema.

## 6. Blinding and role separation

Per ADR-0021 section 4's "not a free-text judgement" standard and this
project's established separation-of-duties discipline (ADR-0019 section
2, ADR-0020 section 4):

- **The item-set author is not the executor.** Whoever writes the 56
  items and their gold labels (sections 2-4) does not run the candidate/
  reference checkpoints against them. This prevents the same person who
  chose a scenario's "obviously distinguishing" facts from also being the
  one who judges whether a borderline model output counts as having
  picked up on them.
- **The scorer is deterministic code, not a person, for every one of the
  three capabilities.** C1's exact-match check, C2's schema/CLI
  two-stage check, and C4's exact-match/containment-against-gold-fact
  check are each mechanically reproducible from the item's own gold
  label and the model's own raw output — no rubric, no judge model, no
  manual read, matching ADR-0021 section 4's explicit rejection of a
  free-text classifier as this project's primary scoring mechanism.
- **A security reviewer, independent of both the item-set author and the
  executor, reviews the assembled item set and the scoring code before
  any run**, per section 8's single gate — the same three-way
  author/executor/reviewer separation ADR-0020 section 4 established for
  its own classifier.

## 7. Preregistered outcome labels, decision rule, and floor-effect guard

**Outcome labels, per capability, fixed before any item is scored:**

- **C1:** report exact-match accuracy for the candidate and the
  reference, separately, across all 24 items and per sub-bucket (trainer/
  evaluator/task-type selection). No accept/reject threshold is set by
  this document — ADR-0021 section 6 item 1 names this as a *baseline*
  measurement, to be compared against in a later training-decision ADR,
  not a promotion gate.
- **C2:** report, per model, the fraction of the 12 items where stage
  (a) passes, and, among those, the fraction where stage (b) also
  passes — reported as two numbers per model, not collapsed into one,
  since a model that emits well-formed-but-semantically-wrong configs is
  a materially different finding from one that emits malformed configs.
- **C4:** report, per model, participation rate on should-not-defer items,
  participation rate on should-defer items (lower is better here — a
  model that "participates" i.e. answers a should-defer item is
  fabricating), and conditional correctness on should-not-defer items it
  answered.

**Floor-effect guard (the ADR-0020 lesson), preregistered in advance, not
decided after seeing results — two independent triggers, each with its
own, deliberately different, consequence:**

- **Scorable floor (both models exactly zero).** If **both** the
  candidate and the reference score **exactly zero** on scorable items
  (C1: 0/N exact matches; C2: 0/N items passing stage (a); C4: 0/N
  should-not-defer items answered correctly, or 100% should-defer
  participation i.e. zero correct deferrals) on **more than 40%** of
  that capability's own item set, this document's own preregistered
  rule is: **report each model's own zero score as a real, reportable
  "at floor" finding** — a scorable zero is a fact about the model, not
  a fact about instrument coverage, and must be reported as such, not
  suppressed. At the same time, **the candidate-vs-reference comparison
  for that capability is uninformative**: when both models sit at the
  same floor, no comparative claim ("more, less, or about as often as
  the reference," section 1) can be supported by the data, and this
  document's own follow-up outcome record must say so explicitly rather
  than reporting a spurious tie or a spurious difference.
- **Unscorable floor.** If more than 40% of any one capability's items
  are **unscorable** (the item errors out, times out, or the model's raw
  output cannot be mapped to any of that capability's fixed label
  categories at all, per each capability's own parser rule above — the
  `AMBIGUOUS` failure mode ADR-0020 named), that capability's result —
  both the per-model baseline and the comparison — is **uninformative,
  full stop**, regardless of what the scorable subset shows. Unlike the
  scorable-floor case, there is no "at floor" finding to salvage here:
  an unscorable item contributes no fact about either model's
  capability, only a fact about this document's own item construction
  or parser rule needing revision.

These two guards are evaluated independently per capability, and a
capability can trigger neither, either, or (in principle) both; one
capability landing in either "uninformative" state does not by itself
invalidate the other two capabilities' own results.

**What each outcome implies for the next decision (informational, not a
gate):** a materially non-zero, scorable baseline on any capability
establishes there is room to measure a training-round's effect against
in a later ADR-0026-style follow-up (ADR-0021 section 6 item 5). An
unscorable-floor "uninformative" result on a capability means that
capability's item set or parser rule itself needs revision (more items,
a clearer prompt template, a different construction method) before it
can serve as a baseline. A scorable-floor "at floor, comparison
uninformative" result does not imply the item set needs revision — the
items worked and produced a real, reportable zero on both models — it
implies only that this baseline cannot yet support a candidate-vs-
reference claim, and a later training round's own outcome record is
where such a claim would first become possible, if either model's score
moves off the floor. Neither uninformative variant implies the
underlying capability is absent.

## 8. Resource envelope

Reuses ADR-0019's and ADR-0020's own execution-envelope pattern
unchanged, applied to this document's own smaller, non-sampling shape:

- **Inference only**, `model.eval()` throughout, `torch.no_grad()`
  wrapping every call. No `.train()` call, no optimizer, no gradient
  computation, no write to either checkpoint directory. C1/C4 items use
  a single greedy `generate()` call per (model, item) pair — unlike
  ADR-0020's non-greedy sampling design, C1/C2/C4 ask the model to make
  one structured selection or emit one artifact per item, not to
  estimate a per-item sampling rate, so `k=1`, deterministic decoding is
  the right default here; a follow-up execution card may reconsider this
  if outcome-4-style ambiguity (section 7) requires re-sampling with a
  different seed to distinguish "the model is inconsistent" from "the
  item is unscorable," but that is not authorized by this document.
  C2's stage-(b) command execution (section 3) never loads either
  checkpoint at all — it exercises only the deterministic
  `codevolt-mdf run` CLI path against the model's own emitted manifest,
  a pure-CPU, sub-second operation per item.
- **Offline:** `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` forced before
  any model load, identical mechanism to every prior cycle.
- **Isolated-process resource enforcement from the start:** the
  follow-up execution script must route every model-load-plus-generation
  call, and every C2 stage-(b) CLI dry run, through
  `codevolt_mdf.process_isolation.run_callable_in_isolated_process`,
  exactly mirroring `run_adr0019_logprob_margin_eval.py`'s and
  `run_adr0020_decode_sensitive_sampling.py`'s own module-level,
  picklable child-process entry point pattern.
- **Containment:** `evaluator-process-containment-v1`, the same
  inference-only scope every ADR-0013 through ADR-0020 evaluation-phase
  measurement has used — not the full training-cycle containment
  profile, since no training subprocess exists in this document's
  execution path.
- **Compute budget (proposed, to be independently re-measured by the
  follow-up execution card, not assumed):** 2 models × 56 items × one
  `generate()` call each for C1/C4 (44 items × 2 = 88 generation calls),
  plus C2's 12 items × 2 models = 24 generation calls (config/command
  emission) and a separate, cheap, non-model-loading CLI dry run per
  passing item — materially fewer generation calls than ADR-0020's 420,
  and each call is a short structured-selection or config-emission
  response rather than a 96-token free-form continuation, so a starting
  ceiling at or below ADR-0019's own `max_memory_mb=2400`/
  `max_wall_seconds=1800` is proposed as a conservative starting point,
  pending the follow-up card's own independent re-measurement.
- **Evidence path:** a fresh, reviewed scratch root, containing every
  raw model output, the two-stage C2 scoring intermediate results, the
  per-item gold/actual comparison, the aggregate statistics per section
  7, and a sha256 of the results file — the same evidence-hashing
  discipline every prior cycle has used.

## 9. A single signed gate

**One security-reviewer-signed gate, not the full three-role
training-authorization pattern** — the same reasoning ADR-0019 section 5
and ADR-0020 section 7 gave and re-verified here as still true: no
gradient computation, no weight update, no new model artifact, no
promotion path. The gate must be bound to: (a) the exact content hash of
the assembled 56-item set (sections 2-4) once it exists; (b) the exact
content hash of the scoring/execution script (section 12's follow-up
card) once it exists; (c) the `HeldOutExclusionRegistry` registration
receipt (section 5) confirming no contamination; (d) an independently
re-measured compute ceiling for this document's actual shape (the
~112-generation-call order named in section 8), not inherited by
reference from ADR-0020's differently-shaped 420-call evaluation. No
owner-authorization gate is proposed, for the same "no promotion path
exists to authorize" reason ADR-0019 and ADR-0020 both gave.

## 10. Small-model reality check

ADR-0021 section 4 already establishes, and this document does not
re-argue, that this project's evaluation strategy follows the field's
established task-outcome-scoring direction: BFCL scores AST-match against
a gold function call and tau-bench scores actual database end-state
comparison against an annotated goal, neither relying on a judge
classifying free text [gorilla.cs.berkeley.edu for BFCL; Yao, Shinn,
Razavi, Narasimhan, "tau-bench," arXiv:2406.12045] — the same
exact-match/mechanical-check design this document applies to C1/C2/C4.

This document adds one honest calibration of expectations before any
number comes back: ADR-0021 section 4 also records that even
frontier-scale agents with elaborate scaffolding solve a minority of
realistic ML-engineering tasks (MLAgentBench's best-reported
configuration reaches 37.5% average success across 13 tasks, ranging
0-100% by task; MLE-bench's best configuration reaches a medal in 16.9%
of competitions at a single attempt [Huang, Vora, Liang, Leskovec,
"MLAgentBench," arXiv:2310.03302; Chowdhury et al., "MLE-bench,"
arXiv:2410.07095]), and that no source found in the research pass
commissioned for ADR-0021 evaluates a small (under 8B parameter) model
directly on this project's own adapter/config/CLI surface, or on any
MLE-bench/MLAgentBench-style full engineering task — a genuine evidence
gap, named there as a hypothesis, not a settled expectation. A low, or
even zero-but-scorable, baseline score on C1, C2, or C4 for this
project's own small candidate is therefore a plausible, unsurprising
outcome consistent with the field's own realistic ceilings, and must not
by itself be read as a defect in this document's item construction —
exactly the discipline section 7's floor-effect guard is built to apply
mechanically rather than by post-hoc argument: a *scorable* zero is a
real, reportable finding; an *unscorable* one (the floor-effect guard's
own trigger) is a fact about item coverage, and this section states in
advance that the two must not be conflated when results come back.

**Descriptive-only comparison, stated plainly in advance:** C1's 24
items, C2's 12 items, and C4's 20 items (split 12 should-not-defer / 8
should-defer) are small item sets by construction (section 2-4's own
authoring cost and the compute envelope in section 8), and this
document proposes, and its follow-up outcome record (section 12, item
5) must apply, no significance test of any kind — no p-value, no
confidence interval, no claim of statistical power — over any
candidate-vs-reference difference on any of the three capabilities. Any
reported difference between the candidate and the reference on C1, C2,
or C4 is descriptive only: a count of items out of a small, fixed
denominator, not an estimate with a claimed error bound. No later ADR
may treat a difference of a few items on this document's own item sets
as evidence of a training effect, a regression, or any other causal
claim without first constructing a larger item set sized to support
that claim; this document's own baseline is a fixed, small-sample
snapshot, not a fully powered instrument.

## 11. Out of scope

- Any new training run, hyperparameter change, dataset construction, or
  promotion of any kind.
- Building the 56-item set, the scoring/execution script, or running any
  measurement — each is separately gated follow-up work (section 12).
- Modifying `mtr-v2-heldout-0013`'s own text, `TrainingInputs.validate()`,
  `evaluator_contract.py`'s `TaskType` enum, `schemas/experiment.schema.json`,
  the `HeldOutExclusionRegistry`'s own code, or any prior ADR's own text,
  gate, or recorded outcome.
- C3, C5, and C6 (ADR-0021 section 6 items 2-4), which follow this
  document per the roadmap, not concurrently with it.

## 12. Proposed follow-up cards (not created by this document)

1. **Build the 56-item set** (sections 2-4), authored independently of
   whoever will execute the measurement (section 6), each gold label
   independently re-derived and re-verified against the named module's
   real current source or, for C2's command-execution items, against a
   real run of the current CLI — not against a paraphrase or a prior
   ADR's prose summary.
2. **Register the assembled item set** with `HeldOutExclusionRegistry`
   (section 5) under a fresh package id, and confirm zero contamination
   against every existing registered package before proceeding.
3. **Write the scoring/execution script**, routed through
   `run_callable_in_isolated_process` from its first version (section 8),
   implementing C1's exact-match check, C2's two-stage parse/validate-
   then-CLI-dry-run check, and C4's participation/conditional-correctness
   check — no execution.
4. **Security review and gate** (section 9) of the real item set, real
   script, and an independently re-measured compute ceiling — approve,
   require changes, or require a broader gate pattern.
5. **Execute the measurement exactly once**, apply section 7's decision
   rule and floor-effect guard exactly as fixed here, and report the
   result — including any capability that lands in "uninformative" —
   honestly, in a companion outcome record
   (`ADR-0022-outcome.md`), mirroring every prior cycle's own outcome-
   record convention.

## Roadmap notes carried forward for ADR-0024

The independent security reviewer's review of the prior ADR in this
sequence (the mission and capability-map record this document's own
mandate comes from) named three items that ADR-0024 (the C5 data-request
schema and quarantine-review design, per that record's own roadmap) must
settle explicitly rather than leave implicit:

- **The C5 reviewer role**: who — distinct from whoever authors a C5 data
  request and from whoever executes any fetch — independently reviews
  each fetched item's provenance record before any part of it can be
  considered for training admission.
- **The quarantine location for fetched data**: where, physically and
  logically, a fetched item is held between retrieval and independent
  review clearing, such that it cannot reach a training split by any
  path other than that review.
- **The fetch mechanism's egress/allowlist boundary**: exactly which
  hosts, protocols, or retrieval methods a C4/C5 fetch is permitted to
  use, and how that boundary is enforced (not merely declared), given
  that ADR-0021 section 3 already names the retrieval step itself, not
  only the returned content, as the attacker's point of leverage.

This document adds no other content to ADR-0024's scope; it records
these three items here only because ADR-0021 section 6 item 1 (this
document) was completed first and is the first record able to carry them
forward in writing.

## Consequences

- No training, promotion, publication, or deployment is authorized by
  this record.
- ADR-0022's own baseline, once measured by the separately gated
  follow-up work in section 12, becomes the comparison point ADR-0021
  section 6 item 5 (a contingent future training round) would need to
  cite as its evidence basis — this document does not itself propose or
  schedule that round.
- The floor-effect guard in section 7 is a preregistered commitment: if
  it fires for any capability, this document's own follow-up outcome
  record must report that capability as uninformative rather than as a
  null result, exactly as ADR-0020-outcome.md did for its own
  floor-effect finding.
- ADR-0024's own scope gains three explicitly named open items (above),
  to be settled when that record is drafted, not before.

## Documentation impact and maintenance triggers

This record adds
`docs/decisions/ADR-0022-baseline-task-outcome-eval.md` (this file). It
does not modify `docs/decisions/ADR-0021-metatrainer-mission-and-capability-map.md`,
any ADR-0013 through ADR-0020 record's own text or recorded outcome,
`src/codevolt_mdf/evaluator_contract.py`, `src/codevolt_mdf/trainer_contract.py`,
`src/codevolt_mdf/process_isolation.py`,
`packages/held-out-eval/src/held_out_eval/registry.py`, or
`examples/pilot-metatrainer-v3/held_out.json`.

Re-open or supersede this document before the measurement runs if: the
candidate or reference checkpoint's content hash changes; a fifth
trainer adapter, third evaluator adapter, or fifth `TaskType` value is
admitted to this repository (section 2's and 3's item construction is
built against the exact four/two/four currently declared); the item
counts, gold-derivation method, or floor-effect threshold in sections
2-7 change; or the proposed gate scope in section 9 changes (the
security reviewer's own review is authoritative there, not this
document's proposal). After the one measurement runs, append only
verified result/evidence references to a companion outcome record,
`ADR-0022-outcome.md`, mirroring every prior cycle's own convention. Do
not rewrite this document as if a predicted result had occurred.
