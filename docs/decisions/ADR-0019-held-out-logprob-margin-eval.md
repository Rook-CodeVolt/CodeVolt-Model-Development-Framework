# ADR-0019: preregistered held-out log-probability margin evaluation of the ADR-0018 DPO candidate

- Status: preregistered design document — fixes the question, the held-out
  data contract, the metric, the statistic, and the decision rule *before*
  any measurement is taken. **This document runs no code, builds no
  dataset, computes no margin, and authorizes no new training run.** It
  authorizes exactly one thing, once its own gate (section 5) clears: a
  read-only, offline scoring pass over the already-existing ADR-0018
  candidate checkpoint and an as-yet-unbuilt held-out preference-pair set.
- Date: 2026-09-23
- Tracking: internal execution records (see linked PR numbers above where applicable)
- Scope: this document proposes and preregisters **one** evaluation of the
  **existing** ADR-0018 candidate checkpoint (already trained, already
  merged, already the subject of a negative outcome record). It does not
  reopen ADR-0018's own verdict, does not modify
  `docs/decisions/ADR-0018-outcome.md`'s text, does not touch
  `examples/pilot-metatrainer-v3-dpo/` or any of its locked hashes, and does
  not propose, authorize, or configure any new training run of any kind.

## Independent re-verification performed for this document

- `docs/decisions/ADR-0018-outcome.md` re-read in full on `main` at
  `fdac4652` (this document's own base commit): verdict **NEGATIVE**,
  generations 65/73 (89.0%) byte-identical across the three evaluation
  suites, while the run's own internal `rewards/margins` metric rose
  `0.044` (step 5) → `0.126` (step 10) → `0.113` (step 15) → `0.253`
  (step 20) — net increase, not monotonic. Three untested hypotheses are
  named there for the divergence between the training-internal signal
  moving and the generation-level signal barely moving: (a) `beta=0.3` /
  `lr=5e-7` / 22 steps may be too conservative a combination to move
  greedy-decoded output measurably even while moving the internal
  preference-margin metric; (b) DPO's loss/margin signal is computed over
  log-probabilities of the existing `chosen`/`rejected` continuations,
  which can shift without necessarily changing which token greedy decoding
  selects first at each position; (c) the 22-record training package may be
  too small relative to full-parameter DPO's capacity on a 135M model to
  produce a decode-level effect in this few steps.
- Candidate checkpoint confirmed present on this host, read-only, at
  `./local-evidence/adr0018/scratch/adr0018-metatrainer-dpo-20260923/trainer_work/adr0018-metatrainer-dpo-20260923/final/`
  — `config.json` confirms `LlamaForCausalLM`, `hidden_size=576`,
  `num_hidden_layers=30` (the `SmolLM2-135M-Instruct` architecture); the
  directory also carries the base model's own `chat_template.jinja`
  (`<|im_start|>{role}\n{content}<|im_end|>\n` ChatML-style template,
  unchanged from base — DPO training does not alter a tokenizer/template).
  Reference model is the same immutable base checkpoint ADR-0018 pinned,
  content hash `43752b3f39894c0122d9a94f3b4e64ad2d76e43d25c2a08aa360ad17a1a0145c`
  (`HuggingFaceTB/SmolLM2-135M-Instruct`).
- `examples/pilot-metatrainer-v3-dpo/preference_pairs.jsonl` re-read fresh:
  22 records, all `"split": "train"`, fields
  `pair_id`/`prompt`/`chosen`/`rejected`/`direction`/`semantic_family`/
  `content_class`/`source_scope`/`citations`. This is the exact training
  set this evaluation's held-out set must not overlap.
- `examples/pilot-metatrainer-v3/held_out.json` re-read fresh: 48 records
  (`example_id` prefixed `mtr-v2-heldout-*` and `mtr-v3n-heldout-*`),
  including `mtr-v2-heldout-0013` (`semantic_family:
  synthetic_data_tradeoffs`) — the single item this whole
  ADR-0015→0016→0017→0018 chain has targeted, and the specific item
  `examples/pilot-metatrainer-v3-dpo/validate_dataset.py` already treats as
  forbidden-by-paraphrase for the training package.
- `packages/held-out-eval/` (`HeldOutExclusionRegistry`) re-read: a
  standalone, dependency-free bidirectional train/held-out contamination
  registry already used elsewhere in this project (issue #11 precedent: 67
  ids silently shared between a train split and a later held-out split,
  undetected until this bidirectional check existed). This document adopts
  it as the mandatory contamination-audit mechanism for the new held-out
  set (section 2) rather than inventing a new ad-hoc check.
- `src/codevolt_mdf/hf_local_evaluator_adapter.py`'s existing
  `_choice_log_likelihood` method (lines ~637–700) already implements
  exactly the per-token log-probability computation this evaluation needs
  (teacher-forced single forward pass under `torch.no_grad()`, chat-template
  or bare-text rendering matching `_generate`'s own choice, length
  normalization by dividing summed log-probs by the number of choice
  tokens) — re-read in full to confirm this evaluation should reuse that
  existing, already-tested code path rather than write a new one, and to
  confirm it currently returns a **length-normalized (mean)** log-prob, not
  a raw sum, which section 3 below addresses directly.
- No `--execute` runner, dataset file, or evaluation script for this
  proposal exists yet anywhere in this repository as of this document's
  base commit (`git log`/`find` both confirm). This document authorizes
  none of those three artifacts by itself; each is separate follow-up work
  (section 6).

## 1. Question, stated precisely

Did the ADR-0018 DPO run shift the candidate model's **held-out**
chosen-minus-rejected log-probability margin, relative to the reference/base
checkpoint, even though its greedy-decoded generations on held-out
evaluation items were 65/73 (89.0%) byte-identical to baseline?

This is deliberately narrower than "did DPO work." ADR-0018's outcome record
already answered the decode-level question (negative, per its own four
pre-stated criteria). This document asks only whether there is a
*measurable, held-out* trace of the training-internal signal (which did move
on the 22 *training* pairs) that decode-level scoring cannot see — the
specific gap named as hypothesis (b) in ADR-0018's outcome record. It is not
a re-litigation of ADR-0018's verdict and does not change it either way; a
positive result here would motivate hypothesis (b)/(c) as the next thing to
test, not overturn the NEGATIVE verdict already recorded.

## 2. Held-out preference-pair set: construction, ownership, and the contamination audit it must pass

- **Size:** 20 pairs. Rationale: matches this project's own standing
  fixed-rubric-set convention (`mtr-v2-heldout-0001`..`0020`, used by every
  manual rubric review to date) for continuity of scale, and is
  large enough to support the paired non-parametric test in section 3 at
  the (limited) power this document states honestly below, without being so
  large that construction cost or contamination-audit burden becomes
  disproportionate to a single confirmatory measurement.
- **Construction rules, fixed in advance:**
  1. Same triple shape as the ADR-0017 training package: `prompt`, `chosen`
     (a corpus-consistent hedged/cited refusal or calibrated-confidence
     answer), `rejected` (a plausible, confidently-worded fabrication in the
     same register), `direction` (`"refusal"` or `"counter"`),
     `semantic_family`, `content_class`, `source_scope`, `citations`,
     `pair_id`, and `"split": "held_out"` (not `"train"` — this field is
     load-bearing for the contamination registry in the next bullet).
  2. **Counter-direction share ≥ 20%**, computed and reported the same way
     ADR-0017's training package reports it (`counts.counter_share`), not
     merely asserted — the same over-refusal-guard minimum Maya's ADR-0017
     gate (b)(2) imposed on the training set, applied identically here so
     the held-out set cannot itself introduce a one-sided bias into the
     margin measurement.
  3. Every `content_class` value must be **disjoint** from the 22 training
     pairs' `content_class` values (re-read from
     `examples/pilot-metatrainer-v3-dpo/preference_pairs.jsonl` at
     construction time, not assumed) — this held-out set must probe the
     *same target behaviour* (confabulated-recipe/fabricated-specific-figure
     refusal under the `confabulated_recipe_refusal_dpo` /
     `calibrated_confidence_dpo` semantic families) without reusing any of
     the 22 training prompts' specific scenarios.
- **Who builds it:** a separate, explicitly scoped follow-up card (section
  6) — not this document, and not the same task/session that will later run
  the scoring pass. Owner decision (Rook, accepting Maya's review of this
  PR): the pair author and the eventual scoring executor (section 6, card 4)
  must be different people, so a result cannot be shaped, even
  unintentionally, by whoever built the test; and the contamination audit
  below is performed by **Maya** specifically, not merely "a reviewer
  distinct from the constructor" — the same separation-of-duties logic
  extended to the audit role, not just the execution role.
- **Contamination audit it must pass, before it is used for anything:**
  1. **Zero overlap with the 22 training pairs**
     (`examples/pilot-metatrainer-v3-dpo/preference_pairs.jsonl`): exact/
     normalized-string match on `prompt` text, zero hits required.
  2. **Zero overlap with the 48 held-out prompts**
     (`examples/pilot-metatrainer-v3/held_out.json`, all
     `mtr-v2-heldout-*`/`mtr-v3n-heldout-*` items): exact/normalized-string
     match, zero hits required.
  3. **Zero overlap with `mtr-v2-heldout-0013`** specifically, including the
     same ≥60%-token-overlap near-paraphrase heuristic
     `examples/pilot-metatrainer-v3-dpo/validate_dataset.py` already applies
     to the training package — this is the single most sensitive item in
     this project's history and must not leak into a new held-out set via
     paraphrase.
  4. **Registration through `HeldOutExclusionRegistry`**
     (`packages/held-out-eval`): the new 20-id set is registered as
     `held_out` under its own fresh package name, and the registry's own
     bidirectional check (against every other package's registered `train`
     ids, not just the two files named above) must raise nothing. This is
     the belt to the token-overlap-heuristic braces above — a structural
     check, not a duplicate of the same heuristic.
  5. **Counter-direction share** re-verified programmatically (bullet 2
     above) as part of the same audit run, not asserted separately.
  All five checks must be re-run and pass by **Maya**, distinct from
  whoever constructed the set (per the owner decision above), immediately
  before the set is used for scoring — the same "re-verify fresh, never
  trust the committed report alone" discipline every prior ADR on this
  project has applied to its own training-set contamination checks.
- **Does it become a permanent held-out asset?** **Yes, if it passes the
  audit above.** Once sealed and registered in `HeldOutExclusionRegistry`,
  this 20-pair set is added to this project's standing held-out inventory
  (alongside the existing 48-item `mtr-v2-heldout-*`/`mtr-v3n-heldout-*`
  set) for reuse by future preference-training evaluations on this same
  axis, rather than being rebuilt from scratch each time — consistent with
  this project's own registry-based contamination model, whose entire
  purpose is to let a legitimate split be reused safely across packages. It
  is explicitly **not** added to `examples/pilot-metatrainer-v3-dpo/`'s own
  `train.jsonl`/hash-locked contents, and its own future re-use as
  held-out data by a later evaluation does not itself authorize any new
  training run.

## 3. Metric, exact computation, statistic, and decision rule — fixed before any measurement

- **Per-pair margin, per model (candidate and reference/base), computed
  identically for both:**
  `margin = logP(chosen | prompt) − logP(rejected | prompt)`,
  where `logP(x | prompt)` is computed via the **same** teacher-forced,
  single-forward-pass method `hf_local_evaluator_adapter.py`'s existing
  `_choice_log_likelihood` already implements (reused, not reimplemented) —
  same chat-template/bare-text rendering choice as that method already
  makes, same tokenizer, same `torch.no_grad()` inference, offline.
- **Summed vs mean log-probs — decided in advance: report both, treat the
  raw summed log-probability as primary.** `_choice_log_likelihood`
  currently returns a length-normalized **mean**; this evaluation computes
  the **sum** (mean × token count) as the primary quantity, for two stated
  reasons: (a) DPO's own training objective and the `rewards/margins`
  metric ADR-0018's outcome record already cites are themselves computed
  over **summed** sequence log-probabilities (TRL's `DPOTrainer`, per its
  own published loss definition), so the primary held-out quantity should
  be the same quantity the training signal that moved is stated in terms
  of, for a fair like-for-like comparison against ADR-0018's own reported
  `0.044→0.253` figures; (b) `chosen` and `rejected` continuations in this
  dataset shape are not guaranteed equal length, and summing without
  normalization is the standard DPO-reward convention this project's own
  training run already used, not an invented alternative. The
  length-normalized **mean** margin is reported as a secondary figure
  specifically to check whether any observed summed-margin shift is a real
  per-token effect or an artifact of a length imbalance between `chosen`
  and `rejected` on the constructed set — if the two diverge in direction,
  that divergence itself is reported as a finding, not resolved by picking
  whichever number looks better.
- **Paired statistic:** for each of the 20 pairs, compute
  `delta_i = margin_candidate_i − margin_reference_i` (one signed value per
  pair). Report: (a) the mean and standard deviation of `delta_i` across
  the 20 pairs; (b) a **Wilcoxon signed-rank test** on `delta_i` against a
  null of zero median shift (chosen over a paired t-test because `n=20` is
  small and this document makes no advance claim that `delta_i` is
  normally distributed — the same posture as not assuming a decode-level
  effect exists at all); (c) the same test and summary statistic computed
  separately for the `refusal`-direction and `counter`-direction subsets,
  reported as secondary breakdowns (not a stopping rule — see below), to
  check whether any aggregate shift is one-directional or a genuine
  bidirectional calibration effect.
- **Minimal-effect threshold, fixed in advance, chosen from cited evidence
  (not a post-hoc number):** `θ = 0.843` **summed-log-prob nats**
  (`|delta_i|` scale — i.e. the same raw-sum units section 3's primary
  quantity is already defined in above). Derivation: TRL's `DPOTrainer`
  reports `rewards/margins` as
  `mean(beta * (logp_policy(chosen) − logp_ref(chosen)) − beta *
  (logp_policy(rejected) − logp_ref(rejected)))`, which is algebraically
  `beta * mean(margin_policy_i − margin_ref_i)` over the batch it is
  computed on — i.e. exactly `beta * mean(delta_i)` in this document's own
  notation, just measured on the **training** pairs instead of the
  held-out pairs. ADR-0018's outcome record (re-verified above, `main` at
  `fdac4652`) reports this run's own final `rewards/margins` value as
  `0.253` (step 20) at `beta=0.3` (ADR-0018's pinned execution-config
  value). Solving for the training-pair-level `mean(delta_i)` that
  produced that reported figure: `0.253 / 0.3 = 0.8433...`, rounded to
  `0.843` nats. This is not an invented number: it is the exact raw
  summed-log-prob-margin shift ADR-0018's own training run is already
  reported to have produced on the pairs it trained on. Using it as the
  held-out minimal-effect threshold operationalises the question this
  document exists to ask (section 1) precisely: *did the held-out shift
  reach the same order of magnitude the training-set shift already did*,
  or is it categorically smaller/absent. `θ` applies symmetrically in the
  wrong-direction case (`-θ`).
- **Effect-size interval, fixed in advance:** the **Hodges–Lehmann
  estimator** of the median of `delta_i` (the median of all pairwise
  Walsh averages `(delta_i + delta_j)/2`, the same estimator that is the
  natural point-estimate companion to the Wilcoxon signed-rank test this
  document already commits to computing) with a **95% confidence interval
  via percentile bootstrap** (10,000 resamples of the 20 `delta_i` values,
  with replacement; the same bootstrap-CI mechanism Maya's review names as
  an acceptable alternative to the Hodges–Lehmann estimate is used here to
  build the interval *around* that estimate, not as a separate competing
  method). Denote the interval `[lo, hi]`.
- **Decision rule, fixed in advance (95% CI vs. `±θ`, not adjusted post
  hoc) — four mutually exclusive, exhaustive outcomes covering every
  possible position of `[lo, hi]` relative to `−θ` and `+θ`:**
  1. **Shift present, supports hypothesis (b)/(c):** `lo > θ` — the entire
     95% CI lies above the minimal-effect threshold in the
     training-consistent direction. A held-out probability-space shift at
     least as large as the training-set shift exists and decode-level
     scoring did not surface it.
  2. **Wrong-direction shift:** `hi < −θ` — the entire 95% CI lies below
     `−θ`, a meaningful shift **opposite** the training signal's
     direction. Reported as its own outcome, not folded into "no shift of
     training-set magnitude": this is evidence the training signal
     actively moved held-out margins the wrong way, a materially
     different finding from no generalisation at all.
  3. **No shift of training-set magnitude:** `−θ ≤ lo` and `hi ≤ θ` — the
     entire 95% CI is contained within `[−θ, θ]`. At this sample's power,
     an effect at least as large as the pre-specified minimal-effect
     threshold can be positively ruled out in either direction; a smaller
     real held-out shift below `θ` is not addressed by this outcome and
     remains possible.
  4. **Ambiguous / underpowered:** none of the above — the CI straddles at
     least one of `−θ`/`+θ` without being entirely on one side, so this
     measurement can neither confirm nor rule out an effect of at least
     magnitude `θ`. This is the *only* condition under which "ambiguous"
     may be reported, and it is now defined purely in terms of the CI
     boundary test above, so it cannot overlap with outcome 3 — section 4's
     table maps one row to each of these four outcomes exactly.
  - The Wilcoxon signed-rank test and its p-value (defined above) are
    still computed and reported for every one of these four outcomes, as
    corroborating evidence of statistical significance vs. the
    zero-shift null, but the **CI-vs-`θ` rule above, not the p-value, is
    what selects the outcome** — this removes the previous overlap between
    a `p≥0.05` result and an undefined "ambiguous" category by making
    "ambiguous" a positive, prespecified condition on the CI rather than a
    leftover label applied after seeing the data.
  - All four outcomes, `θ`, and the CI methodology are fixed **before**
    the 20-pair set exists or is scored; no threshold or method in this
    section may be adjusted after seeing the data.
- **Statistical power, stated honestly:** at `n=20` paired observations, a
  Wilcoxon signed-rank test at α=0.05 two-sided has roughly 80% power to
  detect only a **large** paired effect (conventionally, a paired Cohen's
  d around 0.65–0.75 or higher, depending on the true distribution shape),
  and a percentile-bootstrap CI at this same `n` is correspondingly wide;
  both are materially underpowered to detect a small-to-medium shift with
  confidence. This document does not inflate `n` to manufacture power it
  does not have, and does not treat outcome 3 ("no shift of training-set
  magnitude") as evidence of a true zero effect beyond the specific claim
  it makes: that an effect of magnitude `≥θ` is ruled out at this sample's
  power, while a smaller real held-out shift below `θ` remains possible.
  Outcome 4 ("ambiguous / underpowered") exists precisely to hold the
  honest middle case — a result this sample size cannot resolve either
  way — rather than rounding it into outcome 1 or outcome 3.

## 4. What each outcome implies for the next decision

| Outcome (per section 3's four mutually exclusive, exhaustive outcomes) | Implication for the next decision |
|---|---|
| **1. Shift present** (`lo > θ`: 95% CI entirely above the minimal-effect threshold, training-consistent direction) | The training signal partially generalises to held-out data in probability space even though it does not move greedy decoding. This shifts the live hypothesis space toward (a) conservative `beta`/`lr` and/or (b) a log-prob-vs-greedy-decode mismatch (ADR-0018's hypotheses a/b) — i.e. the mechanism is real but too weak, or invisible to argmax decoding at this dataset size/step count. Next step (separate, not authorized here): a fresh execution-config ADR proposing a different `beta`/`lr`/step-count combination or a decode-sensitive scoring method (e.g. sampling-based decode diversity), justified by *this* measurement, not by re-running the same config. |
| **2. Wrong-direction shift** (`hi < −θ`: 95% CI entirely below `−θ`) | The training signal moved held-out margins meaningfully in the direction **opposite** the training-set shift — a materially different, more concerning finding than simple non-generalisation. Next step (separate, not authorized here): treat this as a priority investigation, not a routine follow-up — before any further training-config change, an ADR-scoped root-cause review of whether the training pairs themselves (or the loss/beta interaction) are actively degrading held-out calibration, independent of the greedy-decode-level metric. |
| **3. No shift of training-set magnitude** (`−θ ≤ lo` and `hi ≤ θ`: 95% CI entirely within `[−θ, θ]`) | An effect at least as large as the training-set shift (`θ`) is positively ruled out in held-out probability space; a smaller real held-out shift is still possible and is not addressed by this outcome. This result does not, by itself, decide between hypothesis (b) (log-prob-vs-greedy-decode mismatch) and hypothesis (c) (22-record training package too small to produce a generalising effect at any magnitude) — hypothesis (b) stays live for any sub-`θ` shift this measurement cannot detect. Next step (separate, not authorized here): scaling the training-pair count is the leading hypothesis to pursue first (still per this project's own "no in-run sweep, no unrecorded retry" discipline, its own fresh ADR), but it is not the only next step — a decode-sensitive scoring method or a re-check of hypothesis (b) against this specific result remains a live alternative if scaling is not pursued or does not resolve the question. |
| **4. Ambiguous / underpowered** (CI straddles `−θ` or `+θ` without lying entirely on one side) | Neither hypothesis is meaningfully favoured over the other, nor is an effect of magnitude `≥θ` ruled out, by this measurement alone. Report the CI, the point estimate, and the direction honestly; the next step (separate, not authorized here) is a power discussion — whether a larger held-out set is worth the added construction/contamination-audit cost before spending another training cycle, rather than picking a hypothesis on underpowered evidence. |

## 5. Execution envelope

- **Read-only on the checkpoint:** the evaluation loads
  `AutoModelForCausalLM.from_pretrained(...)` on the existing candidate and
  reference checkpoints exactly as `hf_local_evaluator_adapter.py` already
  does for every prior real cycle's scoring pass, under `torch.no_grad()`
  throughout, `model.eval()`, no `.train()` call, no optimizer, no gradient
  computation, no write to either checkpoint directory. Nothing about this
  evaluation trains, fine-tunes, or modifies model weights in any way.
- **Offline:** `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` forced before
  any model load, identical in mechanism to `_get_model`'s existing defence
  in depth (`hf_local_evaluator_adapter.py`, already re-read above) — no
  network access to any model/dataset hub regardless of ambient environment
  variables.
- **Containment:** this evaluation runs **inference only** (no subprocess
  training, no `trl.DPOTrainer`), so it is scoped to the existing
  `evaluator-process-containment-v1` profile alone — the same containment
  scope every prior real cycle's baseline/candidate evaluation phase has
  already run under (ADR-0013 through ADR-0018), not the full
  `complete-cycle-host-containment-v1` training-cycle profile, since no
  training subprocess exists in this evaluation's execution path. This is
  a narrower containment scope than a training cycle needs, chosen because
  the evaluation's actual capability surface (load two checkpoints
  read-only, run forward passes, write a JSON results file to a reviewed
  scratch path) is narrower than a training cycle's, not because this
  evaluation is being treated as lower-risk by assumption.
- **Compute budget:** bounded and modest relative to any prior training
  cycle — one forward pass per (model × pair × {chosen, rejected}) = 2
  models × 20 pairs × 2 continuations = 80 forward passes total on a 135M
  parameter model, no backward pass, no optimizer step. Proposed ceiling
  (to be set exactly by the follow-up execution card, not this document):
  `max_wall_seconds` well under ADR-0018's `1800`-second training ceiling
  (this is 80 forward passes, not a 22-step training loop); `max_memory_mb`
  at or below ADR-0018's own measured **evaluation-phase** peak of
  `1,052.875` MB (all three of ADR-0018's own baseline/candidate evaluation
  phases stayed at or under that figure, per that execution's own reported
  measurement) — this evaluation's own execution card must independently
  re-measure and confirm, not assume, this ceiling holds for 20 pairs
  instead of ADR-0018's 48+10+15-item suites.
- **Evidence path:** a fresh, reviewed scratch root
  (`./local-evidence/adr0019/`, mirroring every prior
  cycle's own reviewed-path convention), containing: the sealed, contamination-audited
  held-out pair set; per-pair raw `logP(chosen)`/`logP(rejected)` for both
  models (summed and mean); the computed `delta_i` per pair; the Wilcoxon
  test statistic and p-value for the full set and each direction subset;
  and a `sha256` of the results file for independent re-verification, the
  same evidence-hashing discipline every prior ADR-0013 through ADR-0018
  cycle has used.
- **Does the three-gate signing apply?** **Partially — one proportionate
  gate, not the full three-gate training-authorization pattern, with
  justification below; Maya reviews this specific choice before the
  follow-up execution card runs anything.**
  - This is **not** a training run: no gradient computation, no weight
    update, no new model artifact produced, no promotion path exercised or
    proposed (same "measurement-only, no promotion mechanism exists" fact
    every prior ADR-0013 through ADR-0018 cycle has stated about its own
    training runs, true here even more directly since there is no training
    at all). The full three-role signing pattern
    (`security-reviewer`/`dataset-rights-reviewer`/`project-owner`) that ADR-0013
    through ADR-0018 each required exists specifically to gate an
    irreversible, resource-consuming, weight-modifying action against a
    real trust root — none of those three properties apply here.
  - It **does** run real model-inference code (`transformers`/`torch`
    forward passes) against **new, not-yet-existing data** (the to-be-built
    20-pair held-out set) on this host, which is exactly the category this
    project's `AGENTS.md`/ADR precedent treats as needing independent
    review before execution, even without a training component — the
    precedent is the same reasoning `evaluator-process-containment-v1`
    itself already codifies for every baseline/candidate evaluation phase
    in every prior real cycle (Maya's security review of the containment
    profile already covers *how* an evaluation-only inference pass is
    contained; it does not by itself cover *what specific new data* is fed
    into it).
  - **Proposed proportionate gate:** a single **Gate — dataset admissibility
    and evaluation-scope review (Maya)**, covering: (a) the sealed 20-pair
    held-out set's contamination-audit results (section 2) and its
    counter-direction share; (b) confirmation that the evaluation script
    changes nothing about `hf_local_evaluator_adapter.py`'s existing,
    already-reviewed inference/containment path beyond reusing
    `_choice_log_likelihood`; (c) the compute-budget ceiling proposed above.
    **No `project-owner` promotion-authorization gate is proposed**, because
    this evaluation has no promotion path to authorize (identical fact
    to every prior cycle's own `promotion_decision: null` design) — Rook's
    role here is the ordinary PR-merge/task-acceptance authority this
    document already operates under, not a fourth signed document. **No
    `security-reviewer` host-containment gate is proposed as a *separate*
    signature**, because the containment profile being reused
    (`evaluator-process-containment-v1`) is already independently
    security-approved and unmodified by this proposal — re-signing an
    unmodified, already-approved containment profile for every new use
    would be gate-fatigue without a corresponding safety gain; the single
    proposed gate folds the "is this containment profile still the right
    one for this new use" question into the same dataset-admissibility
    review Maya performs, rather than manufacturing a duplicate signature
    over an artifact that has not changed. This is a proposal, not a
    self-certification — **Maya makes the final call on whether one gate
    is sufficient or whether this in fact warrants the full three-role
    pattern**, and no execution proceeds until she has reviewed this exact
    section against the actual held-out set and evaluation script that
    will exist by then.
  - **Maya's binding condition on this gate (recorded here per her
    CHANGES_REQUESTED review of this PR, `PRR_kwDOUT4pss8AAAABOz7iAg`):**
    the eventual single gate (follow-up card 3) must be issued as a real
    **signed artifact**, mirroring the form of the ADR-0018 Gate 1/Gate 2
    documents already signed for this project, and it must be bound to:
    (a) the **exact content hash** of the evaluation script that will
    exist by the time the gate is reviewed (not a description of the
    script — the literal hash of the file executed); (b) the **sealed
    `HeldOutExclusionRegistry` entry** for the 20-pair set (section 2),
    i.e. the registry's own record of the sealed set, not a snapshot taken
    before sealing; and (c) an **independently re-measured** compute
    ceiling for this evaluation's actual `n=20` shape, not the
    `1,052.875` MB figure inherited by reference from ADR-0018's
    differently-sized (48+10+15-item) evaluation suites above — the
    follow-up execution card must measure this evaluation's own peak
    memory/wall-time freshly and the gate must be bound to that
    independently-measured figure, not to ADR-0018's. This condition is
    binding on follow-up card 3 (Maya's gate review) and card 4 (the
    single execution); it does not block merging this document.

## 6. Explicitly out of scope

- **Any new training run**, of any kind, on any checkpoint, at any
  hyperparameter setting. This document evaluates the checkpoint ADR-0018
  already produced; it does not train a new one.
- **Any hyperparameter change** to `beta`, `learning_rate`, `max_steps`, or
  any other ADR-0018 execution-config value — those remain exactly as
  ADR-0018 recorded them; this document does not revisit or re-litigate
  them, and any future change to them requires its own fresh
  execution-config ADR, per this project's standing "no in-run sweep, no
  unrecorded retry" discipline.
- **Promotion, deployment, or any change to a threshold, scorer, or dataset
  hash** anywhere in this project.
- **Rewriting or reinterpreting** `docs/decisions/ADR-0018-outcome.md` or
  any other prior ADR's own text — each remains its own authoritative
  record; this document only proposes a *follow-up measurement*, per
  ADR-0018's own "named here as a recommendation only" framing of exactly
  this next step.

## Proposed follow-up cards (not created by this document)

1. **Build and seal the held-out preference-pair set** (section 2) —
   author 20 new pairs, register via `HeldOutExclusionRegistry`, report
   counter-direction share. Suggested owner: **Marcus** (backend/
   dataset-construction depth, continuing this same evaluation thread).
   The five-part contamination audit (section 2) is performed separately
   by **Maya**, not by Marcus, per the owner decision recorded in section
   2 — Marcus builds and proposes the set; Maya independently audits and
   seals it.
2. **Write the held-out log-prob margin evaluation script** (section 3),
   reusing `hf_local_evaluator_adapter.py`'s existing
   `_choice_log_likelihood` rather than reimplementing it, computing both
   summed and mean margins, the Hodges–Lehmann/bootstrap-CI decision rule
   against `θ = 0.843` nats, the Wilcoxon corroborating statistic, and the
   full/subset breakdowns — gated, no execution, mirroring every prior
   `run_bounded_cycle_adrXXXX.py`'s own "write-only, no execution" PR
   pattern. Suggested owner: **Marcus**.
3. **Maya's gate review** of section 5's proposed proportionate gate
   (against the real script and sealed dataset from cards 1–2) — approve
   the one-gate proposal, require the full three-role pattern instead, or
   require a different scope; issue it as the signed artifact bound to the
   eval-script hash, the sealed registry entry, and the independently
   re-measured `n=20` compute ceiling, per Maya's binding condition
   recorded in section 5. Suggested owner: **Maya**.
4. **Execute the evaluation exactly once** (read-only, offline, per
   section 5's envelope, once card 3's gate clears) and report the result
   against section 3's pre-fixed CI-vs-`θ` decision rule and section 4's
   four-outcome table, honestly, including if the result lands in the
   "ambiguous / underpowered" row. Owner decision (Rook, accepting Maya's
   review of this PR): the single execution must be run by a **different
   specialist than the pair author of card 1** — Rook will assign the
   executor at that time — so the person who built the held-out set does
   not also control the one scoring pass measured against it. Same
   "report a negative/ambiguous result as a valid result" discipline every
   prior ADR-0013 through ADR-0018 execution card has already used.

## Documentation impact and maintenance triggers

This document adds `docs/decisions/ADR-0019-held-out-logprob-margin-eval.md`
(this file). It does not modify `docs/decisions/ADR-0018-outcome.md`,
`docs/decisions/ADR-0018-dpo-execution-config.md`,
`examples/pilot-metatrainer-v3-dpo/`, `src/codevolt_mdf/dpo_adapter.py`,
`src/codevolt_mdf/hf_local_evaluator_adapter.py`, or any prior ADR's own
text or locked hash.

Re-open or supersede this document before the evaluation runs if any of the
following changes: the candidate/reference checkpoint's content hash or
path; `hf_local_evaluator_adapter.py`'s `_choice_log_likelihood`
implementation or contract version; the chosen statistic, decision-rule
thresholds, or sample size in section 3; the proposed gate scope in section
5 (Maya's review is authoritative here, not this document's own proposal);
or the held-out set's construction rules in section 2. After the one
evaluation runs (once a script exists, a gate clears, and it executes),
append only verified result/evidence references to a companion outcome
record, `ADR-0019-outcome.md`, mirroring
`ADR-0015-outcome.md`/`ADR-0016-outcome.md`/`ADR-0018-outcome.md`. Do not
rewrite this document as if a predicted result had occurred.
