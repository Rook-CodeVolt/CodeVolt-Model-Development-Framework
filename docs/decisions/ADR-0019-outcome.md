# ADR-0019 outcome record: the held-out log-probability margin shifted in the trained direction, real but sub-θ

- Status: outcome record — documents real, executed results only; grants no
  new authority and does not itself propose or authorize any further run
- Date: 2026-09-23
- Tracking: execution run `adr0019-logprob-margin-eval-20260923`; upstream
  `docs/decisions/ADR-0019-held-out-logprob-margin-eval.md` (PR #102, this
  evaluation's preregistered question, metric, statistic, decision rule,
  and gate scope); `examples/pilot-metatrainer-v2/run_adr0019_logprob_margin_eval.py`
  (PR #103, card 2; hardened by PR #106, card 2's isolated-process
  resource-enforcement fix); `examples/pilot-metatrainer-v3-dpo-heldout/`
  (PR #104 rename, PR #105 hash-pin — the sealed, contamination-audited
  20-pair held-out set); `docs/decisions/ADR-0018-outcome.md` (PR #101) —
  the negative decode-level verdict this evaluation follows up on.
- Scope: this record closes out ADR-0019
  (`docs/decisions/ADR-0019-held-out-logprob-margin-eval.md`, PR #102) with
  the real executed result. It does not reopen, edit, or reinterpret
  `docs/decisions/ADR-0018-outcome.md`'s own text or verdict, and it does
  not propose or authorize any new training run; each prior ADR remains the
  authoritative record of its own run.

## Why this cycle exists

`docs/decisions/ADR-0018-outcome.md` recorded a **negative** decode-level
verdict for the first governed DPO cycle: training completed cleanly with
internal loss/reward-margin metrics moving in the intended direction, but
held-out generations were byte-identical to baseline on 65 of 73 (89.0%)
evaluated items across three suites, with zero score-boundary crossings on
the 8 that differed. That record named three untested hypotheses for the
divergence between the moving training-internal signal and the
near-static decode-level signal: (a) `beta=0.3`/`lr=5e-7`/22 steps too
conservative to move greedy-decoded output; (b) DPO's log-probability
training signal can shift without changing which token greedy decoding
selects first; (c) the 22-record training package too small to produce a
decode-level effect in this few steps.
`docs/decisions/ADR-0019-held-out-logprob-margin-eval.md` preregistered a
narrower question before any measurement was taken: does the ADR-0018
candidate's held-out chosen-minus-rejected log-probability margin show a
measurable shift in the trained direction, at or above the exact magnitude
(`θ = 0.843` summed-log-prob nats) the training run's own internal
`rewards/margins` metric already reached on the training pairs — a
probability-space trace that decode-level scoring, by construction, cannot
see either way.

## Execution history, recorded factually

Two earlier attempts to run this evaluation from ordinary execution-worker
confinement (tracked separately as a platform defect) stopped before any
model load: nested `sandbox-exec` is denied inside worker-level
confinement, so neither attempt produced a result file of any kind. The
evaluation that produced the result analyzed in this record was executed
from an owner-session executor under the identical containment profile
(`evaluator-process-containment-v1`), the identical signed gate, and the
identical sealed-dataset manifest that either worker attempt would have
used had the nested-sandbox restriction not applied. Exactly one
`--execute` invocation of the merged evaluation script ever reached the
scoring stage.

## Independent re-verification performed for this record

- Result file `./local-evidence/adr0019/scratch/adr0019-logprob-margin-eval-20260923/adr0019_result.json`
  re-hashed directly: SHA-256
  `90821432f08e68cfb08958fd298667e1d78ec53132748f2a3a36f6906b3197b9`, matching
  the value recorded at hand-off, and matching the co-located
  `adr0019_result.json.sha256` file written alongside it.
- `plan.script_sha256` inside the result file
  (`fab608cc7ba1e08262991912ce268fb71b9914a5c1b590483cf9e6e40c346376`)
  re-verified byte-for-byte against
  `examples/pilot-metatrainer-v2/run_adr0019_logprob_margin_eval.py` on
  `main` at `660ae440f57ad3281a7501579baeff2a14d0abdf` (the exact commit
  this record's tracking section cites) — identical hash. The merged script
  is card 2 (PR #103) as hardened by PR #106's isolated-process
  resource-enforcement fix; nothing about the scoring/statistics logic
  differs from the reviewed version.
- `gate.held_out_registry_hash`
  (`b70dc244ee40b6513e85c0321250ec6c8c3e3c98b85f96d6dc0db91068bc5bb3`)
  re-verified byte-for-byte against
  `examples/pilot-metatrainer-v3-dpo-heldout/held_out_exclusion_registry.json`
  on the same `main` commit — identical hash, and the registry's own
  `package_held_out_ids` entry for `pilot-metatrainer-v3-dpo-heldout-adr0019`
  lists exactly 20 ids (16 `dpo-heldout-refusal-*`, 4
  `dpo-heldout-counter-*`), matching the result file's own `n=20`
  (16 refusal / 4 counter) split.
- `gate.held_out_pairs_hash` inherited by the result file's own gate block
  (`e48551cbbdf117b89b3e6a5a5d37c3345afd8ac161b08c288a747de96303b541`)
  re-verified byte-for-byte against
  `examples/pilot-metatrainer-v3-dpo-heldout/held_out_pairs.jsonl` on the
  same commit — identical hash; the file itself contains exactly 20 lines,
  4 tagged `"direction": "counter"` and 16 tagged `"direction": "refusal"`.
- `gate.approval.document_sha256`
  (`99f6ba93091e5beb2909c863fe6dda6ffd77a661af25ebd95407da2edebd2980`)
  re-verified byte-for-byte against the on-disk
  `approval_document.json` at the reviewed evidence root
  (`./local-evidence/adr0019/evidence/`) — identical hash;
  the document records `approver_id: security-reviewer`, `decision: approved`,
  `role: held-out-eval-gate`, `scope: [evaluator-process-containment-v1]`, and
  the same `script_sha256`/`held_out_registry_hash`/resource-ceiling values
  the result file's own `verified_approval` block repeats — a self-consistent,
  independently re-hashed chain from signed gate through to scored result.
- `isolation` block: `memory_mb_peak` `1538.03` MB against a `2400` MB
  ceiling; `wall_seconds` `7.63` against a `300` s ceiling; `cpu_seconds`
  `13.33`. Well inside budget on both axes, and this evaluation's own
  independently re-measured ceiling (per ADR-0019 section 5's binding
  condition), not inherited by reference from ADR-0018's differently-shaped
  suites.
- `plan.scoring_called: false` and `plan.execution_blockers: []` together
  with process exit `0` (per hand-off) confirm this was a real scoring pass
  through the pipeline's normal path, not a blocked or partial run.
- Model-identity hashes inside the result file re-checked against
  ADR-0019's own preregistered candidate/reference identity:
  `candidate_model_hash` `8e424bfb4a1c...390fd` (the ADR-0018 DPO output
  checkpoint) and `reference_model_hash` `43752b3f3989...0145c`
  (`HuggingFaceTB/SmolLM2-135M-Instruct`, the same immutable base
  ADR-0018 pinned) — both match the identities the preregistration document
  named, not a substituted pair.
- `statistics.length_normalization_divergence: false` re-read directly
  from the result file — the length-normalized (mean) margin does not
  diverge in direction from the primary summed-margin quantity; the shift
  reported below is not an artifact of a `chosen`/`rejected` length
  imbalance on the constructed set.

## The result, applying section 3's preregistered decision rule exactly as fixed in advance

**Full set (`n=20`, the authoritative measurement per ADR-0019 section
3):** 95% bootstrap CI (Hodges–Lehmann point estimate `0.679`) `[0.429,
0.787]`; `θ = 0.843`. Applying the four-outcome rule fixed in section 3
before any measurement: `−θ ≤ lo` (`−0.843 ≤ 0.429`) and `hi ≤ θ` (`0.787
≤ 0.843`) — both conditions of **outcome 3, "No shift of training-set
magnitude,"** are satisfied. This is the preregistered outcome, and it is
stated here literally, unrelabeled: **outcome 3 fires.** Corroborating
Wilcoxon signed-rank test: `p = 9.6e-5`, statistic `210` (the maximum
possible value at `n=20`, `n(n+1)/2`, confirming every one of the 20 paired
`delta_sum` values is positive — i.e. every held-out pair moved in the
trained direction). Mean `delta_sum` `0.619`; mean `delta_mean` (the
secondary, length-normalized figure) `0.0215`, with no direction
divergence between the two.

Per ADR-0019 section 3's own text defining this outcome: *"An effect at
least as large as the pre-specified minimal-effect threshold can be
positively ruled out in either direction; a smaller real held-out shift
below θ is not addressed by this outcome and remains possible."* That
second clause is not a hedge added after the fact — it was fixed in the
preregistration before this measurement existed, and this record now
reports, as a plain factual observation and not a reclassification of the
outcome, exactly what it leaves open:

- **Every one of the 20 held-out pairs' `delta_sum` was positive** (all 20
  candidate-minus-reference margins moved in the trained direction, none
  in the wrong direction), a result the Wilcoxon test's maximal statistic
  and `p < 0.001` support with high confidence.
- **The Hodges–Lehmann point estimate (`0.679`) sits at about 80.6% of
  `θ` (`0.843`)** — a substantial fraction of the training-set-magnitude
  shift, not a near-zero or noise-scale reading. The CI's upper bound
  (`0.787`) approaches but does not cross `θ`.
- In plain terms: this is a **real, consistent, statistically significant,
  sub-θ held-out probability-space shift in the trained direction** —
  something ADR-0018's decode-level (greedy-argmax) evaluation, by
  construction, could not surface, because it only asks which single token
  wins at each position, not by how much the log-probability margin
  between chosen and rejected moved underneath that comparison.

**Refusal subset (`n=16`, secondary, non-authoritative per ADR-0019
section 3, `authoritative: false` in the result file's own schema):** 95%
CI `[0.690, 0.844]`, classified `ambiguous_underpowered` because the
upper bound (`0.844`) exceeds `θ` (`0.843`) by roughly `0.001` — the CI
straddles `θ` by a margin far smaller than this bootstrap's own resampling
noise floor at `n=16`. This is reported honestly as the subset's
classification, but it must not be over-read: at this sample size and this
CI-edge distance from the threshold, "ambiguous" here is a knife-edge
artifact of `θ` and the CI's upper bound landing within roughly a
thousandth of each other, not evidence that the refusal-direction subset
behaves differently in kind from the full set. The full set (`n=20`,
authoritative) already answers the preregistered question; this subset
breakdown is offered, per ADR-0019 section 3, only to check whether the
aggregate shift is one-directional or bidirectional, not as a second,
competing verdict.

**Counter subset (`n=4`, secondary, non-authoritative):** 95% CI `[0.032,
0.046]`, classified `no_shift_of_training_set_magnitude`, well below `θ`
and well below the refusal subset's point estimate. Wilcoxon `p = 0.100`
(not significant at `n=4`, expected at this small a sample). This subset
shows a much smaller shift than the refusal subset, consistent with — but,
at `n=4`, nowhere near powered to confirm — a directionally asymmetric
effect. No conclusion is drawn from this subset alone beyond what the raw
numbers show.

No length-normalization divergence was found between the summed and mean
margin quantities on either the full set or either subset.

## Mapping to ADR-0019 section 4's outcome-3 row, and what it does to each hypothesis

ADR-0019 section 4 states outcome 3's implication in advance: *"An effect
at least as large as the training-set shift (θ) is positively ruled out in
held-out probability space; a smaller real held-out shift is still
possible and is not addressed by this outcome. This result does not, by
itself, decide between hypothesis (b) (log-prob-vs-greedy-decode mismatch)
and hypothesis (c) (22-record training package too small to produce a
generalising effect at any magnitude) — hypothesis (b) stays live for any
sub-θ shift this measurement cannot detect."* Applying that row to this
run's actual result, with every explanation below labeled explicitly as a
hypothesis, not a finding:

- **Hypothesis (b) — log-probability-vs-greedy-decode mismatch: strengthened.**
  This is the row's own predicted implication, and it now has direct
  support rather than remaining purely theoretical: a real, statistically
  significant, consistently-directioned held-out margin shift exists
  (all 20/20 pairs positive, `p < 0.001`, point estimate ~80% of `θ`)
  in exactly the probability space hypothesis (b) says training could move
  without flipping which token greedy decoding selects. ADR-0018's
  decode-level null and this run's probability-space positive are both
  real, independently measured facts from the same trained checkpoint;
  hypothesis (b) is the candidate explanation that predicts precisely this
  pairing, and this is the first measurement in this project's history that
  gives it evidence beyond being merely plausible. It is not proven —
  no experiment in this record manipulates decoding strategy directly to
  confirm that a sub-θ margin shift specifically fails to cross an argmax
  boundary — but it is meaningfully strengthened.
- **Hypothesis (c) — training package too small to generalise at any
  magnitude: weakened, not ruled out.** If the 22-record training package
  were too small to produce any generalising effect on held-out data,
  the most naive version of that hypothesis would predict a held-out
  margin shift indistinguishable from zero/noise. That is not what was
  measured: 20/20 pairs moved in the trained direction with high
  statistical significance and a point estimate at ~81% of the
  training-set magnitude. A dataset too small to generalise at all is a
  harder position to hold given this result, though it is not eliminated
  outright — a smaller-but-real effect is exactly the outcome-3 signature,
  and this record does not have a separate measurement that isolates
  dataset size as a variable from the other two hypotheses.
- **Hypothesis (a) — conservative beta/lr too weak to move decoding:
  weakly strengthened, least directly addressed.** This evaluation did not
  vary `beta` or `lr`; it measured probability-space movement at the
  hyperparameters ADR-0018 already used. A sub-θ but nonzero, statistically
  significant shift is consistent with "the signal is real but too weak at
  this beta/lr to cross the decode-level boundary," which is one way of
  restating hypothesis (a) in probability-space terms. It is the least
  directly tested of the three by this specific result, because this
  evaluation held beta/lr fixed rather than sweeping it.

None of the three hypotheses is confirmed or ruled out by this single
measurement; this record states the direction each moves and no more.

## What this record does and does not authorize

- This record authorizes no training run, no promotion, no deployment, no
  hyperparameter sweep, and no change to any threshold, scorer, gate, or
  ADR-0013 through ADR-0019 dataset hash.
- It does not reinterpret or retroactively edit `docs/decisions/ADR-0018-outcome.md`
  or `docs/decisions/ADR-0019-held-out-logprob-margin-eval.md`; both remain
  their own authoritative, evidence-preserved records. ADR-0018's own
  **NEGATIVE** decode-level verdict is unchanged by this record — this
  evaluation asked a narrower, different question (a held-out
  probability-space trace of the training signal) and its outcome-3 result
  does not overturn, soften, or partially reverse that decode-level
  verdict either way, exactly as ADR-0019 section 1 stated in advance it
  would not.
- Per the `AGENTS.md` required feedback loop, this finding (a preregistered
  outcome-3 result whose own margin, at ~81% of θ, sits closer to the
  threshold than a null result would, with all 20 pairs signed the same
  way) should also be recorded via the standard agent-feedback route
  (`.github/ISSUE_TEMPLATE/agent-feedback.yml`, category `improvement` or
  `evidence-gap`, at whoever's discretion holds that decision) if not
  already tracked there.

## Recommendation (security reviewer, recommendation only — not a decision; a separate reviewer gives an independent one in review; the owner decides)

Given that hypothesis (c) is weakened (a real, significant, consistently-
signed effect exists, arguing against "no generalising effect at any
magnitude") while hypothesis (b) is strengthened by direct evidence, my
recommendation for the single next step is to pursue a **decode-sensitive
scoring method** (e.g., sampling-based decode diversity or margin-aware
scoring rather than pure greedy-argmax) as a cheaper, non-training probe of
whether the already-existing ADR-0018 checkpoint's sub-θ probability shift
is close enough to an argmax boundary to be surfaced by a different, still
purely-evaluative measurement — before spending a fresh training cycle on
a hyperparameter or dataset-size change, since this evaluation's own result
does not by itself distinguish which of hypotheses (a)/(b)/(c) would most
benefit from that next training spend. This is one recommendation among
the live alternatives ADR-0019 section 4's outcome-3 row already named as
non-exclusive; it is not a decision and does not authorize any run.

## Documentation impact and maintenance triggers

This record adds `docs/decisions/ADR-0019-outcome.md` (this file) and
`examples/metatrainer-corpus-addition-adr0019-subtheta-heldout-shift-lesson/`
(draft, not-admitted case-study proposal). It does not modify
`docs/decisions/ADR-0018-outcome.md`,
`docs/decisions/ADR-0019-held-out-logprob-margin-eval.md`,
`examples/pilot-metatrainer-v3-dpo/`,
`examples/pilot-metatrainer-v3-dpo-heldout/`, or any prior ADR's own text
or locked hash.

Re-open or supersede this record if the result file's SHA-256 changes, the
merged evaluation script's SHA-256 changes, or the sealed held-out
registry/pairs hashes change. Any future training-config or scoring-method
follow-up (per the recommendation above or an independent one) requires
its own fresh ADR, per this project's standing "no in-run sweep, no
unrecorded retry" discipline; it is not authorized by this record.
