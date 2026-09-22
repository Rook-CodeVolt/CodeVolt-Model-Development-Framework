# ADR-0015-execution-config: hyperparameter selection for the ADR-0015 bounded execution runner

- Status: execution-config note, not itself an ADR — records the reasoning
  behind `examples/pilot-metatrainer-v2/run_bounded_cycle_adr0015.py`'s
  hyperparameter choices for the ADR-0015 corpus v3 bounded cycle
- Date: 2026-09-22
- Tracking:
- Scope: this note only explains the training-hyperparameter deltas
  (`MAX_STEPS`, `SAVE_STEPS`) versus `run_bounded_cycle_adr0014.py`. It does
  not authorize a training run by itself — the runner remains fail-closed
  pending the three-gate signed approval described in ADR-0015's own "Review
  path" section, exactly as ADR-0014's runner was.

## Inputs carried forward unchanged from ADR-0014

- `LEARNING_RATE = 5e-6` — ADR-0014's corrected value, which fixed the
  catastrophic-forgetting/degeneration regression seen in ADR-0013's real run
  (repetitive token-loop outputs, capability and safety score collapse).
  Per this task's explicit instruction, LR is held unchanged: this cycle
  changes exactly one variable (corpus) relative to ADR-0014's proven-safe
  configuration, so any observed effect can be attributed to the corpus
  change rather than confounded with a simultaneous LR change.
- Resource budget: `max_memory_mb=8192` (ADR-0013's corrected ceiling, raised
  from 4096 in commit 13b316e), `max_wall_seconds=1800`, `max_cpu_seconds=3600`,
  `network_policy=offline`. Unchanged — no evidence motivates a different
  budget for this corpus size, and the same OS-level Seatbelt sandbox,
  resource monitor, signature verification, and fail-closed `--execute` gate
  are reused with no security-logic changes.
- `BATCH_SIZE = 1` (`per_device_train_batch_size`), `gradient_accumulation_steps=1`
  (verified by `_runtime_config()`'s `expected_runtime` check) — unchanged.
  This means each training step consumes exactly one training example; there
  is no batching/accumulation multiplier to account for when relating
  "steps" to "examples" or "epochs".

## MAX_STEPS: 40 -> 112

ADR-0013 trained 3 epochs over its 40-example corpus v2 train split
(`max_steps=120`, `batch_size=1`, no accumulation → 120 steps = 3 passes over
40 examples) and produced the catastrophic-forgetting regression: zero-to-
negative results on all three held-out suites (`meta_trainer` 0.0%->0.0%,
`capability_retention` 30%->0.0%, `safety` 46.7%->33.3%) with raw candidate
outputs showing textbook overfitting/degeneration (repetitive token loops
like "46-46-46-..." and "PEPFET, a variant of PEPFET, ..."). ADR-0014's
correction was to cut this to `max_steps=40` — exactly one epoch over the
same 40-example split — and that one-epoch regime is what fixed the
regression (capability and safety retention held or nearly held in the real
ADR-0014 run).

This task's instruction was to scale `MAX_STEPS` "modestly" for the larger
112-example corpus v3 train split, reasoning about "~3 epochs at batch size
1" but explicitly leaving the exact step count to judgement, given that 40
steps was calibrated for the smaller corpus specifically.

A literal "~3 epochs over 112 examples" reading would give `max_steps=336`
(3 x 112), which is 8.4x ADR-0014's step count and would exactly reproduce
the three-full-passes, full-parameter-SFT, small-corpus regime that ADR-0013's
real run showed causes degenerate overfitting. Naively scaling the *epoch
count* would silently reintroduce the mechanism ADR-0014 corrected, not scale
with the corpus. The corpus growing from 40 to 112 examples (2.8x) is itself
a factor that should make overfitting somewhat less likely per step (more
distinct examples, less repeated exposure to the same small item set within
a pass), which argues against reintroducing multi-epoch training now, not for
it.

This script therefore scales the *step count within ADR-0014's proven one-
epoch regime* to match the new corpus size: `MAX_STEPS = 112`, i.e. exactly
one full pass over the 112-example corpus v3 train split, at
`batch_size=1`/no accumulation (1 step = 1 example, so 112 steps = 1 epoch
over 112 examples, the same "one epoch" invariant that ADR-0014 established
as safe, just re-measured against the new corpus size: 40 -> 112 steps
tracks 40 -> 112 examples 1:1). This is "modest" scaling in the sense that
asked for — the step count grows with the corpus (2.8x, matching the corpus
growth exactly) — while deliberately not scaling the epoch count, which is
the dimension ADR-0013's evidence specifically implicates.

If a future proposal wants to test multiple epochs over the corpus v3
89.6%-larger dataset, given the different (larger, more diverse,
calibrated-refusal-weighted) corpus composition, that should be its own
freshly gated execution proposal with its own explicit evidence basis, not a
silent side effect of this corpus-size adaptation.

## SAVE_STEPS: 20 -> 56

ADR-0014 set `SAVE_STEPS = 20`, exactly 50% of its `MAX_STEPS = 40` — one
checkpoint at the run's midpoint plus the final checkpoint (`SAVE_TOTAL_LIMIT`
stays at 2, unchanged). This script keeps the same fraction of total steps
(0.5) rather than an absolute step count, so the checkpoint cadence tracks
the new run length the same way: `0.5 * 112 = 56`.

## Everything else

`SAVE_TOTAL_LIMIT=2`, `MAX_NEW_TOKENS=160`, `SEED=20260922`, `MODEL_REPO`,
`MODEL_REVISION`, `EXPECTED_MODEL_HASH`, `EXPECTED_RENDERER_ID`,
`EXPECTED_VERSIONS`, and the `capability_retention`/`safety` evaluation
suites are unchanged from ADR-0014 — none of these depend on corpus size,
and no evidence basis exists to justify a change to any of them as part of
this mechanical adaptation.
