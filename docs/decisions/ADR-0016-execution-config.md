# ADR-0016-execution-config: hyperparameter selection for the ADR-0016 bounded execution runner

- Status: execution-config note, not itself an ADR — records the reasoning
  behind `examples/pilot-metatrainer-v2/run_bounded_cycle_adr0016.py`'s
  hyperparameter choices for the ADR-0016 corpus v3
  confabulated_recipe_detection rewrite bounded cycle
- Date: 2026-09-22
- Tracking: internal execution records (see linked PR numbers above where applicable)
- Scope: this note only explains the training-hyperparameter deltas
  (`MAX_STEPS`, `SAVE_STEPS`) versus `run_bounded_cycle_adr0015.py`. It does
  not authorize a training run by itself — the runner remains fail-closed
  pending the three-gate signed approval described in this project's
  standing ADR-0013/0014/0015 execution-gating convention, exactly as
  ADR-0015's runner was.

## Background: why this cycle exists

Maya's ADR-0015 review found that held-out item `mtr-v2-heldout-0013`
(family `synthetic_data_tradeoffs`, tests whether the model correctly
refuses to invent a plausible-sounding numeric training recipe) got *worse*
in the ADR-0015 candidate than in ADR-0014's, despite corpus v3 adding 8
dedicated `confabulated_recipe_detection` training examples specifically
targeting that failure family. A follow-up diagnosis identified the root cause
directly from the artifacts: those 8 examples were long (512-724 char),
third-person meta-commentary about the historical ADR-0014 incident — none
rehearsed the held-out item's own short, first-person, closed yes/no
"Can X? -> No, because..." shape. This plausibly reinforced confident,
discursive prose rather than terse refusal, consistent with the observed
regression (the ADR-0015 candidate went from merely *demonstrating* the
fabrication to explicitly *endorsing* it as normative policy).

PR #93 (task, merged as commit
`16b0108291f549d5e5f6b0e1f409a65c3d6fdd92`, Maya-reviewed and approved,
independently re-verified by Rook) rewrote the `confabulated_recipe_detection`
family: kept 2 of the original 8 records as accurate scaffolding/context,
removed the other 6 (pure retrospective third-person analysis), and added 9
new short, first-person, closed-question-shaped records directly mirroring
the held-out item's own prompt/answer shape (varying the invented content
class: numeric recipe, cost figure, date, percentage, citation, GPU-memory
figure, rounded percentage, citation locator, benchmark score). Train/held-out
split boundaries were verified untouched: `mtr-v2-heldout-0013` stays
`held_out`, `confabulated_recipe_detection` stays train-only
(`validate_dataset.py`'s `semantic_family_disjoint` check, 16/16 checks PASS).

Net corpus v3 change: `confabulated_recipe_detection` family 8 -> 11 records;
corpus totals 160 -> 163 (train 112 -> 115, held-out unchanged at 48).

## Inputs carried forward unchanged from ADR-0015

- `LEARNING_RATE = 5e-6` — ADR-0014's corrected value, unchanged through
  ADR-0015 and this cycle. This cycle changes exactly one variable (corpus
  content, specifically the `confabulated_recipe_detection` family) relative
  to ADR-0015's configuration, so any observed effect on item 0013 can be
  attributed to the corrected training examples rather than confounded with
  a simultaneous LR change.
- Resource budget: `max_memory_mb=16384` (raised from ADR-0013/0014's 8192 in
  PR #89 after ADR-0015's real run measured a 12,119.8MB training-phase peak,
  root-caused to legitimate MPS caching-allocator high-water-mark growth
  under full-parameter SFT), `max_wall_seconds=1800`, `max_cpu_seconds=3600`,
  `network_policy=offline`. Unchanged from ADR-0015 — no evidence motivates a
  different value; this cycle's corpus-size delta (112 -> 115, +2.7%) is far
  smaller than ADR-0015's own step-count-driving delta (40 -> 112) and the
  real measured peak already carries ~35% headroom under the 16384MB ceiling.
- `BATCH_SIZE = 1` (`per_device_train_batch_size`),
  `gradient_accumulation_steps=1` — unchanged. One training step consumes
  exactly one training example.

## MAX_STEPS: 112 -> 115

Following the same "one step per training example, one epoch per corpus
size" invariant ADR-0014 established as safe and ADR-0015 re-confirmed at a
larger corpus size: `MAX_STEPS = 115`, exactly one full pass over the
corpus v3 train split as rewritten by PR #93 (115 examples), at
`batch_size=1`/no accumulation. This is a much smaller step-count delta than
ADR-0015's own (40 -> 112, i.e. 2.8x) because the PR #93 corpus change is a
targeted rewrite of one family (net +3 records: -6, +9, kept 2), not a
corpus-wide expansion. There is no basis to depart from the one-epoch
invariant for a corpus-size change this small, and reverting to multi-epoch
training would reintroduce exactly the overfitting/degeneration mechanism
ADR-0013's evidence showed and ADR-0014 corrected.

## SAVE_STEPS: 56 -> 58

ADR-0015 set `SAVE_STEPS = 56`, 50% of its `MAX_STEPS = 112` — one
checkpoint at the run's midpoint plus the final checkpoint
(`SAVE_TOTAL_LIMIT` stays at 2, unchanged). This script keeps the same
fraction of total steps (0.5) rather than an absolute step count:
`round(0.5 * 115) = 58`.

## Everything else

`SAVE_TOTAL_LIMIT=2`, `MAX_NEW_TOKENS=160`, `SEED=20260922`, `MODEL_REPO`,
`MODEL_REVISION`, `EXPECTED_MODEL_HASH`, `EXPECTED_RENDERER_ID`,
`EXPECTED_VERSIONS`, and the `capability_retention`/`safety` evaluation
suites are unchanged from ADR-0015 — none of these depend on corpus content,
and no evidence basis exists to justify a change to any of them as part of
this mechanical adaptation. The dataset file hashes in
`EXPECTED_FILE_HASHES` are updated to the PR #93-rewritten corpus v3
content (independently recomputed via `shasum` against the checked-out repo
at commit `16b0108291f549d5e5f6b0e1f409a65c3d6fdd92`, matching
`examples/pilot-metatrainer-v3/VALIDATION_REPORT.json`'s recorded `sha256`
block).
