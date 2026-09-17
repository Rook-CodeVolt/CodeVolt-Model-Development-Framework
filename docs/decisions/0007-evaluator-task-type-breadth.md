# ADR-0007: EvaluatorAdapterV1 task-type breadth (multiple-choice, format-conformance)

- Status: accepted
- Date: 2026-09-17

## Context

`docs/PROGRESSION.md`'s research pass (2026-09-17) names evaluation-suite
depth as this framework's highest-priority next investment, ahead of a
second trainer adapter or interoperability exports, because it is the
least-crowded and most defensible ground given this framework's
contamination-resistant, by-construction evaluation design. Issue #24
scoped that work into WP-A (task-type breadth), WP-B (safety/red-team
probing), and WP-C (regression tracking) as independently gateable work
packages. This ADR covers WP-A only.

Before this change, `hf_local_evaluator_adapter.py`'s `score_example`
supported exactly one scoring mode: exact-match/containment against a
greedy-decoded continuation. `docs/ROADMAP.md`'s v0.2 line names
"evaluation suites covering capability, safety, and regression" as the
target; a single exact-match scorer against one model family does not
meet that bar. Multiple-choice/likelihood scoring (the standard approach
`lm-evaluation-harness` and similar tools use for MMLU-style tasks) and
basic format/structured-output conformance checking are the two most
common task shapes exact-match alone cannot express.

## Decision

Add two additional scoring modes to the same `EvaluatorAdapterV1`
contract and the same concrete adapters, rather than introducing a new
contract or a parallel evaluator class:

- **`multiple_choice`**: the held-out example's metadata carries a
  prompt and a list of choices. Scored via teacher-forced,
  length-normalized per-option log-likelihood against the pinned
  checkpoint (own reimplementation, standard-library plus the
  `transformers`/`torch` this adapter already depends on — no new
  dependency added).
- **`format_conformance`**: the held-out example's metadata carries a
  JSON-schema-shaped or regex format spec. Scored via greedy generation
  (the same decoding path exact-match already uses) followed by a
  conformance check against that spec.

New shared, stdlib-only validation/scoring logic for both modes lives in
a new module, `src/codevolt_mdf/scoring_modes.py`, imported by both the
real adapter (`hf_local_evaluator_adapter.py`) and the fake adapter
(`fake_evaluator_adapter.py`), so the fake adapter's deterministic test
double genuinely exercises the same shape/validation rules the real
adapter enforces, not a separately-maintained approximation.

Mode selection is additive: a new `TaskType` enum
(`exact_match` / `multiple_choice` / `format_conformance`) and a
`task_type_of()` helper read an **optional** `"task_type"` key from
`HeldOutExample.metadata`, defaulting to `exact_match` when absent. No
existing dataclass field changed shape, no existing default changed
behavior, and every pre-existing exact-match example (real or in tests)
continues to score exactly as before with zero metadata changes
required. `CONTRACT_VERSION` therefore stays `"1.0.0"` — this is a
strictly additive extension of the v1 contract, not a breaking change.

Every new mode goes through the exact same pipeline the existing
exact-match mode does before it is allowed to score anything:

- The same checkpoint-hash verification (`_hash_model_dir`) the adapter
  already performs.
- The same `HeldOutExclusionRegistry.check_held_out_not_trained`
  bidirectional contamination check, called once per run regardless of
  which task types are present in the held-out set being scored — there
  is no code path where a `multiple_choice` or `format_conformance`
  example bypasses this check while an `exact_match` example would not.
- Input-shape validation now happens for every task type **before**
  any model load, matching the ordering the original exact-match path
  already used (this ADR's implementation corrected a regression where
  the new modes were validated after the model load, which would have
  broken the adapter's existing "reject malformed input cheaply, before
  paying for inference" behavior — fixed before merge, not shipped).

No method on either adapter gained accept/reject/promote authority.
`EvaluationOutput` gained no new field beyond what scoring already
produced; `aggregate_score` remains the only summary value, exactly as
`docs/EVALUATION_POLICY.md` and `docs/ARCHITECTURE.md` core contract
#3/#6 require. Role separation from `trl_adapter.py`/`trainer_contract.py`
is unchanged and still independently proven by the existing AST-import
test.

## What this does not do (explicitly out of scope, tracked separately)

- **WP-B (safety/red-team probing)** and **WP-C (regression tracking
  across candidate revisions)** from issue #24 are not attempted here.
  `docs/ROADMAP.md`'s v0.2 evaluation line remains `[partially done]`
  after this change — task-type breadth is wider, but capability +
  safety + regression coverage is still not complete.
- No new held-out dataset, probe set, or production example is added by
  this ADR; new task types are proven only against the same pinned
  `HuggingFaceTB/SmolLM2-135M` checkpoint and synthetic fixtures the
  existing test suite already uses.
- No change to the trainer adapter, ADR-0006's pilot, or any
  promotion/acceptance threshold.

## Evidence

- `tests/test_scoring_modes.py`: unit tests for the shared validation/
  scoring helpers, plus an AST-import test proving `scoring_modes.py`
  has no import relationship with `trl_adapter.py`/`trainer_contract.py`.
- `tests/test_evaluator_contract.py`: ~20 new fake-adapter conformance
  tests (success, missing-response, invalid-spec, unsupported-task-type
  cases), including an explicit test proving both new modes still go
  through `HeldOutExclusionRegistry` contamination rejection before
  scoring.
- `tests/test_hf_local_evaluator_adapter.py`: 10 new real-inference
  tests against the pinned checkpoint covering multiple-choice
  log-likelihood scoring, format-conformance generation scoring,
  determinism, and malformed-input/spec rejection for both modes.
- Full suite: 169 passed, 0 failed. `ruff check .` clean.
