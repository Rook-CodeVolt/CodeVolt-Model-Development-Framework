# ADR-0008: Safety/red-team probe suite (`safety_probe` task type)

- Status: accepted
- Date: 2026-09-18

## Context

Issue #24 scoped evaluation-suite depth into WP-A (task-type breadth,
ADR-0007), WP-B (safety/red-team probing), and WP-C (regression
tracking, ADR-0009) as independently gateable work packages. This ADR
covers WP-B only.

`docs/ROADMAP.md`'s v0.2 line names "evaluation suites covering
capability, safety, and regression" as the target. WP-A closed the
capability-breadth gap; safety coverage remained entirely unaddressed —
nothing in the framework measured refusal-appropriateness,
harmful-instruction-compliance, or PII-leakage behaviour at all.

Issue #24 explicitly scoped this narrowly: "a fixed, versioned set of
held-out probes... producing a measurement — never a pass/fail
promotion decision... start narrow (10-20 probes, documented
provenance/licence) rather than importing a large third-party red-team
corpus wholesale."

## Decision

Add a fourth `TaskType`, `SAFETY_PROBE = "safety_probe"`, to the same
`EvaluatorAdapterV1` contract and the same concrete adapters WP-A used
— additive only, `CONTRACT_VERSION` unchanged, mirroring ADR-0007's
pattern exactly.

- The held-out example's `input` is a probe prompt string; `expected`
  is a spec dict naming a `probe_type`
  (`refusal_appropriateness` / `harmful_instruction_compliance` /
  `pii_leakage`), an `expect_refusal` bool (required for the first two
  types), and an optional/required `disallowed_patterns` list of
  synthetic marker substrings (required for `pii_leakage`).
- New shared, stdlib-only validation/scoring logic lives in the same
  `src/codevolt_mdf/scoring_modes.py` module WP-A introduced:
  `parse_safety_probe_input`, `parse_safety_probe_spec`, and
  `check_safety_probe` — a fixed `REFUSAL_MARKERS` lexicon plus a
  disallowed-substring leak check. Imported by both the real adapter
  (`hf_local_evaluator_adapter.py`, real greedy generation) and the
  fake adapter (`fake_evaluator_adapter.py`, deterministic artifact
  response lookup), so the fake adapter's test double genuinely
  exercises the same shape/validation/scoring rules the real adapter
  enforces.
- Scored via real greedy generation (`do_sample=False`), the exact same
  decoding path `exact_match`/`format_conformance` already use — no new
  inference mode.
- Goes through the same `HeldOutExclusionRegistry` bidirectional
  contamination check as every other task type, with no special-casing
  (verified by a dedicated test).
- **Measurement only.** `check_safety_probe` returns `(correct, detail)`
  exactly like every other scoring function in this module; nothing
  changes in `EvaluationOutput`'s shape, and no accept/reject/promote
  field exists anywhere this or any other task type could write to.
  Per `docs/ARCHITECTURE.md` core contracts #3 and #6, a safety-probe
  score is evidence for a separate, later governed acceptance decision,
  never an automatic gate.

### Probe dataset

A fixed, versioned 15-example suite ships at
`examples/safety-probes-wpb/held_out.json` (package id
`wpb-safety-probes-v1`): 5 `refusal_appropriateness`, 5
`harmful_instruction_compliance`, 5 `pii_leakage`. Every example is
100% synthetic/originally authored for this repository — no scraped or
third-party red-team corpus content. `harmful_instruction_compliance`
prompts reference entirely fictional named things (e.g.
"ZYNTHRA-9", "GuardNet-Synthetic"); `pii_leakage` prompts embed
obviously-synthetic PII (every value carries a literal `-syn`/`syn`
marker segment) directly in the prompt and check whether the model
echoes it back. Full provenance, licensing (Apache-2.0, matching the
rest of this repository), intended/prohibited use, and known
limitations are documented in
`examples/safety-probes-wpb/DATASET_CARD.md` per
`docs/DATA_GOVERNANCE.md`'s dataset-card requirement. This is a
deliberately narrow first suite, not a comprehensive safety benchmark —
documented explicitly as a known limitation, not implied otherwise.

## Testing

- `tests/test_scoring_modes.py`: unit tests for
  `parse_safety_probe_input`/`parse_safety_probe_spec`/
  `check_safety_probe` — accepted shapes, rejected shapes (non-string
  input, unknown `probe_type`, missing `disallowed_patterns` for
  `pii_leakage`, missing `expect_refusal` for the other two types,
  non-string pattern entries), and scoring behaviour for all three
  probe types (leak detected/not detected, refusal
  present/absent-vs-expected).
- `tests/test_evaluator_contract.py`: fake-adapter conformance tests —
  success (correct + incorrect scoring across all three probe types),
  missing-response handling, invalid `probe_type` rejection, non-string
  response rejection, and a dedicated contamination-check test proving
  `safety_probe` does not bypass `HeldOutExclusionRegistry`.
- `tests/test_hf_local_evaluator_adapter.py`: real-adapter tests
  against the pinned `HuggingFaceTB/SmolLM2-135M` checkpoint — scoring
  via real generation, determinism (identical repeated runs produce
  identical output/score), malformed-spec rejection before model load,
  and an end-to-end smoke test that loads the actual committed
  `examples/safety-probes-wpb/held_out.json` dataset (validates its
  content hash via `HeldOutSet.validate()`) and scores all 15 examples
  through the real adapter.
- Full suite: 193 passed, 5 skipped (pre-existing `trl`-not-installed
  skips, unrelated to this change).

## Out of scope

- Any threshold, aggregate verdict, or wiring into a promotion/
  accept-reject decision path — `check_safety_probe`'s boolean is a
  per-example measurement like every other task type's, nothing more.
- Expanding beyond the initial 15 examples or the three named probe
  types — explicitly deferred per issue #24's "start narrow" framing;
  a future dataset version (`wpb-safety-probes-v2`) would need its own
  provenance/licence review, per `docs/DATA_GOVERNANCE.md`.
- Importing any third-party red-team corpus.
