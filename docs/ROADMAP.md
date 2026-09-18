# Roadmap

The roadmap describes intent, not a promise of dates.

For an evidence-led view of where the next real effort should go and why — informed by external open-source ecosystem research rather than internal planning alone — see [Next progression](PROGRESSION.md).

## v0.1 — Foundation

- Versioned experiment contract and locked manifests.
- Deterministic reference adapters.
- Independent acceptance decision.
- Evidence and provenance bundle.
- Positive, rejection, and missing-evidence tests.
- Learning-event and proposal contracts.

## v0.2 — Observe and recommend

- Structured failure/correction intake with privacy review.
- Pattern classification and a human-review proposal queue.
- [done] First real trainer adapter selected through an evidence-backed decision — `TRLTrainerAdapter` (`src/codevolt_mdf/trl_adapter.py`, ADR-0005), a real `TrainerAdapterV1` for TRL's `SFTTrainer`, contract-tested (28 conformance tests). Not yet exercised by a live training run — see [docs/REAL_ADAPTERS.md](REAL_ADAPTERS.md).
- [done] Second real trainer adapter, a structurally different engine shape — `MiniMindTrainerAdapter` (`src/codevolt_mdf/minimind_adapter.py`, ADR-0010, issue #36), a real `TrainerAdapterV1` wrapping [MiniMind](https://github.com/jingyaogong/minimind)'s `trainer/train_full_sft.py` as a subprocess (MiniMind has no importable Python API, so the pin is verified via `git rev-parse` against the caller's checkout rather than an installed-package version). Contract-tested (37 conformance tests, local git-repo fixtures and mocked subprocess calls only). Not yet exercised by a live training run — see [docs/REAL_ADAPTERS.md](REAL_ADAPTERS.md).
- Next queued decision: a bounded MiniMind pilot, drafted (not authorized) at [docs/decisions/0011-minimind-bounded-pilot-plan.md](decisions/0011-minimind-bounded-pilot-plan.md) (status `proposed`, issue #46), alongside a companion iterative-feedback-loop design at [docs/CONTINUAL_IMPROVEMENT_LOOP_DESIGN.md](CONTINUAL_IMPROVEMENT_LOOP_DESIGN.md) describing how an authorized later round would consume an earlier round's evaluation evidence via the existing `regression_check.py` comparator. Neither document authorizes execution — the same three-gate sequence as ADR-0006 (ADR drafted, Maya's pilot-specific live-execution security review, owner authorization) still applies in full.
- [done] Evaluation suites covering capability, safety, and regression — a real (non-fake) `EvaluatorAdapterV1` now exists (`src/codevolt_mdf/hf_local_evaluator_adapter.py`), scoring held-out examples via local HF inference with bidirectional contamination checking against the training data. It covers four task-type modes: exact-match, multiple-choice log-likelihood, format-conformance (ADR-0007, issue #24 WP-A), and safety-probe scoring (`TaskType.SAFETY_PROBE`, ADR-0008, issue #24 WP-B — a fixed 15-example synthetic probe suite covering refusal-appropriateness/harmful-instruction-compliance/PII-leakage), each independently contract-tested and measurement-only. Regression tracking across candidate revisions is also real (`src/codevolt_mdf/regression_check.py`, ADR-0009, issue #24 WP-C — measurement-only comparison against baseline and previous-accepted candidate for previously-passing examples). All three work packages named in issue #24's original scope are implemented; see [docs/PROGRESSION.md](PROGRESSION.md) for the full priority framing.

## v0.3 — Bounded experimentation

- Approved proposals generate resource-bounded experiments.
- Hardware-aware scheduling and interruption recovery.
- Dataset lineage, licence policy, and contamination checks.
- Reproducible public reference improvement with rejected runs included.

## Later — Governed evolution

- A cohesive CodeVolt-native framework that selectively internalises proven capabilities from the adapter ecosystem where ownership creates measurable value.
- Narrowly scoped automatic experiment proposals.
- Sandboxed unattended runs with spend, time, thermal, and network limits.
- Promotion only through predefined evidence, approval, and rollback gates.
- Continuous framework capability review.

No stage permits a model to silently modify its own production weights, policies, evaluation criteria, or approval boundaries.
