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
- [partially done] Evaluation suites covering capability, safety, and regression — a real (non-fake) `EvaluatorAdapterV1` now exists (`src/codevolt_mdf/hf_local_evaluator_adapter.py`), scoring held-out examples via local HF inference with bidirectional contamination checking against the training data. It now covers three task-type modes (exact-match, multiple-choice log-likelihood, format-conformance — ADR-0007, issue #24 WP-A), each independently contract-tested. Safety/red-team probing (WP-B) and regression tracking across candidate revisions (WP-C) — both named in this line's original scope — are still not implemented; see [docs/PROGRESSION.md](PROGRESSION.md) for the full priority framing and issue #24 for the remaining work packages.

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
