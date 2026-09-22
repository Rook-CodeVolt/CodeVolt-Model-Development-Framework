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
- [done] Second real trainer adapter, a structurally different engine shape — `MiniMindTrainerAdapter` (`src/codevolt_mdf/minimind_adapter.py`, ADR-0010, issue #36), a real `TrainerAdapterV1` wrapping [MiniMind](https://github.com/jingyaogong/minimind)'s `trainer/train_full_sft.py` as a subprocess (MiniMind has no importable Python API, so the pin is verified via `git rev-parse` against the caller's checkout rather than an installed-package version). Contract-tested (37 conformance tests, local git-repo fixtures and mocked subprocess calls only). Now exercised by a real, evaluated bounded pilot (`artifact-adr0011-pilot-20260920`, 2026-09-20, `aggregate_score=0.0` due to severe undertraining) — see [docs/REAL_ADAPTERS.md](REAL_ADAPTERS.md) and `docs/decisions/0011-minimind-bounded-pilot-plan.md`.
- [done, rejected] ADR-0013's one authorized pretrained meta-trainer SFT cycle ([ADR-0013](decisions/0013-bounded-pretrained-metatrainer-cycle.md), issue #79) ran end to end (`adr0013-metatrainer-sft-20260920`, commit `13b316e`): pipeline, security gates, and evaluation all worked correctly, but the candidate scored worse than baseline on all three suites (meta_trainer 0.0%->0.0%, capability_retention 30.0%->0.0%, safety 46.7%->33.3%), matching ADR-0013's own predeclared "stop and reject" trigger — classic full-SFT overfitting on 40 examples. Evidence preserved; no promotion, retry, or extra epoch performed under ADR-0013 itself.
- Next queued decision: a corrected bounded meta-trainer SFT cycle is proposed (not authorized) at [ADR-0014](decisions/0014-corrected-bounded-metatrainer-cycle.md), tracked by issue #85, citing ADR-0013's rejection above as its evidence basis. It reuses ADR-0013's model, corpus, evaluator, contamination machinery, and resource budget unchanged, correcting only `max_steps` (120->40, one epoch instead of three) and `learning_rate` (1e-5->5e-6) under a fresh run id and approval namespace. Live execution remains blocked on independent dataset rights/privacy admission, Maya's exact-candidate security review, and owner confirmation for this one run; no automatic retry, promotion, deployment, or next cycle is authorized.
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
