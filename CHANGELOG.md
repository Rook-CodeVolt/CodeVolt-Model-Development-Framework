# Changelog

This project follows [Semantic Versioning](https://semver.org/) and keeps notable changes in this file.

## [Unreleased]

### Added

- Initial experiment, evidence, decision, and learning contracts.
- Deterministic reference adapters and CLI.
- Governance, contribution, security, support, and architectural documentation.
- Repository-level agent instructions and a structured agent-feedback route.
- Evaluation, data-governance, experiment, dataset, model-card, and release-readiness templates.
- Governed research-source admission, dataset-version contracts, and continual-improvement lifecycle.
- Agent resumption, backlog reconciliation, and experimental-repository migration work packages.
- TrainerAdapterV1 real-engine implementation for TRL (`TRLTrainerAdapter`), gated by ADR-0005; contract conformance tests reused wherever provable without a real training run. No real pilot run performed or authorised by this change.
- `TRLTrainerAdapter`'s TRL version bound pinned from a range to the exact verified version (`trl==0.24.0`), per ADR-0005's amendment addressing a medium finding in Maya's PR #15 review.
- `EvaluatorAdapterContract` v1 (`src/codevolt_mdf/evaluator_contract.py`), a separate, contract-tested independent-evaluation harness for issue #7 step 4, plus a deterministic `FakeEvaluatorAdapter` for its conformance tests. Not wired into `TRLTrainerAdapter` or `trainer_contract.py` (verified by a structural test).
- `HeldOutExclusionRegistry` (`src/codevolt_mdf/held_out_registry.py`), a bidirectional, per-package held-out/train exclusion registry reusing the design pattern from issue #11/`docs/DATA_GOVERNANCE.md`; used by the evaluator contract to reject a held-out set contaminated by any package's prior train usage before scoring.
- ADR-0006 (`docs/decisions/0006-bounded-real-trl-pilot-plan.md`): drafts the bounded real TRL pilot's exact configuration (model, dataset, concurrency, resource limits, gating). Does not authorise pilot execution.
- ADR-0013 (`docs/decisions/0013-bounded-pretrained-metatrainer-cycle.md`) and its gated runner/plan: propose one pretrained SmolLM2-135M-Instruct full-SFT meta-trainer cycle with immutable hashes, baseline-before-training, identical post-evaluation, semantic rubric, retention gates, resource ceilings, and exact-SHA review evidence. Validation-only mode is exercised; no training is performed or authorized by this change.
- ADR-0013 Gate 1 remediation: adds macOS Seatbelt enforcement for the complete live runner and descendants, exact reviewed host paths, a second mandatory security scope token, and role-specific signer-principal governance. The exact Python 3.9.6 environment passes validation-only with `training_called=false`; live execution remains blocked until holder-supplied public keys and independent approvals are admitted.
- ADR-0013's live cycle ran to completion (`adr0013-metatrainer-sft-20260920`, commit `13b316e`): the pipeline, security gates, and evaluation all worked correctly, and produced an honest negative result — candidate scored worse than baseline on all three suites (meta_trainer 0.0%->0.0%, capability_retention 30.0%->0.0%, safety 46.7%->33.3%), matching ADR-0013's own predeclared "stop and reject" trigger. Raw outputs show classic full-parameter-SFT overfitting/degeneration on the 40-example corpus. No promotion, retry, or in-place parameter change performed under ADR-0013.
- ADR-0014 (`docs/decisions/0014-corrected-bounded-metatrainer-cycle.md`) and its gated runner `examples/pilot-metatrainer-v2/run_bounded_cycle_adr0014.py`: propose a corrected bounded pretrained meta-trainer cycle, citing ADR-0013's rejection above as the evidence basis. Reuses ADR-0013's model identity, corpus, evaluator, contamination machinery, and resource budget unchanged; changes only `max_steps` (120->40, one epoch instead of three) and `learning_rate` (1e-5->5e-6, halved), with documented rationale, under a fresh run id (`adr0014-metatrainer-sft-20260922`) and approval namespace (`codevolt-adr0014`). Validation-only mode is exercised (`status: PASS`, `training_called: false`, all dataset/renderer/held-out hashes verified identical to ADR-0013's); no training is performed or authorized by this change.
