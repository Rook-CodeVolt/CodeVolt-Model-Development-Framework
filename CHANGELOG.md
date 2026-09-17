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
