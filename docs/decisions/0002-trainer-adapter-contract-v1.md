# ADR-0002: TrainerAdapterContract v1

- Status: accepted
- Date: 2026-09-12

## Context

`core.py` ships a two-method `TrainerAdapter`/`EvaluatorAdapter` `Protocol`
and a runtime that rejects any adapter other than `deterministic-demo`
(issue #7). Contributors integrating a real training engine (TRL, Unsloth,
Axolotl, LLaMA-Factory, MiniMind, or another) have no precise boundary to
implement against: no defined immutable inputs, typed outputs, resource
budget model, timeout/cancellation/safe-halt/resume semantics, network/
filesystem policy, error taxonomy, upstream compatibility bounds,
provenance/licence/contamination handling, role separation, or rollback
requirement.

Integrating a real engine without first defining this boundary would make
the contract hard to falsify (per the issue's own risk analysis) and would
conflate "trainer produced an artifact" with "artifact is good" or "run is
trustworthy" — violating `docs/ARCHITECTURE.md` core contracts #2 and #3.

## Decision

Add `TrainerAdapterContract v1` (`src/codevolt_mdf/trainer_contract.py`,
`CONTRACT_VERSION = "1.0.0"`) as a new, explicitly versioned module,
separate from and non-breaking to `core.py`'s existing
`deterministic-demo` path. The contract defines:

- immutable typed inputs (`TrainingInputs`) and outputs (`TrainingOutput`,
  `TrainingStatus`, `CheckpointHandle`);
- a `ResourceBudget` / `ResourceUsage` pair with post-hoc enforcement;
- stdlib-only cooperative timeout/cancellation (`CancellationToken`,
  documented as a limitation, not hidden);
- checkpoint/resume semantics that preserve (rather than discard) a
  safe-halted run's state;
- an error-class taxonomy mapped onto `accepted`/`rejected`/`invalid`/
  `interrupted`;
- `UpstreamRequirement` version-bound checking;
- a `run_trainer_contract` runner enforcing all of the above around any
  conforming adapter.

A deterministic, stdlib-only `FakeTrainerAdapter`
(`src/codevolt_mdf/fake_adapter.py`) exercises every scenario the contract
needs to prove: success, rejection, invalid input, timeout, cancellation,
resource overrun, checkpoint/resume, and evidence tamper detection
(`tests/test_trainer_contract.py`, eight conformance cases).

No real training engine is imported, called, or integrated by this
decision. The nine (pre-existing) tests in `tests/test_core.py` and the
`deterministic-demo` runtime path are unchanged.

## Consequences

- Future contract changes are diffable: bump `CONTRACT_VERSION`, add an
  ADR, keep `SUPPORTED_CONTRACT_VERSIONS` accurate.
- A real engine integration is now a bounded, reviewable follow-up: wire
  one adapter to `TrainerAdapterV1`, document it in a new ADR, then route
  it through independent evaluation (issue #7 step 4) and the security
  review (issue #7 step 5) before any pilot run, per the bounded pilot
  plan in `docs/TRAINER_ADAPTER_CONTRACT.md`.
- The stdlib-only timeout mechanism cannot force-terminate a
  non-cooperative adapter's thread; this is documented and pushed onto
  conformance testing plus the real-integration's security review rather
  than silently assumed to be airtight.
- `core.py`'s manifest-driven `run_experiment` flow is not yet wired to
  `TrainerAdapterV1`; that wiring is deferred to the real-integration
  follow-up so this PR stays scoped to contract + fake adapter, as issue
  #7 requires.
