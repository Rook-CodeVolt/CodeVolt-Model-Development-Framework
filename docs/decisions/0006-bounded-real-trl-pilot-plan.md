# ADR-0006: Bounded real TRL pilot plan (drafted, NOT authorised to execute)

- Status: proposed — drafted per issue #7 step 4/5 sequencing; execution
  is explicitly **not** authorised by this ADR (see "Gating" below).
- Date: 2026-09-17

## Context

`docs/TRAINER_ADAPTER_CONTRACT.md`'s "Real bounded pilot plan" section
(added by ADR-0002, restated by ADR-0005) has always described the
shape of the one bounded real-engine training run issue #7 asks for —
fixed small model, concurrency one, explicit CPU/GPU/time/storage
limits, synthetic/admitted data only — as prose inside that document,
not as its own reviewable, versioned decision record. ADR-0005 closed
step (a) of that plan (choose and implement the adapter: TRL,
`TRLTrainerAdapter`, `UpstreamRequirement` pinned to `trl==0.24.0` per
this ADR's sibling amendment). This ADR closes the remaining
formalisation gap for steps (b) and (c)'s *preconditions* — it writes
down the exact bounded pilot configuration as a reviewable decision
record, the same discipline every other real-engine decision in this
project has gone through (ADR-0002 for the contract, ADR-0003/0004 for
OS-level enforcement, ADR-0005 for the engine choice) — **without**
itself satisfying steps (b) or (c), and **without** authorising
execution.

This ADR is drafted, not executed, as an explicit, separately
requested work package: "draft (do not execute) the bounded real pilot
ADR ... Stop there. Do NOT execute a real pilot run yet." Its own
`Status` field reflects that: `proposed`, not `accepted`, because
acceptance of a *pilot-execution* decision is not this ADR's call to
make alone — it requires the pilot-specific Maya review named in
"Gating" below, distinct from her PR #15 code review of the adapter
itself.

## Decision (the bounded configuration this ADR proposes, if approved)

### Model

One fixed, small, pinned model revision — proposed:
`HuggingFaceTB/SmolLM2-135M` (Apache-2.0, 135M parameters), the same
model family already used and soak-tested on the same host class (Mac
M4 Pro, no CUDA GPU) by the private cumulative-training-engine
research referenced in issue #7's 2026-09-14 comment (800 kill/resume
cycles, 872+ rounds, ~2h53m). Reusing an already-evidenced-compatible
model on this exact host class is a deliberate risk reduction, not a
requirement of this ADR's design — any other small, licence-clear,
CPU/MPS-compatible model would satisfy this section equally, provided
it is hash-pinned before the pilot runs. Whichever model is actually
used, `TrainingInputs.model_hash` must be the SHA-256 of its actual
downloaded weights (verified via `TRLTrainerAdapter.prepare()`'s
`_hash_path_identity()` check, already implemented and tested), not a
name or revision string trusted on its own.

### Dataset

Synthetic or explicitly admitted data only, per `docs/DATA_GOVERNANCE.md`.
No production or customer data under any circumstance — this is a hard
requirement already enforced structurally by `TrainingInputs.validate()`
requiring `contamination_checked=True` and a non-empty `dataset_licence`
before any adapter is reached at all. The pilot's specific dataset
must be constructed new for this pilot (a handful of synthetic
prompt/completion pairs, e.g. a fixed-format arithmetic or
classification task mirroring the cumulative-engine research's own
tick-classification curriculum) — reusing any held-out set from
`evaluator_contract.py`'s test fixtures or from any other package's
registered held-out ids is explicitly disallowed, and would itself be
caught by `HeldOutExclusionRegistry.check_held_out_not_trained` if
attempted after this pilot's dataset is registered as train data
anywhere.

### Concurrency

Exactly one concurrent run. No parallel pilot runs, no second pilot
started before the first reaches a terminal `TrainingStatus`
(`ACCEPTED`/`REJECTED`/`INVALID`/`INTERRUPTED`) and its evidence is
recorded. This is caller-level discipline the pilot's own run harness
must enforce (unchanged from ADR-0005's own disclosure that neither
`TRLTrainerAdapter` nor `run_trainer_contract` enforce concurrency
themselves).

### Resource limits (`ResourceBudget`, explicit values)

| Field | Proposed value | Rationale |
|---|---|---|
| `max_wall_seconds` | `1800` (30 min) | Generous enough for a `max_steps`-bounded CPU/MPS SFT run on a 135M model to complete normally; short enough that a stuck run is caught quickly. |
| `max_cpu_seconds` | `3600` (60 min, allows some multi-core parallelism above wall-clock) | OS-measured (v1.1 enforcement), not adapter self-reported. |
| `max_memory_mb` | `8192` (8 GB) | Bounded well under this host class's typical available memory; a genuine leak or a much larger model swapped in by mistake is caught here, not by exhausting the host. |
| `max_gpu_count` | `0` | This host has no CUDA GPU (see `Local-LLM-Feasibility/docs/HARDWARE_DECISION_GATE.md`); MPS is not counted by `torch.cuda.device_count()`, so this is deliberately `0`, not an MPS-specific count this project does not currently measure. |
| `max_storage_mb` | `2048` (2 GB) | Checkpoints + final artifact + evidence for a 135M model; generous headroom over the model's own on-disk size. |
| `network_policy` | `"offline"` | Already the only policy `TRLTrainerAdapter.prepare()` accepts (`RejectedInputError` otherwise); restated here for completeness of the pilot's own declared budget. |
| `training_params["max_steps"]` | `50` | Adapter-enforced (in addition to the wall-clock budget, per ADR-0005's "Adapter-enforced step bound"); a genuinely small, bounded pilot, not an open-ended training run. |

These are proposed starting values for review, not final until the
pilot-specific security review (see "Gating") signs off on them
specifically — that review is expected to scrutinise these numbers as
part of its scope, not merely accept this table's arithmetic.

### Filesystem

`ResourceBudget.filesystem_root` set to a dedicated, empty, per-pilot
scratch directory (not the repository checkout, not any other
project's data directory). `TRLTrainerAdapter.work_dir` points at the
same root. Cleaned up (`cleanup()`, already implemented and idempotent)
after the pilot's evidence is copied out, not left as a persistent
mount.

### Independent evaluation wiring (issue #7 step 4)

The pilot's trained artifact must be scored through
`evaluator_contract.run_evaluator_contract` against a `HeldOutSet`
constructed and registered (`HeldOutExclusionRegistry.register_package_held_out`)
*before* the pilot's training data is finalised, using a package id
distinct from the pilot's own training-data package id, so the
bidirectional contamination check in `held_out_registry.py` has
something real to check against rather than an empty registry that
trivially passes. The evaluator adapter used for the pilot's actual
scoring is **not** decided by this ADR — `FakeEvaluatorAdapter` proves
the contract but is not a real capability-scoring adapter; a real
scoring adapter (even a minimal exact-match or perplexity check against
the pilot's tiny synthetic task) is separate, not-yet-authorised
follow-up work this ADR does not perform. This section records the
*wiring requirement* — every pilot run must go through
`run_evaluator_contract` against a real `HeldOutSet` inaccessible to
the trainer, using `HeldOutExclusionRegistry` to make that inaccessibility
enforced rather than declared — not a claim that a real evaluator
adapter already exists.

### Kill criteria (unchanged from issue #7's own list, restated for this pilot)

Stop on: missing provenance, unbounded resources, trainer access to
held-out data, unverifiable artifacts, unsafe network/filesystem
access, or any path that treats completion as improvement. Concretely
for this pilot: a `TrainingOutput.status` other than `ACCEPTED`, an
evidence-hash mismatch (`TamperDetectedError`), a
`ContaminationDetectedError` from the evaluator contract, or a resource
budget violation each independently halt the pilot and are recorded as
evidence, not silently retried or reinterpreted as success.

## Gating (this ADR authorises none of it)

This ADR does not authorise the pilot to run. Before execution:

1. **Independent evaluation is wired and tested** — this ADR's own
   scope (drafting the config) plus the separately delivered
   `evaluator_contract.py`/`held_out_registry.py` components satisfy
   the *contract-and-tests* half of issue #7 step 4. A real (non-fake)
   evaluator adapter and an actual `HeldOutSet` for this specific
   pilot's task remain to be built when the pilot is scheduled.
2. **Maya's pilot-specific security review** (issue #7 step 5) —
   sandbox, egress, secrets, artifact hashes, supply-chain identity,
   and safe-stop behaviour of the *actual live TRL process*, not the
   adapter code review already completed in PR #15. This is a
   distinct review scope from PR #15's: that review explicitly stated
   "I have not evaluated live execution behaviour because none
   occurred here" — this pilot is exactly the live execution that
   review did not cover.
3. **Owner authorisation to schedule the pilot itself**, separate from
   the standing continual-training-framework authorisation that
   covers drafting this ADR and building the evaluation harness.

No training engine is invoked, no pilot is scheduled, and no resource
is reserved as a result of this ADR. `adapter.train()` remains
untested by any run in this repository, exactly as ADR-0005 left it.

## Consequences

- CodeVolt MDF now has a reviewable, versioned decision record for the
  bounded pilot's exact configuration, closing the "formalize what's
  currently just pilot-plan prose" gap — the next reviewer (Maya, for
  the pilot-specific review; the owner, for scheduling) has concrete
  numbers and a concrete wiring requirement to review against, not
  prose to interpret.
- This ADR's proposed values (model, dataset shape, resource limits)
  are a starting point for that review, not immutable — the
  pilot-specific security review is expected to challenge or tighten
  them as part of its own scope.
- Nothing in this ADR changes `CONTRACT_VERSION`, `TRL_MIN_VERSION`/
  `TRL_MAX_VERSION`, or any code path. It is a planning document only.
