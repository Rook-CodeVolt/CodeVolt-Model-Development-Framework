# TrainerAdapterContract v1

Status: implemented (fake adapter only). No real training engine is
integrated by this document or its code. See "Real bounded pilot plan"
below for the one documented (not executed) real-integration path.

Module: `src/codevolt_mdf/trainer_contract.py`
Version constant: `codevolt_mdf.trainer_contract.CONTRACT_VERSION = "1.0.0"`
Reference (fake) implementation: `src/codevolt_mdf/fake_adapter.py`
Conformance tests: `tests/test_trainer_contract.py`

This document describes the boundary a trainer adapter must cross to be
admitted into CodeVolt MDF, replacing informal expectations with an
executable contract. It supersedes the two-method `TrainerAdapter`
`Protocol` in `core.py` as the boundary for a *real* engine integration;
`core.py`'s `deterministic-demo` path is untouched and still the only
adapter the `run_experiment` CLI/manifest flow accepts (see "Relationship
to core.py" below).

## Versioning

`CONTRACT_VERSION` is a plain semantic-version string. Any change to the
shapes, semantics, or enforcement behaviour described here must:

1. Bump `CONTRACT_VERSION` (patch = clarification/doc-only, minor =
   backward-compatible addition, major = breaking change to an existing
   adapter's obligations).
2. Add an ADR under `docs/decisions/` describing what changed and why
   (per `docs/GOVERNANCE.md`, new contracts require an ADR).
3. Keep `SUPPORTED_CONTRACT_VERSIONS` accurate so the contract runner
   rejects adapters written against an incompatible version instead of
   silently misinterpreting them.

This document and the module are versioned together; do not edit the
enforcement code without updating this file's description of it, or vice
versa.

## Exact immutable inputs — `TrainingInputs`

| field | type | meaning |
|---|---|---|
| `model_revision` | `str` | pinned model identity (not a mutable "latest" pointer) |
| `model_hash` | `str` (sha256 hex) | content hash of the model checkpoint being trained from |
| `dataset_version` | `str` | pinned dataset identity |
| `dataset_hash` | `str` (sha256 hex) | content hash of the dataset |
| `seed` | `int` | deterministic seed |
| `training_params` | mapping, stored as a sorted tuple of items | hyperparameters; immutable once constructed |
| `dataset_licence` | `str` | licence identifier for the dataset |
| `contamination_checked` | `bool` | must be `True`; `TrainingInputs.validate()` raises `InvalidInputError` otherwise, keeping held-out/contamination separation a precondition rather than an afterthought |
| `run_id` | `str` | unique identity for this run, used for evidence/checkpoint paths and cleanup |

`TrainingInputs` is an immutable (frozen) dataclass; use
`TrainingInputs.create(...)` to build one from plain keyword arguments.
`validate()` performs static, adapter-independent checks (hash shape,
required non-empty strings, contamination flag) and raises
`InvalidInputError` — this always happens *before* any adapter code runs.

## Typed outputs — `TrainingOutput`

| field | meaning |
|---|---|
| `status` | one of `TrainingStatus`: `accepted`, `rejected`, `invalid`, `interrupted` |
| `reason` | human-readable explanation |
| `artifact_id` | identity of the produced artifact/checkpoint (accepted runs) |
| `checkpoint` | `CheckpointHandle` when the run safe-halted and can be resumed |
| `evidence_locator` | filesystem path (or future URI) to the evidence bundle |
| `evidence_hash` | sha256 the adapter claims for that evidence at write time |
| `resource_usage` | measured `ResourceUsage` |
| `error_class` | the contract error class name, when applicable |

`status` here describes whether the *trainer run* produced trustworthy
evidence — it is distinct from `core.Decision.status`, which is an
*evaluation acceptance* decision made later by an evaluator/assurance gate.
A trainer's own `accepted` status is never itself a promotion decision
(architecture core contract #2 and #3 continue to hold).

## Error classes and status mapping

| Python exception | run status | when |
|---|---|---|
| `InvalidInputError` | `invalid` | malformed/incomplete inputs, or evidence-hash mismatch found after the run (tamper/corruption) |
| `RejectedInputError` | `rejected` | well-formed inputs the adapter declines (unsupported revision, incompatible upstream version, disallowed network policy) |
| `ResourceBudgetExceededError` | `interrupted` | measured usage exceeds the declared budget |
| `TrainerTimeoutError` | `interrupted` | adapter did not return within `budget.max_wall_seconds` |
| `TrainerCancelledError` | `interrupted` | run cancelled cooperatively via `CancellationToken` |
| `TamperDetectedError` | `invalid` | evidence on disk does not match its declared hash |
| `CheckpointError` | adapter-defined | checkpoint could not be created/read/resumed |

All of these derive from `TrainerContractError`. `run_trainer_contract`
never lets an adapter's unexpected exception escape as a raw traceback to
the caller in the run pathway (`prepare`/`train`) — it is caught, mapped
to a status, and cleanup is invoked; an unrecognised exception maps to
`interrupted` rather than being hidden as `accepted`.

## Resource budget enforcement and measured reporting

`ResourceBudget` is declared before a run starts and validates itself
(positive limits, offline network policy requires no allow-list host,
allow-list network policy requires at least one). `ResourceUsage` is
measured/reported by the adapter after the run and checked by the
contract runner (`ResourceUsage.exceeds(budget)`) against every dimension:
wall time, CPU time, peak memory, GPU count, storage. A violation on an
otherwise-successful run downgrades it to `interrupted` with the specific
violated dimensions named in `reason`, and triggers cleanup. This check is
post-hoc (the fake adapter self-reports usage) because the contract is
stdlib-only; a real integration is expected to source `ResourceUsage` from
actual OS/process measurements, not adapter self-assessment claims alone
(see Threat model note below).

## Timeout, cancellation, safe-halt, checkpoint/resume semantics

**Timeout semantics (documented limitation).** This contract is
dependency-light by design (`docs/ARCHITECTURE.md`, `CONTRIBUTING.md`) and
implements timeout/cancellation with stdlib only
(`threading` + `threading.Event`). Python cannot force-terminate a thread.
The contract runner therefore:

1. Runs `adapter.train(...)` in a daemon thread and joins it with a
   `budget.max_wall_seconds` timeout.
2. If the thread is still alive at the deadline, it signals the
   `CancellationToken` and gives the adapter one more `max_wall_seconds`
   window to notice and return, then reports `interrupted` regardless of
   whether the thread has actually exited. Because the thread is a daemon
   thread, it will not block process exit, but a non-conforming adapter
   that ignores cancellation can continue running in the background after
   the contract has already reported `interrupted`.
3. **A conforming adapter MUST poll `cancel_token.is_cancelled()` or use
   `cancel_token.wait(interval)` frequently** (the fake adapter polls
   every 10ms) so real cancellation is prompt. This is an adapter
   obligation the contract cannot mechanically force with stdlib alone; it
   is verified by the timeout and cancellation conformance tests, and any
   real integration's security review (issue #7 evidence-plan step 5)
   must independently verify this property under process/OS-level
   isolation rather than trusting the thread alone.

**Cancellation** is the same mechanism, triggered by a caller (e.g. a
human-in-the-loop stop command) rather than the wall-clock deadline;
`CancellationToken.cancel(reason=...)` records why.

**Safe-halt / checkpoint / resume**: an adapter may return `interrupted`
with a populated `checkpoint: CheckpointHandle` (`checkpoint_id`, `step`,
`state_locator`, `state_hash`) instead of running to completion. The
contract runner treats a checkpointed interruption as *preserved evidence
and resumable state*, not a failure to discard — `cleanup()` is skipped
when `output.checkpoint is not None`. A later call passes that same
`CheckpointHandle` as `resume_from`; a conforming adapter must
independently re-verify `state_hash` against the file it reads (the fake
adapter does this and raises `InvalidInputError` on mismatch — see the
tamper-on-resume test) rather than trusting the handle blindly.

## Offline/network policy and filesystem boundaries

`ResourceBudget.network_policy` is `"offline"` (default) or
`"allow-list"` with an explicit `allowed_hosts` tuple; `"allow-list"` with
no hosts is rejected as invalid input at budget-construction time.
`ResourceBudget.filesystem_root` declares the one root an adapter may
read/write under. The fake adapter only ever writes under its own
`work_dir` and rejects any budget that is not `offline`
(`RejectedInputError`) — this is deliberately stricter than the contract
requires, modelling the safest-by-default posture `docs/THREAT_MODEL.md`
and `CONTRIBUTING.md` ask for ("avoid network access by default").
Enforcing `filesystem_root` and `allowed_hosts` at the OS/sandbox level
(not just as declared fields) is real-engine-integration and
security-review work, explicitly out of scope here (see "Real bounded
pilot plan").

## Dependency / upstream-version compatibility bounds

`UpstreamRequirement(engine_name, min_version, max_version,
installed_version)` declares an inclusive compatibility bound; `run_trainer_contract`
rejects (`RejectedInputError`) any adapter whose `installed_version` falls
outside `[min_version, max_version]` before doing anything else, so an
incompatible engine version never gets a chance to run. The fake adapter
declares itself as its own upstream (`fake-engine`) purely to exercise this
check deterministically. A real adapter must declare the actual engine name
and pin real bounds (e.g. a specific TRL/Unsloth release range).

## Provenance / licence / privacy / contamination / held-out separation

- `model_hash` / `dataset_hash` make provenance checkable independent of
  mutable names.
- `dataset_licence` is a required field; `validate()` rejects an empty one.
- `contamination_checked` must be `True` before any run reaches the
  adapter — enforced as a static, adapter-independent precondition. This
  contract does not perform the contamination check itself (that remains
  `DatasetVersion`'s job per `docs/ARCHITECTURE.md`'s research/knowledge
  plane); it refuses to run without the flag being set upstream.
- Held-out evaluation separation is preserved structurally: this contract
  only concerns the *trainer* side. `core.py`'s architecture contract #3
  ("An evaluator produces measurements; it does not promote the
  candidate") and the roles below keep the trainer from ever seeing or
  producing its own acceptance decision.

## Trainer / evaluator / security-review role separation

This contract governs only `TrainerAdapterV1`. It intentionally:

- Has no method that scores, evaluates, or accepts/rejects the *quality*
  of an artifact — only whether the *run itself* is trustworthy evidence
  (`TrainingStatus`).
- Never calls or depends on `EvaluatorAdapter` from `core.py`.
- Does not implement or simulate independent evaluation, security review,
  or promotion — those remain the separate gated steps in
  `docs/ARCHITECTURE.md`'s capability lifecycle and issue #7's evidence
  plan steps 4-5, owned by an independent evaluator and by Maya's security
  review, not by this PR.

## Rollback / cleanup after failure

`TrainerAdapterV1.cleanup(run_id)` must be idempotent and safe to call
when there is nothing to clean up. `run_trainer_contract` calls it
automatically whenever a run does not end `accepted` **and** did not leave
a checkpoint behind (a checkpointed safe-halt preserves its state
deliberately; see above). The fake adapter's `cleanup` removes its
per-run working directory with `shutil.rmtree(..., ignore_errors=True)`.

## Deterministic FAKE adapter

`FakeTrainerAdapter` (`src/codevolt_mdf/fake_adapter.py`) is a
`stdlib`-only, in-process adapter selected purely by a `scenario` training
parameter (`success`, `timeout`, `cancel`, `resource_overrun`,
`checkpoint_resume`, `tamper`). Every scenario is a deterministic pure
function of its inputs — same `seed`/`run_id`/`scenario` always produces
the same evidence file contents. It never imports a real training engine
and grants no training/data-admission/deployment authority; it exists only
to give the contract's conformance tests something concrete, safe, and
inspectable to run against.

## Relationship to `core.py`

`core.py`'s `run_experiment` CLI/manifest flow and its
`deterministic-demo` gate (`Experiment`, `TrainerAdapter`/
`EvaluatorAdapter` `Protocol`, `decide`) are **unchanged** by this PR. The
nine (now: existing) tests in `tests/test_core.py` continue to pass
untouched. `trainer_contract.py` is a new, separate module: the next real
engine integration is expected to implement `TrainerAdapterV1` and be
driven through `run_trainer_contract`, and to be wired into `core.py`'s
manifest flow as a **follow-up** change (tracked separately; not part of
this PR) once independent evaluation and security review (issue #7 steps
4-5) have signed off on a specific engine.

## Real bounded pilot plan (documented, not executed)

This PR documents — and does **not** implement or run — the one bounded
real-integration path issue #7 asks for. No training engine is invoked by
this repository as a result of this document.

- **Scope**: exactly one real engine adapter (e.g. TRL or Unsloth — the
  specific engine choice is a separate follow-up ADR, not decided here),
  implementing `TrainerAdapterV1` against `CONTRACT_VERSION = "1.0.0"`.
- **Data**: synthetic or explicitly admitted data only, per
  `docs/DATA_GOVERNANCE.md`. No production or customer data under any
  circumstance.
- **Model**: one fixed, small, pinned model revision (hash-pinned), chosen
  for the pilot so evidence stays reproducible and cheap to inspect.
- **Concurrency**: exactly one concurrent run. No parallel pilot runs.
- **Resource limits**: explicit CPU, GPU, wall-clock, and storage limits
  declared as a `ResourceBudget` and enforced through
  `run_trainer_contract`, with real (not self-reported) OS/process-level
  resource measurement replacing the fake adapter's self-reported
  `ResourceUsage` — this measurement upgrade is itself pilot-scoped work.
- **Network/filesystem**: offline by default; any allow-list is the
  smallest set of hosts the real engine's pinned dependency resolution
  strictly requires, reviewed before the pilot runs.
- **Gating**: the pilot does not run until (a) an ADR records the chosen
  engine and its `UpstreamRequirement` bounds, (b) independent evaluation
  is configured with a held-out set inaccessible to the trainer (issue #7
  step 4), and (c) Maya's independent security review of sandbox, egress,
  secrets, artifact hashes, supply-chain identity, and safe-stop behaviour
  is complete (issue #7 step 5). None of that review is performed by this
  PR.
- **Kill criteria** (unchanged from the issue): stop on missing
  provenance, unbounded resources, trainer access to held-out data,
  unverifiable artifacts, unsafe network/filesystem access, or any path
  that treats completion as improvement.

This PR grants no training, Test1/Test2 transfer, data admission,
deployment, or publication authority. It defines and tests a contract and
a fake adapter only.
