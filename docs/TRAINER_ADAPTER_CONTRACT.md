# TrainerAdapterContract v1

Status: implemented (fake adapter only). No real training engine is
integrated by this document or its code. See "Real bounded pilot plan"
below for the one documented (not executed) real-integration path.

Module: `src/codevolt_mdf/trainer_contract.py`
Version constant: `codevolt_mdf.trainer_contract.CONTRACT_VERSION = "1.1.0"`
(`SUPPORTED_CONTRACT_VERSIONS` still accepts `"1.0.0"` adapters
unchanged — see "OS-level enforcement (v1.1)" below for what changed
and why.)
Reference (fake) implementation: `src/codevolt_mdf/fake_adapter.py`
Process isolation: `src/codevolt_mdf/process_isolation.py`
Conformance tests: `tests/test_trainer_contract.py` (10 cases: the 8
original v1.0 scenarios plus 2 new v1.1 OS-enforcement cases exercised
against deliberately-misbehaving test-only adapters in
`src/codevolt_mdf/testing_adapters.py`)
Design record: `docs/decisions/0003-trainer-contract-os-level-enforcement.md`

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
checked by the contract runner (`ResourceUsage.exceeds(budget)`) against
every dimension: wall time, CPU time, peak memory, GPU count, storage. A
violation on an otherwise-successful run downgrades it to `interrupted`
with the specific violated dimensions named in `reason`, and triggers
cleanup.

**As of v1.1, wall time, CPU time, and peak memory are no longer taken
from the adapter's self-report.** `run_trainer_contract` now runs
`train()` inside a real child OS process (see "OS-level enforcement
(v1.1)" below) and overrides those three fields with what the operating
system actually measured for that process, before the budget check
runs. An adapter that under-reports its own resource consumption — by
mistake or by design — can no longer evade the budget check on those
three dimensions, because its self-report for them is discarded, not
merely cross-checked. GPU count and storage remain adapter self-reported
(there is no portable stdlib way to measure GPU usage, and storage is
computed from the declared `filesystem_root` directory size — see
below).

## Timeout, cancellation, safe-halt, checkpoint/resume semantics

**Timeout and cancellation are now enforced at the OS process level
(v1.1)**, not merely by a cooperative in-process signal. See "OS-level
enforcement (v1.1)" below for the full mechanism and its honest limits.
The adapter-facing API is unchanged:

1. `run_trainer_contract` runs `adapter.train(...)` inside a spawned
   child process (not a thread in the parent) and waits for it to
   finish, subject to `budget.max_wall_seconds`.
2. If the deadline is reached, or a caller triggers cooperative
   cancellation, the parent signals the child's `CancellationToken` and
   gives it a short, bounded grace period (`DEFAULT_KILL_GRACE_SECONDS`)
   to notice and return on its own.
3. If the child has not exited by the end of that grace period, the
   parent sends it `SIGKILL` directly. This is real, unconditional
   process termination — unlike the pre-1.1 thread-based approach, a
   non-conforming adapter cannot keep running in the background after
   the contract has reported `interrupted`, because the process it was
   running in no longer exists.
4. A conforming adapter should still poll `cancel_token.is_cancelled()`
   or use `cancel_token.wait(interval)` frequently, so that a
   cooperative, clean stop (with a checkpoint, see below) is preferred
   over an abrupt kill whenever possible. The grace period exists
   precisely to give a well-behaved adapter that chance before the
   parent escalates.

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
tamper-on-resume test) rather than trusting the handle blindly. A run
that is `SIGKILL`ed for exceeding `max_wall_seconds` or a live resource
budget did not get the chance to checkpoint; it is reported `interrupted`
with no checkpoint, and cleanup runs as normal.

## OS-level enforcement (v1.1)

Module: `src/codevolt_mdf/process_isolation.py`. Full rationale in
`docs/decisions/0003-trainer-contract-os-level-enforcement.md`.

An independent review of the v1.0 contract (issue #7, PR #9) identified
three preconditions that had to be closed before any real training
engine could be trusted to run under this contract: self-reported
resource usage, cooperative-only cancellation, and unenforced
filesystem/network boundaries. v1.1 closes them to the extent a
stdlib-only, no-elevated-privileges implementation can, by running
`train()` in a real child OS process instead of an in-process thread.
Being a separate process (not a thread) is what makes each of the
following possible, because the parent can now measure and terminate it
from outside using kernel-level facilities:

**Enforced:**

- **Process-level cancellation** — real `SIGKILL`, not a cooperative
  request (see previous section).
- **Process-GROUP-level cancellation** — the child calls `os.setsid()`
  on startup, becoming the leader of a new OS process group; the parent
  kills that whole group with `os.killpg(..., SIGKILL)` (`_kill_group`
  in `process_isolation.py`), not just the direct child pid. Any
  subprocess the adapter itself spawns (e.g. via `subprocess.Popen`)
  inherits the group and is terminated along with it, closing what was
  previously an orphaned-grandchild-process leak.
- **PID-tree-lineage cancellation** — before the process-group kill,
  `_kill_group` also walks the full `ppid` descendant tree of the
  isolated child (`ps -eo pid=,ppid=`) and `SIGKILL`s each discovered
  pid individually. This closes the gap process-group-based termination
  alone cannot: a descendant that itself calls `os.setsid()` (or
  otherwise leaves its process group) keeps its `ppid` unchanged, so the
  lineage walk still finds and kills it even though `os.killpg` no
  longer would. See `docs/decisions/0004-pid-tree-walk-setsid-escape-fix.md`
  and the `SetsidEscapingAdapter` conformance test. **Residual gap,
  stated plainly:** a descendant that **double-forks** to be reparented
  to PID 1 before this walk runs is not found by a `ppid`-lineage walk
  either (its `ppid` genuinely becomes PID 1) — that case needs real
  kernel-enforced containment (cgroups v2 `cgroup.kill` on Linux, a
  container/VM boundary on macOS) and remains out of scope.
- **Sanitised inter-process communication (IPC).** The child process
  runs untrusted adapter code, so nothing it returns is pickled and
  unpickled as-is across the `multiprocessing.Queue` back to the
  parent. Exceptions are reduced to plain strings (type name, message,
  traceback text) and rebuilt as a safe `ChildProcessError` in the
  parent; `TrainingOutput` and its nested dataclasses/enum are
  recursively rebuilt from only JSON-safe leaf types and an explicit
  allow-list, checked by exact `type()` rather than `isinstance` so a
  subclass cannot sneak a malicious object through. This closes a real
  pickle-deserialization vulnerability found during an independent
  security review: a crafted exception with a hand-written `__reduce__`
  raised by the child would, pre-fix, execute arbitrary code (proven
  with `os.system(...)`) in the trusted parent process the instant
  `Queue.get()` unpickled it. A legitimate adapter exception's type name
  and message still cross the boundary intact — only the raw object
  identity/behaviour is stripped.
- **OS-measured resource usage** — wall/CPU/peak-memory come from `ps`
  polling while the child runs plus `resource.getrusage(RUSAGE_CHILDREN)`
  after it exits, and these override the adapter's self-report (see
  "Resource budget enforcement" above).
- **Environment stripping** — the child's environment variables are
  replaced with a small allow-list (`PATH`, `PYTHONPATH`, `HOME`,
  `LANG`, `LC_ALL`, `TMPDIR`) before `train()` runs, so ambient
  secrets/credentials in the parent's environment are not implicitly
  exposed to the adapter.

**Best-effort only, NOT a real OS sandbox — documented, not hidden:**

- **Filesystem containment is Python-interpreter-level only.** When
  `budget.filesystem_root` is set, the child wraps `builtins.open` and
  `os.open` to reject paths outside that root. This does not stop the
  adapter from shelling out to a subprocess, from a compiled C
  extension, or from any code path that touches the filesystem without
  going through Python's own `open`/`os.open`.
- **Network containment is Python-interpreter-level only.** The child
  wraps Python's `socket` module (`connect`, `create_connection`,
  `getaddrinfo`) to enforce `network_policy`. This does not stop the
  adapter from shelling out to `curl`/`wget`, from a compiled extension
  opening raw sockets, or from DNS resolution outside Python's
  resolver.
- **GPU usage is not measured at all** and remains fully adapter
  self-reported; there is no portable stdlib way to measure GPU
  utilisation.
- **`prepare()` still runs unisolated, in the parent process.** Only
  the `train()` call is sandboxed. A malicious adapter could still act
  during import or during `prepare()`, before any isolation applies.

None of this is a substitute for genuine OS-level sandboxing
(containers, gVisor, seccomp, network namespaces), which needs elevated
privileges or platform-specific tooling this project does not assume
are present. That gap is exactly what Maya's independent security
review (issue #7 step 5) is expected to close or explicitly accept
before any real training engine is admitted to run under this contract.

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
As of v1.1, `filesystem_root` and `network_policy` are enforced at
Python-interpreter scope inside the child process (see "OS-level
enforcement (v1.1)" above) — this is real enforcement, but is explicitly
not equivalent to OS-level sandboxing (chroot/namespaces/seccomp), which
remains real-engine-integration and security-review work, out of scope
here (see "Real bounded pilot plan").

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

**Status update (2026-09-17):** the adapter side of this plan's step (a)
is now implemented — see `src/codevolt_mdf/trl_adapter.py`
(`TRLTrainerAdapter`) and
`docs/decisions/0005-trl-trainer-adapter-v1.md` (engine choice: TRL,
`UpstreamRequirement` bounds pinned to the exact verified version
`0.24.0` — originally a range `[0.20.0, 0.24.0)`, narrowed per Maya's
PR #15 review; see ADR-0005's "Exact pin, not a range" amendment). Its
contract tests (`tests/test_trl_adapter.py`, 28 cases) reuse the
fake-adapter-style
conformance pattern wherever the contract allows proving behaviour
without an actual training run (rejection, invalid-input, provenance/
hash-tamper, offline-policy, upstream-version-incompatibility). No real
pilot run has occurred: `adapter.train()` is fully implemented but never
called by any test, script, or CI step. Steps (b) and (c) below remain
unmet and are the explicit condition for any pilot run.

**Status update (2026-09-17, package 2):** `pyproject.toml`'s
`trl-adapter` extra and `TRLTrainerAdapter`'s `UpstreamRequirement` are
now pinned to the exact verified version `trl==0.24.0` (previously a
range) — see ADR-0005's "Exact pin, not a range" amendment, addressing
a medium finding from Maya's PR #15 review. Issue #7 step 4
(independent evaluation) now has a separate, contract-tested harness:
`src/codevolt_mdf/evaluator_contract.py`
(`EvaluatorAdapterV1`/`run_evaluator_contract`) and
`src/codevolt_mdf/held_out_registry.py`
(`HeldOutExclusionRegistry`, reusing the bidirectional, per-package
pattern from issue #11/`docs/DATA_GOVERNANCE.md`). This is a separate
component from `trl_adapter.py` — no import relationship in either
direction, enforced by a structural test
(`tests/test_evaluator_contract.py`).
`docs/decisions/0006-bounded-real-trl-pilot-plan.md` drafts (does not
authorise execution of) the bounded pilot's exact configuration. No
pilot has run; a real (non-fake) scoring adapter and Maya's
pilot-specific review of the live TRL process (step (c) below) remain
unmet and are the explicit condition for any pilot run.

- **Scope**: exactly one real engine adapter (e.g. TRL or Unsloth — the
  specific engine choice is a separate follow-up ADR, not decided here),
  implementing `TrainerAdapterV1` against `CONTRACT_VERSION = "1.1.0"`
  (or later; `SUPPORTED_CONTRACT_VERSIONS` is checked at run time).
- **Data**: synthetic or explicitly admitted data only, per
  `docs/DATA_GOVERNANCE.md`. No production or customer data under any
  circumstance.
- **Model**: one fixed, small, pinned model revision (hash-pinned), chosen
  for the pilot so evidence stays reproducible and cheap to inspect.
- **Concurrency**: exactly one concurrent run. No parallel pilot runs.
- **Resource limits**: explicit CPU, GPU, wall-clock, and storage limits
  declared as a `ResourceBudget` and enforced through
  `run_trainer_contract`. Wall/CPU/memory measurement is now real
  OS/process-level measurement (v1.1, see "OS-level enforcement" above)
  rather than self-reported; GPU measurement remains self-reported and
  is separate follow-up work for a real engine on GPU hardware.
- **Network/filesystem**: offline by default; any allow-list is the
  smallest set of hosts the real engine's pinned dependency resolution
  strictly requires, reviewed before the pilot runs.
- **Gating**: the pilot does not run until (a) an ADR records the chosen
  engine and its `UpstreamRequirement` bounds — done, ADR-0005, amended
  to an exact pin; (b) independent evaluation is configured with a
  held-out set inaccessible to the trainer (issue #7 step 4) — the
  contract-tested harness (`evaluator_contract.py`/
  `held_out_registry.py`) is done; a real (non-fake) scoring adapter
  and an actual pilot-specific held-out set are not yet built; and
  (c) Maya's independent security review of sandbox, egress, secrets,
  artifact hashes, supply-chain identity, and safe-stop behaviour of
  the live TRL process is complete (issue #7 step 5) — not started,
  distinct from her PR #15 code review of the adapter itself.
  `docs/decisions/0006-bounded-real-trl-pilot-plan.md` drafts the
  bounded pilot's exact configuration for that review to evaluate; it
  does not itself authorise execution.
- **Kill criteria** (unchanged from the issue): stop on missing
  provenance, unbounded resources, trainer access to held-out data,
  unverifiable artifacts, unsafe network/filesystem access, or any path
  that treats completion as improvement.

This PR grants no training, Test1/Test2 transfer, data admission,
deployment, or publication authority. It defines and tests a contract and
a fake adapter only.
