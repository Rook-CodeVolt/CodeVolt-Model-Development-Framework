# ADR-0003: TrainerAdapterContract OS-level enforcement (v1.1)

- Status: accepted
- Date: 2026-09-14

## Context

ADR-0002 introduced `TrainerAdapterContract v1` (`CONTRACT_VERSION =
"1.0.0"`): a versioned boundary for a real training-engine integration,
proven against a deterministic, stdlib-only `FakeTrainerAdapter`. That
ADR was explicit that the contract's enforcement was declared but not
mechanically forced, and issue #7's own risk analysis flagged this as
something a real-engine pilot could not proceed past.

An independent review of that work, filed against issue #7 as PR #9,
identified three specific preconditions that had to be closed before any
real training engine could be trusted to run under this contract, rather
than merely declare compliance with it:

1. **Self-reported resource usage.** `ResourceUsage` (wall time, CPU
   time, peak memory, GPU count, storage) was whatever the adapter
   claimed about itself. A misbehaving or buggy real adapter could
   under-report its own consumption and never trip the budget check —
   the contract runner had no independent way to know it was being lied
   to.
2. **Cooperative-only cancellation.** Timeout and cancellation ran
   `adapter.train(...)` in a Python thread and relied on the adapter
   polling a `CancellationToken`. Python cannot force-terminate a
   thread. A non-conforming or hung adapter could keep running
   indefinitely in the background after the contract had already
   reported the run as `interrupted`, with no way for the parent to stop
   it.
3. **Unenforced filesystem/network boundaries.** `ResourceBudget`
   declared a `filesystem_root` and a `network_policy`, but nothing
   checked that a real adapter actually stayed inside them. Declaring a
   boundary and enforcing one are different things; the contract only
   did the former.

Any one of these gaps is disqualifying for a real training engine,
because a real engine is untrusted third-party code by construction:
the whole point of the contract is to make claims about resource
consumption and containment falsifiable rather than trusted on the
adapter's word.

## Decision

Add `src/codevolt_mdf/process_isolation.py` and route
`run_trainer_contract`'s `train()` call through it. This runs
`adapter.train(...)` in a real child **process** (via
`multiprocessing` with the `spawn` start method, chosen so behaviour is
consistent across Linux and macOS rather than depending on `fork`'s
copy-on-write semantics) instead of a thread in the parent process.
Bump `CONTRACT_VERSION` to `"1.1.0"` (`SUPPORTED_CONTRACT_VERSIONS`
still includes `"1.0.0"`, so this is additive, not breaking, to an
already-conforming adapter's obligations). Two new conformance tests
(`test_runaway_adapter_is_sigkilled_within_bound`,
`test_lying_adapter_self_report_is_overridden_by_os_measurement`) prove
the new behaviour against two new test-only adapters
(`RunawayAdapter`, `LyingAdapter` in `src/codevolt_mdf/testing_adapters.py`)
that deliberately misbehave the way a real adapter might. All 8
original v1 conformance cases are unchanged in behaviour; the full suite
is 33 tests, run and passing (`python3 -m pytest tests/ -v`: **33
passed**), and `ruff check .` reports zero errors.

Being a real OS process rather than a thread is what makes the fix
possible: the parent can now ask the kernel — not the adapter — what
happened, and can kill the process outright rather than asking it
nicely to stop.

### What this now enforces (mechanically, not by adapter cooperation)

- **Process-level cancellation.** On timeout, cancellation, or a live
  resource-budget overrun, the parent sends `SIGKILL` to the child
  process. This is real termination, not a cooperative request; a
  non-conforming adapter cannot keep running past it the way it could
  keep running past a cancelled thread.
- **Real, OS-measured resource usage.** While the child runs, the
  parent polls its live memory (RSS) and CPU time via `ps`; after it
  exits, the parent reads `resource.getrusage(RUSAGE_CHILDREN)`. These
  OS-sourced numbers *replace* the adapter's self-reported wall/CPU/
  memory figures before the contract runner checks them against the
  budget — an adapter that under-reports its own usage can no longer
  hide an overrun, because its self-report is discarded for those
  three dimensions, not merely cross-checked.
- **Environment stripping.** The child's environment variables are
  replaced with a small allow-list (`PATH`, `PYTHONPATH`, `HOME`,
  `LANG`, `LC_ALL`, `TMPDIR`) before `train()` runs, so ambient
  secrets or credentials present in the parent's environment are not
  implicitly handed to the adapter.

### What this still does NOT enforce (documented, not hidden)

This is a real, measurable improvement in the floor of trust the
contract can place in an adapter — it is deliberately **not** described
as a full sandbox, because it isn't one:

- **Filesystem containment is Python-interpreter-level only.** When
  `budget.filesystem_root` is set, the child wraps `builtins.open` and
  `os.open` to reject paths outside that root. This stops ordinary
  Python file I/O in the child interpreter. It does **not** stop the
  adapter from shelling out to a subprocess, a compiled C extension, or
  any code path that touches the filesystem without going through
  Python's own `open`/`os.open`.
- **Network containment is Python-interpreter-level only.** When
  `network_policy` is `"offline"` or `"allow-list"`, the child wraps
  Python's `socket` module (`connect`, `create_connection`,
  `getaddrinfo`). This stops ordinary Python networking. It does
  **not** stop the adapter from shelling out to `curl`/`wget`, from a
  compiled extension opening raw sockets directly, or from DNS
  resolution performed outside Python's resolver.
- **GPU usage remains fully adapter self-reported.** There is no
  portable stdlib way to measure GPU utilisation; a real integration
  targeting GPU hardware needs a vendor-specific measurement (e.g.
  NVML) as separate follow-up work, out of scope here.
- **`prepare()` still runs unisolated, in the parent process.** Only
  the `train()` call is sandboxed by this change. A malicious adapter
  module could still do damage during import or during `prepare()`,
  before any of the above protections take effect.

None of this is a substitute for genuine OS-level sandboxing
(containers, gVisor, seccomp profiles, network namespaces), which
would need elevated privileges or platform-specific tooling this
project does not assume are present, and remains explicitly out of
scope for this change. That is precisely the kind of control Maya's
independent security review (issue #7 step 5) is expected to require
before any real training engine is admitted to run under this
contract.

## Consequences

- The three preconditions PR #9's review raised against issue #7 are
  now addressed at the level described above (real process kill,
  OS-measured usage overriding self-report, Python-level filesystem/
  network containment) rather than left as declared-but-unchecked
  fields. They are **not** closed to the standard a full OS sandbox
  would provide, and this ADR does not claim otherwise.
- This is still a contract-and-fake-adapter change only. No real
  training engine is imported, called, or integrated by this decision.
  The real bounded pilot described in `docs/TRAINER_ADAPTER_CONTRACT.md`
  ("Real bounded pilot plan") remains gated on independent evaluation
  (issue #7 step 4) and Maya's independent security review (issue #7
  step 5); this ADR and its PR do not authorize that pilot.
- Future contract changes remain diffable: bump `CONTRACT_VERSION`, add
  an ADR, keep `SUPPORTED_CONTRACT_VERSIONS` accurate — this ADR follows
  that same discipline set out in ADR-0002.
- Because `train()` now runs in a spawned child process, an adapter
  written against the pre-1.1 in-process thread model continues to work
  unchanged: the contract constructs the `CancellationToken` the
  adapter expects inside the child and forwards the cross-process
  cancellation signal into it, so adapter code does not need to change
  to run under process isolation.
