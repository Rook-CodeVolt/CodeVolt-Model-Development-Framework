# ADR-0004: PID-tree-walk fix for the disclosed `os.setsid()` process-group escape

- Status: accepted
- Date: 2026-09-15

## Context

PR #13 (`docs/decisions/0003-trainer-contract-os-level-enforcement.md`)
added OS-level process isolation to `TrainerAdapterContract` v1.1,
including process-**group**-level cancellation: the isolated child calls
`os.setsid()` on startup so it leads a new process group, and the parent
kills the whole group with `os.killpg(..., SIGKILL)` (`_kill_group`) so
any subprocess the adapter itself spawns dies with it.

That PR explicitly disclosed, rather than hid, one residual gap: a
grandchild process that itself calls `os.setsid()` leaves the process
group `os.killpg` targets and would survive the group kill. This is a
real, general limitation of process-group-based termination, not a bug
specific to the PR #13 implementation — `os.setsid()` changes a
process's process-group id and session id, but it does **not** change
its `ppid` (parent process id), which the kernel assigns once at fork
time and never revises because of `setsid()`.

Issue #7's engine-selection/sandboxing-design task
(`t_4d1933f5`, posted to issue #7's comment thread) proposed closing this
literal gap with a two-layer design and explicitly scoped Layer 2
(double-fork/orphan-adoption via cgroups/a container boundary) out, per
the same "containers/gVisor/seccomp... out of scope" framing this
project has used since ADR-0003. This ADR implements Layer 1 only.

## Decision

Add a ppid-lineage walk-and-kill to `src/codevolt_mdf/process_isolation.py`
(`_list_pid_ppid_pairs`, `_descendant_pids`, `_kill_pid_tree`), and call it
from `_kill_group` **before** the existing `os.killpg` sweep:

1. `_list_pid_ppid_pairs()` runs `ps -eo pid=,ppid=` — a single,
   privilege-free, dependency-free snapshot of every `(pid, ppid)` pair
   currently visible on the host.
2. `_descendant_pids()` does a breadth-first walk of that snapshot from
   the isolated child's pid, following `ppid` links rather than
   process-group membership. A descendant that has called `os.setsid()`
   to leave its process group is still found here, because leaving a
   process group does not change `ppid`.
3. `_kill_pid_tree()` re-walks up to three times (`_MAX_PID_TREE_WALK_PASSES`),
   individually `os.kill(pid, SIGKILL)`-ing every discovered pid each
   pass, tolerating `ProcessLookupError`/`OSError` for pids that already
   exited. The bounded re-walk closes the narrow race where a process
   forks a new child between one `ps` snapshot and the kill of its
   parent; it is not an unbounded retry loop.
4. `_kill_group()` calls `_kill_pid_tree` **first**, then `os.killpg` as a
   redundant safety net. Order matters: `_child_worker` is itself a
   process-group leader (it called `os.setsid()` on startup, same as
   before this change), so killing the group first would SIGKILL the
   isolated child before the lineage walk ran — and the moment a parent
   process is reaped, the kernel immediately reparents its still-live
   children, corrupting the very `ppid` lineage the walk depends on.
   Walking and killing by lineage first, while every process in the tree
   is still alive and still shows its real parentage in `ps`, avoids
   that race. `os.killpg` afterwards is now a strictly redundant
   second sweep for the common (non-escaping) case.

A new TEST-ONLY adapter, `SetsidEscapingAdapter`
(`src/codevolt_mdf/testing_adapters.py`), spawns a grandchild that calls
`os.setsid()` on its own startup before hanging — precisely the disclosed
gap — and writes the grandchild's pid to a file so the test can check its
liveness via a real OS `os.kill(pid, 0)` call, not an internal accounting
flag. `tests/test_trainer_contract.py::test_setsid_escaping_grandchild_is_still_killed`
exercises it through the real `run_in_isolated_process` path (not a
standalone script) and asserts the escaped grandchild is dead after the
contract runner's timeout-triggered kill.

**Verified the test actually exercises the fix, not just passes
coincidentally:** stashing only the `process_isolation.py` change (fix
reverted, old `killpg`-only `_kill_group`, new test and new
`SetsidEscapingAdapter` both kept) reproduces a failing
`test_setsid_escaping_grandchild_is_still_killed` — the escaped
grandchild is confirmed still alive. Restoring the fix makes it pass
again, three consecutive runs, with no other test regressed.

## What this does NOT do (out of scope, per issue #7's Layer 2 design)

This closes exactly the literal disclosed gap — a single `os.setsid()`
call by a grandchild — using only `ps` and `os.kill`: no new privileges,
no new dependency, and it is exercised on this project's actual macOS
host (not just asserted to work there).

It does **not** defend against a descendant that **double-forks** to be
reparented to PID 1 (`launchd` on macOS / `init` or a container's PID 1
on Linux) before this walk runs: at that point the descendant's `ppid`
genuinely points at PID 1, not at anything under the isolated child, so
a `ppid`-lineage walk starting from the isolated child's pid will not
find it either. Closing that harder case needs real kernel-enforced
containment — cgroups v2 `cgroup.kill` on Linux (which kills by cgroup
membership, not `ppid`, so a double-fork does not escape it), or a
container/VM boundary on macOS (no cgroup equivalent exists there) — and
remains explicitly out of scope here, consistent with ADR-0003's
"containers/gVisor/seccomp/network namespaces... out of scope" framing
and with the Layer 2 design posted to issue #7.

## Evidence

```
python3 -m pytest tests/ -v
...
37 passed in 9.78s
```
37 passed: the 36 pre-existing tests (all behaviour unchanged) plus the
one new `test_setsid_escaping_grandchild_is_still_killed` case. Reran
the `setsid`/`grandchild` subset three consecutive times with no
flakiness.

```
ruff check .
All checks passed!
```

## Consequences

- The one residual gap PR #13 disclosed (a grandchild escaping the
  process group via its own `os.setsid()` call) is now closed for the
  single-`setsid()`-hop case, with a conformance test proving it against
  a real OS process, not an internal flag.
- `_kill_group`'s call sites and signature are unchanged; this is
  purely an internal strengthening of the existing kill path. No public
  contract surface, `CONTRACT_VERSION`, or adapter-facing API changes.
- The harder double-fork/orphan-adoption case (Layer 2 of the original
  design) remains open and explicitly out of scope, same as before this
  change. No real training engine is imported, called, or integrated by
  this decision, and it does not authorize engine integration, Test 2
  porting, or a training pilot — issue #7 steps 4 (independent
  evaluation) and 5 (Maya's independent security review) remain
  separately gated.

## Addendum: failure/exhaustion observability (Maya's PR #14 review)

Maya's independent review of the PR implementing this ADR (PR #14,
review comment
https://github.com/Rook-CodeVolt/CodeVolt-Model-Development-Framework/pull/14#issuecomment-5683138092)
issued REQUEST CHANGES on exactly one required item: the design in the
"Decision" section above (step 3, `_kill_pid_tree`'s bounded re-walk)
was implemented without the logging/evidence-bundle visibility that
same design called for when the walk exhausts its passes without fully
reaping a target, or when the underlying `ps` call itself fails. Before
this addendum, both failure modes silently degraded `_kill_group` to a
root-pid-only kill with zero observability — a real gap between the
approved design and the shipped code, not merely an omitted nice-to-have.

This is now closed, not deferred, by:

- `process_isolation._list_pid_ppid_pairs()` now returns `(pairs, ok)`
  instead of silently returning `[]` on a `ps` failure; `ok=False` is
  distinguishable from a genuine empty snapshot and is logged via the
  module's `_logger.warning(...)`.
- `process_isolation._kill_pid_tree()` now returns a
  `PidTreeWalkOutcome(ps_call_failed, exhausted_without_confirmed_reap)`
  instead of `None`, logging a warning for each failure mode as it
  happens (per-pass for a `ps` failure, once if all
  `_MAX_PID_TREE_WALK_PASSES` passes exhaust without any pass reporting
  zero live targets).
- `_kill_group()` propagates that outcome to its caller;
  `run_in_isolated_process()` threads it into a new
  `MeasuredUsage.pid_tree_walk_outcome` field (`None` when no kill ever
  ran, so the common non-degraded path is unaffected).
- `trainer_contract.run_trainer_contract()` appends a
  `_pid_tree_walk_reason_suffix(measured)` string to `TrainingOutput.reason`
  on the timeout and resource-overrun paths (the two paths that call
  `_kill_group`) whenever `PidTreeWalkOutcome.degraded` is true — this is
  the evidence-bundle-visible half of the requirement, not just a log
  line nobody reads afterward.

Evidence (added in the PR #14 follow-up commit, `tests/test_trainer_contract.py`
section 11):

```
test_kill_pid_tree_reports_and_logs_when_ps_call_fails            PASSED
test_kill_pid_tree_reports_and_logs_when_passes_exhausted         PASSED
test_evidence_bundle_reason_reflects_degraded_pid_tree_walk       PASSED
```

Negative control: stashing only the `process_isolation.py`/
`trainer_contract.py` changes (new tests kept) reproduces all three
tests failing — `AttributeError: 'NoneType' object has no attribute
'exhausted_without_confirmed_reap'` for the two unit-level tests, and a
`reason` string with no `[pid-tree-walk degraded` marker for the
end-to-end one. Restoring the changes makes all three pass again, with
the full 40-test suite (37 pre-existing + 3 new) green and `ruff check .`
clean.

This addendum does not change the Decision, the closed literal gap, or
the explicitly-out-of-scope double-fork/orphan-adoption case above; it
only adds the observability the original design specified for the
walk's own failure/exhaustion, per Maya's required remediation.

## Addendum 2: cancellation hard-kill parity (Maya's 2nd PR #14 review)

Maya's re-review of the addendum above (PR #14, review comment
https://github.com/Rook-CodeVolt/CodeVolt-Model-Development-Framework/pull/14#issuecomment-5683727075)
issued REQUEST CHANGES again, narrowly scoped to one remaining call
site: `_kill_group()` runs at three places in
`process_isolation.run_in_isolated_process()` -- the timeout path, the
resource-overrun path, and the cancellation-triggered hard-kill path
(a non-cooperative adapter, e.g. `RunawayAdapter`, does not honour
`cancel_token` within `DEFAULT_KILL_GRACE_SECONDS` and the runner
escalates to a real `SIGKILL`). Addendum 1 above threaded the degraded-
walk signal into `TrainingOutput.reason` for the first two call sites
only. The third call site correctly logged the degraded walk via the
module logger (Addendum 1's logging is call-site-agnostic), but its
result fell through into the generic
`exc_payload`/`RuntimeError("child process exited without reporting a
result")` branch in `trainer_contract.run_trainer_contract()`, which
discards the signal before it reaches the evidence bundle. Maya
independently reproduced this with a real `cancel_token` fire against
a monkeypatched failing `subprocess.run`.

Closed by giving the cancellation hard-kill path the same explicit-flag
treatment `killed_for_timeout`/`killed_for_overrun` already had, rather
than trying to special-case it inside the generic exception-formatting
branch:

- `process_isolation.MeasuredUsage` gained a `killed_for_cancellation`
  field (default `False`, so existing call sites/tests are unaffected).
- `run_in_isolated_process()`'s cancellation branch sets
  `killed_for_cancellation = True` before calling `_kill_group()` when
  the adapter is still alive after the cooperative grace period, mirroring
  the timeout/overrun branches exactly.
- The `if killed_for_overrun or killed_for_timeout:` short-circuit (which
  returns `(None, None, measured)` before the generic
  `exc_payload`/`RuntimeError` fallback can ever run) now also checks
  `killed_for_cancellation`, so this path never reaches that fallback.
- `trainer_contract.run_trainer_contract()` gained a third
  `if measured.killed_for_cancellation:` branch, positioned after the
  overrun branch and before the `exc_payload` check, that builds a
  `TrainingOutput(status=INTERRUPTED, error_class=TrainerCancelledError.__name__, ...)`
  with the cancellation reason, the grace-period bound, and the same
  `_pid_tree_walk_reason_suffix(measured)` call the other two branches
  use -- so all three `_kill_group()` call sites now share one evidence
  formatting helper rather than duplicating the degraded-walk text.

Evidence (added in the PR #14 follow-up commit,
`tests/test_trainer_contract.py` section 12):

```
test_cancellation_hard_kill_of_noncooperative_adapter_reaches_reason   PASSED
```

This single test asserts both halves of the requirement: at the
`process_isolation` level, `measured.killed_for_cancellation is True`
while `killed_for_timeout`/`killed_for_overrun` stay `False` and both
`output`/`exc` stay `None` (proving the generic `RuntimeError` fallback
is never reached); at the evidence-bundle level, an end-to-end
`run_trainer_contract()` call against a real, SIGKILLed `RunawayAdapter`
process with a forced-failing `ps` call asserts `output.error_class ==
"TrainerCancelledError"`, the cancellation reason and `SIGKILL` text
appear in `output.reason`, the `[pid-tree-walk degraded` marker is
present, and the old generic fallback text ("child process exited
without reporting a result") is absent.

Negative control: stashing only the `process_isolation.py`/
`trainer_contract.py` changes (new test kept) reproduces the test
failing -- `AttributeError: 'MeasuredUsage' object has no attribute
'killed_for_cancellation'` on the first assertion. Restoring the
changes makes it pass again; two consecutive full-suite runs afterward
are 41/41 green (37 pre-existing + 3 from Addendum 1 + 1 new) with
`ruff check .` clean and no new `mypy` findings (the same 4 pre-existing
findings from Addendum 1 are unchanged).

This addendum does not change the Decision, either previously-closed
gap, or the explicitly-out-of-scope double-fork/orphan-adoption case
above; it only extends Addendum 1's evidence-bundle-visibility
requirement to the third and last `_kill_group()` call site.
