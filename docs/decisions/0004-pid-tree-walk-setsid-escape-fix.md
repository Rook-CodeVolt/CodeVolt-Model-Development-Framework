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
