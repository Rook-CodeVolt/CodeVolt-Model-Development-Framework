"""RED/GREEN coverage for the large-result Queue deadlock in process_isolation.

Evidence: ADR-0020 sampling attempts under the composed sandbox were each
killed just past their wall-clock budget, whatever that budget was, while a
smaller-result timing harness completed well inside the same budget on the
same workload shape.
Hypothesis: run_callable_in_isolated_process (and its siblings
run_in_isolated_process / run_evaluator_in_isolated_process) only call
``result_queue.get_nowait()`` *after* the child process is no longer alive.
A child that ``put()``s a payload larger than the OS pipe buffer blocks
inside its own ``multiprocessing.Queue`` feeder thread at interpreter exit,
so it never becomes not-alive -- the documented "joining processes that use
queues" deadlock. The parent then spins until its wall-clock budget expires
and reports a false timeout, even though the child actually finished its
work and was just stuck trying to hand back the result.

These tests exercise the real ``spawn``-context multiprocessing path (no
mocking of Process/Queue) so they fail against the unpatched deadlock and
pass once the poll loop drains the result queue before checking liveness.
"""

from __future__ import annotations

import time

from codevolt_mdf.process_isolation import (
    run_callable_in_isolated_process,
)
from codevolt_mdf.trainer_contract import ResourceBudget

# A large payload increases both the queued item's pickled size well past a
# typical pipe buffer (~64KB on Linux, ~16KB-ish on macOS) and its transfer
# time, without wasting test wall time on volume for its own sake.
_LARGE_PAYLOAD_ITEMS = 300_000  # ~2MB+ once JSON/pickle-serialized as ints


def _compute_large_payload(**_kwargs) -> dict:
    """Module-level compute_fn: must be picklable by reference for spawn."""
    return {"values": list(range(_LARGE_PAYLOAD_ITEMS))}


def _make_budget(max_wall_seconds: float) -> ResourceBudget:
    return ResourceBudget(
        max_wall_seconds=max_wall_seconds,
        max_cpu_seconds=60.0,
        max_memory_mb=1024.0,
        max_gpu_count=0,
        max_storage_mb=1024.0,
    )


def test_large_result_returns_well_within_a_generous_budget():
    """A >=1MB result must come back promptly, not hang to the wall budget.

    Before the fix: the child finishes and put()s ~2MB into the queue, then
    blocks in its Queue feeder thread trying to flush it through the pipe;
    the parent's poll loop only ever checks is_alive() (never true while the
    feeder thread is blocked) so it spins until max_wall_seconds and reports
    killed_for_timeout=True with no payload -- a false timeout on a run that
    actually completed its work.

    After the fix: the parent drains the queue inside the poll loop (before
    the liveness check), unblocking the child's feeder thread immediately,
    so the call returns the full payload in well under the wall budget.
    """
    budget = _make_budget(max_wall_seconds=25.0)

    start = time.monotonic()
    output, error, measured = run_callable_in_isolated_process(
        _compute_large_payload,
        kwargs={},
        budget=budget,
        poll_interval=0.05,
    )
    elapsed = time.monotonic() - start

    assert error is None, f"expected no error, got {error!r} (measured={measured!r})"
    assert output is not None, f"expected the payload, got None (measured={measured!r})"
    assert output["values"] == list(range(_LARGE_PAYLOAD_ITEMS))
    assert not measured.killed_for_timeout, (
        "large result falsely reported as a timeout kill -- this is the "
        "Queue-feeder-thread deadlock: the child finished and blocked "
        "trying to hand back its result, but the parent only checked "
        "is_alive() and never drained the queue before the child's exit"
    )
    assert not measured.killed_for_overrun
    assert elapsed < 10.0, (
        f"call took {elapsed:.2f}s against a {budget.max_wall_seconds}s budget "
        "-- should return promptly once the child's result is available, "
        "not spin until the wall-clock budget nearly expires"
    )
