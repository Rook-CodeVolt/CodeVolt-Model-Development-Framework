"""OS-level process isolation for TrainerAdapterContract v1.1.

This module is what turns the contract's three previously-documented
gaps (self-reported resource usage, cooperative-only cancellation,
undeclared-but-unenforced filesystem/network boundaries) into real
process-level enforcement, without integrating any training engine.

It runs ``adapter.train(...)`` in a genuine child OS process
(``multiprocessing`` with the ``spawn`` start method, so behaviour is the
same on Linux and macOS and does not depend on ``fork``'s copy-on-write
quirks). Because it is a real process, not a thread, the parent can:

- measure its *actual* CPU time and peak memory from the kernel
  (``resource.getrusage(RUSAGE_CHILDREN)`` after it exits, plus live
  polling of ``ps`` while it runs) instead of trusting whatever the
  adapter chooses to report;
- terminate it with ``SIGKILL`` (``Process.kill()``) the instant it
  either exceeds a live-polled budget dimension or fails to exit after a
  cooperative-cancellation grace period -- something a Python thread
  fundamentally cannot do to itself or another thread.

Filesystem and network boundaries are enforced *inside* the child, at
Python-interpreter scope, before ``adapter.train`` runs: this is honestly
a partial mitigation (see module docstring section "What this does not
enforce" below), not an OS sandbox, container, or firewall. It is
documented as such rather than oversold.

## What this enforces

- **Process-level cancellation**: SIGKILL, not a cooperative request.
- **Real resource measurement**: wall/CPU/peak-memory come from the OS
  (`ps` while running, `getrusage` after exit), not the adapter's return
  value. The adapter's self-reported ``ResourceUsage`` is discarded for
  wall/CPU/memory and replaced with the measured figures before the
  contract runner checks them against the budget.
- **Filesystem containment (Python-level, writes only)**: when
  ``budget.filesystem_root`` is set, the child's cwd is pinned there and
  ``builtins.open`` / ``os.open`` are wrapped to reject any *write*
  (create/truncate/append/read-write) that resolves outside that root,
  except ``/dev/null`` (explicitly allowlisted -- see
  ``_DEVNULL_RESOLVED``; a write there cannot persist or exfiltrate
  anything). Reads are not restricted to the root -- see "What this does
  NOT enforce" below for why, and Finding 1 of Maya's pilot-specific
  live-execution review (issue #7 step 5, PR #18) for the concrete
  real-execution failures a read-restrictive guard caused.
- **Network containment (Python-level)**: when ``budget.network_policy``
  is ``"offline"``, ``socket.socket.connect``/``create_connection``/
  ``getaddrinfo`` are wrapped to raise immediately. When it is
  ``"allow-list"``, connections are permitted only to
  ``budget.allowed_hosts``.
- **Environment stripping**: the child's environment is replaced with a
  small allow-list (``PATH``, ``PYTHONPATH``, ``HOME``, ``LANG``,
  ``LC_ALL``, ``TMPDIR``) before ``adapter.train`` runs, so ambient
  secrets/credentials in the parent's environment are not implicitly
  visible to the adapter.

## What this does NOT enforce (documented, not hidden)

- **Not a real network sandbox.** The socket wrapper only intercepts
  Python's own ``socket`` module inside the child interpreter. It does
  **not** stop the adapter from shelling out to `curl`/`wget`/a compiled
  binary, does not intercept a C-extension that opens raw sockets via
  its own syscalls, and does not touch DNS resolution performed outside
  Python's resolver. Genuine OS-level network blocking (netns, pf/
  iptables rules, a seccomp/sandbox profile) needs elevated privileges
  or platform-specific tooling this project does not assume are present
  and is explicitly out of scope for this change.
- **Not a real filesystem jail.** The `open`/`os.open` wrapper only
  covers Python-level file access inside the child interpreter. It does
  not stop `os.system`/`subprocess` calls to external tools, does not
  use `chroot`/mount namespaces/`sandbox-exec`, and a sufficiently
  determined adapter written in a compiled extension could still touch
  paths outside the declared root. As of the write-only scoping above,
  it also does not restrict *reads* to the declared root at all: a real
  training run needs to read its pinned model/dataset checkpoint
  (validated by content hash in ``prepare()``, but not necessarily
  located under ``filesystem_root`` -- ADR-0006 deliberately keeps
  ``filesystem_root`` as an empty per-pilot scratch directory, separate
  from model/dataset storage) plus hundreds of interpreter/site-packages
  files transitively imported by ``datasets``/``transformers``/``trl``
  (e.g. ``dill``'s module-load-time ``/dev/null`` probe, `transformers`'
  lazy ``_LazyModule`` machinery reading arbitrary
  ``site-packages/transformers/models/.../configuration_*.py`` files at
  import time) -- none of which are writes and none of which are
  meaningfully contained by rejecting them, since a read cannot itself
  exfiltrate data anywhere the process couldn't already reach via other
  means (subprocess, sockets already governed separately, etc). Trying
  to keep reads root-restricted while training-engine imports work at
  all was tried and rejected: pre-importing the ML dependency chain
  before installing the guard only pushes the failure one import
  deeper, and a curated site-packages/stdlib/devnull-only allowlist
  still breaks on the model/dataset read the adapter must legitimately
  perform. See Maya's pilot-specific live-execution review (issue #7
  step 5, PR #18), Finding 1, for the concrete reproduction.
- **Not GPU-usage measurement.** There is no portable stdlib way to
  measure GPU utilisation; ``gpu_count_used`` remains adapter
  self-reported. A real engine integration on GPU hardware needs a
  vendor-specific measurement (e.g. NVML) as follow-up work, out of
  scope here.
- **Not a defence against a malicious adapter's code itself** (e.g. an
  adapter module doing damage during import, before ``train`` even
  runs). The contract only wraps the ``train`` call; ``prepare``
  currently still runs in the parent process (see
  ``docs/TRAINER_ADAPTER_CONTRACT.md``).

Real, OS-level sandboxing (containers, gVisor, a signed seccomp profile,
network namespaces) is exactly the kind of control Maya's independent
security review (issue #7 step 5) is expected to require before any real
training engine is admitted; this module raises the floor from "nothing
enforced, adapter self-discipline only" to "real process isolation for
timeout/cancellation/measurement, partial Python-level containment for
filesystem/network" -- it does not claim to reach full sandbox parity.
"""

from __future__ import annotations

import builtins
import logging
import multiprocessing
import os
import re
import resource
import signal
import socket
import subprocess
import time
import traceback
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)
"""Module logger for the two originally-approved-but-silent failure modes
of the pid-tree walk (issue #7 Layer 1 design step 3, flagged by Maya's
PR #14 review): the underlying ``ps`` call failing/timing out, and the
bounded walk-and-kill loop exhausting all its passes without confirming a
full reap. Both are logged here (host-side observability) and also
surfaced into ``MeasuredUsage.pid_tree_walk_incomplete`` -> the
``TrainingOutput.reason`` evidence field (see ``trainer_contract.py``),
so the gap between "walk failed/degraded silently" and "root-pid-only
kill happened without anyone knowing" is closed in both places, not just
one. See docs/decisions/0004-pid-tree-walk-setsid-escape-fix.md."""

_ENV_ALLOWLIST = ("PATH", "PYTHONPATH", "HOME", "LANG", "LC_ALL", "TMPDIR")

# How often the parent polls the child's live resource usage via `ps`.
DEFAULT_POLL_INTERVAL_SECONDS = 0.05

# Grace period given to a cooperative cancellation signal (via
# ``mp_cancel_event``) before the parent escalates to SIGKILL on a
# **wall-clock timeout**. Bounded and short: a timed-out adapter is
# treated by the contract runner as having no usable checkpoint
# regardless (see ``run_trainer_contract``, "timed out with no
# checkpoint recorded: nothing to resume"), so there is no reason to
# wait long enough for a full checkpoint save here -- only enough for
# an unresponsive process to notice the event and exit promptly.
DEFAULT_KILL_GRACE_SECONDS = 1.0

# Grace period given specifically to a **cooperative cancellation**
# request (distinct from the timeout grace above) before the parent
# escalates to SIGKILL. This is the path where a real checkpoint save
# is expected to complete and be reported back -- unlike the timeout
# path, the contract runner keeps and trusts a checkpoint produced
# here (see ``run_trainer_contract``'s ``killed_for_cancellation``
# handling only discarding it when the adapter never got the chance).
#
# Raised from an original single shared 1.0s during Elias's
# real-execution verification of the trl_adapter.py checkpoint/resume
# fix (Maya's pilot-specific live-execution review, issue #7 step 5,
# PR #18, Finding 3): once ``_safe_halt_output`` genuinely persists a
# resumable checkpoint (``save_model`` + optimizer/scheduler/scaler/RNG
# state + ``save_state()``, not just ``save_model`` alone), that save
# itself measured ~1.4s wall-clock for a 135M-parameter model on this
# host (dominated by the fp32 Adam optimizer state, ~2x model size) --
# already longer than the previous shared 1.0s grace period on its own,
# before accounting for the training loop's own per-step
# callback-detection latency on top. At 1.0s, a real
# cooperative-cancel-then-checkpoint cycle was reliably SIGKILLed
# mid-save before it could report a checkpoint at all, silently
# defeating the fix this constant is meant to support. 8.0s keeps
# meaningful margin over the measured real save time (~5.5x) while
# staying a small fraction (<0.5%) of ADR-0006's locked 1800s
# wall-clock budget, so cancellation remains bounded and fast relative
# to a real run, not indefinite. Kept as a separate constant from
# ``DEFAULT_KILL_GRACE_SECONDS`` (rather than raising that one
# directly) so the *timeout* path -- which discards any checkpoint
# regardless and has no real-save-completion reason to wait longer --
# is not slowed down by a change that exists purely to let a genuine
# cancellation-triggered checkpoint save finish.
DEFAULT_CANCELLATION_KILL_GRACE_SECONDS = 8.0


@dataclass(frozen=True)
class MeasuredUsage:
    """OS-measured resource consumption of the child process.

    Distinct from ``trainer_contract.ResourceUsage``: this is what the
    *operating system* reported, not what the adapter claimed. The
    contract runner uses this to override wall/CPU/memory in the
    adapter's self-reported ``ResourceUsage`` before checking the budget.
    """

    wall_seconds: float
    cpu_seconds: float
    memory_mb_peak: float
    storage_mb_used: float | None  # None when filesystem_root was not declared
    killed_for_overrun: bool
    killed_for_timeout: bool
    killed_for_cancellation: bool = False
    """True only when the adapter did not honour ``cancel_token``
    cooperatively within ``DEFAULT_KILL_GRACE_SECONDS`` and the runner
    escalated to a real ``_kill_group()`` SIGKILL -- i.e. the 3rd
    ``_kill_group()`` call site, distinct from ``killed_for_timeout`` and
    ``killed_for_overrun``. See
    ``docs/decisions/0004-pid-tree-walk-setsid-escape-fix.md``, Addendum:
    cancellation hard-kill parity."""
    pid_tree_walk_outcome: PidTreeWalkOutcome | None = None
    """Set only when a kill (``_kill_group``) actually ran -- i.e. only
    meaningful alongside ``killed_for_overrun``/``killed_for_timeout``/
    ``killed_for_cancellation``. ``None`` when the child exited on its own
    and no kill was ever attempted, in which case the pid-tree walk's
    failure modes do not apply. See ``PidTreeWalkOutcome.degraded`` and
    ``docs/decisions/0004-pid-tree-walk-setsid-escape-fix.md``."""


class SandboxViolationError(PermissionError):
    """Raised inside the child process when code crosses a declared boundary."""


class ChildProcessError(RuntimeError):
    """Safe, parent-reconstructed stand-in for an exception raised in the child.

    The child process is untrusted (it runs arbitrary adapter code), so its
    raw exception object is never pickled across the ``multiprocessing.Queue``
    boundary -- an exception class with a hand-written ``__reduce__``/
    ``__reduce_ex__`` could otherwise make unpickling in the (trusted) parent
    execute arbitrary code. See module docstring, "What this does NOT
    enforce" / IPC payload sanitisation, and
    ``docs/decisions/0003-trainer-contract-os-level-enforcement.md``.

    Only plain strings cross the boundary (exception type name, ``str()`` of
    the exception, and formatted traceback text); this object is
    reconstructed fresh in the parent from those strings.
    """

    def __init__(self, original_type_name: str, message: str, traceback_text: str) -> None:
        super().__init__(message)
        self.original_type_name = original_type_name
        self.traceback_text = traceback_text


# --------------------------------------------------------------------------
# IPC payload sanitisation
#
# The child process runs untrusted adapter code. Anything that crosses the
# ``multiprocessing.Queue`` boundary back to the parent is pickled by the
# child and unpickled by the parent; unpickling an object whose class
# defines ``__reduce__``/``__reduce_ex__`` can execute arbitrary code
# *in the parent's trusted context* the moment ``Queue.get`` deserialises
# it (a `PoC <../../tests/test_process_isolation_ipc_security.py>`_ in this
# repository's own test suite proves this concretely against an
# unrestricted queue). To close that hole, nothing adapter-controlled is
# ever pickled directly:
#
# - Exceptions are converted to three plain ``str`` values (type name,
#   message, traceback text) before being queued, and reconstructed as a
#   ``ChildProcessError`` (or a known, argument-free contract error class)
#   in the parent -- never by unpickling the original object.
# - ``TrainingOutput`` (and its nested ``ResourceUsage``/``CheckpointHandle``/
#   ``TrainingStatus``) are recursively re-validated and rebuilt from
#   scratch in the child using only JSON-safe leaf types (``None``, ``bool``,
#   ``int``, ``float``, ``str``, ``bytes``) plus the explicit allow-listed
#   dataclasses/enum, checked by *exact* ``type()`` (not ``isinstance``) so
#   a subclass cannot sneak a malicious ``__reduce__`` through disguised as
#   an allowed type. Anything else raises ``TypeError`` inside the child and
#   is reported back as a safe error instead of ever being queued.
# --------------------------------------------------------------------------

_JSON_SAFE_LEAF_TYPES = (type(None), bool, int, float, str, bytes)


def _sanitize_ipc_value(value: Any, *, allowed_dataclasses: tuple[type, ...], allowed_enums: tuple[type, ...]) -> Any:
    """Recursively validate/rebuild ``value`` from only safe types, or raise.

    Uses exact ``type(value) is X`` checks (not ``isinstance``) throughout:
    a subclass overriding ``__reduce__``/``__reduce_ex__`` must not be able
    to pass this check just because it *is a* ``str``/allowed dataclass.
    """
    t = type(value)
    if t in _JSON_SAFE_LEAF_TYPES:
        return value
    if t is list or t is tuple:
        rebuilt = [
            _sanitize_ipc_value(item, allowed_dataclasses=allowed_dataclasses, allowed_enums=allowed_enums)
            for item in value
        ]
        return rebuilt if t is list else tuple(rebuilt)
    if t is dict:
        return {
            _sanitize_ipc_value(k, allowed_dataclasses=allowed_dataclasses, allowed_enums=allowed_enums): (
                _sanitize_ipc_value(v, allowed_dataclasses=allowed_dataclasses, allowed_enums=allowed_enums)
            )
            for k, v in value.items()
        }
    if t in allowed_enums:
        return t(value.value)
    if t in allowed_dataclasses and is_dataclass(t):
        kwargs = {
            f.name: _sanitize_ipc_value(
                getattr(value, f.name), allowed_dataclasses=allowed_dataclasses, allowed_enums=allowed_enums
            )
            for f in fields(t)
        }
        return t(**kwargs)
    raise TypeError(
        f"value of type {t.__module__}.{t.__qualname__} is not permitted to cross the "
        "process-isolation IPC boundary (only JSON-safe leaf types and explicitly "
        "allow-listed TrainingOutput/ResourceUsage/CheckpointHandle/TrainingStatus values "
        "are); rejecting to avoid pickling an adapter-controlled object in the parent process"
    )


def _sanitize_training_output(output: Any) -> Any:
    """Validate and rebuild an adapter's ``TrainingOutput`` for safe IPC.

    Imported lazily to avoid a hard import-time dependency cycle with
    ``trainer_contract`` (same reasoning as the lazy import in
    ``_child_worker``/``run_trainer_contract``).
    """
    from .trainer_contract import CheckpointHandle, ResourceUsage, TrainingOutput, TrainingStatus

    if type(output) is not TrainingOutput:
        raise TypeError(
            f"adapter.train() must return a TrainingOutput, got {type(output).__module__}."
            f"{type(output).__qualname__}"
        )
    return _sanitize_ipc_value(
        output,
        allowed_dataclasses=(TrainingOutput, CheckpointHandle, ResourceUsage),
        allowed_enums=(TrainingStatus,),
    )


def _safe_exception_tuple(exc: BaseException) -> tuple[str, str, str]:
    """Reduce an (untrusted, child-raised) exception to three plain strings.

    ``str(exc)`` and ``type(exc).__name__`` are guaranteed by the language
    to yield real ``str`` instances (CPython raises ``TypeError`` if
    ``__str__`` returns anything else), so this never accidentally smuggles
    an attacker-controlled object across the boundary.
    """
    return (type(exc).__name__, str(exc), traceback.format_exc())


_RECONSTRUCTABLE_CONTRACT_ERRORS = ("RejectedInputError", "InvalidInputError")


def _reconstruct_exception(payload: tuple[str, str, str]) -> BaseException:
    """Rebuild a safe exception in the parent from a child-reported tuple.

    Never unpickles the child's original exception object. For the small,
    fixed set of contract error classes whose constructor takes only a
    message string (and which are defined in our own trusted
    ``trainer_contract`` module), the real class is reconstructed so
    existing ``isinstance`` based status-mapping keeps working. Anything
    else becomes a ``ChildProcessError`` carrying the original type name as
    plain data (not as a class to instantiate).
    """
    type_name, message, traceback_text = payload
    if type_name in _RECONSTRUCTABLE_CONTRACT_ERRORS:
        from . import trainer_contract

        cls = getattr(trainer_contract, type_name)
        exc = cls(message)
        exc.traceback_text = traceback_text  # type: ignore[attr-defined]
        return exc
    return ChildProcessError(type_name, message, traceback_text)


def _sanitize_environment() -> None:
    """Replace the child's environment with a minimal allow-list."""
    kept = {k: v for k, v in os.environ.items() if k in _ENV_ALLOWLIST}
    os.environ.clear()
    os.environ.update(kept)


def _open_mode_is_write(mode: str) -> bool:
    """Return True if a ``builtins.open`` mode string requests any write access.

    Read-only reads (``"r"``, ``"rb"``, ``"rt"``) must never be blocked outside
    ``filesystem_root`` -- real training imports (``dill``'s ``/dev/null``
    probe, ``transformers``' lazy ``_LazyModule`` site-packages reads, and
    the adapter's own read of a model/dataset checkpoint that may live
    outside the per-pilot scratch root) are all reads. Anything requesting
    create/truncate/append/update access (``w``, ``a``, ``x``, or a ``+``
    update flag in any combination) is a write and stays root-restricted.
    """
    return any(flag in mode for flag in ("w", "a", "x", "+"))


def _os_open_flags_are_write(flags: int) -> bool:
    """Return True if raw ``os.open`` flags request any write access.

    Mirrors ``_open_mode_is_write`` for the lower-level ``os.open`` API:
    checks the access-mode bits (``O_RDONLY``/``O_WRONLY``/``O_RDWR``) and
    the creation/mutation bits (``O_CREAT``/``O_TRUNC``/``O_APPEND``/
    ``O_EXCL``) that can accompany ``O_RDONLY`` to still mutate the
    filesystem (e.g. creating an empty file read-only).
    """
    accmode = flags & os.O_ACCMODE
    if accmode != os.O_RDONLY:
        return True
    mutating_bits = 0
    for name in ("O_CREAT", "O_TRUNC", "O_APPEND", "O_EXCL"):
        mutating_bits |= getattr(os, name, 0)
    return bool(flags & mutating_bits)


_DEVNULL_RESOLVED = Path(os.devnull).resolve()
"""``/dev/null`` (POSIX) explicitly allowlisted for both reads and writes.

``dill``'s module-load-time probe (``dill/_objects.py``:
``open(os.devnull, 'wb', buffering=0).close()``) *writes* to
``/dev/null`` to build its type-introspection table -- this is a write
by mode, so it is not covered by the read/write split above, but it is
categorically harmless regardless of ``filesystem_root``: the kernel
discards everything written to it, nothing is created, persisted, or
exfiltrated anywhere. Blocking it serves no containment purpose and
was the literal first failure a real pilot run hit (Maya's
pilot-specific live-execution review, issue #7 step 5, PR #18, Finding
1). Explicitly allowlisted rather than silently exempted by the
write-detection logic above, so the exemption is visible and
auditable in one place.
"""


def _pin_filesystem_root(filesystem_root: str) -> None:
    """Pin cwd to ``filesystem_root`` and reject Python-level *writes* outside it.

    Deliberately scoped to writes only (not reads) -- see module
    docstring, "What this enforces" and "What this does NOT enforce" for
    the full rationale. A read-restrictive guard is fail-closed in
    intent but fails the pilot outright in practice: real training
    imports (``dill``, ``transformers``' lazy module machinery) and the
    adapter's own read of a model/dataset checkpoint outside
    ``filesystem_root`` are all legitimate reads that a training run
    cannot proceed without. Containing exfiltration/tamper risk from
    writes -- checkpoints, evidence, any file the untrusted adapter code
    creates or mutates -- is the actual security property this boundary
    is for; reads outside the root are not a filesystem-containment gap
    on their own (see Maya's pilot-specific live-execution review,
    issue #7 step 5, PR #18, Finding 1).

    Honest scope: only intercepts ``builtins.open`` and ``os.open`` inside
    this interpreter. See module docstring, "What this does NOT enforce".
    """
    root = Path(filesystem_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)

    real_open = builtins.open
    real_os_open = os.open

    def _check(path: Any) -> None:
        candidate = Path(os.fspath(path))
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved == _DEVNULL_RESOLVED:
            return
        if root not in resolved.parents and resolved != root:
            raise SandboxViolationError(
                f"write to path {resolved} is outside the declared filesystem_root {root}"
            )

    def _guarded_open(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)) and _open_mode_is_write(mode):
            _check(file)
        return real_open(file, mode, *args, **kwargs)

    def _guarded_os_open(path, flags, *args, **kwargs):
        if _os_open_flags_are_write(flags):
            _check(path)
        return real_os_open(path, flags, *args, **kwargs)

    builtins.open = _guarded_open  # type: ignore[assignment]
    os.open = _guarded_os_open  # type: ignore[assignment]


def _apply_network_policy(network_policy: str, allowed_hosts: tuple[str, ...]) -> None:
    """Wrap Python's ``socket`` module to enforce ``network_policy``.

    Honest scope: only intercepts connections made through Python's own
    ``socket`` module. See module docstring, "What this does NOT enforce".
    """

    real_connect = socket.socket.connect
    real_create_connection = socket.create_connection
    real_getaddrinfo = socket.getaddrinfo

    def _host_allowed(host: Any) -> bool:
        if host in (None, "", "localhost", "127.0.0.1", "::1"):
            return network_policy != "offline"
        return network_policy == "allow-list" and host in allowed_hosts

    def _guarded_connect(self, address, *a, **kw):
        host = address[0] if isinstance(address, tuple) else address
        if not _host_allowed(host):
            raise SandboxViolationError(
                f"network_policy={network_policy!r} blocked outbound connection to {host!r}"
            )
        return real_connect(self, address, *a, **kw)

    def _guarded_create_connection(address, *a, **kw):
        host = address[0] if isinstance(address, tuple) else address
        if not _host_allowed(host):
            raise SandboxViolationError(
                f"network_policy={network_policy!r} blocked outbound connection to {host!r}"
            )
        return real_create_connection(address, *a, **kw)

    def _guarded_getaddrinfo(host, *a, **kw):
        if not _host_allowed(host):
            raise SandboxViolationError(
                f"network_policy={network_policy!r} blocked DNS resolution for {host!r}"
            )
        return real_getaddrinfo(host, *a, **kw)

    socket.socket.connect = _guarded_connect  # type: ignore[assignment]
    socket.create_connection = _guarded_create_connection  # type: ignore[assignment]
    socket.getaddrinfo = _guarded_getaddrinfo  # type: ignore[assignment]


def _child_worker(
    adapter: Any,
    inputs: Any,
    budget: Any,
    resume_from: Any,
    cancel_event: Any,
    reason_buf: Any,
    result_queue: Any,
) -> None:
    """Entry point run inside the child process (module-level: picklable target).

    Builds a fresh, thread-based ``CancellationToken`` local to this
    process (adapters are written against that API) and forwards the
    cross-process ``cancel_event`` into it via a small daemon thread, so
    adapter code is unchanged by running under process isolation.
    ``reason_buf`` is a shared ``multiprocessing.Array('c', ...)`` the
    parent writes the human-readable cancellation reason into before
    setting ``cancel_event``, so it survives the process boundary.

    Calls ``os.setsid()`` first so this process becomes the leader of a
    new process group: any subprocess the adapter spawns (e.g. via
    ``subprocess.Popen``) inherits that group, so the parent's
    ``os.killpg`` can terminate the whole tree together instead of only
    this direct child (see ``run_in_isolated_process`` / ``_kill_group``
    and ``docs/decisions/0003-trainer-contract-os-level-enforcement.md``).
    """
    import threading

    from .trainer_contract import CancellationToken

    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            # Already a session/group leader (can happen under some test
            # harnesses); nothing else to do -- killpg below still targets
            # this process's own pgid either way.
            pass

    try:
        _sanitize_environment()
        if budget.filesystem_root:
            _pin_filesystem_root(budget.filesystem_root)
        _apply_network_policy(budget.network_policy, budget.allowed_hosts)
    except Exception as exc:  # noqa: BLE001 - sandbox setup failure must surface
        result_queue.put(("error", _safe_exception_tuple(exc)))
        return

    token = CancellationToken()

    def _forward_cancel() -> None:
        cancel_event.wait()
        reason = reason_buf.value.decode("utf-8", errors="replace") or "cancelled"
        token.cancel(reason=reason)

    forwarder = threading.Thread(target=_forward_cancel, daemon=True)
    forwarder.start()

    try:
        output = adapter.train(inputs, budget, token, resume_from)
        # Never pickle the adapter's raw return value across the queue: it
        # is untrusted, and a class with a hand-written __reduce__ could
        # execute arbitrary code when unpickled in the trusted parent.
        # Rebuild it from scratch using only JSON-safe/allow-listed types.
        safe_output = _sanitize_training_output(output)
        result_queue.put(("output", safe_output))
    except BaseException as exc:  # noqa: BLE001 - surfaced to parent, not swallowed
        result_queue.put(("exception", _safe_exception_tuple(exc)))


_PS_TIME_RE = re.compile(r"^(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)$")


def _parse_ps_cputime(raw: str) -> float:
    """Parse `ps -o time=` output (``[[dd-]hh:]mm:ss``) into seconds."""
    raw = raw.strip()
    match = _PS_TIME_RE.match(raw)
    if not match:
        return 0.0
    days, hours, minutes, seconds = match.groups()
    total = int(minutes) * 60 + int(seconds)
    if hours:
        total += int(hours) * 3600
    if days:
        total += int(days) * 86400
    return float(total)


def _poll_live_usage(pid: int) -> tuple[float, float] | None:
    """Return (memory_mb, cpu_seconds) for a live pid via `ps`, or None if gone."""
    try:
        proc = subprocess.run(
            ["ps", "-o", "rss=,time=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = proc.stdout.strip()
    if not line:
        return None
    parts = line.split(None, 1)
    if len(parts) != 2:
        return None
    rss_kb_str, cputime_str = parts
    try:
        memory_mb = float(rss_kb_str) / 1024.0
    except ValueError:
        return None
    cpu_seconds = _parse_ps_cputime(cputime_str)
    return memory_mb, cpu_seconds


def _list_pid_ppid_pairs() -> tuple[list[tuple[int, int]], bool]:
    """Return (``(pid, ppid)`` pairs, ``ok``) for every process visible on the host.

    ``ok`` is ``False`` when the underlying ``ps`` call itself failed or
    timed out (as opposed to succeeding with zero/malformed lines, which
    is a normal empty snapshot). Callers must check ``ok`` rather than
    inferring failure from an empty list, because this failure mode was
    previously silent: ``_kill_pid_tree`` would treat a failed ``ps`` the
    same as "no descendants exist" and quietly degrade to a root-pid-only
    kill with zero observability (Maya's PR #14 review, required
    remediation item). See
    ``docs/decisions/0004-pid-tree-walk-setsid-escape-fix.md``.

    Uses ``ps -eo pid=,ppid=`` (no elevated privileges, no new
    dependency, works the same on macOS and Linux) rather than any
    process-group/session primitive: the point of this helper is to
    reconstruct parent/child *lineage*, which ``os.setsid()`` does not
    change (it changes the caller's process group and session id, not
    its ``ppid``).
    """
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid=,ppid="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _logger.warning(
            "_list_pid_ppid_pairs: 'ps -eo pid=,ppid=' failed or timed out (%s: %s); "
            "pid-tree lineage snapshot unavailable for this pass",
            type(exc).__name__,
            exc,
        )
        return [], False
    pairs: list[tuple[int, int]] = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        pairs.append((pid, ppid))
    return pairs, True


def _descendant_pids(root_pid: int, pairs: list[tuple[int, int]]) -> list[int]:
    """Return every pid transitively descended from ``root_pid`` via ppid lineage.

    Breadth-first over the ``(pid, ppid)`` snapshot ``pairs``. This is
    deliberately independent of process-group/session membership: a
    descendant that calls ``os.setsid()`` to leave its process group and
    become the leader of a new session keeps the ``ppid`` the kernel
    assigned it at fork time -- ``setsid()`` does not reparent a process
    -- so walking ``ppid`` lineage still finds it even though
    ``os.killpg`` would not.
    """
    children_by_ppid: dict[int, list[int]] = {}
    for pid, ppid in pairs:
        children_by_ppid.setdefault(ppid, []).append(pid)

    descendants: list[int] = []
    seen = {root_pid}
    frontier = [root_pid]
    while frontier:
        next_frontier: list[int] = []
        for parent_pid in frontier:
            for child_pid in children_by_ppid.get(parent_pid, ()):
                if child_pid in seen:
                    continue
                seen.add(child_pid)
                descendants.append(child_pid)
                next_frontier.append(child_pid)
        frontier = next_frontier
    return descendants


# Bounds the walk-then-kill loop below. Each pass can only discover
# descendants that existed in that pass's `ps` snapshot; a process that
# forks a new child in the brief window between snapshot and SIGKILL
# would be missed by a single pass. Re-walking a few times closes that
# race without an unbounded loop; SIGKILL is not interruptible so a
# process discovered in an earlier pass and already killed simply will
# not appear (or will report ProcessLookupError, handled below) in a
# later pass.
_MAX_PID_TREE_WALK_PASSES = 3


@dataclass(frozen=True)
class PidTreeWalkOutcome:
    """Result of one ``_kill_pid_tree`` call: did the walk fully confirm the kill?

    Both fields are the two failure modes Maya's PR #14 review required
    visibility for (issue #7's approved Layer 1 design, step 3):

    - ``ps_call_failed``: the underlying ``ps -eo pid=,ppid=`` call
      itself errored or timed out on at least one pass, so that pass's
      lineage snapshot was empty/unavailable and the walk could only
      target the already-known root pid, not any descendant, for that
      pass.
    - ``exhausted_without_confirmed_reap``: the walk used all
      ``_MAX_PID_TREE_WALK_PASSES`` passes without any pass reporting
      zero live targets, so a full reap of the tree is not confirmed
      (some descendant may still be alive).

    ``degraded`` is true if either happened. This does not mean nothing
    was killed -- ``os.kill(SIGKILL)`` was still attempted for every
    target found, and ``_kill_group``'s ``os.killpg`` fallback still runs
    unconditionally afterwards -- it means the walk cannot *positively
    confirm* every descendant was found and reaped, which is exactly the
    silent-failure gap flagged in review.
    """

    ps_call_failed: bool
    exhausted_without_confirmed_reap: bool

    @property
    def degraded(self) -> bool:
        return self.ps_call_failed or self.exhausted_without_confirmed_reap


def _kill_pid_tree(root_pid: int) -> PidTreeWalkOutcome:
    """SIGKILL ``root_pid`` and every descendant found by walking ppid lineage.

    Complements (does not replace) process-group-based termination: a
    process that calls ``os.setsid()`` leaves the process group
    ``os.killpg`` targets, but its ``ppid`` is untouched by ``setsid()``,
    so a fresh ``ps -eo pid=,ppid=`` walk from ``root_pid`` still finds
    it and every process it in turn spawned. Individually SIGKILLs each
    discovered pid -- not a single group-wide signal -- since an escaped
    descendant is, by construction, no longer reachable by one.

    This closes the literal disclosed gap (a single ``os.setsid()`` call
    detaching a descendant from the process group) using only ``ps`` and
    ``os.kill``: no new privileges, no new dependency, works on macOS
    and Linux alike. It does **not** defend against a descendant that
    double-forks to be reparented to PID 1 (init/launchd) before this
    walk runs -- closing that case needs real kernel-enforced
    containment (cgroups v2 ``cgroup.kill`` on Linux, or a container/VM
    boundary on macOS) and is explicitly out of scope here, consistent
    with ADR-0003's existing "containers/gVisor/seccomp... out of scope"
    framing. See ``docs/decisions/0004-pid-tree-walk-setsid-escape-fix.md``.

    Returns a ``PidTreeWalkOutcome`` so callers (``_kill_group`` and, via
    ``MeasuredUsage``, ``run_trainer_contract``) can surface both
    originally-silent failure modes -- a failed/timed-out ``ps`` call
    degrading a pass to a root-pid-only kill, and the bounded retry
    loop exhausting without a pass confirming zero live targets -- into
    both the module logger and the evidence-visible ``TrainingOutput``
    reason string, per Maya's PR #14 review (required remediation item,
    the approved issue #7 Layer 1 design's step 3).
    """
    ps_call_failed = False
    for pass_num in range(_MAX_PID_TREE_WALK_PASSES):
        pairs, ps_ok = _list_pid_ppid_pairs()
        if not ps_ok:
            ps_call_failed = True
            _logger.warning(
                "_kill_pid_tree: pid-tree lineage snapshot unavailable on pass %d/%d "
                "for root_pid=%d ('ps' call failed/timed out); this pass can only "
                "target the already-known root pid, not any descendant -- degraded, "
                "not a full pid-tree kill for this pass. os.killpg fallback in "
                "_kill_group still runs afterwards regardless.",
                pass_num + 1,
                _MAX_PID_TREE_WALK_PASSES,
                root_pid,
            )
        targets = [root_pid, *_descendant_pids(root_pid, pairs)]
        any_alive = False
        for pid in targets:
            try:
                os.kill(pid, signal.SIGKILL)
                any_alive = True
            except ProcessLookupError:
                continue
            except OSError:
                continue
        if not any_alive:
            return PidTreeWalkOutcome(
                ps_call_failed=ps_call_failed, exhausted_without_confirmed_reap=False
            )
    _logger.warning(
        "_kill_pid_tree: exhausted all %d passes for root_pid=%d without any pass "
        "reporting zero live targets; full reap of the pid tree is not confirmed "
        "(a descendant may still be alive). os.killpg fallback in _kill_group still "
        "runs afterwards regardless.",
        _MAX_PID_TREE_WALK_PASSES,
        root_pid,
    )
    return PidTreeWalkOutcome(ps_call_failed=ps_call_failed, exhausted_without_confirmed_reap=True)


def _kill_group(process: Any) -> PidTreeWalkOutcome | None:
    """SIGKILL ``process``, its process group, and its full ppid-lineage tree.

    Two independent mechanisms, both applied in a specific order because
    each closes a gap the other does not:

    - ``_kill_pid_tree`` walks ``ps -eo pid=,ppid=`` lineage from
      ``process.pid`` and SIGKILLs every discovered pid one at a time,
      runs FIRST. This is what catches a descendant that itself calls
      ``os.setsid()`` to escape the process group ``killpg`` targets --
      ``setsid()`` changes process-group/session membership but never
      the kernel-recorded ``ppid``, so the lineage walk still finds it.
      It must run before anything is killed: once a parent process is
      reaped, the kernel immediately reparents its still-live children,
      which would corrupt the very lineage this walk depends on. See
      ``docs/decisions/0004-pid-tree-walk-setsid-escape-fix.md`` and the
      ``SetsidEscapingAdapter`` conformance test.
    - ``os.killpg`` targets the whole OS process group ``process`` leads
      (``_child_worker`` calls ``os.setsid()`` so it becomes a group
      leader; any subprocess it spawns via e.g. ``subprocess.Popen``
      inherits that group), runs SECOND as a redundant safety net for
      anything the walk's pid snapshot happened to miss (e.g. a process
      forked in the narrow window between the walk's last ``ps`` call
      and its kill).

    Falls back to ``process.kill()`` only when ``process.pid`` is
    unknown (should not happen once ``process.start()`` has returned).

    Wrapped for the inherent race: any of these processes may have
    already exited between the liveness check and this call, in which
    case the OS reports ``ProcessLookupError`` / ``ESRCH`` -- not a real
    failure, just confirmation there is nothing left to kill.

    Returns the ``PidTreeWalkOutcome`` from ``_kill_pid_tree`` (or
    ``None`` on the ``process.pid is None``/no-``os.kill`` fallback
    paths, where the pid-tree walk never ran at all) so the caller can
    surface degraded-walk visibility all the way into
    ``MeasuredUsage``/``TrainingOutput.reason`` -- see
    ``docs/decisions/0004-pid-tree-walk-setsid-escape-fix.md``.
    """
    pid = process.pid
    if pid is None:
        process.kill()
        return None

    # Order matters: the ppid-lineage walk must run BEFORE anything is
    # killed. `_child_worker` itself calls `os.setsid()`, so `process`
    # (the tracked isolated child) is already its own process-group
    # leader; killing that group first (as a naive implementation might)
    # would SIGKILL `process` itself before the walk below runs. Once a
    # parent is reaped, the kernel immediately reparents its live
    # children (to PID 1 / the nearest reaper) -- which would change the
    # very ppid lineage `_kill_pid_tree` depends on to find an escaped
    # grandchild, turning a real fix into one that only works when
    # nothing has actually escaped yet. Snapshotting and killing by
    # ppid lineage first (while `process` and everything it spawned are
    # still alive and still show their real lineage in `ps`) avoids that
    # race; `os.killpg` afterwards is then a harmless, redundant safety
    # net for the still-common case where nothing escaped the group.
    walk_outcome: PidTreeWalkOutcome | None = None
    if hasattr(os, "kill") and hasattr(subprocess, "run"):
        walk_outcome = _kill_pid_tree(pid)
    else:  # pragma: no cover - no platform in this project's support matrix hits this
        process.kill()

    if hasattr(os, "killpg") and hasattr(os, "getpgid"):
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            pass

    return walk_outcome


def _directory_size_mb(root: str) -> float:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            try:
                total += os.path.getsize(path)
            except OSError:
                continue
    return total / (1024.0 * 1024.0)


def run_in_isolated_process(
    adapter: Any,
    inputs: Any,
    budget: Any,
    resume_from: Any,
    cancel_event: Any,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    kill_grace_seconds: float = DEFAULT_KILL_GRACE_SECONDS,
    cancellation_kill_grace_seconds: float = DEFAULT_CANCELLATION_KILL_GRACE_SECONDS,
) -> tuple[Any | None, BaseException | None, MeasuredUsage]:
    """Run ``adapter.train`` in a real child process with enforcement.

    Returns ``(output_or_None, exception_or_None, measured_usage)``.
    Exactly one of the first two is non-``None`` unless the child was
    hard-killed (timeout, resource overrun, or a non-cooperative
    cancellation escalating to SIGKILL), in which case both are ``None``
    and ``measured_usage.killed_for_*`` explains why.

    ``kill_grace_seconds`` applies to the wall-clock-timeout path (a
    timed-out run's checkpoint, if any, is discarded regardless, so no
    real-save-completion grace is needed there); ``cancellation_kill_grace_seconds``
    applies to the cooperative-cancellation path, where a genuine
    checkpoint save is expected to complete and be kept -- see the two
    constants' own docstrings for why they differ.
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = ctx.Queue()
    mp_cancel_event = ctx.Event()
    reason_buf = ctx.Array("c", 256)
    process = ctx.Process(
        target=_child_worker,
        args=(adapter, inputs, budget, resume_from, mp_cancel_event, reason_buf, result_queue),
        daemon=True,
    )

    rusage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    process.start()

    peak_memory_mb = 0.0
    live_cpu_seconds = 0.0
    killed_for_overrun = False
    killed_for_timeout = False
    killed_for_cancellation = False
    pid_tree_walk_outcome: PidTreeWalkOutcome | None = None

    while True:
        elapsed = time.monotonic() - start
        if not process.is_alive():
            break

        live = _poll_live_usage(process.pid)  # type: ignore[arg-type]
        if live is not None:
            memory_mb, cpu_seconds = live
            peak_memory_mb = max(peak_memory_mb, memory_mb)
            live_cpu_seconds = max(live_cpu_seconds, cpu_seconds)
            if memory_mb > budget.max_memory_mb or cpu_seconds > budget.max_cpu_seconds:
                killed_for_overrun = True
                pid_tree_walk_outcome = _kill_group(process)
                process.join(timeout=5)
                break

        if elapsed > budget.max_wall_seconds:
            killed_for_timeout = True
            mp_cancel_event.set()
            process.join(timeout=kill_grace_seconds)
            if process.is_alive():
                pid_tree_walk_outcome = _kill_group(process)
                process.join(timeout=5)
            break

        if cancel_event is not None and cancel_event.is_cancelled():
            reason_bytes = cancel_event.reason.encode("utf-8")[:255]
            reason_buf.value = reason_bytes
            mp_cancel_event.set()
            process.join(timeout=cancellation_kill_grace_seconds)
            if process.is_alive():
                killed_for_cancellation = True
                pid_tree_walk_outcome = _kill_group(process)
                process.join(timeout=5)
            break

        time.sleep(poll_interval)

    # Process has exited (or was just killed and reaped above). Join once
    # more defensively so RUSAGE_CHILDREN below reflects a reaped child.
    process.join(timeout=5)
    wall_seconds = round(time.monotonic() - start, 6)

    rusage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    measured_cpu_seconds = max(
        live_cpu_seconds,
        (rusage_after.ru_utime + rusage_after.ru_stime)
        - (rusage_before.ru_utime + rusage_before.ru_stime),
    )
    # ru_maxrss is bytes on macOS, kilobytes on Linux; normalise heuristically.
    maxrss_delta = max(rusage_after.ru_maxrss - rusage_before.ru_maxrss, 0)
    ru_maxrss_mb = maxrss_delta / (1024.0 * 1024.0 if maxrss_delta > 10_000_000 else 1024.0)
    measured_memory_mb = max(peak_memory_mb, ru_maxrss_mb)

    storage_mb_used = None
    if budget.filesystem_root:
        storage_mb_used = _directory_size_mb(budget.filesystem_root)

    measured = MeasuredUsage(
        wall_seconds=wall_seconds,
        cpu_seconds=round(measured_cpu_seconds, 6),
        memory_mb_peak=round(measured_memory_mb, 6),
        storage_mb_used=storage_mb_used,
        killed_for_overrun=killed_for_overrun,
        killed_for_timeout=killed_for_timeout,
        killed_for_cancellation=killed_for_cancellation,
        pid_tree_walk_outcome=pid_tree_walk_outcome,
    )

    if killed_for_overrun or killed_for_timeout or killed_for_cancellation:
        return None, None, measured

    try:
        kind, payload = result_queue.get_nowait()
    except Exception:  # noqa: BLE001 - empty queue (child died without reporting)
        return None, RuntimeError("child process exited without reporting a result"), measured

    if kind == "output":
        return payload, None, measured
    return None, _reconstruct_exception(payload), measured
