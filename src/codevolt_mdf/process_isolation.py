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
- **Filesystem containment (Python-level)**: when ``budget.filesystem_root``
  is set, the child's cwd is pinned there and ``builtins.open`` /
  ``os.open`` are wrapped to reject any path that resolves outside that
  root, for code running inside the child interpreter.
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
  paths outside the declared root.
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

_ENV_ALLOWLIST = ("PATH", "PYTHONPATH", "HOME", "LANG", "LC_ALL", "TMPDIR")

# How often the parent polls the child's live resource usage via `ps`.
DEFAULT_POLL_INTERVAL_SECONDS = 0.05

# Grace period given to a cooperative cancellation signal before the
# parent escalates to SIGKILL. Bounded and short: this is what makes
# cancellation "real" rather than an indefinite cooperative wait.
DEFAULT_KILL_GRACE_SECONDS = 1.0


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


def _pin_filesystem_root(filesystem_root: str) -> None:
    """Pin cwd to ``filesystem_root`` and reject Python-level I/O outside it.

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
        if root not in resolved.parents and resolved != root:
            raise SandboxViolationError(
                f"path {resolved} is outside the declared filesystem_root {root}"
            )

    def _guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            _check(file)
        return real_open(file, *args, **kwargs)

    def _guarded_os_open(path, *args, **kwargs):
        _check(path)
        return real_os_open(path, *args, **kwargs)

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


def _kill_group(process: Any) -> None:
    """SIGKILL the entire process group led by ``process``, not just it.

    ``_child_worker`` calls ``os.setsid()`` so it leads a new process
    group; any subprocess it spawns (e.g. ``subprocess.Popen``) inherits
    that group. Killing only ``process.pid`` (the old behaviour) leaves
    such grandchildren running as orphans. ``os.killpg`` targets the whole
    group at once. Falls back to ``process.kill()`` when the pid is
    unknown or the process never became a group leader (e.g. platforms
    without ``os.setsid``/``os.killpg``, such as native Windows).

    Wrapped for the inherent race: the process (and therefore its group)
    may have already exited between the liveness check and this call, in
    which case the OS reports ``ProcessLookupError`` / ``ESRCH`` -- not a
    real failure, just confirmation there is nothing left to kill.
    """
    pid = process.pid
    if pid is not None and hasattr(os, "killpg") and hasattr(os, "getpgid"):
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    process.kill()


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
) -> tuple[Any | None, BaseException | None, MeasuredUsage]:
    """Run ``adapter.train`` in a real child process with enforcement.

    Returns ``(output_or_None, exception_or_None, measured_usage)``.
    Exactly one of the first two is non-``None`` unless the child was
    killed for a timeout/overrun, in which case both are ``None`` and
    ``measured_usage.killed_for_*`` explains why.
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
                _kill_group(process)
                process.join(timeout=5)
                break

        if elapsed > budget.max_wall_seconds:
            killed_for_timeout = True
            mp_cancel_event.set()
            process.join(timeout=kill_grace_seconds)
            if process.is_alive():
                _kill_group(process)
                process.join(timeout=5)
            break

        if cancel_event is not None and cancel_event.is_cancelled():
            reason_bytes = cancel_event.reason.encode("utf-8")[:255]
            reason_buf.value = reason_bytes
            mp_cancel_event.set()
            process.join(timeout=kill_grace_seconds)
            if process.is_alive():
                _kill_group(process)
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
    )

    if killed_for_overrun or killed_for_timeout:
        return None, None, measured

    try:
        kind, payload = result_queue.get_nowait()
    except Exception:  # noqa: BLE001 - empty queue (child died without reporting)
        return None, RuntimeError("child process exited without reporting a result"), measured

    if kind == "output":
        return payload, None, measured
    return None, _reconstruct_exception(payload), measured
