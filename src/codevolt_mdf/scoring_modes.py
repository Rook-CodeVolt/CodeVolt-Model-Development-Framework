"""Shared, stdlib-only parsing/validation helpers for the non-exact-match scoring modes.

WP-A (issue #24) task-type breadth: ``multiple_choice`` (per-option
likelihood scoring, the standard MMLU-style approach ``lm-evaluation-
harness`` and others use) and ``format_conformance`` (structured-output
checks such as "is this valid JSON"). This module defines the *shape*
both new modes expect ``HeldOutExample.input``/``expected`` to take and
validates it -- it does not itself run any model or decide correctness
against model output. Both ``fake_evaluator_adapter.py`` (the
deterministic test double) and ``hf_local_evaluator_adapter.py`` (the
real adapter) import from here, so a held-out example's own
well-formedness is validated identically regardless of which adapter
scores it, and a fake-adapter test proves the same input-shape
contract a real-adapter test relies on.

Deliberately separate from ``evaluator_contract.py`` itself (which
stays a pure contract definition) and, like every other module in this
evaluator family, has zero import relationship with
``trl_adapter.py``/``trainer_contract.py`` -- see
``tests/test_scoring_modes.py``'s AST-import test.

Stdlib-only, matching ``evaluator_contract.py``'s own dependency-light
discipline: this module never imports ``torch``/``transformers``, even
though the real adapter that calls it does.
"""

from __future__ import annotations

import json
import re
import signal
import threading
from typing import Any

from .evaluator_contract import EvaluatorContractError, HeldOutExample, InvalidInputError

# Hard wall-clock bound for scoring a single regex against a single output.
# Dataset-author-controlled patterns (HeldOutExample.expected["pattern"]) are
# untrusted input: a pathological pattern like ``(a+)+$`` can trigger
# catastrophic backtracking in Python's re engine and hang effectively
# forever on a short string. See issue #26.
_REGEX_MATCH_TIMEOUT_SECONDS = 2.0

# signal.alarm/setitimer only works in the main thread of the main
# interpreter on POSIX. This process is a POSIX (macOS/Linux) dev/CI
# target, and re.search's C-level backtracking loop does not otherwise
# yield to check for a thread-join timeout (a plain worker-thread timeout
# would appear to "work" but never actually preempt the stuck match --
# verified empirically), so SIGALRM is the one mechanism that reliably
# interrupts it.
# signal.alarm/setitimer only works in the main thread of the main
# interpreter on POSIX. Whether the *current* call is happening on that
# thread can change between calls (e.g. a worker-pool caller), so this is
# a function checked fresh on every call, not a module-level constant
# computed once at import time.
def _sigalrm_usable() -> bool:
    return (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )

# --------------------------------------------------------------------------
# Multiple-choice / likelihood scoring
# --------------------------------------------------------------------------
#
# HeldOutExample shape expected for TaskType.MULTIPLE_CHOICE:
#   input:    {"prompt": str, "choices": [str, str, ...]} (>= 2 distinct,
#             non-empty choices)
#   expected: either the correct choice's exact text (str) or its
#             0-based index into "choices" (int)
#   metadata: must include ("task_type", "multiple_choice")


def parse_multiple_choice_input(example: HeldOutExample) -> tuple[str, list[str]]:
    """Validate and unpack a multiple-choice ``HeldOutExample.input``.

    Returns ``(prompt, choices)``. Raises ``InvalidInputError`` for any
    malformed shape -- never raises anything else, so callers can treat
    ``InvalidInputError`` as the complete set of failure modes here.
    """
    payload = example.input
    if not isinstance(payload, dict):
        raise InvalidInputError(
            f"example {example.example_id!r}: multiple_choice HeldOutExample.input must be a "
            f"dict with 'prompt' and 'choices' keys, got {type(payload).__name__}"
        )
    prompt = payload.get("prompt")
    choices = payload.get("choices")
    if not isinstance(prompt, str) or not prompt.strip():
        raise InvalidInputError(
            f"example {example.example_id!r}: multiple_choice input['prompt'] must be a "
            "non-empty string"
        )
    if not isinstance(choices, list) or len(choices) < 2:
        raise InvalidInputError(
            f"example {example.example_id!r}: multiple_choice input['choices'] must be a list "
            "of at least 2 candidate strings"
        )
    if not all(isinstance(choice, str) and choice.strip() for choice in choices):
        raise InvalidInputError(
            f"example {example.example_id!r}: every multiple_choice choice must be a "
            "non-empty string"
        )
    if len(set(choices)) != len(choices):
        raise InvalidInputError(
            f"example {example.example_id!r}: multiple_choice input['choices'] must not "
            "contain duplicate choice text"
        )
    return prompt, list(choices)


def resolve_expected_choice(expected: Any, choices: list[str]) -> str:
    """Resolve a multiple-choice ``expected`` value (index or text) to choice text.

    ``expected`` may be the correct choice's exact text (``str``) or its
    0-based index into ``choices`` (``int``, excluding ``bool`` which is
    a ``bool``/``int`` subclass in Python and would be a confusing
    silent index). Raises ``InvalidInputError`` if ``expected`` is
    neither, or an index/text that isn't actually one of ``choices``.
    """
    if isinstance(expected, bool):
        raise InvalidInputError(
            f"multiple_choice expected={expected!r} must not be a bool; use the choice's "
            "exact text or its integer index"
        )
    if isinstance(expected, int):
        if not (0 <= expected < len(choices)):
            raise InvalidInputError(
                f"multiple_choice expected index {expected} is out of range for "
                f"{len(choices)} choice(s)"
            )
        return choices[expected]
    if isinstance(expected, str):
        if expected not in choices:
            raise InvalidInputError(
                f"multiple_choice expected={expected!r} is not one of the declared choices "
                f"{choices!r}"
            )
        return expected
    raise InvalidInputError(
        f"multiple_choice expected must be a str (exact choice text) or int (choice index), "
        f"got {type(expected).__name__}"
    )


# --------------------------------------------------------------------------
# Structured-output / format-conformance scoring
# --------------------------------------------------------------------------
#
# HeldOutExample shape expected for TaskType.FORMAT_CONFORMANCE:
#   input:    str, a non-empty prompt
#   expected: a format spec dict, one of:
#               {"format": "json"}
#               {"format": "json", "required_keys": [str, ...]}
#               {"format": "regex", "pattern": str}
#   metadata: must include ("task_type", "format_conformance")

_SUPPORTED_FORMATS = ("json", "regex")


class RegexScoringTimeoutError(EvaluatorContractError):
    """A format_conformance regex pattern did not finish matching in time.

    Raised instead of letting a pathological, dataset-author-controlled
    pattern (e.g. one with nested quantifiers like ``(a+)+$``) hang the
    evaluation run via catastrophic backtracking. See issue #26.
    """


class RegexScoringUnsupportedContextError(EvaluatorContractError):
    """format_conformance regex scoring has no way to bound this call's wait time.

    Raised immediately (never after waiting out a timeout) when SIGALRM
    isn't usable -- i.e. this call is happening off the POSIX main thread,
    or on a platform without SIGALRM/setitimer (e.g. Windows). See the
    PR #27 review discussion: a thread-join timeout does *not* actually
    bound wait time for a catastrophically-backtracking pattern, because
    ``re.search``'s C loop never yields the GIL, so the joining thread
    can't get scheduled to notice its timeout elapsed either. Rather than
    silently returning a result from an unbounded match (which looks
    handled but isn't), we fail closed here: callers must not invoke
    format_conformance regex checks from a context where SIGALRM can't be
    used, or must catch this and handle it explicitly (e.g. route the
    check to a real subprocess-based executor of their own).
    """


def parse_format_conformance_input(example: HeldOutExample) -> str:
    """Validate a format-conformance ``HeldOutExample.input`` is a non-empty prompt string."""
    prompt = example.input
    if not isinstance(prompt, str) or not prompt.strip():
        raise InvalidInputError(
            f"example {example.example_id!r}: format_conformance HeldOutExample.input must be "
            "a non-empty string prompt"
        )
    return prompt


def parse_format_spec(example: HeldOutExample) -> dict[str, Any]:
    """Validate and return a format-conformance ``HeldOutExample.expected`` spec."""
    spec = example.expected
    if not isinstance(spec, dict):
        raise InvalidInputError(
            f"example {example.example_id!r}: format_conformance HeldOutExample.expected must "
            f"be a dict format spec (e.g. {{'format': 'json'}}), got {type(spec).__name__}"
        )
    fmt = spec.get("format")
    if fmt not in _SUPPORTED_FORMATS:
        raise InvalidInputError(
            f"example {example.example_id!r}: format_conformance expected['format'] must be "
            f"one of {_SUPPORTED_FORMATS}, got {fmt!r}"
        )
    if fmt == "regex":
        pattern = spec.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise InvalidInputError(
                f"example {example.example_id!r}: format_conformance regex spec requires a "
                "non-empty string 'pattern'"
            )
        try:
            re.compile(pattern)
        except re.error as exc:
            raise InvalidInputError(
                f"example {example.example_id!r}: format_conformance regex pattern "
                f"{pattern!r} does not compile: {exc}"
            ) from exc
    if fmt == "json" and "required_keys" in spec:
        required_keys = spec["required_keys"]
        if not isinstance(required_keys, list) or not all(
            isinstance(key, str) for key in required_keys
        ):
            raise InvalidInputError(
                f"example {example.example_id!r}: format_conformance json spec's "
                "'required_keys' must be a list of strings"
            )
    return spec


def check_format_conformance(text: str, spec: dict[str, Any]) -> tuple[bool, str]:
    """Check whether ``text`` conforms to a validated format ``spec``.

    Returns ``(conforms, detail)``; ``detail`` is a human-readable
    explanation either way, carried into ``ExampleResult.raw_output``
    by callers so a non-conforming result is diagnosable, not just a
    bare ``False``. Assumes ``spec`` already passed ``parse_format_spec``
    -- does not re-validate the spec's own shape.
    """
    fmt = spec["format"]
    if fmt == "json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return False, f"output is not valid JSON: {exc}"
        required_keys = spec.get("required_keys")
        if required_keys:
            if not isinstance(parsed, dict):
                return False, (
                    "output is valid JSON but not a JSON object, so required_keys cannot "
                    "be checked"
                )
            missing = [key for key in required_keys if key not in parsed]
            if missing:
                return False, f"output JSON object is missing required key(s): {missing}"
            return True, f"valid JSON object with all required key(s) {required_keys}"
        return True, "valid JSON"
    if fmt == "regex":
        pattern = spec["pattern"]
        if _regex_search_bounded(pattern, text):
            return True, f"output matches pattern {pattern!r}"
        return False, f"output does not match pattern {pattern!r}"
    # Unreachable if callers always validate via parse_format_spec first.
    raise InvalidInputError(f"unsupported format_conformance spec format {fmt!r}")


# SIGALRM interrupts re.search's C-level backtracking loop directly, so
# it is the only mechanism here that reliably bounds wait time for a
# catastrophically-backtracking pattern. It only works on POSIX and only
# from the interpreter's main thread. A plain thread-join timeout looks
# like a fallback but is NOT one: re.search's C loop never releases the
# GIL while backtracking, so a joining thread can't get scheduled to
# even notice its timeout elapsed -- it silently waits out the full
# match just like an unguarded call would (empirically confirmed: 44.5s+
# on a pathological input that SIGALRM correctly bounds to 2s; see the
# PR #27 review). Rather than ship a "guard" that looks handled but
# isn't, we fail closed: when SIGALRM isn't usable we raise immediately
# instead of attempting an unbounded wait.
def _regex_search_bounded(pattern: str, text: str) -> bool:
    """Run ``re.search(pattern, text)`` with a hard wall-clock timeout.

    Raises ``RegexScoringTimeoutError`` if the match does not finish
    within ``_REGEX_MATCH_TIMEOUT_SECONDS``, instead of letting a
    catastrophically-backtracking pattern hang the caller indefinitely.
    See issue #26.

    Raises ``RegexScoringUnsupportedContextError`` immediately (not
    after any wait) if this call cannot use SIGALRM -- i.e. it is
    running off the POSIX main thread, or on a platform without
    SIGALRM/setitimer -- since no other mechanism available here can
    actually bound the wait for a catastrophically-backtracking
    pattern. See the PR #27 review.
    """
    if _sigalrm_usable():
        return _regex_search_bounded_via_signal(pattern, text)
    return _regex_search_bounded_via_thread(pattern, text)


def _regex_search_bounded_via_signal(pattern: str, text: str) -> bool:
    def _on_alarm(signum: int, frame: Any) -> None:
        raise RegexScoringTimeoutError(
            f"format_conformance regex pattern {pattern!r} did not finish matching within "
            f"{_REGEX_MATCH_TIMEOUT_SECONDS}s (possible catastrophic backtracking); "
            "rejecting this example instead of hanging the evaluation run"
        )

    previous_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, _REGEX_MATCH_TIMEOUT_SECONDS)
    try:
        return re.search(pattern, text) is not None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _regex_search_bounded_via_thread(pattern: str, text: str) -> bool:
    """Fail closed: this call cannot bound a catastrophically-backtracking match.

    Deliberately does NOT attempt a thread-join timeout. A join-based
    "fallback" was tried and empirically does not work: ``re.search``'s
    C loop never yields the GIL while backtracking, so the calling
    thread can't get scheduled to observe its own join timeout elapsing
    either -- it just waits out the full unbounded match, silently. That
    is worse than no guard at all because it looks handled. Only
    process-level preemption (``multiprocessing`` + ``.terminate()``)
    can interrupt a GIL-holding C loop from outside, and spawning a
    subprocess per regex check is a complexity/reliability cost this
    narrow edge case (off-main-thread or non-POSIX callers) does not
    currently justify. So: raise immediately instead of pretending to
    guard anything. See the PR #27 review (Maya-CodeVolt) for the full
    empirical finding (44.5s-180s+ observed wait on pathological input
    via the old thread-join code).
    """
    raise RegexScoringUnsupportedContextError(
        f"format_conformance regex pattern {pattern!r} cannot be safely bounded here: "
        "SIGALRM is not usable in this context (this call is off the POSIX main thread, "
        "or running on a platform without SIGALRM/setitimer), and a thread-join timeout "
        "cannot actually interrupt a catastrophically-backtracking match -- re.search's C "
        "loop never yields the GIL, so a joining thread cannot observe its own timeout "
        "elapsing either. Call format_conformance regex scoring from the POSIX main thread, "
        "or handle this error explicitly (e.g. dispatch it to a subprocess-based executor)."
    )

