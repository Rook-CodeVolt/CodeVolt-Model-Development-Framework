"""Deterministic FAKE evaluator adapter used to exercise EvaluatorAdapterContract.

This adapter never imports or calls a real model-inference engine. It is
deterministic given an artifact's declared "responses" map (a plain
dict baked into the artifact locator's contents, see ``make_fake_artifact``
in the test suite), and exists solely so ``evaluator_contract.py`` has
something concrete and safe to test against. It grants no evaluation,
promotion, or deployment authority. Mirrors the pattern established by
``fake_adapter.py`` for the trainer side.

WP-A (issue #24) task-type breadth: this adapter now dispatches on
``task_type_of(example)`` (default ``exact_match``, unchanged from
before) to also fake-exercise ``multiple_choice`` and
``format_conformance`` scoring -- the deterministic test double for
each new mode, so the contract shape both new modes expect is proven
here before/alongside ``hf_local_evaluator_adapter.py``'s real
(non-fake) implementation of the same two modes. Still stdlib-only, no
real model inference of any kind.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .evaluator_contract import (
    EvaluatorAdapterV1,  # noqa: F401 - imported for documentation/typing clarity
    ExampleResult,
    HeldOutExample,
    InvalidInputError,
    RejectedInputError,
    TaskType,
    task_type_of,
)
from .scoring_modes import (
    check_format_conformance,
    parse_format_conformance_input,
    parse_format_spec,
    parse_multiple_choice_input,
    resolve_expected_choice,
)


@dataclass
class FakeEvaluatorAdapter:
    """Deterministic, in-process, stdlib-only evaluator adapter.

    Reads a JSON "artifact" file at ``artifact_locator`` shaped
    ``{"responses": {example_id: predicted_value, ...}}`` and scores
    each example against ``example.expected`` using the scoring mode
    declared by ``task_type_of(example)``:

    - ``exact_match`` (default, unchanged behaviour): ``correct =
      responses[example_id] == example.expected``.
    - ``multiple_choice``: ``responses[example_id]`` is the adapter's
      chosen answer (choice text or index, same shape as ``expected``);
      resolved to choice text and compared against the resolved
      expected choice.
    - ``format_conformance``: ``responses[example_id]`` is the
      artifact's raw text output for that prompt; checked against
      ``example.expected``'s format spec via
      ``scoring_modes.check_format_conformance``.

    A missing ``example_id`` in ``responses`` scores 0.0/incorrect
    rather than raising, modelling an artifact that simply didn't
    answer that item -- the adapter's job is to score what's there, not
    to require completeness (that's a caller-level acceptance-threshold
    concern, per ``docs/EVALUATION_POLICY.md``, not this contract's).
    """

    name: str = "fake-deterministic-evaluator-v1"
    contract_version: str = "1.0.0"

    def score_example(
        self, artifact_id: str, artifact_locator: str, example: HeldOutExample
    ) -> ExampleResult:
        path = Path(artifact_locator)
        if not path.exists():
            raise RejectedInputError(f"artifact_locator {path} does not exist")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InvalidInputError(f"artifact at {path} is not valid JSON: {exc}") from exc
        responses = payload.get("responses", {})
        if not isinstance(responses, dict):
            raise InvalidInputError(f"artifact at {path} has a non-object 'responses' field")

        task_type = task_type_of(example)
        if task_type == TaskType.MULTIPLE_CHOICE.value:
            return self._score_multiple_choice(example, responses)
        if task_type == TaskType.FORMAT_CONFORMANCE.value:
            return self._score_format_conformance(example, responses)
        if task_type != TaskType.EXACT_MATCH.value:
            raise InvalidInputError(
                f"example {example.example_id!r}: unsupported task_type {task_type!r} "
                f"(fake adapter supports {[t.value for t in TaskType]})"
            )

        predicted = responses.get(example.example_id, None)
        correct = predicted == example.expected
        return ExampleResult(
            example_id=example.example_id,
            correct=correct,
            score=1.0 if correct else 0.0,
            raw_output=json.dumps(predicted),
        )

    # -- new scoring modes (WP-A, issue #24) -----------------------------

    def _score_multiple_choice(self, example: HeldOutExample, responses: dict) -> ExampleResult:
        _prompt, choices = parse_multiple_choice_input(example)
        expected_choice = resolve_expected_choice(example.expected, choices)

        predicted_raw = responses.get(example.example_id, None)
        if predicted_raw is None:
            return ExampleResult(
                example_id=example.example_id, correct=False, score=0.0, raw_output="null"
            )
        try:
            predicted_choice = resolve_expected_choice(predicted_raw, choices)
        except InvalidInputError:
            # The fake artifact's own answer doesn't name a real choice --
            # that is simply an incorrect answer, not a contract violation
            # (mirrors exact_match's "missing/unrecognised response scores
            # 0.0, doesn't raise" discipline).
            return ExampleResult(
                example_id=example.example_id,
                correct=False,
                score=0.0,
                raw_output=json.dumps(predicted_raw),
            )
        correct = predicted_choice == expected_choice
        return ExampleResult(
            example_id=example.example_id,
            correct=correct,
            score=1.0 if correct else 0.0,
            raw_output=predicted_choice,
        )

    def _score_format_conformance(self, example: HeldOutExample, responses: dict) -> ExampleResult:
        parse_format_conformance_input(example)
        spec = parse_format_spec(example)

        predicted_text = responses.get(example.example_id, None)
        if predicted_text is None:
            return ExampleResult(
                example_id=example.example_id,
                correct=False,
                score=0.0,
                raw_output="(no response)",
            )
        if not isinstance(predicted_text, str):
            raise InvalidInputError(
                f"example {example.example_id!r}: format_conformance fake artifact response "
                f"must be a string, got {type(predicted_text).__name__}"
            )
        conforms, detail = check_format_conformance(predicted_text, spec)
        return ExampleResult(
            example_id=example.example_id,
            correct=conforms,
            score=1.0 if conforms else 0.0,
            raw_output=f"{predicted_text} | {detail}",
        )
