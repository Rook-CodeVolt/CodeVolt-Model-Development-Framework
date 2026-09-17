"""Deterministic FAKE evaluator adapter used to exercise EvaluatorAdapterContract.

This adapter never imports or calls a real model-inference engine. It is
deterministic given an artifact's declared "responses" map (a plain
dict baked into the artifact locator's contents, see ``make_fake_artifact``
in the test suite), and exists solely so ``evaluator_contract.py`` has
something concrete and safe to test against. It grants no evaluation,
promotion, or deployment authority. Mirrors the pattern established by
``fake_adapter.py`` for the trainer side.
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
)


@dataclass
class FakeEvaluatorAdapter:
    """Deterministic, in-process, stdlib-only evaluator adapter.

    Reads a JSON "artifact" file at ``artifact_locator`` shaped
    ``{"responses": {example_id: predicted_value, ...}}`` and scores
    each example by exact-match against ``example.expected``. A missing
    ``example_id`` in ``responses`` scores 0.0/incorrect rather than
    raising, modelling an artifact that simply didn't answer that item
    -- the adapter's job is to score what's there, not to require
    completeness (that's a caller-level acceptance-threshold concern,
    per ``docs/EVALUATION_POLICY.md``, not this contract's).
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

        predicted = responses.get(example.example_id, None)
        correct = predicted == example.expected
        return ExampleResult(
            example_id=example.example_id,
            correct=correct,
            score=1.0 if correct else 0.0,
            raw_output=json.dumps(predicted),
        )
