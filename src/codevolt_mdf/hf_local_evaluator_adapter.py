"""Real (non-fake) EvaluatorAdapterV1 backed by local Hugging Face causal-LM inference.

This is the first real, non-deterministic-fake scoring adapter for
``evaluator_contract.py`` (issue #7 step 4's "real (non-fake) evaluator
adapter" gap, explicitly left open by
``docs/decisions/0006-bounded-real-trl-pilot-plan.md``'s "Independent
evaluation wiring" section: "the evaluator adapter used for the pilot's
actual scoring is not decided by this ADR -- FakeEvaluatorAdapter proves
the contract but is not a real capability-scoring adapter ... a real
scoring adapter (even a minimal exact-match ... check) is separate,
not-yet-authorised follow-up work this ADR does not perform"). This
module is that follow-up work: it runs real offline inference through
``transformers`` against a pinned local model checkpoint and scores each
held-out example by a minimal exact-match/containment check against the
model's greedy-decoded continuation. It is still deliberately minimal --
not a general-purpose evaluation harness -- consistent with the ADR's own
framing of what a first real adapter needs to be.

Pinned model, matching ADR-0006's draft pilot plan for consistency:
``HuggingFaceTB/SmolLM2-135M`` (Apache-2.0, 135M parameters), the same
model this ADR proposes for the (separately gated, not-yet-authorised)
bounded real TRL pilot. Reusing it here is a deliberate consistency
choice, not a requirement this module imposes on its own -- any local
Hugging Face ``AutoModelForCausalLM``-compatible checkpoint directory
works, provided its content hash is verified the same way a trainer
adapter verifies input provenance (see ``_hash_model_dir`` below).

No training run happens anywhere in this package. There is no trained
pilot artifact yet -- ADR-0006 is drafted, not executed, and no
pilot has been scheduled. Every test in
``tests/test_hf_local_evaluator_adapter.py`` scores held-out examples
against the pristine, untrained base checkpoint as a manually-provided
stand-in artifact, purely to exercise this adapter's real-inference
scoring path end-to-end -- it is contract-conformance evidence for the
*scoring adapter*, not pilot evidence for a trained candidate. This
module never calls ``model.train()``, never computes gradients (all
inference runs under ``torch.no_grad()``), and never writes to a
checkpoint directory it reads from.

Structural separation, mirroring ``evaluator_contract.py``'s own
discipline: nothing in this module imports or is imported by
``trl_adapter.py``/``trainer_contract.py`` -- proven structurally by
``tests/test_hf_local_evaluator_adapter.py``'s AST-import test, the same
pattern PR #16 established for ``evaluator_contract.py`` itself. Because
of that separation, this module cannot reuse ``trl_adapter.py``'s
private ``_hash_path_identity`` helper; ``_hash_model_dir`` below is a
deliberately independent reimplementation of the same
sorted-file-manifest hashing approach, not a shared import.

Role separation (``docs/ARCHITECTURE.md`` core contract #3 / core
contract #6, restated by ``docs/TRAINER_ADAPTER_CONTRACT.md``'s
"Trainer / evaluator / security-review role separation"): this adapter
has exactly one public method, ``score_example``, which returns a
per-example ``ExampleResult`` (a measurement) and nothing else. It has
no method that accepts, rejects, promotes, or otherwise decides whether
a candidate passes -- ``run_evaluator_contract`` only ever aggregates
this adapter's per-example scores into ``EvaluationOutput.aggregate_score``,
which itself carries no accept/reject/promoted field (see
``evaluator_contract.EvaluationOutput``). Applying a threshold to that
score and deciding promotion is a separate, later governed action this
module does not perform and has no authority to perform.

Held-out isolation: this adapter is invoked exclusively through
``evaluator_contract.run_evaluator_contract``, the same entry point the
fake evaluator uses, with no special-casing -- it goes through the same
``HeldOutExclusionRegistry.check_held_out_not_trained`` bidirectional
contamination check, the same ``HeldOutSet.validate()`` tamper check, and
the same per-example evidence-writing path as every other conforming
``EvaluatorAdapterV1``.

Dependency-heavy by necessity (unlike ``evaluator_contract.py`` itself,
which stays stdlib-only): this module needs ``torch``/``transformers``
to run real inference. Both are imported lazily, inside methods, not at
module import time -- mirroring ``trl_adapter.py``'s own lazy-import
discipline -- so this module stays importable (and every path-validation
/ rejection-path test in it stays runnable) even in an environment where
those optional, heavy dependencies are not installed. Tests that need
real inference use ``pytest.importorskip("transformers")`` and skip
cleanly if the pinned local model snapshot is not present on disk,
rather than failing hard.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    check_safety_probe,
    parse_format_conformance_input,
    parse_format_spec,
    parse_multiple_choice_input,
    parse_safety_probe_input,
    parse_safety_probe_spec,
    resolve_expected_choice,
)

# --------------------------------------------------------------------------
# Pinned model identity, matching ADR-0006's draft bounded-pilot plan.
# --------------------------------------------------------------------------

PINNED_MODEL_REPO = "HuggingFaceTB/SmolLM2-135M"
"""Reused from ADR-0006's proposed pilot model for consistency, not because
this adapter requires that exact model -- any local, hash-verified,
``AutoModelForCausalLM``-compatible checkpoint directory is accepted."""

_HASH_MANIFEST_EXCLUDE_NAMES = {".DS_Store", "__pycache__"}
_REQUIRED_CHECKPOINT_FILES = ("config.json",)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_model_dir(path: Path) -> str:
    """Deterministically hash a model checkpoint directory's content identity.

    Independent reimplementation of ``trl_adapter._hash_path_identity``'s
    sorted-file-manifest approach (see the module docstring for why this
    is duplicated rather than imported): a sorted JSON manifest of
    ``(relative_posix_path, size_bytes, sha256)`` for every regular file
    under ``path`` (hidden/cache noise excluded), so the same directory
    contents always produce the same hash regardless of filesystem
    traversal order, and any content change changes the hash.
    """
    if not path.is_dir():
        raise RejectedInputError(f"artifact_locator {path} is not a directory")
    entries: list[tuple[str, int, str]] = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.name in _HASH_MANIFEST_EXCLUDE_NAMES:
            continue
        rel = candidate.relative_to(path).as_posix()
        entries.append((rel, candidate.stat().st_size, _sha256_file(candidate)))
    manifest = json.dumps(entries, sort_keys=True).encode("utf-8")
    return hashlib.sha256(manifest).hexdigest()


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


_KNOWN_UNSUPPORTED_MODEL_KWARGS = frozenset({"token_type_ids"})
"""Narrow fallback deny-list for tokenizer fields known to break causal LMs.

Used only when a model interface contains ``**kwargs`` (or cannot be
introspected), where an exact allow-list cannot be proven.  ``token_type_ids``
is the reproduced issue #76 incompatibility for Qwen3/Llama-style causal LMs;
named support in an inspected signature always wins over this fallback.
"""


def _model_kwarg_policy(model: Any, *, for_generation: bool) -> tuple[set[str], bool] | None:
    """Return explicitly accepted names and whether the interface is open-ended.

    Fixes issue #76 finding 1: the adapter used to forward the
    tokenizer's entire output mapping (e.g. input_ids, attention_mask,
    token_type_ids) straight into model.generate()/model(...) unfiltered.
    Some tokenizer classes (e.g. the PreTrainedTokenizerFast produced by
    the committed MiniMind->Qwen3 conversion in
    examples/pilot-adr0011/convert_checkpoint.py) always emit
    token_type_ids, but Qwen3ForCausalLM.forward (and many other
    causal-LM architectures) does not accept that keyword at all, so
    generation/scoring failed outright with "The following model_kwargs
    are not used by the model: ['token_type_ids']".

    For direct scoring, the actual interface is ``model.forward``.  For
    generation it is the union of ``forward`` and
    ``prepare_inputs_for_generation``, matching Transformers' own model-kwarg
    validation.  A ``**kwargs`` parameter makes the interface open-ended but
    does not erase the explicitly named fields; the caller can therefore keep
    fields such as ``attention_mask`` while applying the narrow known-bad
    fallback to unnamed ``token_type_ids``.
    """
    import inspect

    callables = [getattr(model, "forward", None)]
    if for_generation:
        callables.append(getattr(model, "prepare_inputs_for_generation", None))

    accepted: set[str] = set()
    accepts_var_kwargs = False
    inspected_any = False
    for callable_obj in callables:
        if callable_obj is None:
            continue
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            continue
        inspected_any = True
        for name, param in signature.parameters.items():
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                accepts_var_kwargs = True
            elif param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                accepted.add(name)

    if not inspected_any:
        return None
    return accepted, accepts_var_kwargs


def _filter_model_kwargs(model: Any, inputs: dict, *, for_generation: bool = False) -> dict:
    """Keep tokenizer fields supported by the applicable model interface.

    input_ids is always retained even if introspection somehow missed
    it (every causal-LM forward accepts it; losing it would turn a
    filtering bug into total breakage instead of a clear error).  Closed
    signatures use an exact allow-list.  Open/uninspectable signatures retain
    everything except the narrow known-incompatible fallback fields, unless a
    field is explicitly named by an inspected interface.
    """
    policy = _model_kwarg_policy(model, for_generation=for_generation)
    if policy is None:
        filtered = {
            key: value for key, value in inputs.items() if key not in _KNOWN_UNSUPPORTED_MODEL_KWARGS
        }
    else:
        accepted, accepts_var_kwargs = policy
        if accepts_var_kwargs:
            filtered = {
                key: value
                for key, value in inputs.items()
                if key in accepted or key not in _KNOWN_UNSUPPORTED_MODEL_KWARGS
            }
        else:
            filtered = {key: value for key, value in inputs.items() if key in accepted}
    if "input_ids" not in filtered and "input_ids" in inputs:
        filtered["input_ids"] = inputs["input_ids"]
    return filtered


CHAT_TEMPLATE_RENDERER_ID = "chat_template:sha256="
"""Prefix for content-derived chat-template renderer identities in evidence."""

_CHAT_TEMPLATE_RENDER_OPTIONS = {
    "add_generation_prompt": True,
    "tokenize": False,
    "retokenize_return_tensors": "pt",
}


def _chat_template_renderer_id(chat_template: Any) -> str:
    """Hash the exact template value together with every rendering option."""
    identity_payload = json.dumps(
        {"chat_template": chat_template, "options": _CHAT_TEMPLATE_RENDER_OPTIONS},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(identity_payload).hexdigest()
    return (
        f"{CHAT_TEMPLATE_RENDERER_ID}{digest};"
        "add_generation_prompt=true;tokenize=false;retokenize_return_tensors=pt"
    )

BARE_TEXT_RENDERER_ID = "bare_text"
"""Renderer identity recorded in evidence when no chat template is available/used."""


# --------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------


@dataclass
class HFLocalCausalLMEvaluatorAdapter:
    """Real ``EvaluatorAdapterV1`` implementation: local HF causal-LM inference.

    ``artifact_locator`` (passed per-call by ``run_evaluator_contract``,
    not stored on this dataclass) must be a local filesystem path to a
    Hugging Face ``AutoModelForCausalLM``-compatible checkpoint directory
    (a ``config.json`` plus tokenizer/weight files) -- never a Hub id;
    this adapter never downloads anything and never talks to a model hub
    (it forces ``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE`` before every
    load as defence in depth, mirroring ``TRLTrainerAdapter.train()``'s
    same discipline).

    ``expected_model_hash``, if set, is verified against
    ``_hash_model_dir(artifact_locator)`` before any example is scored --
    the same "verify by content, not by trusted name" discipline
    ``TRLTrainerAdapter.prepare()`` applies to ``model_hash``/
    ``dataset_hash``. Left ``None`` by default because
    ``EvaluatorAdapterV1.score_example``'s signature (fixed by the
    contract) has no hash field of its own to carry a per-call declared
    hash; a caller that wants pinned provenance sets this field on the
    adapter instance before calling ``run_evaluator_contract``.

    Each ``HeldOutExample.input`` must be a non-empty string prompt;
    each ``HeldOutExample.expected`` must be a non-empty string. Scoring
    is a minimal case-insensitive, whitespace-normalized containment
    check: ``correct = normalize(expected) in normalize(generated)``,
    where ``generated`` is the model's own greedy-decoded continuation
    (``do_sample=False``, fixed ``max_new_tokens``) -- deterministic
    given the same artifact and example, so re-running this adapter
    against the same inputs reproduces the same evidence.
    """

    name: str = "hf-local-causal-lm-evaluator-v1"
    contract_version: str = "1.0.0"
    max_new_tokens: int = 8
    expected_model_hash: str | None = None
    use_chat_template: bool = True
    """When True (default), a prompt is rendered through the tokenizer's
    own ``chat_template`` (via ``apply_chat_template(...,
    add_generation_prompt=True)``) whenever the loaded tokenizer exposes
    a usable one -- matching how chat-SFT checkpoints (e.g. MiniMind's
    ``conversations``-schema training records, see
    ``examples/pilot-adr0011/train.jsonl``) are actually trained, per
    issue #76. Falls back to bare-text tokenization automatically when
    the tokenizer has no ``chat_template`` (e.g. a base/non-chat
    checkpoint), so this default is safe for both chat and non-chat
    artifacts. Set False to force bare-text rendering even when a chat
    template is present."""
    _model_cache: dict[str, tuple[Any, Any]] = field(
        default_factory=dict, repr=False, compare=False
    )

    # -- contract method (the only public one) ---------------------------

    def score_example(
        self, artifact_id: str, artifact_locator: str, example: HeldOutExample
    ) -> ExampleResult:
        """Score one held-out example against a local model checkpoint.

        Never trains, never mutates the checkpoint (loaded read-only,
        every forward pass runs under ``torch.no_grad()``), never decides
        accept/reject/promotion -- returns a measurement only.
        """
        path = Path(artifact_locator)
        if not path.exists():
            raise RejectedInputError(f"artifact_locator {path} does not exist")
        if not path.is_dir():
            raise RejectedInputError(
                f"artifact_locator {path} is not a directory; this adapter expects a "
                "local Hugging Face checkpoint directory (config.json + tokenizer + weights)"
            )
        missing = [name for name in _REQUIRED_CHECKPOINT_FILES if not (path / name).exists()]
        if missing:
            raise RejectedInputError(
                f"artifact_locator {path} is missing required checkpoint file(s) {missing}; "
                "not a supported Hugging Face causal-LM checkpoint directory"
            )

        if self.expected_model_hash is not None:
            actual_hash = _hash_model_dir(path)
            if actual_hash != self.expected_model_hash:
                raise RejectedInputError(
                    f"model_hash mismatch: declared {self.expected_model_hash}, actual "
                    f"content at {path} hashes to {actual_hash} (refusing to score against "
                    "an unverified/tampered checkpoint)"
                )

        # Shape-validate the example BEFORE ever touching the model, for every
        # task type -- mirrors the original exact-match ordering (input/expected
        # validated before model load) so a malformed example is reported as
        # InvalidInputError without paying for (or masking behind) a model-load
        # attempt, and so a bogus/unloadable checkpoint doesn't turn a clear
        # input-shape error into a confusing "could not be loaded" one instead.
        task_type = task_type_of(example)
        if task_type == TaskType.MULTIPLE_CHOICE.value:
            mc_prompt, mc_choices = parse_multiple_choice_input(example)
            mc_expected_choice = resolve_expected_choice(example.expected, mc_choices)
        elif task_type == TaskType.FORMAT_CONFORMANCE.value:
            fc_prompt = parse_format_conformance_input(example)
            fc_spec = parse_format_spec(example)
        elif task_type == TaskType.SAFETY_PROBE.value:
            sp_prompt = parse_safety_probe_input(example)
            sp_spec = parse_safety_probe_spec(example)
        elif task_type == TaskType.EXACT_MATCH.value:
            if not isinstance(example.input, str) or not example.input.strip():
                raise InvalidInputError(
                    f"example {example.example_id!r}: HeldOutExample.input must be a "
                    "non-empty string prompt for this adapter"
                )
            if not isinstance(example.expected, str) or not example.expected.strip():
                raise InvalidInputError(
                    f"example {example.example_id!r}: HeldOutExample.expected must be a "
                    "non-empty string for this adapter"
                )
        else:
            raise InvalidInputError(
                f"example {example.example_id!r}: unsupported task_type {task_type!r} "
                f"(this adapter supports {[t.value for t in TaskType]})"
            )

        try:
            model, tokenizer = self._get_model(path)
        except (OSError, ValueError) as exc:
            raise InvalidInputError(
                f"artifact_locator {path} could not be loaded as a Hugging Face "
                f"causal-LM checkpoint: {exc}"
            ) from exc

        if task_type == TaskType.MULTIPLE_CHOICE.value:
            return self._score_multiple_choice(
                model, tokenizer, example, mc_prompt, mc_choices, mc_expected_choice
            )
        if task_type == TaskType.FORMAT_CONFORMANCE.value:
            return self._score_format_conformance(model, tokenizer, example, fc_prompt, fc_spec)
        if task_type == TaskType.SAFETY_PROBE.value:
            return self._score_safety_probe(model, tokenizer, example, sp_prompt, sp_spec)

        try:
            generated, renderer_id = self._generate(model, tokenizer, example.input)
        except Exception as exc:  # any inference failure is INVALID, not a crash
            raise InvalidInputError(
                f"inference failed for example {example.example_id!r}: {exc}"
            ) from exc

        correct = _normalize(example.expected) in _normalize(generated)
        return ExampleResult(
            example_id=example.example_id,
            correct=correct,
            score=1.0 if correct else 0.0,
            raw_output=f"{generated} | renderer={renderer_id}",
        )

    # -- new scoring modes (WP-A, issue #24) ------------------------------

    def _score_multiple_choice(
        self,
        model: Any,
        tokenizer: Any,
        example: HeldOutExample,
        prompt: str,
        choices: list[str],
        expected_choice: str,
    ) -> ExampleResult:
        """Score a multiple-choice example by per-option log-likelihood.

        Standard lm-evaluation-harness-style MMLU approach: for each
        candidate choice, compute the model's total log-probability of
        that choice's tokens conditioned on the prompt (teacher-forced,
        no sampling, ``torch.no_grad()``), and pick the choice with the
        highest length-normalized average log-probability -- length
        normalization (dividing by token count) avoids systematically
        favouring shorter choices purely because they have fewer terms
        in the log-probability sum. Correct iff the argmax choice
        equals the resolved expected choice. Deterministic given the
        same artifact and example, matching the exact-match path's own
        determinism guarantee. ``prompt``/``choices``/``expected_choice``
        are pre-parsed by the caller (before any model load) so a
        malformed example is reported without needing a model at all.
        """
        try:
            scored_with_renderer = [
                (choice, *self._choice_log_likelihood(model, tokenizer, prompt, choice))
                for choice in choices
            ]
        except Exception as exc:  # any inference failure is INVALID, not a crash
            raise InvalidInputError(
                f"multiple_choice inference failed for example {example.example_id!r}: {exc}"
            ) from exc

        scored = [(choice, logprob) for choice, logprob, _renderer in scored_with_renderer]
        renderer_id = scored_with_renderer[0][2] if scored_with_renderer else BARE_TEXT_RENDERER_ID
        predicted_choice = max(scored, key=lambda pair: pair[1])[0]
        correct = predicted_choice == expected_choice
        detail = ", ".join(f"{choice!r}={logprob:.4f}" for choice, logprob in scored)
        return ExampleResult(
            example_id=example.example_id,
            correct=correct,
            score=1.0 if correct else 0.0,
            raw_output=(
                f"chose {predicted_choice!r} | avg log-likelihoods: {detail} | "
                f"renderer={renderer_id}"
            ),
        )

    def _score_format_conformance(
        self,
        model: Any,
        tokenizer: Any,
        example: HeldOutExample,
        prompt: str,
        spec: dict[str, Any],
    ) -> ExampleResult:
        """Score a structured-output/format-conformance example.

        Generates greedy-decoded output the same way exact-match does
        (``do_sample=False``, ``self.max_new_tokens``), then checks the
        generated text against the declared format spec via
        ``scoring_modes.check_format_conformance`` instead of comparing
        it to a fixed expected string -- correctness here means "the
        model's output has the right shape" (valid JSON with required
        keys, or matches a regex), not "the model's output equals a
        specific value." ``prompt``/``spec`` are pre-parsed by the
        caller (before any model load).
        """
        try:
            generated, renderer_id = self._generate(model, tokenizer, prompt)
        except Exception as exc:  # any inference failure is INVALID, not a crash
            raise InvalidInputError(
                f"format_conformance inference failed for example {example.example_id!r}: {exc}"
            ) from exc

        conforms, detail = check_format_conformance(generated, spec)
        return ExampleResult(
            example_id=example.example_id,
            correct=conforms,
            score=1.0 if conforms else 0.0,
            raw_output=f"{generated} | {detail} | renderer={renderer_id}",
        )

    def _score_safety_probe(
        self,
        model: Any,
        tokenizer: Any,
        example: HeldOutExample,
        prompt: str,
        spec: dict[str, Any],
    ) -> ExampleResult:
        """Score a safety-probe example (WP-B, issue #24).

        Generates greedy-decoded output the same way exact_match and
        format_conformance do (``do_sample=False``, ``self.max_new_tokens``),
        then scores it via ``scoring_modes.check_safety_probe`` -- a
        measurement only (refusal-marker lexicon match plus disallowed-
        substring leak check), never a pass/fail promotion decision; see
        ``docs/decisions/0008-safety-probe-suite.md``. ``prompt``/``spec``
        are pre-parsed by the caller (before any model load), matching
        every other mode's validate-before-load discipline.
        """
        try:
            generated, renderer_id = self._generate(model, tokenizer, prompt)
        except Exception as exc:  # any inference failure is INVALID, not a crash
            raise InvalidInputError(
                f"safety_probe inference failed for example {example.example_id!r}: {exc}"
            ) from exc

        correct, detail = check_safety_probe(generated, spec)
        return ExampleResult(
            example_id=example.example_id,
            correct=correct,
            score=1.0 if correct else 0.0,
            raw_output=f"{generated} | {detail} | renderer={renderer_id}",
        )

    # -- helpers (private: not part of the adapter's public contract surface) --

    def _get_model(self, path: Path) -> tuple[Any, Any]:
        cache_key = str(path)
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        # Defence in depth: force offline mode before every load, exactly as
        # TRLTrainerAdapter.train() does -- this adapter never downloads
        # anything and never talks to a model/dataset hub regardless of
        # ambient environment variables.
        import os

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32)
        model.eval()
        self._model_cache[cache_key] = (model, tokenizer)
        return model, tokenizer

    def _render_prompt_inputs(self, tokenizer: Any, prompt: str) -> tuple[dict[str, Any], str]:
        """Render ``prompt`` to the tokenizer's complete model-input mapping.

        Uses the tokenizer's chat template when available and enabled, else
        bare-text tokenization.  ``attention_mask``, ``token_type_ids``, and
        architecture-specific fields are deliberately retained here so the
        model-interface filter -- not the renderer -- decides support.

        Returns ``(model_inputs, renderer_id)``. Fixes issue #76 finding 2:
        this adapter used to always bare-tokenize the raw prompt string,
        even against chat-SFT checkpoints trained on the ``conversations``
        schema (see ``examples/pilot-adr0011/train.jsonl``: a list of
        ``{"role": "user"/"assistant", "content": ...}`` turns) whose own
        tokenizer chat template wraps a user turn in role/special-token
        markup the model was actually trained to expect. Rendering the bare
        prompt instead of that markup meant the model saw a different input
        distribution than it was trained on.
        """
        chat_template = getattr(tokenizer, "chat_template", None)
        if self.use_chat_template and chat_template:
            messages = [{"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = dict(tokenizer(rendered, return_tensors="pt"))
            renderer_id = _chat_template_renderer_id(chat_template)
        else:
            inputs = dict(tokenizer(prompt, return_tensors="pt"))
            renderer_id = BARE_TEXT_RENDERER_ID
        if "input_ids" not in inputs:
            raise InvalidInputError("tokenizer output did not contain required input_ids")
        return inputs, renderer_id

    def _generate(self, model: Any, tokenizer: Any, prompt: str) -> tuple[str, str]:
        import torch

        rendered_inputs, renderer_id = self._render_prompt_inputs(tokenizer, prompt)
        input_ids = rendered_inputs["input_ids"]
        inputs = _filter_model_kwargs(model, rendered_inputs, for_generation=True)
        pad_token_id = tokenizer.eos_token_id
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        prompt_len = input_ids.shape[1]
        continuation_ids = output_ids[0][prompt_len:]
        decoded = tokenizer.decode(continuation_ids, skip_special_tokens=True)
        return decoded, renderer_id

    def _choice_log_likelihood(
        self, model: Any, tokenizer: Any, prompt: str, choice: str
    ) -> tuple[float, str]:
        """Length-normalized average log-probability of ``choice`` given ``prompt``.

        Teacher-forced (no sampling, no ``generate()`` loop): renders the
        prompt (chat-template or bare-text, matching ``_generate``'s own
        rendering choice per issue #76), appends ``choice``'s tokens, runs a
        single forward pass under ``torch.no_grad()``, and sums the model's
        own log-probability of each of ``choice``'s tokens conditioned on
        everything before it (prompt plus any preceding choice tokens) --
        the standard per-option-likelihood approach
        ``lm-evaluation-harness`` and similar MMLU-style evaluators use,
        reimplemented directly here with no dependency on that project.
        Divides by the number of choice tokens so a longer choice is not
        penalised purely for having more tokens to sum log-probabilities
        over. Also applies ``_filter_model_kwargs`` (issue #76 finding 1)
        before the forward pass so tokenizers that always emit
        ``token_type_ids`` (e.g. the MiniMind->Qwen3 conversion) don't
        crash model families whose ``forward`` doesn't accept that kwarg.

        Returns ``(avg_log_prob, renderer_id)``.
        """
        import torch
        import torch.nn.functional as F

        prompt_inputs, renderer_id = self._render_prompt_inputs(tokenizer, prompt)
        prompt_ids = prompt_inputs["input_ids"]
        choice_ids = tokenizer(choice, return_tensors="pt", add_special_tokens=False)["input_ids"]
        full_ids = torch.cat([prompt_ids, choice_ids], dim=1)
        prompt_len = prompt_ids.shape[1]
        choice_len = full_ids.shape[1] - prompt_len
        if choice_len <= 0:
            raise InvalidInputError(
                f"choice {choice!r} tokenizes to zero additional tokens beyond the prompt; "
                "cannot score its likelihood"
            )

        full_inputs = dict(prompt_inputs)
        full_inputs["input_ids"] = full_ids
        for key, value in prompt_inputs.items():
            if key == "input_ids" or not torch.is_tensor(value):
                continue
            if value.ndim != prompt_ids.ndim or value.shape != prompt_ids.shape:
                continue
            if key == "attention_mask":
                extension = torch.ones_like(choice_ids, dtype=value.dtype, device=value.device)
            else:
                extension = value[:, -1:].expand(-1, choice_len)
            full_inputs[key] = torch.cat([value, extension], dim=1)

        inputs = _filter_model_kwargs(model, full_inputs)
        with torch.no_grad():
            logits = model(**inputs).logits  # (1, seq_len, vocab)

        # logits[i] predicts token i+1, so the logits that predict the
        # choice's tokens are indices [prompt_len - 1, full_len - 2].
        relevant_logits = logits[0, prompt_len - 1 : full_ids.shape[1] - 1, :]
        target_ids = full_ids[0, prompt_len:]
        log_probs = F.log_softmax(relevant_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
        avg_log_prob = float(token_log_probs.sum().item() / choice_len)
        return avg_log_prob, renderer_id
