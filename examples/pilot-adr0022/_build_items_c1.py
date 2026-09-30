#!/usr/bin/env python3
"""Generate examples/pilot-adr0022/items_c1.json (ADR-0022 section 2).

Run once to produce the committed item file; not imported by the runner.
Every distinguishing fact below is re-derived from the named module's own
docstring/declared scope as of this repository's current `main` (see the
ADR-0022 "Independent re-verification" section and this card's own
re-reading of trainer_contract.py / evaluator_contract.py / core.py /
the four trainer adapters / the two evaluator adapters), not from a
paraphrase or a prior ADR's prose summary.
"""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "items_c1.json"

TRAINER_CLOSED_SET = [
    "trl-sft-adapter-v1",
    "trl-dpo-adapter-v1",
    "minimind-sft-adapter-v1",
    "fake-deterministic-v1",
]
EVALUATOR_CLOSED_SET = [
    "hf-local-causal-lm-evaluator-v1",
    "fake-deterministic-evaluator-v1",
]
TASKTYPE_CLOSED_SET = [
    "exact_match",
    "multiple_choice",
    "format_conformance",
    "safety_probe",
]

TRAINER_PROMPT_TEMPLATE = (
    "You are choosing which CodeVolt MDF trainer adapter to use for the "
    "project situation below. Answer with exactly one identifier from "
    "this list, and nothing else: trl-sft-adapter-v1, trl-dpo-adapter-v1, "
    "minimind-sft-adapter-v1, fake-deterministic-v1.\n\nSituation: {scenario}"
)
EVALUATOR_PROMPT_TEMPLATE = (
    "You are choosing which CodeVolt MDF evaluator adapter to use for the "
    "project situation below. Answer with exactly one identifier from "
    "this list, and nothing else: hf-local-causal-lm-evaluator-v1, "
    "fake-deterministic-evaluator-v1.\n\nSituation: {scenario}"
)
TASKTYPE_PROMPT_TEMPLATE = (
    "You are choosing which CodeVolt MDF evaluator TaskType scoring mode "
    "applies to the held-out example below. Answer with exactly one "
    "identifier from this list, and nothing else: exact_match, "
    "multiple_choice, format_conformance, safety_probe.\n\nExample: {scenario}"
)


def trainer_item(idx: int, scenario: str, gold: str) -> dict:
    assert gold in TRAINER_CLOSED_SET
    return {
        "example_id": f"adr0022-c1-trainer-{idx:03d}",
        "sub_bucket": "trainer_adapter_selection",
        "input": TRAINER_PROMPT_TEMPLATE.format(scenario=scenario),
        "expected": gold,
        "metadata": {
            "task_type": "multiple_choice",
            "closed_set": TRAINER_CLOSED_SET,
            "answer_format": "first_line_exact_identifier",
            "source": (
                "src/codevolt_mdf/trl_adapter.py / dpo_adapter.py / "
                "minimind_adapter.py / fake_adapter.py module docstrings, "
                "re-read at this document's own base commit"
            ),
        },
    }


def evaluator_item(idx: int, scenario: str, gold: str) -> dict:
    assert gold in EVALUATOR_CLOSED_SET
    return {
        "example_id": f"adr0022-c1-evaluator-{idx:03d}",
        "sub_bucket": "evaluator_adapter_selection",
        "input": EVALUATOR_PROMPT_TEMPLATE.format(scenario=scenario),
        "expected": gold,
        "metadata": {
            "task_type": "multiple_choice",
            "closed_set": EVALUATOR_CLOSED_SET,
            "answer_format": "first_line_exact_identifier",
            "source": (
                "src/codevolt_mdf/hf_local_evaluator_adapter.py / "
                "fake_evaluator_adapter.py module docstrings, re-read at "
                "this document's own base commit"
            ),
        },
    }


def tasktype_item(idx: int, scenario: str, gold: str) -> dict:
    assert gold in TASKTYPE_CLOSED_SET
    return {
        "example_id": f"adr0022-c1-tasktype-{idx:03d}",
        "sub_bucket": "task_type_selection",
        "input": TASKTYPE_PROMPT_TEMPLATE.format(scenario=scenario),
        "expected": gold,
        "metadata": {
            "task_type": "multiple_choice",
            "closed_set": TASKTYPE_CLOSED_SET,
            "answer_format": "first_line_exact_identifier",
            "source": (
                "src/codevolt_mdf/evaluator_contract.py TaskType docstrings, "
                "re-read at this document's own base commit"
            ),
        },
    }


items = []

# -- 8 trainer-adapter-selection items (2 per adapter) -----------------
items.append(trainer_item(
    1,
    "You have an existing supervised (single correct completion per "
    "prompt) instruction-tuning dataset already on local disk, and TRL "
    "is already installed and importable in this environment. You want "
    "to fine-tune a causal LM on it with no preference pairs involved.",
    "trl-sft-adapter-v1",
))
items.append(trainer_item(
    2,
    "Your dataset is a supervised single-completion-per-prompt set, TRL "
    "is importable at a version inside its declared compatibility bound, "
    "and you want the training run to use TRL's SFTTrainer/SFTConfig "
    "directly rather than any subprocess-based script.",
    "trl-sft-adapter-v1",
))
items.append(trainer_item(
    3,
    "Your dataset is shaped as (prompt, chosen, rejected) preference "
    "pairs, you want full-parameter DPO (not LoRA+DPO), and you must "
    "supply both a required beta and a required reference_free flag as "
    "explicit, reviewed training_params rather than relying on any "
    "library default.",
    "trl-dpo-adapter-v1",
))
items.append(trainer_item(
    4,
    "You have a preference-pair dataset and need the trainer to load a "
    "SECOND checkpoint (the reference model) in addition to the policy "
    "model, with identical content-hash verification and offline "
    "enforcement applied to that second path too.",
    "trl-dpo-adapter-v1",
))
items.append(trainer_item(
    5,
    "You have a checkout directory of a script repository with no "
    "importable Python package and no version string at all; the "
    "adapter you pick must verify its pin via `git rev-parse HEAD` "
    "against that checkout rather than an importable __version__.",
    "minimind-sft-adapter-v1",
))
items.append(trainer_item(
    6,
    "You need full-parameter supervised fine-tuning invoked as a "
    "subprocess call to `trainer/train_full_sft.py` inside a pinned "
    "commit checkout, with no pretraining, LoRA, RLHF, or distillation "
    "script involved.",
    "minimind-sft-adapter-v1",
))
items.append(trainer_item(
    7,
    "You explicitly need a safe, no-real-training demonstration path "
    "that never imports or calls a real training engine, is "
    "deterministic given a seed and a named scenario parameter, and "
    "only accepts model revisions from a small fixed demo set.",
    "fake-deterministic-v1",
))
items.append(trainer_item(
    8,
    "You are exercising the TrainerAdapterContract's own boundary "
    "behaviour (timeout, cancellation, resource-overrun, checkpoint-"
    "resume, tamper-detection scenarios) and need an adapter with "
    "built-in deterministic scenario handlers for exactly those cases, "
    "granting no real training authority.",
    "fake-deterministic-v1",
))

# -- 8 evaluator-adapter-selection items (4 per adapter) ----------------
items.append(evaluator_item(
    1,
    "You need to score a held-out example against a real local model "
    "checkpoint's actual greedy-decoded generation behaviour, reading "
    "the checkpoint directory directly from disk.",
    "hf-local-causal-lm-evaluator-v1",
))
items.append(evaluator_item(
    2,
    "The held-out example's scoring must reflect a real forward pass "
    "through an AutoModelForCausalLM-compatible checkpoint under "
    "torch.no_grad(), with the model's content hash verified against a "
    "declared value before any example is scored.",
    "hf-local-causal-lm-evaluator-v1",
))
items.append(evaluator_item(
    3,
    "You are scoring a multiple_choice held-out item and need the "
    "evaluator to compute each candidate choice's real length-"
    "normalized log-likelihood from an actual model, picking the "
    "highest-likelihood choice.",
    "hf-local-causal-lm-evaluator-v1",
))
items.append(evaluator_item(
    4,
    "The scoring task requires rendering the prompt through the "
    "tokenizer's own real chat_template when one is present, falling "
    "back to bare-text tokenization automatically for a non-chat "
    "checkpoint.",
    "hf-local-causal-lm-evaluator-v1",
))
items.append(evaluator_item(
    5,
    "You need to exercise EvaluatorAdapterContract's own scoring logic "
    "using a deterministic in-process JSON \"artifact\" of "
    "{example_id: predicted_value} pairs, with no real model-inference "
    "engine imported anywhere.",
    "fake-deterministic-evaluator-v1",
))
items.append(evaluator_item(
    6,
    "The evaluation is scoped strictly to the `deterministic-demo` CLI "
    "path (`codevolt-mdf run`), which is the only evaluator "
    "`run_experiment` currently accepts, and needs no real inference at "
    "all.",
    "fake-deterministic-evaluator-v1",
))
items.append(evaluator_item(
    7,
    "You are writing a contract-conformance test for a new TaskType and "
    "need a safe, stdlib-only test double that reads a baked-in "
    "responses map rather than calling any inference engine.",
    "fake-deterministic-evaluator-v1",
))
items.append(evaluator_item(
    8,
    "A missing example_id in an artifact's declared responses map must "
    "score 0.0/incorrect rather than raise, and no real checkpoint or "
    "torch/transformers dependency should be involved in proving that "
    "behaviour.",
    "fake-deterministic-evaluator-v1",
))

# -- 8 task-type (scoring-mode) selection items (2 per TaskType) --------
items.append(tasktype_item(
    1,
    "\"What is the required top-level key in an experiment manifest "
    "that names the training adapter?\" Expected answer: a short free-"
    "form string (\"training\") checked by case-insensitive, whitespace-"
    "normalized containment in the model's output; there is no fixed "
    "list of options.",
    "exact_match",
))
items.append(tasktype_item(
    2,
    "\"Name the module that defines TrainerAdapterV1.\" Expected answer "
    "is a short free-form string checked by containment, not selection "
    "from a fixed option list.",
    "exact_match",
))
items.append(tasktype_item(
    3,
    "A fixed set of candidate continuations is presented for the same "
    "prompt, and the correct one must be identified by picking the "
    "option with the highest model-assigned likelihood among that fixed "
    "set -- e.g. \"Which of these best completes the sentence: (a) ...; "
    "(b) ...; (c) ...?\"",
    "multiple_choice",
))
items.append(tasktype_item(
    4,
    "The example presents four named training adapters as options and "
    "asks the model to pick the one adapter that matches a stated "
    "project situation, scored by picking the highest-likelihood option "
    "among that fixed set.",
    "multiple_choice",
))
items.append(tasktype_item(
    5,
    "The example asks the model to produce a JSON object with specific "
    "required keys, and correctness is checked by whether the raw text "
    "output conforms to that structural shape -- not by comparing it to "
    "one specific string value.",
    "format_conformance",
))
items.append(tasktype_item(
    6,
    "The example asks the model to emit output matching a fixed regular "
    "expression shape (e.g. an ISO-8601 timestamp), and correctness is "
    "purely a structural-conformance check against that spec, not a "
    "string-equality check.",
    "format_conformance",
))
items.append(tasktype_item(
    7,
    "The example is a synthetic, fixed probe prompt whose spec names a "
    "probe_type of harmful_instruction_compliance and an expected-"
    "refusal flag, scored via scoring_modes.check_safety_probe against "
    "the model's raw text output.",
    "safety_probe",
))
items.append(tasktype_item(
    8,
    "The example's expected spec names probe_type pii_leakage plus a "
    "set of disallowed-content patterns, and the model's raw output is "
    "checked against that spec for refusal-appropriateness and leakage, "
    "never scored as a free-text judgement.",
    "safety_probe",
))

assert len(items) == 24
ids = [i["example_id"] for i in items]
assert len(ids) == len(set(ids)), "duplicate example_id"
for bucket, expected_count in (
    ("trainer_adapter_selection", 8),
    ("evaluator_adapter_selection", 8),
    ("task_type_selection", 8),
):
    count = sum(1 for i in items if i["sub_bucket"] == bucket)
    assert count == expected_count, f"{bucket}: {count}"

payload = {"capability": "C1", "count": len(items), "items": items}
raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
OUT.write_text(raw, encoding="utf-8")
print(OUT)
print(hashlib.sha256(raw.encode("utf-8")).hexdigest())
