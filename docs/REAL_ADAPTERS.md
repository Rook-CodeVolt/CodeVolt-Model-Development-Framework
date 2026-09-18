# Real (non-fake) adapters

This page is the cold-reader entry point for "does this framework actually train or evaluate anything yet." Short answer: three real engine integrations exist and are contract-tested, but none has been exercised by a live run in this repository, and no real training pilot has been executed.

## What "real" means here

Every contract in this framework (`TrainerAdapterV1`, `EvaluatorAdapterV1`) ships with a deterministic **fake** reference implementation (`fake_adapter.py`, `fake_evaluator_adapter.py`) used by `codevolt-mdf run` in the [deterministic demo](../README.md#try-the-deterministic-demo). Fakes prove the contract shape and evidence plumbing; they do not train or evaluate anything against a real model.

A **real** adapter wraps an actual training or inference engine. Three now exist:

## TrainerAdapterV1 — TRLTrainerAdapter

- File: `src/codevolt_mdf/trl_adapter.py`
- Contract: `TrainerAdapterV1` (`src/codevolt_mdf/trainer_contract.py`), `contract_version = "1.1.0"`
- Engine: [TRL](https://github.com/huggingface/trl) (Hugging Face Transformer Reinforcement Learning), pinned to exact version `0.24.0`
- Decision record: `docs/decisions/0005-trl-trainer-adapter-v1.md`
- Scope: exactly one training method — supervised fine-tuning via TRL's `SFTTrainer`/`SFTConfig`. No RL trainers (PPO/GRPO/DPO/...), no distributed/multi-node training, no vLLM. Fully offline — both model and dataset must already exist as verified local paths before `prepare()` is called.
- Tests: 28 conformance tests (`tests/test_trl_adapter.py`) exercising the same contract-conformance scenarios (provenance, resource budget, cancellation, tamper detection, etc.) the fake adapter is tested against.
- **Not yet done:** `adapter.train()` is fully implemented against real TRL/transformers APIs but has never been exercised by any run in this repository. No real training has happened.

## TrainerAdapterV1 — MiniMindTrainerAdapter

- File: `src/codevolt_mdf/minimind_adapter.py`
- Contract: `TrainerAdapterV1` (`src/codevolt_mdf/trainer_contract.py`), `contract_version = "1.1.0"`
- Engine: [MiniMind](https://github.com/jingyaogong/minimind) (Apache-2.0, from-scratch native PyTorch, no TRL/PEFT dependency), pinned to exact commit `cc312c1cc614bc371cd85dcbcbc1d3ba1590f364` on `master`
- Decision record: `docs/decisions/0010-minimind-trainer-adapter-v1.md` (issue #36)
- Scope: exactly one training method — full-parameter supervised fine-tuning via MiniMind's `trainer/train_full_sft.py`, invoked as a subprocess (not imported — MiniMind has no PyPI package or importable API). No pretraining, LoRA, RLHF/DPO, or distillation scripts. Fully offline — the MiniMind checkout, model, and dataset must already exist as verified local paths before `prepare()` is called.
- **Deviation from the TRL adapter**: MiniMind has no importable version to check, so the pin is verified by running `git rev-parse HEAD` against the caller-supplied MiniMind checkout directory and requiring exact equality with the pinned commit — not an installed-package `__version__` comparison. See the ADR's "Pin verification" section for the full rationale.
- Tests: 37 conformance tests (`tests/test_minimind_adapter.py`) covering the pin-verification path, offline-policy enforcement, provenance/hash checks, required-parameter validation, and `train()`'s subprocess argument construction/cancellation handling — all against local, throwaway git-repo fixtures and a mocked `subprocess.Popen`, never a real MiniMind checkout or subprocess call.
- **Not yet done:** `adapter.train()` is fully implemented against the real MiniMind CLI but has never been exercised by any run in this repository. No real training has happened.

## EvaluatorAdapterV1 — local HF evaluator

- File: `src/codevolt_mdf/hf_local_evaluator_adapter.py`
- Contract: `EvaluatorAdapterV1` (`src/codevolt_mdf/evaluator_contract.py`)
- Engine: local Hugging Face `transformers` causal-LM inference (greedy decoding, `torch.no_grad()`), pinned model `HuggingFaceTB/SmolLM2-135M` (Apache-2.0) by default — any local `AutoModelForCausalLM`-compatible checkpoint directory works, provided its content hash is verified
- Scope: scores held-out examples one at a time via one of four task-type modes — exact-match/containment (the original mode), multiple-choice via teacher-forced, length-normalized per-option log-likelihood, format-conformance via greedy generation against a JSON/regex spec (both added by ADR-0007, additive to the v1 contract, `CONTRACT_VERSION` unchanged), and safety-probe scoring (`ADR-0008`, issue #24 WP-B — refusal-appropriateness/harmful-instruction-compliance/PII-leakage, also additive). Mode is selected by an optional `task_type` key on the held-out example's metadata, defaulting to exact-match. Still deliberately minimal — not a general-purpose evaluation harness. Regression-across-revisions tracking (issue #24 WP-C) is implemented separately — see [Regression tracking](#regression-tracking--regression_checkpy) below. Every mode goes through the same `HeldOutExclusionRegistry` bidirectional contamination check as every other conforming evaluator adapter, and the adapter has exactly one public method (`score_example`) with no authority to accept, reject, or promote a candidate.
- Tests: `tests/test_hf_local_evaluator_adapter.py`, including an AST-import test proving this module has no import relationship with `trl_adapter.py`/`trainer_contract.py` (trainer/evaluator role separation), plus real-inference tests for all four task-type modes against the pinned checkpoint. `tests/test_scoring_modes.py` covers the shared, stdlib-only validation/scoring helpers (`src/codevolt_mdf/scoring_modes.py`) that both this real adapter and the fake adapter use.
- **Not yet done:** every test in this suite scores held-out examples against the pristine, untrained base checkpoint as a stand-in artifact, to prove the scoring path works — it is contract-conformance evidence for the evaluator adapter, not pilot evidence for an actual trained candidate. There is no trained pilot artifact yet.

## Safety/red-team probing — `TaskType.SAFETY_PROBE`

- Files: `src/codevolt_mdf/scoring_modes.py` (`parse_safety_probe_input`, `parse_safety_probe_spec`, `check_safety_probe`), wired into both `hf_local_evaluator_adapter.py` and `fake_evaluator_adapter.py`
- Decision record: `docs/decisions/0008-safety-probe-suite.md`
- Dataset: `examples/safety-probes-wpb/held_out.json` (package id `wpb-safety-probes-v1`) — a fixed, versioned, 15-example synthetic suite (5 each: refusal-appropriateness, harmful-instruction-compliance, PII-leakage). Provenance/licence/limitations documented in `examples/safety-probes-wpb/DATASET_CARD.md` per `docs/DATA_GOVERNANCE.md`.
- Scope: an additive fourth `TaskType`, scored via the same real greedy-generation path exact-match/format-conformance already use, checked against a fixed refusal-marker lexicon plus disallowed-pattern leak detection. Measurement only — no accept/reject/promote field anywhere, per `docs/ARCHITECTURE.md` core contracts #3/#6.
- Tests: fake-adapter conformance tests in `tests/test_evaluator_contract.py` (success/rejection/invalid-input/contamination), real-adapter tests in `tests/test_hf_local_evaluator_adapter.py` against the pinned checkpoint (including an end-to-end smoke test loading the actual committed dataset file), unit tests for the scoring helpers in `tests/test_scoring_modes.py`.

## Regression tracking — `regression_check.py`

- File: `src/codevolt_mdf/regression_check.py`
- Decision record: `docs/decisions/0009-regression-tracking-across-candidate-revisions.md`
- Scope: a stand-alone, measurement-only module (not a new `TaskType`, not a contract change) that compares already-produced `EvaluationOutput` objects. `compare_for_regressions(baseline, candidate, previous_candidate=None)` restricts attention to examples that were previously passing (in baseline, and in the previous candidate when supplied) and reports whether the new candidate now fails any of them — closing the "Regression checks for retained capabilities" gap named in `docs/EVALUATION_POLICY.md`. It calls no adapter and performs no inference itself; it consumes `EvaluationOutput`s a caller already produced. It has no threshold, no aggregate verdict, and no accept/reject/promote authority — output is evidence for a separate, later governed acceptance step.
- Tests: `tests/test_regression_check.py` (14 fake-data tests: no-regression, regression-detected, already-failing-excluded, 3-way comparison, tamper/malformed-input rejection, AST-import boundary, no-verdict-field). `tests/test_regression_check_real.py` adds two real-adapter integration tests running the pinned `HuggingFaceTB/SmolLM2-135M` checkpoint through the real evaluator twice and feeding genuine `EvaluationOutput`s through the comparator — one confirming no false regressions on identical repeated runs, one confirming a genuinely perturbed candidate is correctly flagged.

## Current gating status: pilot execution is blocked

A real bounded training pilot — actually running `TRLTrainerAdapter.train()` on a real model and scoring the result through the real evaluator — is drafted but **not authorised to execute**:

- The exact bounded configuration (model, dataset shape, resource limits, kill criteria) is written down in `docs/decisions/0006-bounded-real-trl-pilot-plan.md`, status `proposed`, not `accepted`.
- Before it can run, three things are required and none are complete yet:
  1. A real evaluator adapter and a real held-out set for the pilot's specific task (the evaluator adapter itself now exists — see above — but the pilot-specific held-out set does not yet).
  2. **Maya's pilot-specific security review** of the *live* TRL process (sandbox, egress, secrets, artifact hashes, supply-chain identity, safe-stop) — a distinct, broader scope than her PR #15 code review of the adapter, which explicitly did not evaluate live execution behaviour because none occurred.
  3. Owner authorisation to schedule the pilot itself.
- Separately, findings from an initial live-execution security pass are being addressed in an in-flight fix before Maya's re-review can happen; the pilot stays blocked until that work lands and is re-reviewed.

This applies to both real trainer adapters — TRL and MiniMind alike. No training engine has been invoked, no pilot has been scheduled, and no resource has been reserved by anything merged to date. Do not read the existence of these adapters as evidence that training capability is production-ready — it explicitly is not.
