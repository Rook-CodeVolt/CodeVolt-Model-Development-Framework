# Real (non-fake) adapters

This page is the cold-reader entry point for "does this framework actually train or evaluate anything yet." Short answer: two real engine integrations exist and are contract-tested, but neither has been exercised by a live run in this repository, and no real training pilot has been executed.

## What "real" means here

Every contract in this framework (`TrainerAdapterV1`, `EvaluatorAdapterV1`) ships with a deterministic **fake** reference implementation (`fake_adapter.py`, `fake_evaluator_adapter.py`) used by `codevolt-mdf run` in the [deterministic demo](../README.md#try-the-deterministic-demo). Fakes prove the contract shape and evidence plumbing; they do not train or evaluate anything against a real model.

A **real** adapter wraps an actual training or inference engine. Two now exist:

## TrainerAdapterV1 — TRLTrainerAdapter

- File: `src/codevolt_mdf/trl_adapter.py`
- Contract: `TrainerAdapterV1` (`src/codevolt_mdf/trainer_contract.py`), `contract_version = "1.1.0"`
- Engine: [TRL](https://github.com/huggingface/trl) (Hugging Face Transformer Reinforcement Learning), pinned to exact version `0.24.0`
- Decision record: `docs/decisions/0005-trl-trainer-adapter-v1.md`
- Scope: exactly one training method — supervised fine-tuning via TRL's `SFTTrainer`/`SFTConfig`. No RL trainers (PPO/GRPO/DPO/...), no distributed/multi-node training, no vLLM. Fully offline — both model and dataset must already exist as verified local paths before `prepare()` is called.
- Tests: 28 conformance tests (`tests/test_trl_adapter.py`) exercising the same contract-conformance scenarios (provenance, resource budget, cancellation, tamper detection, etc.) the fake adapter is tested against.
- **Not yet done:** `adapter.train()` is fully implemented against real TRL/transformers APIs but has never been exercised by any run in this repository. No real training has happened.

## EvaluatorAdapterV1 — local HF evaluator

- File: `src/codevolt_mdf/hf_local_evaluator_adapter.py`
- Contract: `EvaluatorAdapterV1` (`src/codevolt_mdf/evaluator_contract.py`)
- Engine: local Hugging Face `transformers` causal-LM inference (greedy decoding, `torch.no_grad()`), pinned model `HuggingFaceTB/SmolLM2-135M` (Apache-2.0) by default — any local `AutoModelForCausalLM`-compatible checkpoint directory works, provided its content hash is verified
- Scope: scores held-out examples one at a time via one of three task-type modes — exact-match/containment (the original mode), multiple-choice via teacher-forced, length-normalized per-option log-likelihood, and format-conformance via greedy generation against a JSON/regex spec (both added by ADR-0007, additive to the v1 contract, `CONTRACT_VERSION` unchanged). Mode is selected by an optional `task_type` key on the held-out example's metadata, defaulting to exact-match. Still deliberately minimal — not a general-purpose evaluation harness, and safety/red-team probing and regression-across-revisions tracking (issue #24 WP-B/WP-C) are not yet implemented. Every mode goes through the same `HeldOutExclusionRegistry` bidirectional contamination check as every other conforming evaluator adapter, and the adapter has exactly one public method (`score_example`) with no authority to accept, reject, or promote a candidate.
- Tests: `tests/test_hf_local_evaluator_adapter.py`, including an AST-import test proving this module has no import relationship with `trl_adapter.py`/`trainer_contract.py` (trainer/evaluator role separation), plus real-inference tests for all three task-type modes against the pinned checkpoint. `tests/test_scoring_modes.py` covers the shared, stdlib-only validation/scoring helpers (`src/codevolt_mdf/scoring_modes.py`) that both this real adapter and the fake adapter use.
- **Not yet done:** every test in this suite scores held-out examples against the pristine, untrained base checkpoint as a stand-in artifact, to prove the scoring path works — it is contract-conformance evidence for the evaluator adapter, not pilot evidence for an actual trained candidate. There is no trained pilot artifact yet.

## Current gating status: pilot execution is blocked

A real bounded training pilot — actually running `TRLTrainerAdapter.train()` on a real model and scoring the result through the real evaluator — is drafted but **not authorised to execute**:

- The exact bounded configuration (model, dataset shape, resource limits, kill criteria) is written down in `docs/decisions/0006-bounded-real-trl-pilot-plan.md`, status `proposed`, not `accepted`.
- Before it can run, three things are required and none are complete yet:
  1. A real evaluator adapter and a real held-out set for the pilot's specific task (the evaluator adapter itself now exists — see above — but the pilot-specific held-out set does not yet).
  2. **Maya's pilot-specific security review** of the *live* TRL process (sandbox, egress, secrets, artifact hashes, supply-chain identity, safe-stop) — a distinct, broader scope than her PR #15 code review of the adapter, which explicitly did not evaluate live execution behaviour because none occurred.
  3. Owner authorisation to schedule the pilot itself.
- Separately, findings from an initial live-execution security pass are being addressed in an in-flight fix before Maya's re-review can happen; the pilot stays blocked until that work lands and is re-reviewed.

No training engine has been invoked, no pilot has been scheduled, and no resource has been reserved by anything merged to date. Do not read the existence of these adapters as evidence that training capability is production-ready — it explicitly is not.
