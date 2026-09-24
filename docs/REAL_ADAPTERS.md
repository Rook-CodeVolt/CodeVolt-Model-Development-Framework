# Real (non-fake) adapters

This page is the cold-reader entry point for "does this framework actually train or evaluate anything yet." Short answer: three real engine integrations exist and are contract-tested. One bounded, exploratory TRL training pilot has been executed and accepted (`TrainingOutput.status=ACCEPTED`, evaluator `aggregate_score=0.7`/1.0, zero contamination) per `docs/decisions/0006-bounded-real-trl-pilot-plan.md`'s "Pilot execution result" — a single, scope-limited result that does not authorise a further/repeated TRL pilot or any pilot for MiniMind, which has not been exercised by a live run at all.

## What "real" means here

Every contract in this framework (`TrainerAdapterV1`, `EvaluatorAdapterV1`) ships with a deterministic **fake** reference implementation (`fake_adapter.py`, `fake_evaluator_adapter.py`) used by `codevolt-mdf run` in the [deterministic demo](../README.md#try-the-deterministic-demo). Fakes prove the contract shape and evidence plumbing; they do not train or evaluate anything against a real model.

A **real** adapter wraps an actual training or inference engine. Three now exist:

## Standalone package — `held-out-eval`

- Location: `packages/held-out-eval/` (importable as `held_out_eval`), independently pip-installable, own `pyproject.toml`/`README.md`/`LICENSE`/tests.
- Decision record: [ADR-0012](decisions/0012-extract-held-out-eval-standalone-package.md)
- What it is: `HeldOutExclusionRegistry`, previously a module inside this framework (`src/codevolt_mdf/held_out_registry.py`), extracted because it was already stdlib-only and had zero dependency on any trainer adapter, evaluator contract class, or the `codevolt-mdf` CLI — but had no adoption path for anyone not already using this framework's full adapter architecture.
- Verified, not assumed: building the package into a wheel and installing *only* that wheel into a brand-new virtual environment (no `codevolt-mdf` present) succeeds; the package's own 16 tests pass against the installed wheel; `examples/plain_training_loop.py` runs end to end against a plain PyTorch training loop unrelated to any trainer this framework integrates; and `import held_out_eval` does not import `codevolt_mdf`.
- What this does not establish: no external team has adopted this package yet. This is proof of adoptability, not proof of adoption. The main framework's evaluator adapters below now depend on this package (not the reverse) — no public adapter/CLI contract changed.

## TrainerAdapterV1 — TRLTrainerAdapter

- File: `src/codevolt_mdf/trl_adapter.py`
- Contract: `TrainerAdapterV1` (`src/codevolt_mdf/trainer_contract.py`), `contract_version = "1.1.0"`
- Engine: [TRL](https://github.com/huggingface/trl) (Hugging Face Transformer Reinforcement Learning), pinned to exact version `0.24.0`
- Decision record: `docs/decisions/0005-trl-trainer-adapter-v1.md`
- Scope: exactly one training method — supervised fine-tuning via TRL's `SFTTrainer`/`SFTConfig`. No RL trainers (PPO/GRPO/DPO/...), no distributed/multi-node training, no vLLM. Fully offline — both model and dataset must already exist as verified local paths before `prepare()` is called.
- Tests: 28 conformance tests (`tests/test_trl_adapter.py`) exercising the same contract-conformance scenarios (provenance, resource budget, cancellation, tamper detection, etc.) the fake adapter is tested against.
- **Pilot executed (one bounded run, 2026-09-17):** per ADR-0006's "Pilot execution result" section, `adapter.train()` was exercised for one bounded, gated, 50-`max_steps` SFT run against `HuggingFaceTB/SmolLM2-135M` (`run_id=adr0006-pilot-20260917`). `TrainingOutput.status=ACCEPTED` (wall time 15.7s of a 1800s budget), and the resulting artifact was scored by the real evaluator against a held-out set: `EvaluationOutput.status=SCORED`, `aggregate_score=0.7` (7/10), zero contamination. This is a single bounded exploratory result, not a capability claim or promotion decision, and it does **not** authorise scheduling a further or repeated TRL pilot — that remains a separate, later governed action per ADR-0006's "What this result does and does not establish."

## TrainerAdapterV1 — MiniMindTrainerAdapter

- File: `src/codevolt_mdf/minimind_adapter.py`
- Contract: `TrainerAdapterV1` (`src/codevolt_mdf/trainer_contract.py`), `contract_version = "1.1.0"`
- Engine: [MiniMind](https://github.com/jingyaogong/minimind) (Apache-2.0, from-scratch native PyTorch, no TRL/PEFT dependency), pinned to exact commit `cc312c1cc614bc371cd85dcbcbc1d3ba1590f364` on `master`
- Decision record: `docs/decisions/0010-minimind-trainer-adapter-v1.md` (issue #36)
- Scope: exactly one training method — full-parameter supervised fine-tuning via MiniMind's `trainer/train_full_sft.py`, invoked as a subprocess (not imported — MiniMind has no PyPI package or importable API). No pretraining, LoRA, RLHF/DPO, or distillation scripts. Fully offline — the MiniMind checkout, model, and dataset must already exist as verified local paths before `prepare()` is called.
- **Deviation from the TRL adapter**: MiniMind has no importable version to check, so the pin is verified by running `git rev-parse HEAD` against the caller-supplied MiniMind checkout directory and requiring exact equality with the pinned commit — not an installed-package `__version__` comparison. See the ADR's "Pin verification" section for the full rationale.
- Tests: 37 conformance tests (`tests/test_minimind_adapter.py`) covering the pin-verification path, offline-policy enforcement, provenance/hash checks, required-parameter validation, and `train()`'s subprocess argument construction/cancellation handling — all against local, throwaway git-repo fixtures and a mocked `subprocess.Popen`, never a real MiniMind checkout or subprocess call.
- **Now exercised by a real, evaluated pilot:** `adapter.train()` has been exercised by a real bounded pilot run (`artifact-adr0011-pilot-20260920`, 2026-09-20) — training executed for real via a real `train_full_sft.py` subprocess and was evaluated for real against a real held-out set, scoring `aggregate_score=0.0` due to severe undertraining (40 examples, 1 epoch), not a pipeline or adapter defect. See `docs/decisions/0011-minimind-bounded-pilot-plan.md`'s "Pilot execution result" section for full detail.

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

## Current gating status: one TRL pilot executed and accepted; no further pilot authorised; MiniMind pilot still blocked

One real bounded training pilot — actually running `TRLTrainerAdapter.train()` on a real model and scoring the result through the real evaluator — has executed and was accepted. Per `docs/decisions/0006-bounded-real-trl-pilot-plan.md`'s "Pilot execution result" section (run `adr0006-pilot-20260917`, PR #22, 2026-09-17):

- Gating cleared before execution: the exact bounded configuration (model, dataset shape, resource limits, kill criteria) was locked in ADR-0006; the pilot-specific live-execution security review (issue #7 step 5, PR #18 review) returned clear to execute after its findings were fixed and merged (PR #21); the owner authorised execution of this specific bounded run.
- `TrainingOutput.status=ACCEPTED` (wall time 15.7s of a 1800s budget, all five resource dimensions well inside budget, no kill criterion triggered), and `EvaluationOutput.status=SCORED` (`aggregate_score=0.7`, 7/10, against a real held-out set with zero contamination).
- **This does not authorise anything further.** Per ADR-0006's own "What this result does and does not establish": this is exploratory pilot evidence, not a capability claim, not a promotion decision, and it does not authorise any further pilot, larger run, or production use for TRL. A repeat or expanded TRL pilot would require its own separate gating (fresh security review scope and owner authorisation), the same as this one did.
- **MiniMind has now also been exercised by its own real, evaluated pilot.** Per `docs/decisions/0011-minimind-bounded-pilot-plan.md`'s "Pilot execution result" section (run `artifact-adr0011-pilot-20260920`, 2026-09-20): `MiniMindTrainerAdapter.train()` was invoked for real (accepted, all five resource budgets respected) and the resulting artifact was evaluated for real against a real held-out set with zero contamination, scoring `aggregate_score=0.0` (0/10) — a confirmed artifact of severe undertraining (40 examples, 1 epoch), not a pipeline or adapter defect. This is a single bounded exploratory pilot proving the pipeline runs end-to-end for MiniMind; it is not a capability claim, not a promotion decision, and does not authorise any further MiniMind pilot, larger run, or production use.

Do not read the existence of these adapters, or these two executed pilots, as evidence that training capability is production-ready — it explicitly is not.

A MiniMind-specific bounded pilot configuration is separately drafted (not authorised) at `docs/decisions/0011-minimind-bounded-pilot-plan.md` (status `proposed`, issue #46), alongside a companion design for how an authorized later round would consume an earlier round's evidence via `regression_check.py` at `docs/CONTINUAL_IMPROVEMENT_LOOP_DESIGN.md`. Same gating sequence applies: ADR drafted, the pilot-specific live-execution security review, owner authorisation, in that order — none of which has happened for this ADR.
