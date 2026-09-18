# CodeVolt Model Development Framework

An evidence-led framework for developing, training, evaluating, and safely evolving language models.

> **Project status:** early foundation (`v0.1`). The contracts and contribution model are usable. A real (non-fake) trainer adapter for TRL and a real (non-fake) evaluator adapter backed by local Hugging Face inference are both merged — see [Real adapters](docs/REAL_ADAPTERS.md) for what that does and does not mean. A real bounded training pilot has **run**: after Maya's pilot-specific live-execution security review cleared it, the bounded pilot planned in `docs/decisions/0006-bounded-real-trl-pilot-plan.md` executed (PR #22) with `TrainingOutput.status=ACCEPTED` (wall time 15.7s of a 1800s budget, well within all resource limits) and `EvaluationOutput.status=SCORED`, `aggregate_score=0.7` (7/10) on the real registered synthetic held-out arithmetic set, with zero contamination flagged. This is a single bounded exploratory result, not a capability, promotion, or production claim, and it does not authorise any further or larger pilot — see ADR-0006's "What this result does and does not establish" section. This project does not claim autonomous or production-safe self-improvement.

## Why this exists

Training a model is easy to start and difficult to trust. Results are often separated from the exact data, configuration, baseline, environment, and acceptance rule that produced them. CodeVolt MDF makes those things part of one reviewable experiment.

The framework owns the evolution process—not the underlying engines. MiniMind, TRL, Unsloth, Axolotl, LLaMA-Factory, llama.cpp, and future tools can remain replaceable adapters.

```text
proposal -> bounded experiment -> independent evaluation
         -> evidence record -> accept, reject, or revise
```

## Principles

- Evidence before claims.
- Compare every candidate with an immutable baseline.
- Treat missing or malformed evidence as invalid, never as success.
- Keep training and evaluation independently replaceable.
- Record rejected experiments as useful knowledge.
- Make every promoted change attributable and reversible.
- Observe and recommend before allowing automation to act.

## Try the deterministic demo

Python 3.9 or later is supported.

```bash
python -m pip install -e .
codevolt-mdf validate examples/deterministic-experiment.json
codevolt-mdf run examples/deterministic-experiment.json --output runs
```

The demo performs no model training. It exercises the experiment contract and writes a locked manifest, baseline, candidate evaluation, provenance, and keep/reject decision to a run directory — using the deterministic fake trainer/evaluator adapters (`src/codevolt_mdf/fake_adapter.py`, `fake_evaluator_adapter.py`), not the real engines below.

## Real (non-fake) adapters

Beyond the deterministic demo, two real engine integrations are merged and contract-tested (see [docs/REAL_ADAPTERS.md](docs/REAL_ADAPTERS.md) for the full picture):

- `TRLTrainerAdapter` (`src/codevolt_mdf/trl_adapter.py`) — a real `TrainerAdapterV1` implementation wrapping TRL's `SFTTrainer` for supervised fine-tuning.
- The local HF evaluator (`src/codevolt_mdf/hf_local_evaluator_adapter.py`) — a real `EvaluatorAdapterV1` implementation that scores held-out examples via local Hugging Face causal-LM inference.

Both are implemented against real upstream APIs and pass their own conformance test suites, but neither has been exercised in a live training/evaluation pilot yet. `adapter.train()` has never actually been run by anything in this repository. The next step — a real bounded pilot — is drafted but not authorised to execute; see [docs/REAL_ADAPTERS.md](docs/REAL_ADAPTERS.md) for the exact gating status.

## How it is structured

```text
src/codevolt_mdf/       Core contracts, adapters, gate, evidence, CLI
schemas/                Versioned machine-readable contracts
examples/               Reproducible example experiments
capabilities/           Capability proposals and maturity records
docs/                   Architecture, roadmap, governance and decisions
templates/              Experiment, dataset, model and release records
tests/                  Positive, negative and tamper-oriented tests
```

See [Architecture](docs/ARCHITECTURE.md), [Continual improvement](docs/CONTINUAL_IMPROVEMENT.md), [Roadmap](docs/ROADMAP.md), [Next progression](docs/PROGRESSION.md), [Real adapters](docs/REAL_ADAPTERS.md), [Evidence log](docs/EVIDENCE_LOG.md), [Agent integration](docs/AGENT_INTEGRATION.md), [Agent resumption](docs/AGENT_RESUMPTION_PLAYBOOK.md), [Training-loop reliability](docs/TRAINING_LOOP_RELIABILITY.md), [Data governance](docs/DATA_GOVERNANCE.md), [Feedback](docs/FEEDBACK.md), and [Contributing](CONTRIBUTING.md) for the full design and ways to participate.

For a concrete, link-by-link worked example of this project's evidence-over-theory principle in practice — proposal, contract, independent review, a live-execution security review that found and blocked on real defects, and an independently re-verified real pilot run — see [docs/EVIDENCE_LOG.md](docs/EVIDENCE_LOG.md).

## Make it your own

Ordinary customization belongs in experiment manifests. New training engines, evaluators, data sources, and learning strategies belong in adapters. Organisations needing different governance can fork the project while retaining the same evidence contract.

CodeVolt MDF is a CodeVolt project led by Rook and developed with transparent AI-assisted engineering. Decisions and evidence remain reviewable by people.

## License

Apache License 2.0. See [LICENSE](LICENSE).
