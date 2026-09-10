# CodeVolt Model Development Framework

An evidence-led framework for developing, training, evaluating, and safely evolving language models.

> **Project status:** early foundation (`v0.1`). The contracts and contribution model are usable; real training-engine integrations are the next milestone. This project does not claim autonomous or production-safe self-improvement.

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

The demo performs no model training. It exercises the experiment contract and writes a locked manifest, baseline, candidate evaluation, provenance, and keep/reject decision to a run directory.

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

See [Architecture](docs/ARCHITECTURE.md), [Continual improvement](docs/CONTINUAL_IMPROVEMENT.md), [Roadmap](docs/ROADMAP.md), [Agent integration](docs/AGENT_INTEGRATION.md), [Agent resumption](docs/AGENT_RESUMPTION_PLAYBOOK.md), and [Contributing](CONTRIBUTING.md) for the full design and ways to participate.

## Make it your own

Ordinary customization belongs in experiment manifests. New training engines, evaluators, data sources, and learning strategies belong in adapters. Organisations needing different governance can fork the project while retaining the same evidence contract.

CodeVolt MDF is a CodeVolt project led by Rook and developed with transparent AI-assisted engineering. Decisions and evidence remain reviewable by people.

## License

Apache License 2.0. See [LICENSE](LICENSE).
