# Architecture

## Owned control layer

CodeVolt MDF owns the experiment contract, provenance, evidence, evaluation gate, capability lifecycle, and governance. It deliberately does not absorb upstream training projects into a mega-fork.

```text
Experiment manifest
  |-- dataset and model provenance
  |-- resource budget
  |-- trainer adapter --------> replaceable training engine
  |-- evaluator adapter ------> independent evaluation engine
  `-- acceptance policy ------> accepted / rejected / invalid
                                      |
                                evidence bundle
```

## Core contracts

1. An experiment is immutable once execution begins.
2. A trainer produces an artifact; it does not decide whether the artifact is good.
3. An evaluator produces measurements; it does not promote the candidate.
4. The assurance gate applies declared thresholds to baseline and candidate evidence.
5. Missing, corrupt, or unverifiable evidence produces `invalid`, not `accepted`.
6. Promotion is a separate governed action and is not implemented in `v0.1`.

## Evolution Engine

Learning begins with structured observations. A failure or correction becomes a learning event, related events may become a proposal, and an approved proposal may generate a bounded experiment. Production feedback must be reviewed and stripped of secrets or personal data before entering a dataset.

```text
observe -> classify -> propose -> approve -> experiment -> evaluate -> retain/reject
```

The initial implementation exposes data types for events and proposals. Automatic retraining and automatic promotion are explicitly out of scope.

## Extension points

- `TrainerAdapter`: TRL, Unsloth, Axolotl, LLaMA-Factory, MiniMind, or another engine.
- `EvaluatorAdapter`: task, safety, regression, robustness, and cost suites.
- Dataset adapters: versioned, licensed sources with contamination controls.
- Learning strategies: analysis and proposals, never unreviewed model replacement.

See [ADR-0001](decisions/0001-owned-control-layer.md) for the architectural decision.

The [evaluation policy](EVALUATION_POLICY.md), [data governance policy](DATA_GOVERNANCE.md), and repository templates define what complete evidence and release readiness mean.
