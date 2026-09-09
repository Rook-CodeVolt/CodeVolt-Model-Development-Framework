# ADR-0001: Own the control layer, not a mega-fork

- Status: accepted
- Date: 2026-09-09

## Context

Useful capabilities exist across MiniMind, TRL, PEFT, Unsloth, Axolotl, LLaMA-Factory, lm-evaluation-harness, llama.cpp, vLLM, and other projects. Combining their source into one fork would create licence, upgrade, maintenance, and identity problems.

## Decision

CodeVolt MDF owns stable experiment, evidence, evaluation, and governance contracts. External systems integrate through narrow adapters and retain their own provenance and licences.

## Consequences

Engines remain replaceable and upstream improvements are easier to adopt. Adapter compatibility requires active testing, and some engine-specific functionality will not fit the common contract.
