# Meta-trainer corpus v2 — dataset card

Status: independently audited and accepted for repository inclusion. On 2026-09-20, task completed a separate all-example review: all 60 records passed checks C1-C6 (60/60 for each check). This acceptance permits repository admission only; the corpus has not been used in a training run, and any training use requires a separate governance, authorization, and scheduling decision.

## Scope

This corpus contains 60 reasoning-oriented examples derived only from the designated Seed-rich A and Seed-rich B sections of the source research synthesis (research synthesis comment, 2026-09-20). It intentionally excludes section 3.1 and all other non-designated material.

Every record contains:

- a stable `example_id`;
- an explicit `semantic_family`;
- a `source_scope` naming the authorized source-research section;
- a non-empty `citations` array with direct source ids `[N]` and/or an explicit `research synthesis, SSX.X` locator;
- one user message and one assistant answer.

## Split construction

The split was assigned at the semantic-family level before output generation. Five families are train-only and three families are held-out-only. The validator proves that no exact `semantic_family` key appears on both sides and that the manifest covers every record.

This is a bounded structural claim, not a claim that every possible conceptual relationship has been eliminated. The independent all-example audit recorded in task inspected the semantic split for subtler adjacency and paraphrase leakage and accepted all 60 records against C1-C6.

Train-only families (40 examples):

- zero-score diagnostic ladder;
- SFT objective, masking, and format;
- qualified hyperparameter sweeps;
- contamination and split controls;
- limited-data evaluation.

Held-out-only families (20 examples):

- PEFT/LoRA/QLoRA decisions;
- synthetic-data trade-offs;
- catastrophic forgetting and capability retention.

## Citation policy

Direct factual claims use the source research's numbered sources where those sources directly support the statement. Operational recommendations or cross-source conclusions are labeled `research synthesis, SSX.X` rather than being misattributed to a topically related external source. A corpus-level bibliography is not used as a substitute for per-example locators.

The source of truth remains the source research comment and its 23-source bibliography. No additional repository incident, issue id, or implementation-specific narrative was introduced.

## Explicit exclusions and cautions

The corpus does not assert:

- a universal minimum sample count;
- that 40 examples are sufficient or insufficient without run details;
- LoRA or QLoRA memory ratios extrapolated from GPT-3/65B experiments to a roughly 30M model;
- a universal learning rate, epoch count, or batch size;
- a licensing conclusion.

No licensing determination was made.

## Intended use

The corpus is admitted as one independently reviewed, incremental component of a larger evolving training corpus. It is not sufficient by itself for the long-term goal of a model that assists with training other models. Repository inclusion does not authorize training: no training run has used this corpus, and a later governance decision must separately authorize and schedule any such use.

## Files

- `train.jsonl`: 40 records.
- `held_out.json`: 20 records.
- `semantic_family_manifest.json`: family-to-split and family-to-id mapping.
- `held_out_exclusion_registry.json`: held-out family and id registry.
- `REPRESENTATIVE_EXAMPLES.json`: five exact records copied programmatically from the generated corpus.
- `SOURCE_MAP.md`: the source research's numbered source bibliography, preserved for locator resolution.
- `generate_corpus.py`: deterministic generator.
- `validate_dataset.py`: structural and policy validator.
- `VALIDATION_REPORT.json`: validator output, written after validation.
- `BOUNDED_RUN_PLAN.md`: proposed ADR-0013 baseline/train/evaluate plan; it does not authorize training.
- `run_bounded_cycle.py`: defaults to hash/config validation only and requires an exact-SHA review gate before its live path can run.
