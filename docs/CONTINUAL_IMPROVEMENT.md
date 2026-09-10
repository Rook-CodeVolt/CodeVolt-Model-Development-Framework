# Continual improvement

The framework is never permanently “done.” Individual research items, dataset versions, experiments, releases, and improvement cycles must be done before they can feed the next stage.

## Two governed loops

```text
Continuous discovery
research agent -> source candidates -> rights/privacy review
               -> extraction/deduplication -> approved research catalogue

Controlled model evolution
approved catalogue -> immutable dataset -> bounded experiment
                   -> independent evaluation -> release decision
                   -> monitoring -> new learning events
```

Discovery is not admission. Research agents propose sources; they do not grant permission to train on them. A source may be approved for retrieval or research while remaining prohibited for training or evaluation.

## Done at each boundary

1. **Research item:** provenance, retrieval time, hash, relevance, intended use, rights status, privacy review, and risks are recorded.
2. **Dataset version:** every source is admitted for the declared use; processing and splits are documented; licence, privacy, deduplication, and contamination checks pass; the version is immutable.
3. **Experiment:** the bounded run terminates and produces complete evidence. Rejected and invalid runs are still completed experiments.
4. **Model release:** predetermined capability, regression, safety, and operational gates pass, with limitations, monitoring, ownership, and rollback documented.
5. **Improvement cycle:** the decision and lessons are recorded, unresolved findings become proposals, and an immutable baseline is established for the next cycle.

“Done” therefore means sufficiently evidenced for a declared purpose and risk boundary—not incapable of further improvement.

## Knowledge placement

- Put stable skills, behaviours, and domain reasoning into **weights**.
- Put changing factual knowledge into a versioned **retrieval store**.
- Use **tools** for live authoritative queries and actions.
- Keep permissions and constraints in a separate **policy layer**.
- Preserve failures and proposals in the **learning catalogue**.

This separation reduces unnecessary retraining and makes factual updates cheaper and reversible.

## Role separation

- The research agent discovers, compares, catalogues, and flags uncertainty.
- The data steward admits sources and dataset versions for explicit uses.
- The training agent creates candidates within declared budgets.
- The evaluation agent tests candidates without access to training decisions or hidden holdouts.
- The assurance gate applies predetermined criteria.
- The monitoring agent detects drift and new learning events.
- A human maintainer authorises high-impact transitions.

One deployment may combine roles operationally, but their permissions, inputs, outputs, and evidence must remain logically separate.
