# Evaluation and promotion policy

A completed training run is not an improvement. A candidate is eligible for promotion only when its declared evidence is complete and every mandatory gate passes.

## Required evaluation dimensions

- Target capability compared with an immutable baseline.
- Regression checks for retained capabilities.
- Safety and misuse tests appropriate to the intended use.
- Robustness across representative inputs and fixed seeds where relevant.
- Resource, latency, and size measurements for the target hardware.
- Dataset provenance, licence, privacy, consent, and contamination review.
- Reproduction from the locked manifest and declared environment.

Thresholds must be declared before seeing candidate results. Evaluation data must not be used as training data, and hidden holdouts must remain inaccessible to the trainer and model-development agent.

## Decisions

- **Accepted:** all mandatory evidence exists and all declared thresholds pass.
- **Rejected:** evidence is valid but one or more thresholds fail.
- **Invalid:** evidence is missing, corrupt, contaminated, unverifiable, or produced outside the declared contract.

Only an accepted result is eligible for a separate promotion review. Promotion additionally requires intended-use documentation, limitations, security/privacy review, artifact integrity, a named owner, monitoring, and a tested rollback path.
