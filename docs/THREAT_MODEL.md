# Threat model

## Protected assets

Credentials, private or personal data, training hardware, model artifacts, evaluation holdouts, evidence integrity, and the authority to promote or publish a model.

## Principal risks

- Malicious datasets, model files, adapters, or dependencies executing code.
- Dataset poisoning, benchmark contamination, or fabricated metrics.
- Secrets and personal data entering manifests, logs, artifacts, or contributions.
- Pull requests reaching privileged self-hosted runners.
- An evaluator or trainer altering the acceptance policy or its own evidence.
- Resource exhaustion through unbounded training.
- Unsafe or licence-incompatible model publication.

## Initial controls

- Immutable manifests and hashes for evidence-bearing inputs.
- Separate trainer, evaluator, and assurance responsibilities.
- Fail-closed validation and explicit invalid results.
- No automatic promotion in `v0.1`.
- No secrets or private datasets in public evidence.
- Hosted CI only; privileged training runners are deferred.
- Pinned provenance, declared resource budgets, human approval, and rollback as future integration requirements.

This document is a living model. New adapters must describe their additional trust boundaries and mitigations.
