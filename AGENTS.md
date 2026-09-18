# Instructions for AI agents

These instructions apply to agents working anywhere in this repository.

## Primary obligation

Complete the user's scoped task first. While doing so, preserve useful evidence about framework defects and improvements you directly observe. Do not invent findings merely to produce feedback.

## Agent quick-start

Before doing any work in this repository, an agent (or an agent swarm) should read the following three documents in order: this file first for the operating contract, then the integration and feedback pattern, then the resumption playbook for how to safely pick up or hand off work.

- [`AGENTS.md`](AGENTS.md) — this file: the repository-level operating contract, required feedback loop, and contribution quality bar.
- [`docs/AGENT_INTEGRATION.md`](docs/AGENT_INTEGRATION.md) — how agent observations become reviewable evidence, and how agent feedback relates to model learning.
- [`docs/AGENT_RESUMPTION_PLAYBOOK.md`](docs/AGENT_RESUMPTION_PLAYBOOK.md) — how to resume interrupted work, claim it safely, and hand it off without destroying evidence.

## Required feedback loop

When you identify a reproducible defect, unsafe behaviour, missing contract, evidence-integrity problem, recurring friction, or a concrete improvement:

1. Confirm the observation using the least destructive reproducible check available.
2. Separate observed facts from hypotheses and recommendations.
3. Remove credentials, personal information, private datasets, proprietary prompts, and sensitive model output.
4. Search existing reports or issues where access is available; add evidence to an existing item instead of duplicating it.
5. Create a report using `.github/ISSUE_TEMPLATE/agent-feedback.yml`, or return the same fields clearly to the requesting user when you cannot create an issue.
6. Link relevant experiment manifests, evidence bundles, tests, logs, commits, or adapter versions using safe references.
7. State severity, confidence, user impact, reproduction steps, proposed acceptance criteria, and a safe next action.

Do not silently change governance, evaluation thresholds, protected data, published models, or promotion policy in response to a finding. A report is not authority to implement a materially broader change.

## Finding categories

- `defect`: implemented behaviour contradicts a contract or documented expectation.
- `evidence-gap`: a claim cannot be verified from the recorded bundle.
- `safety`: behaviour may expose data, credentials, systems, people, or model users to harm.
- `capability`: a concrete missing extension blocks or substantially degrades a use case.
- `usability`: avoidable friction or ambiguity creates recurring errors.
- `improvement`: evidence suggests a measurable enhancement to an existing capability.
- `negative-result`: a reasonable approach failed and should not be repeated without new evidence.

## Minimum report

```yaml
title: "Concise observable problem or opportunity"
category: defect | evidence-gap | safety | capability | usability | improvement | negative-result
severity: low | medium | high | critical
confidence: low | medium | high
observed: "What happened, without inference"
expected: "The relevant contract or desired measurable behaviour"
reproduction:
  - "Minimal deterministic step"
evidence:
  - "Safe path, run ID, test, or commit"
hypothesis: "Likely cause, explicitly identified as a hypothesis"
acceptance_criteria:
  - "Condition that would demonstrate resolution"
recommended_next_action: "Smallest safe next step"
privacy_review: "Confirmed no sensitive material is included"
```

Critical security findings must not be filed publicly. Follow `SECURITY.md` and notify the user that a private report is required.

See `docs/FEEDBACK.md` for the full channel-selection, deduplication, triage-status, and closure-loop reference that this loop feeds into.

## Contribution quality

For code changes, run the relevant tests and lint checks. New adapters require provenance, compatibility bounds, failure behaviour, documentation, tests, and reproducible evidence. Never describe a candidate as improved solely because training completed.
