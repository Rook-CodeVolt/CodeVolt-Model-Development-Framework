# Agent integration and feedback

AI agents can help improve the framework when their observations are converted into reviewable evidence rather than disappearing into individual sessions.

`AGENTS.md` is the repository-level operating contract. Compatible coding agents should discover it when they enter the repository. For other agent systems, include or reference its required feedback loop in the agent's project instructions.

## Recommended operating pattern

```text
agent performs scoped work
        |
        +-- no material finding --------------------> finish normally
        |
        `-- verified finding -> sanitise -> deduplicate -> structured report
                                                   -> human triage
                                                   -> proposal/experiment
                                                   -> evidence-led decision
```

The agent should not create noisy reports for preferences, untested speculation, or expected behaviour. It should not expose private task context in a public repository. When it lacks permission to open an issue, it returns a complete report to the user for review.

See [`docs/FEEDBACK.md`](FEEDBACK.md) for channel selection, deduplication, the triage lifecycle and statuses, and the security/private-report boundary that applies to both agents and human reporters.

## Connecting reports to model learning

Agent feedback and model learning events are related but distinct. An agent report concerns the framework or its operation. A learning event records model behaviour. A verified agent report may propose a new evaluation, adapter, guardrail, or learning event, but it does not directly enter a training dataset.

Only reviewed, licensed, privacy-safe examples should enter versioned datasets. Every resulting change still passes through a bounded experiment, independent evaluation, evidence verification, and promotion approval.
