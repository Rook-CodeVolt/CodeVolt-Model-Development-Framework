# Feedback: issues, bugs, and recommendations

This page explains how to report a defect, propose a capability, or share a
finding — for human contributors and for autonomous agents — using the
channels this repository already has. It does not introduce a new service.

## Channel selection

| You have | Use | Why |
| --- | --- | --- |
| A reproducible defect | [Bug report](../.github/ISSUE_TEMPLATE/bug.yml) issue form | Structured reproduction, expected vs. actual behaviour |
| A verified finding from agent-assisted work (defect, evidence gap, safety, capability, usability, improvement, or negative result) | [Agent feedback](../.github/ISSUE_TEMPLATE/agent-feedback.yml) issue form | Matches the report shape required by [`AGENTS.md`](../AGENTS.md) |
| A new trainer, evaluator, dataset, or learning-strategy capability idea | [Capability proposal](../.github/ISSUE_TEMPLATE/capability.yml) issue form | Requires a need statement and an evidence plan, per [`docs/GOVERNANCE.md`](GOVERNANCE.md) |
| A question, design discussion, or experiment write-up | GitHub Discussions | Not a tracked action item; see [`SUPPORT.md`](../SUPPORT.md) |
| A vulnerability, leaked credential, unsafe artifact, or content that could harm people | Private vulnerability reporting (see [`SECURITY.md`](../SECURITY.md)) | Must never be filed as a public issue |

If you are unsure which form fits, open the closest issue form rather than a
blank issue — blank issues are disabled so that every report carries the
minimum structured fields maintainers need to triage it.

## Who can submit, and how

- **Public users** file through the issue forms above or GitHub Discussions.
  You do not need to run the framework, capture logs, or expose any private
  data to report a problem: describe what you expected, what happened, and
  the smallest example that shows it, using synthetic or already-public
  material only.
- **Autonomous agents** follow the mandatory loop in [`AGENTS.md`](../AGENTS.md):
  confirm the observation with the least destructive check available,
  separate fact from hypothesis, strip anything sensitive, search for an
  existing report, then file (or return to the requesting human) the
  `agent-feedback` fields. [`docs/AGENT_INTEGRATION.md`](AGENT_INTEGRATION.md)
  covers the full operating pattern and how a report relates to model
  learning events.

## Safe evidence, without requiring exposure

Never include credentials, personal information, private datasets,
proprietary prompts, or sensitive model output in a public issue. A useful
report does not require logs or production data — it requires:

- What you expected (a contract, doc, or reasonable default) versus what
  happened, stated as observed fact.
- The smallest deterministic reproduction you can construct, preferably from
  a public example (e.g. `examples/deterministic-experiment.json`) or
  synthetic input.
- Safe references only: a run ID, test name, commit hash, or file path —
  never raw sensitive content.

If a finding cannot be shown without exposing something sensitive, describe
the shape of the problem and state that evidence is being withheld for that
reason; a maintainer will ask privately if more is needed.

## Deduplication

Before filing, search open and closed issues and Discussions for the same
symptom. If a match exists, add your reproduction, environment, or evidence
as a comment on the existing item instead of opening a new one. The
`agent-feedback` form makes this an explicit, required checkbox for agents;
human reporters should do the same.

## Triage lifecycle and statuses

Every accepted report moves through the same labelled lifecycle:

```text
new -> triage -> confirmed -> accepted -> in progress -> resolved -> closed
                     `------------------> wontfix / duplicate / invalid -> closed
```

- **new / triage** — filed via an issue form; awaiting a maintainer read.
  Forms apply the `triage` label (and `agent-feedback`, `bug`, or
  `capability`/`proposal`) automatically on creation.
- **confirmed** — a maintainer reproduced the issue or accepted the need.
- **accepted** — scoped for work; may spawn a linked capability proposal or
  architecture decision record under [`docs/decisions/`](decisions/).
- **in progress / resolved** — a pull request references the issue; the issue
  closes when the change and its tests merge.
- **wontfix / duplicate / invalid** — closed with a recorded rationale
  instead of silent deletion, per [`docs/GOVERNANCE.md`](GOVERNANCE.md).

Labels are the source of truth for status because this repository does not
run a separate ticketing system. If a label referenced by an issue form is
not yet visible on the repository's Issues page, treat the form's intent as
authoritative — creating the missing label is a repository-setting action
tracked separately from this document (see "Unresolved repository-setting
actions" in the change that introduced this page).

## Security and private-report boundary

Anything that could expose data, credentials, systems, or people to harm
must go through private vulnerability reporting, never a public issue or
Discussion. See [`SECURITY.md`](../SECURITY.md). Maintainers may redirect a
public report that turns out to be security-sensitive into a private
channel and remove the public copy's sensitive detail.

## Decision and closure loop

Accepted reports do not close silently. Each one reaches one of: fixed by a
merged pull request, converted into a capability proposal that follows the
lifecycle in [`docs/GOVERNANCE.md`](GOVERNANCE.md)
(`proposed -> researched -> experimental -> independently evaluated ->
supported`, or `rejected`), or closed as wontfix/duplicate/invalid with a
recorded rationale. Rejected and invalid outcomes are kept, not deleted,
because they prevent repeated work.

## Accepted feedback does not automatically enter training data

Filing, confirming, or accepting a piece of feedback is a project-management
action. It never by itself admits any content into a dataset used for model
training. Turning a report into a durable artifact means one of:

- a new or updated **test** (see `tests/`),
- a documentation change (this repository's `docs/` or root policy files),
- a **capability proposal** or architecture decision record, which still
  requires its own evidence plan, bounded experiment, and independent
  evaluation before anything is "supported" per
  [`docs/GOVERNANCE.md`](GOVERNANCE.md).

Any example, log snippet, or dataset a reporter attaches to an issue is
discovery material at most. It follows the same admission path as any other
source: quarantined by default, and only usable for training after an
explicit rights/privacy review and intended-use approval, exactly as
described in [`docs/DATA_GOVERNANCE.md`](DATA_GOVERNANCE.md) and
[`docs/CONTINUAL_IMPROVEMENT.md`](CONTINUAL_IMPROVEMENT.md). An issue being
open, accepted, or closed never bypasses that review.
