# Agent resumption playbook

This playbook lets an authorised agent resume interrupted model-development work without destroying local evidence or manufacturing activity.

## Before touching a worktree

1. Read repository `AGENTS.md`, governance, decisions, latest report, and this playbook.
2. Record local `HEAD`, upstream divergence, worktree status, active processes, and relevant scheduler state.
3. Do not pull, rebase, clean, reset, checkout, delete, or bulk-format a dirty worktree.
4. Acquire the repository/project work claim required by the active control plane.
5. Create a hash-bound inventory of modified, deleted, and untracked paths without reading or publishing secrets.
6. Stop if another owner is active, provenance is unknowable, or the required claim cannot be acquired.

## Learning-oriented work pattern

Each run owns one bounded question. It must state:

- the decision the work will enable;
- observed starting state;
- hypothesis or classification rule;
- files and systems in scope;
- protected boundaries;
- evidence and tests required;
- completion and kill criteria;
- expected handoff recipient.

An agent should perform the investigation and make a recommendation, but it must not self-approve security, data admission, evaluation independence, or live deployment where governance assigns those decisions elsewhere.

## Backlog reconciliation

Classify every artifact into exactly one disposition:

- `admit`: provenance, integrity, sanitisation, relevance, and review requirements pass;
- `summarise`: several valid artifacts can be represented by one hash-bound aggregate plus retained source location;
- `quarantine`: potentially useful but one or more admission conditions remain unknown;
- `reject`: invalid, duplicated, out of scope, contaminated, or unable to support a claim;
- `superseded`: replaced by a named later artifact while retained for audit.

Never use `git clean`, destructive reset, or deletion as classification. Physical retention or removal is a later owner-approved storage action.

## Evidence handoff

The completion report must identify exact commits and evidence, distinguish local-only observations from committed evidence, record tests and failures, list remaining dirty paths, and return one of:

- `continue`: a bounded admitted next question exists;
- `hold`: evidence or review is pending;
- `reject`: the line has failed its criteria;
- `owner decision required`: the next transition exceeds delegated authority.

Scheduler existence, file counts, training completion, or agent activity are not substantive progress unless they change a supported decision.
