# Continual improvement loop design: consuming round N's evidence in round N+1

- Status: design proposal — not a decision record, not an
  authorization of any round of training.
- Companion to `docs/decisions/0011-minimind-bounded-pilot-plan.md`
  (issue #46, Part 2).
- Date: 2026-09-18.

## What this document is

A design for how an **already-authorized** round N+1 pilot would
consume round N's evidence — specifically round N's
`EvaluationOutput` and `regression_check.py`'s comparison output — to
inform round N+1's dataset construction or evaluation focus. It
describes a *shape* future ADRs can follow, not a mechanism that runs
on its own.

## What this document is not

- **Not standing or unattended training authority.** Nothing here
  proposes that a round's completion triggers, schedules, or
  pre-approves the next round. Every round — round 1, round 2, round
  N — requires its own ADR (mirroring ADR-0006/ADR-0011's shape), its
  own gating sequence (ADR drafted -> the pilot-specific
  live-execution security review -> owner authorization), and its own
  resource envelope, evaluation criteria, and promotion gate. This
  document does not shorten, automate, or pre-clear any of those
  three gates for any future round.
- **Not a second regression mechanism.** `src/codevolt_mdf/regression_check.py`
  (ADR-0009) is the *only* comparison mechanism this design uses or
  proposes. This document does not define a new comparator, a new
  scoring function, or a new report shape — it describes how an
  already-existing `RegressionReport` (the return value of
  `compare_for_regressions`) gets *read* by a human drafting the next
  round's ADR, nothing more.
- **Not a claim that any round beyond round 1 (ADR-0011) is planned,
  scheduled, or likely.** This is a design for round N+1 *if and when*
  round N is authorized, executed, and produces evidence — a
  conditional design, not a commitment.

## Why this document exists

`docs/decisions/0006-bounded-real-trl-pilot-plan.md` and
`docs/decisions/0011-minimind-bounded-pilot-plan.md` each specify one
bounded pilot. Neither describes what happens *after* — if a pilot's
evidence shows partial capability (ADR-0006's executed TRL pilot
scored `0.7` on 10 held-out arithmetic examples, 3 wrong) or if a
regression comparison flags previously-passing examples now failing,
that evidence is currently a dead end: nothing in this framework's
existing documentation says how a human would use it to shape what
gets tried next. Issue #46 (goal, owner authority, 2026-09-18) is
explicit that the actual objective is "a real feedback loop: a
candidate trains, is independently evaluated against held-out data,
the result feeds an evidence-led decision about the next round — not
a single isolated pilot that stops after one run." This document is
that design.

## The mechanism: `regression_check.py`, unmodified, reused

`compare_for_regressions(baseline, candidate, previous_candidate=None)`
already produces exactly the comparison a round-N+1 decision needs:

- **`baseline`**: the immutable baseline's `EvaluationOutput` — for a
  MiniMind pilot lineage, the pristine/untrained-checkpoint score
  against the pilot's registered held-out set (the same role
  `docs/REAL_ADAPTERS.md` describes every current evaluator test as
  scoring against, "the pristine, untrained base checkpoint as a
  stand-in artifact").
- **`candidate`**: round N+1's newly trained artifact's
  `EvaluationOutput`.
- **`previous_candidate`** (optional): round N's accepted artifact's
  `EvaluationOutput` — the direct N-to-N+1 comparison this document is
  about.

`RegressionReport` (already returned by this call, unmodified) carries
exactly the fields a human drafting round N+1's dataset needs:
`regressed_example_ids` (which specific held-out examples that were
passing in round N are now failing — irrelevant to dataset design,
relevant to *catching a regression before promotion*), and, read the
other direction, the **complement** of `regressed_example_ids` within
`previously_passing_count` (examples still failing in both rounds —
this is the signal this document's "what transfers" section below
actually uses). `regression_check.py` does not itself expose a
"still consistently failing" list as a named field — the reader
derives it by taking round N's raw `EvaluationOutput.results` (already
available; it is the same object fed into `compare_for_regressions`
as `previous_candidate`/`baseline`) and filtering for
`ExampleResult.correct == False`. This document proposes no change to
`regression_check.py` to add such a field — deriving it from existing,
already-produced data is sufficient and keeps `regression_check.py`'s
scope exactly as narrow as ADR-0009 defined it (comparison of
previously-*passing* examples only; it was never meant to be a general
failure-analysis tool, and this design does not ask it to become one).

## What specifically transfers from round N to round N+1

Two distinct signals, both derived from data `regression_check.py`
either produces directly or was already given as input — no new data
source, no new instrumentation:

### 1. Per-example failures -> next dataset's probe targets

Round N's `EvaluationOutput.results` (the full per-example list, not
just the aggregate score) names exactly which held-out examples the
round-N candidate got wrong, and — from ADR-0006's executed pilot as
a concrete precedent — often *how* it got them wrong (ADR-0006's
`0d3f1668...` evaluation evidence shows all three wrong answers were
near-miss arithmetic errors, not formatting failures, a distinction
visible only by reading the per-example `raw_output` field, not the
aggregate `0.7`). A human drafting round N+1's ADR reads this detail
and proposes a dataset shape for round N+1's *training* split that
specifically targets the failure pattern observed — e.g., if round
N's MiniMind pilot (ADR-0011) shows failures clustered on multi-step
unit conversions (km<->kg-style cross-unit-family confusions) rather
than single-step conversions, round N+1's training data construction
would deliberately include more multi-step examples, and round N+1's
*held-out* set would deliberately include a held-out probe of the
same failure family (still generated fresh, still disjoint from
train, per the same discipline ADR-0006/ADR-0011 both require) to
test whether the targeted training actually closed that specific gap
rather than merely re-measuring the same failure differently.

This is a proposal a human writes into round N+1's own ADR — it is
not automated, and no code in this repository reads round N's results
and writes round N+1's dataset unattended. The transfer is: evidence
in, human judgment, ADR text out.

### 2. `regression_check.py`'s comparison -> round N+1's regression-safety net

Independent of what round N+1's *new* dataset targets, round N+1's
ADR should specify that its own acceptance evidence includes a
`compare_for_regressions(baseline=<round-1 baseline eval>,
candidate=<round-N+1 eval>, previous_candidate=<round-N eval>)` call
before any promotion decision is made — exactly the 3-way comparison
`compare_for_regressions` already supports (`previous_candidate`
optional parameter, exercised in
`tests/test_regression_check.py`'s "3-way comparison" case). This
answers a different question than signal 1 above: not "did round N+1
improve on what round N got wrong" but "did round N+1's new training
*break* anything round N (and the immutable baseline) got right." Both
questions matter for a genuine feedback loop — targeting new failures
without a regression check risks a round N+1 that "fixes" the probed
gap while quietly breaking previously-solid capability, exactly the
failure mode `docs/EVALUATION_POLICY.md`'s "Regression checks for
retained capabilities" dimension and ADR-0009 exist to catch.

### What does not transfer

- **No weights, no checkpoints, no optimizer state carries forward
  implicitly.** Whether round N+1 starts from round N's accepted
  artifact, from the same fixed base checkpoint round N started from,
  or from a fresh random init is a per-round ADR decision (see
  ADR-0011's own "Model" section, which explicitly declines to resolve
  this for round 1) — this design does not default to "always
  continue from the last accepted artifact," because that default
  would itself be a form of standing authority this document is
  explicit it does not propose.
- **No evaluation criteria, promotion gate, or resource envelope
  carries forward unchanged by default.** Round N+1's ADR states its
  own `ResourceBudget` table, its own kill criteria, and its own
  acceptance bar, drafted fresh (informed by, but not copy-pasted
  from, round N's actual measured `resource_usage` — round N's
  measured wall/CPU/memory/storage usage, once it exists, is itself
  useful evidence for whether round N+1's proposed budget is
  realistic, the same way ADR-0006's real-inference timing estimate
  informed its own budget before execution).

## What does not change, restated explicitly

Per issue #46's own scope: each future round still requires its own
ADR-authorized bounded pilot (mirroring ADR-0006/ADR-0011's exact
structure — Context, Decision with Model/Dataset/Concurrency/Resource
limits/Filesystem/Independent evaluation wiring/Kill criteria, Gating
naming the same three-step sequence), its own gating (the security reviewer's
live-execution review scoped to that exact run's configuration if it
materially changed from the last cleared one — see "Recommendation"
below for what "materially changed" might mean, proposed, not
decided), and owner authorization. No round is authorized to modify
its own evaluation criteria, promotion gate, or resource envelope —
those remain drafted by a human in that round's own ADR and reviewed
by the same independent parties (the security reviewer, then owner) every prior round
went through. This document is a design for an evidence chain *across*
authorized rounds; it is not, and does not propose to become, standing
or unattended authority to keep training.

## Recommendation (not a decision): a lighter re-review path for materially-unchanged configurations

This is framed as a proposal for the security reviewer to accept or reject, not a
fait accompli — analogous in spirit to issue #37's doc-only
stale-review fast path, but for a *security* review, which is a
higher-stakes class of gate than a documentation-freshness check, so
this recommendation is deliberately narrower and more conservative
than that precedent.

**The proposal**: if round N+1's configuration is *materially
unchanged* from a configuration the security reviewer has already cleared for live
execution in round N — same pinned engine commit/version, same
resource budget table, same filesystem/network posture, same
subprocess-invocation code path (i.e., no adapter code changed between
round N's cleared review and round N+1's proposed run) — the security reviewer
could elect a **lighter-touch re-review**: re-verify the specific deltas
(new dataset content/hash, new held-out package id, the model
checkpoint provenance if it changed) rather than repeating the full
sandbox/egress/secrets/supply-chain/safe-stop review from scratch.

**What would NOT qualify as materially unchanged** (and would
therefore always require the full review, no exceptions under this
proposal): any change to the pinned engine commit or version, any
change to `_build_subprocess_args`/the adapter's subprocess-invocation
code (including the CLI-mismatch fixes ADR-0011 identifies as a
blocking pre-condition for round 1 itself — landing that fix is
itself an adapter-code change and would require full review the first
time it's exercised live), any change to the resource budget table's
values, any change to `network_policy` or `filesystem_root`
structure, or any change to which evaluator adapter scores the
result.

**Why this is a recommendation and not a decision**: unlike issue
#37's precedent (a documentation-freshness fast path — low blast
radius, easily reversible, no safety content), a live-execution
security review exists specifically to catch sandbox escape, secret
exfiltration, and supply-chain risk in an actually-running process —
risks that do not necessarily shrink just because the *configuration*
text is unchanged (e.g., the pinned MiniMind commit's own upstream
content does not change between rounds if the pin doesn't move, but a
reviewer's *confidence* that nothing has drifted in the surrounding
environment, dependency versions, or host state is a judgment call
only the reviewer is positioned to make each time). This document
takes no position on whether that judgment call should ever resolve to
"yes, lighter review is safe here" — it names the option, scopes what
would and would not qualify, and leaves the accept/reject call to
the security reviewer, per round, not as a standing policy this document establishes
unilaterally.

## Summary: the loop, end to end

1. Round N's ADR is drafted, reviewed (the security reviewer's live-execution review,
   full scope, always — round N is round N's own first exercise of
   whatever configuration it proposes), authorized, and executed.
2. Round N's trained artifact is scored via
   `evaluator_contract.run_evaluator_contract` against round N's
   registered held-out set, producing `EvaluationOutput`.
3. A human reads round N's `EvaluationOutput.results` (per-example
   detail) and, if a prior round/baseline exists,
   `compare_for_regressions`'s `RegressionReport` — both already-real,
   already-merged mechanisms, nothing new — to identify (a) specific
   failure patterns worth targeting in round N+1's dataset and (b)
   confirmation that round N did not regress previously-passing
   capability.
4. That human drafts round N+1's own ADR (full ADR-0006/ADR-0011
   structure), informed by step 3's evidence but not automatically
   generated from it, proposing a dataset shape/evaluation focus that
   targets step 3's findings.
5. Round N+1's ADR goes through the same three-gate sequence round N
   did: ADR drafted -> the pilot-specific live-execution security
   review (full scope by default; lighter-touch only if the security reviewer
   affirmatively elects the recommendation above for that specific
   round) -> owner authorization.
6. Round N+1 executes (if authorized), producing its own
   `EvaluationOutput`, and the loop repeats from step 3 with N
   incremented — each iteration a separate, fully gated decision, not
   an accumulating standing authority.

No step in this sequence is unattended. No step skips the security review
by default. No step skips owner authorization. The only thing that
repeats without renegotiation is the *mechanism* used to compare
rounds (`regression_check.py`) and the *shape* of the ADR each round
still individually requires.
