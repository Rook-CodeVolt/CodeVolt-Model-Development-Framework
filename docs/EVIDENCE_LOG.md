# Evidence log: issue #7, the trainer-adapter contract and its first real pilot

This page exists for one reason: to let a cold reader check, from real
commits and real numbers, whether this project's "evidence before
claims" principle (see [README](../README.md#principles)) was actually
followed for one complete piece of work, rather than asserted about it.
Everything below links to a real issue comment, pull request, commit,
or committed evidence file in this repository. Where a number is
reported, it is the number recorded in that committed evidence, not a
summary written after the fact.

This is a log, not a roadmap. For where effort should go *next*, see
[Next progression](PROGRESSION.md). For the current adapter inventory,
see [Real adapters](REAL_ADAPTERS.md). This page's job is narrower:
show the one governance chain, end to end, for
[issue #7](https://github.com/Rook-CodeVolt/CodeVolt-Model-Development-Framework/issues/7).

## What issue #7 asked for

[Issue #7](https://github.com/Rook-CodeVolt/CodeVolt-Model-Development-Framework/issues/7)
(opened 2026-09-12) asked for a real, versioned trainer-adapter
contract before this project integrated any real training engine — not
inferred behaviour from a two-method `Protocol`, and not training
admitted on the strength of a design document alone. It named seven
acceptance steps: contract + fake adapter, contract tests, a real
bounded pilot, independent evaluation against a held-out set, an
independent security review of the live engine, acceptance criteria,
and named kill criteria. The rest of this page is the record of those
seven steps actually closing, in order, with the reviews that gated
each transition.

## The chain

### 1. Contract, fake adapter, first independent review — PR #9

[PR #9](https://github.com/Rook-CodeVolt/CodeVolt-Model-Development-Framework/pull/9)
(merged `1b8a9ea`) added `TrainerAdapterContract` v1
([ADR-0002](decisions/0002-trainer-adapter-contract-v1.md)) and the
deterministic fake adapter, with 11 conformance tests (success,
rejection, invalid-input, timeout, cancellation, resource-overrun,
checkpoint/resume, tamper). An independent review of that PR confirmed
the contract and fake adapter were sound and honestly scoped, but
named three real limitations as *preconditions for any real adapter*,
not defects in this PR: resource usage was adapter-self-reported
rather than OS-measured; cancellation was cooperative-only
(a non-conforming adapter's thread could outlive an "interrupted"
report); and `filesystem_root`/`network_policy` were declared but not
enforced. Issue #7 was left open specifically because those three
preconditions, and the real pilot itself, were not yet done.

### 2. Closing the three preconditions — two more independent review rounds, two real vulnerabilities found — PR #13, PR #14

[PR #13](https://github.com/Rook-CodeVolt/CodeVolt-Model-Development-Framework/pull/13)
(merged `21437a9`,
[ADR-0003](decisions/0003-trainer-contract-os-level-enforcement.md))
replaced adapter self-reported resource usage with OS-measured
wall/CPU/memory (`ps` + `getrusage(RUSAGE_CHILDREN)`), replaced
cooperative cancellation with a real `SIGKILL`, and added
Python-interpreter-scope filesystem/network enforcement — explicitly
documented as *not* a true OS sandbox (does not stop subprocess calls,
compiled extensions, or non-Python DNS resolution). This went through
two independent security review rounds before merge, which found two
real, reproducible defects during review, not merely style comments:
an orphaned-subprocess-survival case, and a working proof-of-concept
pickle-deserialisation remote-code-execution across the process
boundary. Both were fixed and independently re-verified with fresh,
different attack proofs before merge. One residual gap was disclosed,
not hidden: a grandchild process calling `os.setsid()` itself could
still escape the process-group kill.

[PR #14](https://github.com/Rook-CodeVolt/CodeVolt-Model-Development-Framework/pull/14)
(merged `6d078ba`) closed that disclosed gap with a full
ppid-descendant-tree walk. Its own review cycle: round 1 REQUEST
CHANGES (pid-tree-walk failures were silent, no observability); round
2 REQUEST CHANGES (the degraded-walk signal reached
`TrainingOutput.reason` on 2 of 3 kill-trigger paths but not the
third); round 3 APPROVE, independently reproduced (41/41 tests, own
negative control). This is reported here specifically because it is a
good example of review working as intended, not as a rubber stamp —
two rounds of real, specific pushback before approval, on a
security-relevant code path.

At the end of this stage, issue #7's contract-and-isolation
preconditions were closed, but the issue stayed explicitly open: no
real training engine had been integrated, and steps 4–7 (independent
evaluation, security review of a real engine, the bounded pilot, and
final acceptance) had not started.

### 3. Real (non-fake) trainer adapter — PR #15

[PR #15](https://github.com/Rook-CodeVolt/CodeVolt-Model-Development-Framework/pull/15)
(merged `2026-09-17`, [ADR-0005](decisions/0005-trl-trainer-adapter-v1.md))
added `TRLTrainerAdapter`
(`src/codevolt_mdf/trl_adapter.py`), a real `TrainerAdapterV1`
wrapping [TRL](https://github.com/huggingface/trl)'s `SFTTrainer`,
chosen over Unsloth/Axolotl/LLaMA-Factory specifically because it has
no CUDA-only hard dependency and this project's only bounded-inference
evidence at the time was on a CPU/MPS Mac host. 28 conformance tests.
An independent review of this PR raised one medium finding — the
declared upstream version bound (`trl>=0.20.0,<0.25.0`) was a range,
not narrow enough to trust for a pilot — which is why the pin was
narrowed to the exact version `trl==0.24.0` before any pilot ran. That
review explicitly stated it had *not* evaluated live execution
behaviour, because none had occurred — `adapter.train()` had never
been called by anything in the repository. That explicit gap is what
step 5 below exists to close.

### 4. Real (non-fake) evaluator adapter and held-out registry — PR #16, PR #17

[PR #16](https://github.com/Rook-CodeVolt/CodeVolt-Model-Development-Framework/pull/16)
narrowed the TRL pin to `0.24.0` and added
`EvaluatorAdapterContract v1` (`src/codevolt_mdf/evaluator_contract.py`)
— a separately configured evaluator that scores an artifact against a
held-out set the trainer adapter never touches, structurally separate
from `trl_adapter.py`/`trainer_contract.py` (no import relationship
either direction, proven by a structural AST test, not just asserted
in a docstring). It reuses the bidirectional
`HeldOutExclusionRegistry` pattern so a held-out set contaminated by
prior training use anywhere is rejected before scoring, not merely
declared clean. It also drafted, but explicitly did not authorise,
[ADR-0006](decisions/0006-bounded-real-trl-pilot-plan.md), the bounded
pilot's configuration.

[PR #17](https://github.com/Rook-CodeVolt/CodeVolt-Model-Development-Framework/pull/17)
(merged `2026-09-17`) added the actual real (non-fake) evaluator:
`HFLocalCausalLMEvaluatorAdapter`
(`src/codevolt_mdf/hf_local_evaluator_adapter.py`), scoring held-out
examples one at a time via local Hugging Face causal-LM inference
(greedy decoding, `torch.no_grad()`) against the pinned
`HuggingFaceTB/SmolLM2-135M` checkpoint. Every test in that PR's own
suite scored held-out examples against the pristine, *untrained* base
checkpoint — proof the scoring path itself works, explicitly not
pilot evidence for a trained candidate, because no trained candidate
existed yet.

At this point, both real adapters existed and were contract-tested,
but neither had been exercised together, and no live TRL process had
ever actually run — [docs/REAL_ADAPTERS.md](REAL_ADAPTERS.md) recorded
this honestly at the time as "no real training has happened."

### 5. Pilot spec locked, live-execution security review — blocked, three real defects found, fixed, re-cleared — PR #18 (closed, superseded), PR #21

[PR #18](https://github.com/Rook-CodeVolt/CodeVolt-Model-Development-Framework/pull/18)
finalised ADR-0006 from a proposed plan into a concrete, locked spec —
exact model revision and hash
(`HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`),
a newly built synthetic 2-digit-addition dataset (40 train / 10
held-out records, numeric-pair-disjoint, not merely id-disjoint), and
an unchanged resource budget (30 min wall / 60 min CPU / 8 GB memory /
0 GPU / 2 GB storage / offline network / `max_steps=50`) — specifically
so Maya could review a concrete spec instead of prose. **This PR did
not authorise execution.**

Maya's pilot-specific live-execution security review of that spec —
a distinct, broader scope than PR #15's code review, which had
explicitly not covered live execution — found three real, reproducible
HIGH-severity defects by actually running `TRLTrainerAdapter.train()`,
none of which the 28 static conformance tests caught, because those
tests stop before `train()` is ever called by design:

1. The filesystem sandbox guard rejected legitimate read-only imports
   (`dill`'s `/dev/null` probe, `transformers`' lazy-module reads),
   so real training could not start at all with `filesystem_root` set.
2. `SFTConfig` had no `save_total_limit`, so checkpoint storage could
   grow unbounded over a longer run.
3. `_safe_halt_output` did not persist full resumable checkpoint state.

[PR #21](https://github.com/Rook-CodeVolt/CodeVolt-Model-Development-Framework/pull/21)
(merged `c3cbc55`) fixed all three: the filesystem guard now
intercepts writes only, not reads, outside `filesystem_root`; the
`SFTConfig` bound was added; and `_safe_halt_output` now persists full
checkpoint state. All three were self-verified with real
`train()`/kill/resume execution against the pinned checkpoint before
re-review. Maya's re-review returned **CLEAR TO EXECUTE**, with one
residual note explicitly logged as LOW/non-blocking (a pre-existing
macOS `ps`-timing quirk, no action needed) — this is the review
finding a real problem, blocking on it, and then clearing the
*specific fix* rather than the general idea, and it is documented here
precisely because that sequence is the point of having the review
step at all.

PR #18 itself was later closed as superseded — its one commit was
cherry-picked cleanly into PR #22 below, so nothing in it was lost; it
just did not need to exist as a separate merge once the pilot execution
PR carried the same commit forward.

### 6. Real pilot executed — PR #22

[PR #22](https://github.com/Rook-CodeVolt/CodeVolt-Model-Development-Framework/pull/22)
wires the trainer and evaluator together
(`examples/pilot-adr0006/run_pilot.py`) and executes ADR-0006's locked
50-step SFT run: `run_trainer_contract` (real `TRLTrainerAdapter.train()`)
and, only if `TrainingOutput.status == ACCEPTED`, immediately scores
the resulting artifact through `run_evaluator_contract` using the real
`HFLocalCausalLMEvaluatorAdapter` against the registered held-out set
— both calls in the same run, so the evaluator scores the artifact
this specific trainer run actually produced. A cheap 2-step synthetic
dry-run against a disposable, non-registered dataset proved the wiring
end to end before the full 50-step budget was spent.

**Model:** `HuggingFaceTB/SmolLM2-135M` (Apache-2.0, 135M parameters),
revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, `model.safetensors`
269,060,552 bytes, sha256
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
(re-verified independently at spec-lock time and again at pilot
execution time — see
[ADR-0006](decisions/0006-bounded-real-trl-pilot-plan.md)).

**Training run** (`run_id=adr0006-pilot-20260917`), OS-measured, all
five resource dimensions well within ADR-0006's locked budget:

| Field | Result | Budget |
|---|---|---|
| `status` | `ACCEPTED` | — |
| wall time | 15.7 s | 1800 s |
| CPU time | 14.1 s | 3600 s |
| memory peak | 1.6 GB | 8 GB |
| storage used | 518 MB | 2 GB |
| GPU count | 0 | 0 |

Training loss dropped `1.71 → 0.69` over the 50 steps
(committed log history:
`examples/pilot-adr0006/evidence/training-evidence.sanitised.json`).

**Evaluation run**, scored against the real, registered 10-example
held-out set (package `adr0006-pilot-heldout-v1`), zero contamination
flagged by `HeldOutExclusionRegistry.check_held_out_not_trained`
immediately before scoring:

| example | input | expected | model output | correct |
|---|---|---|---|---|
| heldout-0001 | `Add: 27 + 69 =` | 96 | 96 | yes |
| heldout-0002 | `Add: 69 + 28 =` | 97 | 97 | yes |
| heldout-0003 | `Add: 58 + 46 =` | 104 | 94 | **no** |
| heldout-0004 | `Add: 28 + 56 =` | 84 | 84 | yes |
| heldout-0005 | `Add: 51 + 13 =` | 64 | 64 | yes |
| heldout-0006 | `Add: 11 + 49 =` | 60 | 50 | **no** |
| heldout-0007 | `Add: 41 + 45 =` | 86 | 86 | yes |
| heldout-0008 | `Add: 42 + 49 =` | 91 | 181 | **no** |
| heldout-0009 | `Add: 20 + 23 =` | 43 | 43 | yes |
| heldout-0010 | `Add: 64 + 17 =` | 81 | 81 | yes |

**Aggregate score: 0.7 (7/10).** All three wrong answers are named,
not aggregated away — the same three appear in the committed evidence
(`examples/pilot-adr0006/evidence/pilot-result.sanitised.json`) as
they do in this table.

At the time of writing, PR #22 is open (CI green on all three tested
Python versions, mergeable) pending merge — the run described above is
real and already executed and committed to that branch; only the
merge into `main` is outstanding.

### 7. Independent evidence review — PASS

Maya independently reviewed PR #22's evidence rather than trusting its
prose. Concretely, she:

- recomputed the held-out set's content hash and the training dataset's
  content hash directly from the committed files and confirmed both
  match the values locked in ADR-0006 — the held-out set actually
  scored is the one that was actually locked, not a substitute;
- independently verified the train/held-out numeric pairs have zero
  overlap, not just non-overlapping ids;
- reconstructed the evaluator's actual scoring rule from
  `hf_local_evaluator_adapter.py` (normalised containment match) and
  reapplied it herself to all 10 raw model outputs against the
  held-out set's expected values — every one of the declared
  `correct`/`score` flags reproduced exactly; none were mismarked, and
  none of the three wrong examples were softened or excluded;
- recomputed both evidence-bundle hashes from the raw committed bytes
  and confirmed they match the hashes recorded in the PR and ADR;
- grepped the full diff for accept/promotion/production-readiness
  language and found only explicit negations of it, and confirmed
  structurally — by reading `evaluator_contract.py`'s `EvaluationOutput`
  dataclass itself, not its docstring — that it has no field capable of
  holding an accept, promote, or production-readiness claim, so no
  code path in this PR could produce one even by mistake;
- confirmed PR #21's fix commit hash matches what PR #22 claims as its
  gating precondition, and that PR #22's CI passed on the actual merge
  commit, not a stale run.

**Verdict: PASS.** No material findings. One cosmetic, non-security
note: PR #22 duplicates a commit already present in the (now closed,
superseded) PR #18 — self-disclosed in the PR body, no risk.

This closes issue #7's full evidence chain: design proposal → versioned
contract → independent code review (multiple rounds, across PR #9,
#13, #14, #15) → live-execution security review that found and blocked
on three real defects → fixes independently re-verified → real pilot
executed → independent evidence review. Every gate in that chain found
at least one real thing to push back on, and every push-back is
recorded above with the PR that fixed it — this is deliberately not a
chain where every step passed cleanly on the first try, because that
would be a less convincing chain, not a more convincing one.

## What `0.7` does and does not establish

This is the part most likely to be quoted out of context, so it is
stated plainly, in both directions.

**What it does not establish:** `0.7` (7/10) on a 10-example synthetic
two-digit-addition task does not establish general arithmetic
capability, does not establish that this model or this training
recipe is good at anything beyond this exact narrow task, is not an
accept or promotion decision, and does not authorise any further,
larger, or different pilot. `EvaluationOutput` has no field to hold an
accept/promote/production-readiness verdict — this is a structural
fact, independently confirmed by Maya's review above, not a policy
this document is merely asserting. Three of ten answers were wrong,
each by a genuine arithmetic miss (off by 10, 10, and 90 respectively)
rather than a formatting or harness failure — this is exploratory
evidence of a small, expected, partial-generalisation result from a
50-step fine-tune on 40 examples, not a benchmark result to compare
against anything else.

**What it does establish:** that the full pipeline this issue asked
for — hash-pinned model and dataset, OS-measured resource enforcement,
real trainer adapter, real evaluator adapter, bidirectional
contamination checking, live-execution security review, and
independent post-hoc evidence verification — runs end to end, on a
real model, against a real held-out set, producing evidence that
survives someone else trying to break it. That is what this evidence
chain is actually offering as a result: proof that the *governance and
pipeline* work, with a small real number as the concrete artifact that
exercised it — not proof that the *model* is capable of anything in
particular.

## Reproducing this

`examples/pilot-adr0006/run_pilot.py` reproduces the run byte-for-byte
given the same pinned model snapshot and dataset files (fixed seed,
deterministic dataset, `do_sample=False` greedy evaluator decoding).
Exact loss/entropy floating-point values may vary slightly by
PyTorch/MPS backend version; that variance is not load-bearing for any
claim in this document. Full committed evidence:
`examples/pilot-adr0006/evidence/training-evidence.sanitised.json`,
`examples/pilot-adr0006/evidence/evaluation-evidence.sanitised.json`,
`examples/pilot-adr0006/evidence/pilot-result.sanitised.json`. Raw
scratch output (model checkpoint copies, HF cache) is not committed —
only sanitised evidence containing no secrets and no raw model
weights is.
