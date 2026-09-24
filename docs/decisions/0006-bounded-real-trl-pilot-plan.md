# ADR-0006: Bounded real TRL pilot plan (drafted, then executed per gating)

- Status: accepted — drafted per issue #7 step 4/5 sequencing; execution
  was gated on the pilot-specific live-execution security review and
  owner authorisation (see "Gating" below), both of which cleared, and
  the single bounded pilot run this ADR planned has since executed
  (PR #22; see "Pilot execution result" below for the outcome and its
  explicit scope limits).
- Date: 2026-09-17

## Context

`docs/TRAINER_ADAPTER_CONTRACT.md`'s "Real bounded pilot plan" section
(added by ADR-0002, restated by ADR-0005) has always described the
shape of the one bounded real-engine training run issue #7 asks for —
fixed small model, concurrency one, explicit CPU/GPU/time/storage
limits, synthetic/admitted data only — as prose inside that document,
not as its own reviewable, versioned decision record. ADR-0005 closed
step (a) of that plan (choose and implement the adapter: TRL,
`TRLTrainerAdapter`, `UpstreamRequirement` pinned to `trl==0.24.0` per
this ADR's sibling amendment). This ADR closes the remaining
formalisation gap for steps (b) and (c)'s *preconditions* — it writes
down the exact bounded pilot configuration as a reviewable decision
record, the same discipline every other real-engine decision in this
project has gone through (ADR-0002 for the contract, ADR-0003/0004 for
OS-level enforcement, ADR-0005 for the engine choice) — **without**
itself satisfying steps (b) or (c), and **without** authorising
execution.

This ADR is drafted, not executed, as an explicit, separately
requested work package: "draft (do not execute) the bounded real pilot
ADR ... Stop there. Do NOT execute a real pilot run yet." Its own
`Status` field reflects that: `proposed`, not `accepted`, because
acceptance of a *pilot-execution* decision is not this ADR's call to
make alone — it requires the pilot-specific security review named in
"Gating" below, distinct from the earlier PR #15 code review of the adapter
itself.

## Decision (the bounded configuration this ADR proposes, if approved)

### Model

One fixed, small, pinned model revision — proposed:
`HuggingFaceTB/SmolLM2-135M` (Apache-2.0, 135M parameters), the same
model family already used and soak-tested on the same host class (Mac
M4 Pro, no CUDA GPU) by the private cumulative-training-engine
research referenced in issue #7's 2026-09-14 comment (800 kill/resume
cycles, 872+ rounds, ~2h53m). Reusing an already-evidenced-compatible
model on this exact host class is a deliberate risk reduction, not a
requirement of this ADR's design — any other small, licence-clear,
CPU/MPS-compatible model would satisfy this section equally, provided
it is hash-pinned before the pilot runs. Whichever model is actually
used, `TrainingInputs.model_hash` must be the SHA-256 of its actual
downloaded weights (verified via `TRLTrainerAdapter.prepare()`'s
`_hash_path_identity()` check, already implemented and tested), not a
name or revision string trusted on its own.

### Dataset

Synthetic or explicitly admitted data only, per `docs/DATA_GOVERNANCE.md`.
No production or customer data under any circumstance — this is a hard
requirement already enforced structurally by `TrainingInputs.validate()`
requiring `contamination_checked=True` and a non-empty `dataset_licence`
before any adapter is reached at all. The pilot's specific dataset
must be constructed new for this pilot (a handful of synthetic
prompt/completion pairs, e.g. a fixed-format arithmetic or
classification task mirroring the cumulative-engine research's own
tick-classification curriculum) — reusing any held-out set from
`evaluator_contract.py`'s test fixtures or from any other package's
registered held-out ids is explicitly disallowed, and would itself be
caught by `HeldOutExclusionRegistry.check_held_out_not_trained` if
attempted after this pilot's dataset is registered as train data
anywhere.

### Concurrency

Exactly one concurrent run. No parallel pilot runs, no second pilot
started before the first reaches a terminal `TrainingStatus`
(`ACCEPTED`/`REJECTED`/`INVALID`/`INTERRUPTED`) and its evidence is
recorded. This is caller-level discipline the pilot's own run harness
must enforce (unchanged from ADR-0005's own disclosure that neither
`TRLTrainerAdapter` nor `run_trainer_contract` enforce concurrency
themselves).

### Resource limits (`ResourceBudget`, explicit values)

| Field | Proposed value | Rationale |
|---|---|---|
| `max_wall_seconds` | `1800` (30 min) | Generous enough for a `max_steps`-bounded CPU/MPS SFT run on a 135M model to complete normally; short enough that a stuck run is caught quickly. |
| `max_cpu_seconds` | `3600` (60 min, allows some multi-core parallelism above wall-clock) | OS-measured (v1.1 enforcement), not adapter self-reported. |
| `max_memory_mb` | `8192` (8 GB) | Bounded well under this host class's typical available memory; a genuine leak or a much larger model swapped in by mistake is caught here, not by exhausting the host. |
| `max_gpu_count` | `0` | This host has no CUDA GPU (see `Local-LLM-Feasibility/docs/HARDWARE_DECISION_GATE.md`); MPS is not counted by `torch.cuda.device_count()`, so this is deliberately `0`, not an MPS-specific count this project does not currently measure. |
| `max_storage_mb` | `2048` (2 GB) | Checkpoints + final artifact + evidence for a 135M model; generous headroom over the model's own on-disk size. |
| `network_policy` | `"offline"` | Already the only policy `TRLTrainerAdapter.prepare()` accepts (`RejectedInputError` otherwise); restated here for completeness of the pilot's own declared budget. |
| `training_params["max_steps"]` | `50` | Adapter-enforced (in addition to the wall-clock budget, per ADR-0005's "Adapter-enforced step bound"); a genuinely small, bounded pilot, not an open-ended training run. |

These are proposed starting values for review, not final until the
pilot-specific security review (see "Gating") signs off on them
specifically — that review is expected to scrutinise these numbers as
part of its scope, not merely accept this table's arithmetic.

### Filesystem

`ResourceBudget.filesystem_root` set to a dedicated, empty, per-pilot
scratch directory (not the repository checkout, not any other
project's data directory). `TRLTrainerAdapter.work_dir` points at the
same root. Cleaned up (`cleanup()`, already implemented and idempotent)
after the pilot's evidence is copied out, not left as a persistent
mount.

### Independent evaluation wiring (issue #7 step 4)

The pilot's trained artifact must be scored through
`evaluator_contract.run_evaluator_contract` against a `HeldOutSet`
constructed and registered (`HeldOutExclusionRegistry.register_package_held_out`)
*before* the pilot's training data is finalised, using a package id
distinct from the pilot's own training-data package id, so the
bidirectional contamination check in `held_out_registry.py` has
something real to check against rather than an empty registry that
trivially passes. The evaluator adapter used for the pilot's actual
scoring is **not** decided by this ADR — `FakeEvaluatorAdapter` proves
the contract but is not a real capability-scoring adapter; a real
scoring adapter (even a minimal exact-match or perplexity check against
the pilot's tiny synthetic task) is separate, not-yet-authorised
follow-up work this ADR does not perform. This section records the
*wiring requirement* — every pilot run must go through
`run_evaluator_contract` against a real `HeldOutSet` inaccessible to
the trainer, using `HeldOutExclusionRegistry` to make that inaccessibility
enforced rather than declared — not a claim that a real evaluator
adapter already exists.

### Kill criteria (unchanged from issue #7's own list, restated for this pilot)

Stop on: missing provenance, unbounded resources, trainer access to
held-out data, unverifiable artifacts, unsafe network/filesystem
access, or any path that treats completion as improvement. Concretely
for this pilot: a `TrainingOutput.status` other than `ACCEPTED`, an
evidence-hash mismatch (`TamperDetectedError`), a
`ContaminationDetectedError` from the evaluator contract, or a resource
budget violation each independently halt the pilot and are recorded as
evidence, not silently retried or reinterpreted as success.

## Gating (this ADR authorises none of it)

This ADR does not authorise the pilot to run. Before execution:

1. **Independent evaluation is wired and tested** — this ADR's own
   scope (drafting the config) plus the separately delivered
   `evaluator_contract.py`/`held_out_registry.py` components satisfy
   the *contract-and-tests* half of issue #7 step 4. A real (non-fake)
   evaluator adapter and an actual `HeldOutSet` for this specific
   pilot's task remain to be built when the pilot is scheduled.
2. **The pilot-specific security review** (issue #7 step 5) —
   sandbox, egress, secrets, artifact hashes, supply-chain identity,
   and safe-stop behaviour of the *actual live TRL process*, not the
   adapter code review already completed in PR #15. This is a
   distinct review scope from PR #15's: that review explicitly stated
   "I have not evaluated live execution behaviour because none
   occurred here" — this pilot is exactly the live execution that
   review did not cover.
3. **Owner authorisation to schedule the pilot itself**, separate from
   the standing continual-training-framework authorisation that
   covers drafting this ADR and building the evaluation harness.

No training engine is invoked, no pilot is scheduled, and no resource
is reserved as a result of this ADR. `adapter.train()` remains
untested by any run in this repository, exactly as ADR-0005 left it.

## Consequences

- CodeVolt MDF now has a reviewable, versioned decision record for the
  bounded pilot's exact configuration, closing the "formalize what's
  currently just pilot-plan prose" gap — the next reviewer (for
  the pilot-specific review; the owner, for scheduling) has concrete
  numbers and a concrete wiring requirement to review against, not
  prose to interpret.
- This ADR's proposed values (model, dataset shape, resource limits)
  are a starting point for that review, not immutable — the
  pilot-specific security review is expected to challenge or tighten
  them as part of its own scope.
- Nothing in this ADR changes `CONTRACT_VERSION`, `TRL_MIN_VERSION`/
  `TRL_MAX_VERSION`, or any code path. It is a planning document only.

## Locked pilot parameters (2026-09-17, dataset built + dry-validated, NOT executed)

Per owner instruction to finalise this ADR's proposed configuration
into a concrete, ready-to-run spec for the pilot-specific
live-execution security review (issue #7 step 5) — **this section
records that concrete spec. It does not authorise execution.** No
`adapter.train()` call has occurred; only `TrainingInputs.validate()`,
`ResourceBudget` construction, and `TRLTrainerAdapter.prepare()` (static
validation only, performs no training work per its own docstring) were
exercised to confirm the locked values are accepted by the existing,
already-reviewed contract code before asking for review.

### Model (confirmed, matches PR #17's provenance exactly)

- Repo: `HuggingFaceTB/SmolLM2-135M`, Apache-2.0.
- Revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- Local snapshot path (already present, `hf cache list` confirms
  `['main']` ref, offline-resolvable via `local_files_only=True`):
  `~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- `model.safetensors`: 269,060,552 bytes, sha256
  `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1` —
  independently re-verified with `shasum -a 256` against the resolved
  blob, not merely re-quoted from PR #17.
- Directory-identity hash (`trl_adapter._hash_path_identity`,
  independently recomputed against the live local snapshot in this
  session, not copied from PR #17's prose):
  `7f948d54bed4a691a8106c221ecd47b6f03ef34a8059be4b825ebb2d27c793d6` —
  matches PR #17's independently-implemented `hf_local_evaluator_adapter._hash_model_dir`
  output exactly, cross-checked both ways in this session.

### Dataset (newly built for this pilot, per this ADR's own "Dataset" section)

- Generator: `examples/pilot-adr0006/generate_dataset.py` — deterministic,
  fixed seed `20260917`, stdlib-only. Task: 2-digit addition,
  fixed-format SFT text (`"Add: {a} + {b} = {a+b}"`), distinct from
  the cumulative-engine research's own tick-classification curriculum
  and from every existing `evaluator_contract.py` test fixture (new
  content, new package ids, per this ADR's explicit prohibition on
  reusing either).
- Train split: `examples/pilot-adr0006/dataset/train.jsonl`, 40 records,
  `dataset_licence="synthetic-CodeVolt-owned"`, `contamination_checked=True`.
  Content hash (`trl_adapter._hash_path_identity`, single-file sha256):
  `3d290db4cf709add0fc120af59eeadb11ccd73f9418f7bbf0f3853e0f107ad2d`.
- Held-out split: `examples/pilot-adr0006/held_out.json`, 10 examples,
  package id `adr0006-pilot-heldout-v1`, `HeldOutSet.dataset_hash`
  (`evaluator_contract._hash_examples`):
  `c5ade195c2b4b8697b175b8c94e10add6206ad4d72137ea83d24c65dbf489e65`.
  `HeldOutSet.validate()` passes (no tamper, no duplicate ids).
- Disjointness: train and held-out numeric `(a, b)` pairs are generated
  from non-overlapping pools (`generate_dataset.py` excludes every train
  pair before sampling held-out pairs) — held-out is unseen *content*,
  not merely an unseen id over already-trained problems. Verified
  programmatically at generation time (`overlap == 0`).
- Registry: both packages registered in a fresh
  `HeldOutExclusionRegistry` (`examples/pilot-adr0006/held_out_exclusion_registry.json`),
  train ids as `train-0001..train-0040`, held-out ids as
  `heldout-0001..heldout-0010`.
  `registry.check_held_out_not_trained(held_out_set.example_ids)`
  returns an empty set — the exact precondition
  `evaluator_contract.run_evaluator_contract` re-checks at evaluation
  time (issue #11's bidirectional check), confirmed to pass here before
  the pilot is even proposed for review, not left to be discovered
  during the pilot itself.

### Concurrency

Exactly one concurrent run, unchanged from this ADR's original
"Concurrency" section — caller-level discipline (a single foreground
invocation, no background/parallel scheduling) for this specific pilot
run.

### Resource budget (unchanged values, now attached to a real, constructed `ResourceBudget` that passed its own `__post_init__` validation)

| Field | Locked value |
|---|---|
| `max_wall_seconds` | `1800` (30 min) |
| `max_cpu_seconds` | `3600` (60 min) |
| `max_memory_mb` | `8192` (8 GB) |
| `max_gpu_count` | `0` |
| `max_storage_mb` | `2048` (2 GB) |
| `network_policy` | `"offline"` |
| `filesystem_root` | a dedicated empty per-pilot scratch directory, not this checkout |
| `training_params["max_steps"]` | `50` |
| `training_params["model_path"]` | the pinned local snapshot path above |
| `training_params["dataset_path"]` | `examples/pilot-adr0006/dataset/train.jsonl` |

`TrainingInputs.create(...)` with the above fields plus
`model_revision="HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2"`,
`seed=20260917`, and a fresh `run_id`, followed by `.validate()`, plus a
standalone `ResourceBudget(...)` construction with the table above, plus
`TRLTrainerAdapter(work_dir=...).prepare(inputs, budget)` (installed
`trl==0.24.0`, matches the adapter's pinned `[0.24.0, 0.24.0]` bound
exactly) — all three passed without modification to any contract code,
in this session, against the real local host. **`train()` was not
called.**

### Expected runtime (estimate, not a measured training run)

Real-inference timing was measured in this session using the
already-merged, already-reviewed `HFLocalCausalLMEvaluatorAdapter`
(`score_example`, CPU/float32, `do_sample=False`) against the pinned
checkpoint as an honest proxy for this host's per-forward-pass cost,
since no training run has been authorised to measure directly:
model load ≈ 3.0 s (one-time); a single 8-new-token greedy-decode
forward pass ≈ 0.18 s once loaded. A training step additionally needs a
backward pass and optimizer step, typically several times the cost of
an equivalent forward pass alone on CPU for a dense 135M model with
these tiny sequence lengths and `per_device_train_batch_size=1`. Taken
together, 50 steps is expected to complete in low-single-digit minutes
on this host, well inside the 1800 s wall-clock budget above — this is
an order-of-magnitude estimate from a real (non-training) measurement
on the same host and model, not a measured training run; the
`max_wall_seconds`/`max_steps` budgets are the actual enforced backstop
regardless of how this estimate turns out.

### What remains unexecuted

No `adapter.train()` call, no TRL `SFTTrainer` instantiation, no
gradient computation, and no write under `filesystem_root` beyond this
ADR-drafting session's own dataset-generation output has occurred as a
result of this section. This section only confirms the pilot's inputs
are well-formed and accepted by the existing, previously-reviewed
validation code — it is not, and does not claim to be, pilot execution
or pilot evidence. Gating (see above) is unchanged: the pilot-specific
live-execution review and separate owner authorisation to schedule
remain required before `train()` is ever called.

## Pilot execution result (2026-09-17, REAL, EXECUTED)

Gating cleared: the pilot-specific live-execution security review
(issue #7 step 5, PR #18 review) returned CLEAR TO EXECUTE on findings
1-3, all three fixed and merged (PR #21,
`c3cbc5513fff8fc78526403ea75513c43229c1b8`), residual note LOW/
non-blocking (pre-existing macOS `ps`-timing behaviour, no action
needed). Owner authorised execution of this specific bounded run.

Before execution, the trainer-adapter/evaluator-adapter wiring gap this
ADR's "Independent evaluation wiring" section identified as open (a
real, non-fake evaluator adapter scoring the pilot's actual trained
artifact through `run_evaluator_contract` against the registered
`HeldOutSet`, not merely a contract-and-tests demonstration) was closed
by `examples/pilot-adr0006/run_pilot.py`: it runs
`run_trainer_contract` (real `TRLTrainerAdapter.train()`) and, only if
`TrainingOutput.status == ACCEPTED`, immediately scores the resulting
artifact through `run_evaluator_contract` using the already-merged
`HFLocalCausalLMEvaluatorAdapter` (PR #17) against the exact registered
held-out set below -- both calls in the same script, same run, so the
evaluator scores the artifact this specific trainer run actually
produced, not a stand-in. A cheap 2-step synthetic dry-run against a
throwaway dataset/held-out pair (not this pilot's registered packages)
exercised the identical trainer-then-evaluator code path first, to
prove the wiring end-to-end before spending the full 50-step budget.

### Run identity

- `run_id`: `adr0006-pilot-20260917`.
- `model_hash` (independently recomputed via
  `trl_adapter._hash_path_identity` against the live local snapshot at
  execution time): `7f948d54bed4a691a8106c221ecd47b6f03ef34a8059be4b825ebb2d27c793d6`
  -- matches this ADR's earlier "Locked pilot parameters" section exactly.
- `dataset_hash` (same helper, against `examples/pilot-adr0006/dataset/train.jsonl`):
  `3d290db4cf709add0fc120af59eeadb11ccd73f9418f7bbf0f3853e0f107ad2d` --
  matches this ADR's earlier "Locked pilot parameters" section exactly.
- `max_steps=50`, `seed=20260917`, `network_policy="offline"` -- all
  ADR-0006 locked values, unmodified.

### TrainingOutput (real `TRLTrainerAdapter.train()` call)

| Field | Value |
|---|---|
| `status` | `ACCEPTED` |
| `reason` | `trl SFT run completed 50 max_steps` |
| `artifact_id` | `artifact-adr0006-pilot-20260917` |
| `evidence_hash` (independently re-verified with `shasum -a 256` against the evidence file) | `98881a6ce342b2e8c4ceeca333afb37dbcec4e217ec08fd392f992b56c82e636` |
| `resource_usage.wall_seconds` (OS-measured) | `15.708815` (budget: `1800`) |
| `resource_usage.cpu_seconds` (OS-measured) | `14.084922` (budget: `3600`) |
| `resource_usage.memory_mb_peak` (OS-measured) | `1621.5` (budget: `8192`) |
| `resource_usage.storage_mb_used` | `517.76` (budget: `2048`) |
| `resource_usage.gpu_count_used` | `0` (budget: `0`) |

All five resource dimensions well inside budget; no kill criterion
triggered. Sanitised evidence:
`examples/pilot-adr0006/evidence/training-evidence.sanitised.json`
(training log history only -- no secrets, no raw model weights).

### EvaluationOutput (real `HFLocalCausalLMEvaluatorAdapter`, real held-out set)

Scored via `run_evaluator_contract` against `HeldOutSet` package
`adr0006-pilot-heldout-v1`, `dataset_hash`
`c5ade195c2b4b8697b175b8c94e10add6206ad4d72137ea83d24c65dbf489e65`
(re-verified by `HeldOutSet.validate()` at run time -- no tamper), with
`HeldOutExclusionRegistry.check_held_out_not_trained` returning zero
contaminated ids immediately before scoring.

| Field | Value |
|---|---|
| `status` | `SCORED` |
| `aggregate_score` | `0.7` (7/10 exact-match correct) |
| `evidence_hash` (independently re-verified with `shasum -a 256`) | `0d3f1668e4d3b721db16425e71a10cb28d20ae3108cc5f29bc850d5040ecbfa0` |

Per-example results (2-digit addition, exact-match after
normalisation):

| example_id | input | expected | raw_output | correct |
|---|---|---|---|---|
| heldout-0001 | `Add: 27 + 69 =` | `96` | ` 96` | true |
| heldout-0002 | `Add: 69 + 28 =` | `97` | ` 97` | true |
| heldout-0003 | `Add: 58 + 46 =` | `104` | ` 94` | **false** |
| heldout-0004 | `Add: 28 + 56 =` | `84` | ` 84` | true |
| heldout-0005 | `Add: 51 + 13 =` | `64` | ` 64` | true |
| heldout-0006 | `Add: 11 + 49 =` | `60` | ` 50` | **false** |
| heldout-0007 | `Add: 41 + 45 =` | `86` | ` 86` | true |
| heldout-0008 | `Add: 42 + 49 =` | `91` | ` 181` | **false** |
| heldout-0009 | `Add: 20 + 23 =` | `43` | ` 43` | true |
| heldout-0010 | `Add: 64 + 17 =` | `81` | ` 81` | true |

Reported honestly, not selectively: 3/10 wrong (heldout-0003, -0006,
-0008), all three off by a small but non-trivial arithmetic amount (10,
10, 90 respectively) rather than a formatting/parsing failure -- this
looks like exactly what a 50-step SFT run on a 135M base model against
40 training examples of two-digit addition is expected to produce
(partial arithmetic generalisation, not memorisation, not a broken
harness): training loss dropped from `1.71` to `0.69` over the run
(`training-evidence.sanitised.json` log history) while still leaving
30% of held-out arithmetic wrong. Sanitised evidence:
`examples/pilot-adr0006/evidence/evaluation-evidence.sanitised.json`,
full combined result:
`examples/pilot-adr0006/evidence/pilot-result.sanitised.json`.

### What this result does and does not establish

This is exploratory pilot evidence, not a capability claim and not a
promotion decision. `0.7` on a 10-example synthetic arithmetic
held-out set says only: the trainer-adapter -> evaluator-adapter wiring
this ADR's gating required now runs end-to-end against a real held-out
set with real inference, and the one bounded 50-step run this ADR
locked produced a trustworthy (budget-respecting, hash-verified,
contamination-checked) artifact that a real evaluator scored honestly,
including its failures. It does not establish general arithmetic
capability, does not accept or promote the trained artifact, and does
not authorise any further pilot, larger run, or production use --
those remain separate, later governed actions (`docs/ARCHITECTURE.md`
core contract #6) this ADR and this pilot do not perform.

### Reproducibility

`examples/pilot-adr0006/run_pilot.py` reproduces this run byte-for-byte
given the same pinned model snapshot and dataset files (fixed seed,
deterministic dataset, `do_sample=False` greedy evaluator decoding) --
re-running it is expected to reproduce the same `TrainingOutput`/
`EvaluationOutput` structure, though exact loss/entropy floating-point
values may vary slightly by PyTorch/MPS backend version and are not
themselves load-bearing for this ADR's gating.
