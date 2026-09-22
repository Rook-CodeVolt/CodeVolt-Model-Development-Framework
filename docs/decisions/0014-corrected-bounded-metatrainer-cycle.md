# ADR-0014: Corrected bounded pretrained meta-trainer SFT cycle (post ADR-0013 rejection)

- Status: proposed — design and dry validation only; training is blocked by the
  gates in this record
- Date: 2026-09-22
- Tracking: issue #85
- Supersedes: does not repeal ADR-0013. ADR-0013's run is final, evidence-preserved,
  and rejected. This is the new proposal ADR-0013's own text requires for any
  continuation.

## Context

ADR-0013 authorized exactly one bounded pretrained full-SFT cycle for the internal
meta-trainer reasoning assistant described there. That cycle ran end to end under
real gated authorization (commit `13b316e6bea25d141ecf1c89284e40e68a1b3ff9`, run_id
`adr0013-metatrainer-sft-20260920`): `training.status=accepted`, baseline and
candidate evaluation both actually ran, and the pipeline, security gates, and
evaluation all functioned correctly. The result was an honest negative:

| Suite | Baseline | Candidate | Result |
|---|---:|---:|---|
| `meta_trainer` (target task, 20 items) | 0.0% | 0.0% | no improvement |
| `capability_retention` (arithmetic, 10 items) | 30.0% | 0.0% | regressed |
| `safety` (15 items) | 46.7% | 33.3% | regressed |

Raw held-out candidate outputs show classic degeneration: repetitive token loops
such as `"46-46-46-46..."` and `"PEPFET, a variant of PEPFET..."`. This is the
textbook signature of a full-parameter SFT run overfitting a very small corpus.

This exact outcome matches ADR-0013's own predeclared "Stop and reject this
candidate" trigger (`0013-...md` line 292: "zero or less than 0.25/4.0 rubric
improvement with no other material gain"). ADR-0013's own "Continue only through a
new proposal" section (lines 272-283) is explicit that "a continuation requires a
new issue/ADR with a fresh bounded configuration" and that ADR-0013 "does not
authorize an automatic retry, LR sweep, extra epoch, or second seed." This ADR is
that new proposal, opened and reviewed independently, per issue #85 and the owner's
explicit authorization recorded there.

Nothing else about ADR-0013's premises has changed: the corpus, evaluator,
contamination machinery, resource budget shape, and security posture are all reused
byte-identical to the extent listed below. Only the training hyperparameters, run
identity, and reviewed host paths are new, so that ADR-0014 is independently
identifiable, requires its own fresh three-gate signing, and cannot be confused with
ADR-0013's already-consumed run.

## Decision

If and only if every gate in this ADR clears, run exactly one baseline -> bounded
full-SFT -> post-training evaluation cycle against the exact identities below,
through the new runner `examples/pilot-metatrainer-v2/run_bounded_cycle_adr0014.py`.
No parameter, dataset, dependency, renderer, budget, threshold, or command may drift
without a new review of the changed candidate.

### Unchanged from ADR-0013 (verified byte-identical)

- Base model identity, immutable revision, licence, and every content hash listed in
  ADR-0013's "Base model identity and licence" section
  (`HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f77366fa6b3b4b768ec3050bf629380bac`).
- Trainer (`src/codevolt_mdf/trl_adapter.py`, full SFT, `use_lora=False`) and
  evaluator (`src/codevolt_mdf/hf_local_evaluator_adapter.py`) adapter identities,
  contract versions, and exact dependency pins (`trl==0.24.0`,
  `transformers==4.56.1`, `datasets==3.0.0`, `accelerate==1.4.0`, `torch==2.8.0`).
- Dataset identity and all locked file hashes: `train.jsonl` (40 records),
  `held_out.json` (20 records), `semantic_family_manifest.json`,
  `held_out_exclusion_registry.json`, `REPRESENTATIVE_EXAMPLES.json`,
  `SOURCE_MAP.md`, `DATASET_CARD.md`, `VALIDATION_REPORT.json` — same SHA-256 values
  as ADR-0013's table, confirmed by a fresh dry-validation run of the ADR-0014
  runner (`status: PASS`, `training_called: false`, hashes match verbatim).
- Renderer identity:
  `chat_template:sha256=551557e5be16b6241465fab68eb9958e8e953a47ff00bb24e40ab38e76ee8aed;add_generation_prompt=true;tokenize=false;retokenize_return_tensors=pt`.
- Held-out contract hash `3acdaae0a51f29ee42989d8a2161ae8f3205c1662b887e8a3de7e5c89cfa17d7`.
- Capability-retention and safety suites
  (`examples/pilot-adr0006/held_out.json`, `examples/safety-probes-wpb/held_out.json`).
- Contamination checks, pass/continue/stop thresholds, and every mandatory stop
  condition, unchanged verbatim from ADR-0013's "Pass, continue, and stop
  thresholds" section.
- Resource budget ceilings (wall/CPU/memory/storage/network/filesystem), the
  complete-cycle host-containment Seatbelt mechanism, and the fail-closed
  four-probe containment self-test.
- Role-aligned rubric, capability-retention methodology, and the requirement for
  independent Maya security review plus independent dataset-rights admission plus
  owner confirmation before `--execute`.

### Changed: bounded training configuration

| Field | ADR-0013 (rejected) | ADR-0014 (this proposal) |
|---|---|---|
| `max_steps` | `120` (3 epochs at batch 1 over 40 examples) | `40` (1 epoch) |
| Learning rate | `1e-5` | `5e-6` |
| Save cadence | every `40` steps | every `20` steps |
| Seed | `20260920` | `20260922` |
| Run id | `adr0013-metatrainer-sft-20260920` | `adr0014-metatrainer-sft-20260922` |
| Approval namespace | `codevolt-adr0013` | `codevolt-adr0014` |
| Reviewed host root | `./local-evidence/adr0013` | `./local-evidence/adr0014` |

Per-device batch size (`1`), gradient accumulation (`1`), effective batch (`1`),
device (one local Apple MPS device, CUDA count `0`), precision (`bf16=True`,
`fp16=False`), saved-checkpoint limit (`2`), and concurrency (one foreground run) are
all unchanged from ADR-0013.

#### Rationale: `max_steps` 120 -> 40

The real ADR-0013 run trained 3 epochs (`max_steps=120`) over the locked 40-example
train split and produced zero-or-negative results on all three held-out suites, with
raw outputs showing degenerate repetition loops — the textbook signature of
overfitting a full-parameter SFT run on a very small corpus. One epoch (`max_steps=40`)
is the smallest unit that still completes a full pass over the locked 40-example
split; it is the minimum change that directly targets the observed failure mode
without touching the dataset, evaluator, or gates, and without inventing an
unreviewed intermediate step count. It is not a sweep: exactly one new value is
proposed, mirroring ADR-0013's own "no in-run sweep" discipline applied at the
proposal level instead.

#### Rationale: learning rate `1e-5` -> `5e-6`

ADR-0013 already selected `1e-5` as "the low end of the cited full-SFT starting
region," matching the MiniMind project default; it was not itself excessive relative
to common full-SFT practice, and this ADR does not treat LR as the sole cause of the
observed failure. However, at 1 epoch instead of 3, this cycle gets only one bounded
pass over 40 examples, so overfitting must be controlled primarily through the epoch
reduction above, not through further LR reduction alone, or the run risks producing
too small a training signal to observe any candidate/baseline difference at all. LR
is nonetheless lowered modestly (not left unchanged) because full-parameter SFT (not
LoRA/PEFT) on a 135M model updates every parameter every step; halving the LR and
cutting the epoch count are complementary defense-in-depth levers against re-hitting
the same degeneration pattern, not redundant with each other. Halving rather than a
steeper cut is chosen deliberately to keep this cycle a live, comparable second
measurement point in the same experimental family as ADR-0013 — informative evidence
either way — rather than a barely-perturbing no-op run that would not test the
overfitting hypothesis at all.

#### Rationale: fresh run id, approval namespace, and host paths

ADR-0013's own text requires a genuinely new proposal, not an in-place retry. A fresh
`RUN_ID`, `APPROVAL_NAMESPACE`, and reviewed host root ensure this cycle's evidence,
signed approvals, and live-execution scratch directory can never be confused with
ADR-0013's already-consumed, already-rejected run, and that ADR-0013's existing
signed approvals cannot be mistakenly treated as covering this cycle. The three
admitted signer identities (`rook-owner`, `maya-security`, `maya-dataset-rights`) are
unchanged — the same people hold the same roles — but
`examples/pilot-metatrainer-v2/approval_allowed_signers_adr0014` is a fresh copy of
the trust root scoped to this ADR, and a fresh signed approval document bound to
ADR-0014's exact implementation SHA/run id/namespace is required from each holder
before `--execute`; ADR-0013's existing signatures do not verify against ADR-0014's
documents.

### Explicitly not changed

Per the owner's authorization boundary (issue #85) and this task's scope: dataset
content, security/sandboxing/gate-verification code, and resource budgets are
unchanged from ADR-0013. This ADR does not touch
`src/codevolt_mdf/trl_adapter.py`, `src/codevolt_mdf/hf_local_evaluator_adapter.py`,
the contamination/registry machinery, or any resource-budget ceiling.

## Baseline and post-training evaluation

Identical to ADR-0013's "Baseline and post-training evaluation" section: the runner
evaluates the immutable base checkpoint before any training call, trains at most
once, then evaluates the candidate through the identical evaluator path, held-out
set, renderer, decoding settings, and rubric. Capability-retention and contamination
checks are unchanged.

## Pass, continue, and stop thresholds

Unchanged verbatim from ADR-0013's "Pass, continue, and stop thresholds" section,
including the "eligible for independent promotion review (not automatically
promoted)" thresholds, the "continue only through a new proposal" conditions, and
every "stop and reject" trigger. This ADR does not relax, tighten, or otherwise edit
any threshold; only the training configuration under test changes.

## Resource budget

Unchanged verbatim from ADR-0013: `1800 s` wall time and `3600 s` CPU time per
trainer/evaluator phase, `4096 MB` peak memory per phase, `0` CUDA GPUs, exactly `1`
dry-validated Apple MPS device, `1024 MB` storage, fully offline network during
baseline/train/post, and the same macOS Seatbelt complete-cycle host-containment
profile (`evaluator-process-containment-v1` and `complete-cycle-host-containment-v1`),
re-scoped only to the new reviewed root below.

### Reviewed host paths

The reviewed root is `./local-evidence/adr0014`, with the same
mode-`0700`/ownership/no-symlink/2-GiB-free preflight checks as ADR-0013. Exact
paths:

- validation scratch:
  `./local-evidence/adr0014/scratch/adr0014-check`;
- non-secret gate and retained manifest:
  `./local-evidence/adr0014/evidence/adr0014-review-gate.json`;
- live output:
  `./local-evidence/adr0014/scratch/adr0014-metatrainer-sft-20260922`.

## Safe stop, checkpoints, evidence, rollback, and cleanup

Unchanged in mechanism from ADR-0013: baseline-first with a hard stop before training
if baseline evidence is not `SCORED`; `run_trainer_contract`-based cancellation and
OS-level resource enforcement; evidence layout
(`baseline_evaluation/<suite>/...`, `trainer_work/<run_id>/evidence.json`,
`trainer_work/<run_id>/final/`, `candidate_evaluation/<suite>/...`,
`cycle_result.json`); rollback by deletion/quarantine of the candidate scratch
directory with no production change; cleanup only after evidence hashes are copied to
an approved evidence location and independent review records a retain/reject
disposition.

## Review and authority gates

`examples/pilot-metatrainer-v2/run_bounded_cycle_adr0014.py` defaults to
validation-only and prints `training_called=false` — confirmed by the dry-validation
run below. `--execute` additionally requires a review-gate JSON whose implementation
and dataset hashes are exact and whose three approval documents have detached SSH
signatures verified against `approval_allowed_signers_adr0014`. The allowed-signers
principals are exactly `maya-security`, `maya-dataset-rights`, and `rook-owner` — the
same three people ADR-0013 admitted (PRs #82/#83) — but no fresh signature exists yet
against ADR-0014's exact SHA/run_id/namespace, so execution remains fail-closed until:

1. This ADR, the runner, and the live-execution plan are merged or otherwise checked
   out at the exact SHA reviewed by Maya.
2. Maya has approved the sandbox/filesystem boundaries, offline behavior, dependency
   and model identity, evidence handling, MPS/resource limitations, safe-stop
   behavior, and the exact command — including the two changed hyperparameters and
   their rationale above. A general adapter review or ADR-0013's prior approval is
   not sufficient; ADR-0014 requires its own signed document.
3. An independent rights/privacy reviewer re-confirms the dataset admission under
   ADR-0014's namespace (the dataset itself, corpus, and locked hashes are unchanged
   from ADR-0013, but the signed document must bind to this ADR's exact identity).
4. The owner has confirmed use of the already-recorded authorization for this exact
   one run after reviewing the final gate references.

No secret may be placed in the gate file. This ADR grants no publication,
deployment, promotion, repeated scheduling, continuous unattended training, new
credentials, new egress, or new authority — identical to ADR-0013's grant of
authority (none beyond this one bounded cycle).

## Reproducible commands

From a clean checkout of the exact reviewed commit:

```bash
python3 -m venv .venv-adr0014
.venv-adr0014/bin/python -m pip install --upgrade pip==26.0.1
.venv-adr0014/bin/python -m pip install \
  torch==2.8.0 transformers==4.56.1 datasets==3.0.0 \
  accelerate==1.4.0 trl==0.24.0
.venv-adr0014/bin/python -m pip install -e '.[dev]'

.venv-adr0014/bin/python examples/pilot-metatrainer-v2/validate_dataset.py
.venv-adr0014/bin/python examples/pilot-metatrainer-v2/run_bounded_cycle_adr0014.py \
  --scratch-root./local-evidence/adr0014/scratch/adr0014-check
```

The second command block is validation-only and was executed for this proposal on a
freshly installed pinned stack. It reported:

```
status: PASS
training_called: false
train_count: 40
held_out_count: 20
model_hash: 43752b3f39894c0122d9a94f3b4e64ad2d76e43d25c2a08aa360ad17a1a0145c
renderer_id: chat_template:sha256=551557e5be16b6241465fab68eb9958e8e953a47ff00bb24e40ab38e76ee8aed;add_generation_prompt=true;tokenize=false;retokenize_return_tensors=pt
held_out_contract_hash: 3acdaae0a51f29ee42989d8a2161ae8f3205c1662b887e8a3de7e5c89cfa17d7
dataset_hashes: identical to ADR-0013's table (train.jsonl, held_out.json, and every
  support file hash match verbatim, confirming the corpus was reused unchanged)
runtime_config: {device: mps, bf16: true, fp16: false, use_cpu: false,
  gradient_accumulation_steps: 1}
versions: {torch: 2.8.0, transformers: 4.56.1, datasets: 3.0.0,
  accelerate: 1.4.0, trl: 0.24.0}
execution_blockers:
  - role-specific public signer keys are not yet admitted in the trust root
  - Maya exact-candidate live-execution approval
  - owner confirmation for this one run
```

All locked hashes, the renderer identity, and the held-out contract hash exactly
match ADR-0013's values, confirming the dataset and evaluator path are reused
byte-identical and only the training configuration changed.

Only after all gates clear, create the non-secret review-gate JSON at an approved
path and run exactly:

```bash
.venv-adr0014/bin/python examples/pilot-metatrainer-v2/run_bounded_cycle_adr0014.py \
  --execute \
  --review-gate./local-evidence/adr0014/evidence/adr0014-review-gate.json \
  --scratch-root./local-evidence/adr0014/scratch/adr0014-metatrainer-sft-20260922
```

The runner performs baseline, then at most one training run, then post-evaluation. It
does not score the human rubric or decide promotion. There is no retry command in
this ADR either.

## Continuous-training semantics

Unchanged from ADR-0013: a governed sequence (proposal -> baseline -> one bounded
training run -> identical post-evaluation -> independent review -> explicit
promote/reject/hold -> only then a new proposal), never an endless self-modifying
loop. This ADR is itself the "new proposal for another bounded cycle" ADR-0013's own
text anticipated, triggered by that cycle's honest negative result, not by any
unreviewed retry.

## Scale limitation

Unchanged from ADR-0013. This remains a 40-example slice that can test pipeline
integrity and whether a corrected hyperparameter configuration shifts the observed
overfitting behavior on three narrow held-out semantic families. It cannot establish
broad model-training expertise, robust factuality, general reasoning, production
reliability, or a universal SFT recipe. A pass here would be one incremental
experiment; a second negative result would not by itself prove the target task is
unreachable with this corpus/model combination, but would further disfavor naive
full-SFT-on-40-examples as the recipe and should prompt a design-level ADR (corpus
size, LoRA/PEFT path, or a different base model) rather than a third bare
hyperparameter tweak.

## Consequences

- ADR-0013's rejected candidate and its evidence remain the authoritative record;
  this ADR does not retroactively reinterpret that result.
- The pipeline's negative-result handling is validated a second time: honest
  rejection followed by a properly gated, independently reviewed new proposal,
  rather than a silent in-place parameter change.
- Full-SFT's regression risk is controlled through the same explicit safety and
  retention gates as ADR-0013, unweakened.
- Execution remains blocked until security, data admission, and one-run owner gates
  are real and checkable under the fresh ADR-0014 namespace.
- No model is trained, promoted, published, or deployed by accepting this proposal.

## Documentation impact and maintenance triggers

This decision adds `examples/pilot-metatrainer-v2/run_bounded_cycle_adr0014.py`,
`examples/pilot-metatrainer-v2/approval_allowed_signers_adr0014`, and
`tests/test_bounded_metatrainer_cycle_adr0014.py`, updates the changelog and roadmap,
and records the proposal in issue #85. No API or contract version changes; no edit to
`run_bounded_cycle.py`, `approval_allowed_signers`, or
`tests/test_bounded_metatrainer_cycle.py` (ADR-0013's files are left as the untouched
historical record of that rejected cycle).

Re-open or supersede this ADR before any run if any of the following changes:

- model repo/revision/files/hash/licence;
- dataset/support file, split, source, admission, or hash;
- trainer/evaluator/contract implementation;
- Python or dependency pin;
- renderer/template/decoding/rubric/threshold;
- device, precision, host class, budget, filesystem, network, or evidence path;
- review-gate schema or authority boundary;
- a PEFT-aware evaluator/merge path becomes available and changes the chosen method;
- either `max_steps` or `learning_rate` above needs further adjustment — that is a
  new ADR, not an edit to this one, per the same "no in-run sweep, no unrecorded
  retry" discipline ADR-0013 established.

After the one run, append only verified result/evidence references. Do not rewrite
this proposal as if predicted results had occurred.
