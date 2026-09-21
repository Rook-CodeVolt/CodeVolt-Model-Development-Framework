# ADR-0013 bounded cycle execution plan

Status: proposed and dry-validated; live baseline/training/post-evaluation is blocked.
This plan is subordinate to
[`docs/decisions/0013-bounded-pretrained-metatrainer-cycle.md`](../../docs/decisions/0013-bounded-pretrained-metatrainer-cycle.md).
If the two documents differ, the ADR controls and execution stops.

## One-cycle sequence

1. Check out the exact commit named in the review gate.
2. Build the exact pinned environment.
3. Run `validate_dataset.py`.
4. Run `run_bounded_cycle.py` without `--execute`; retain its JSON output.
5. Confirm the immutable model snapshot is already local. No live-cycle network
   access is permitted.
6. Obtain all three role-separated approval documents, each bound to the exact
   implementation SHA, training-dataset hash, admitted licence/use value and run id.
7. Have each role sign its canonical JSON document with an admitted SSH key, then
   create the non-secret review-gate manifest below.
8. Run the exact `--execute` command once, in the foreground, with no parallel run.
9. Hash and preserve the evidence bundle before cleanup.
10. Conduct the independent rubric and retention review. The runner does not decide
    promotion.

## Review-gate JSON

The gate is evidence, not a credential. It must not contain tokens, cookies, private
keys, personal data, model outputs, or private conversation text.

The manifest schema is `schema_version`, `implementation_sha`, `dataset_hash`,
`dataset_licence`, and an `approvals` object containing exactly `security`, `owner`,
and `dataset`. Each role entry contains exactly `document`, `signature`, and
`document_sha256`. Each signed document contains exactly: `schema_version`, `role`,
`approver_id`, `decision`, `implementation_sha`, `dataset_hash`, `dataset_licence`,
`run_id`, and a non-empty `scope` list. The security scope must include
`evaluator-process-containment-v1` and the separately reviewed complete-cycle host
containment profile.

The versioned `approval_allowed_signers` file is the only trust root. It uses
OpenSSH allowed-signers entries and requires three role-specific principals:
`maya-security`, `maya-dataset-rights`, and `rook-owner`. Each holder generates and
retains their own private key; only the public key is proposed in a reviewed change.
Private keys, credentials, and signatures are never committed to the trust root or
embedded in the gate. No public keys are admitted in this revision, so execution
continues to fail closed until those holder-supplied keys are independently reviewed
and merged. Arbitrary strings, local booleans, unsigned JSON, placeholders, missing
roles and signatures from untrusted keys fail.

## Exact environment

```bash
python3 -m venv .venv-adr0013
.venv-adr0013/bin/python -m pip install --upgrade pip==26.0.1
.venv-adr0013/bin/python -m pip install \
  torch==2.8.0 transformers==4.56.1 datasets==3.0.0 \
  accelerate==1.4.0 trl==0.24.0
.venv-adr0013/bin/python -m pip install -e '.[dev]'
```

The live runner refuses version drift. The reviewed dry probe observed Apple MPS,
`bf16=True`, `fp16=False`, `use_cpu=False`, and gradient accumulation `1`. Any changed
runtime result stops the cycle.

## Complete-cycle host containment

Live `--execute` automatically re-executes the whole runner under the macOS Seatbelt
policy supplied to `/usr/bin/sandbox-exec`; it does not rely on the Python socket or
`open()` guards alone. The profile denies `network*` and denies `file-write*` except
beneath the exact live scratch root and `/dev/null`. The policy is inherited by the
runner's subprocesses and native extensions. Before loading the review gate, the
contained process proves that a direct native write, `/usr/bin/touch`, a nested
allow-all `sandbox-exec`, and a native TCP socket cannot escape the policy. Any
unexpected success or non-permission failure stops execution.

The policy intentionally permits reads and process execution. Python, site-packages,
the immutable model snapshot, repository inputs, macOS frameworks, and MPS resources
are outside the scratch root and must remain readable. This is therefore an
OS-enforced outbound-network and filesystem-write boundary, not a read jail, process
allow-list, VM, container, or defence against kernel compromise. Dependency and model
fetches happen before this boundary; the complete live cycle is fully offline.

The security approval document must include both exact scope tokens:
`evaluator-process-containment-v1` and
`complete-cycle-host-containment-v1`.

## Reviewed host paths

The only approved host root is `./local-evidence/adr0013`. Its
`evidence` and `scratch` directories are owned by the invoking uid, mode `0700`, have
no symlink components, and must have at least 2 GiB free. The runner checks those
properties immediately before execution and rejects any other live paths.

- validation scratch: `./local-evidence/adr0013/scratch/adr0013-check`;
- review gate: `./local-evidence/adr0013/evidence/adr0013-review-gate.json`;
- live output: `./local-evidence/adr0013/scratch/adr0013-metatrainer-sft-20260920`.

The operator creates no symlinks below this root. After independent evidence review,
retain the gate and manifest in `evidence`; remove or quarantine the live-output
subdirectory according to the disposition rules below. The pinned model cache is
read-only to the sandbox and is never part of this cleanup.

## Validation-only command

```bash
.venv-adr0013/bin/python examples/pilot-metatrainer-v2/validate_dataset.py
.venv-adr0013/bin/python examples/pilot-metatrainer-v2/run_bounded_cycle.py \
  --scratch-root./local-evidence/adr0013/scratch/adr0013-check
```

Expected properties:

- status `PASS`;
- `training_called=false`;
- model content hash
  `43752b3f39894c0122d9a94f3b4e64ad2d76e43d25c2a08aa360ad17a1a0145c`;
- 40 train and 20 held-out records;
- evaluator-contract held-out hash
  `3acdaae0a51f29ee42989d8a2161ae8f3205c1662b887e8a3de7e5c89cfa17d7`;
- renderer identity
  `chat_template:sha256=551557e5be16b6241465fab68eb9958e8e953a47ff00bb24e40ab38e76ee8aed;add_generation_prompt=true;tokenize=false;retokenize_return_tensors=pt`;
- all unresolved execution blockers printed.

This command is safe to repeat because it does not call `run_trainer_contract` or the
evaluator.

## Exact live command after all gates

```bash
.venv-adr0013/bin/python examples/pilot-metatrainer-v2/run_bounded_cycle.py \
  --execute \
  --review-gate./local-evidence/adr0013/evidence/adr0013-review-gate.json \
  --scratch-root./local-evidence/adr0013/scratch/adr0013-metatrainer-sft-20260920
```

Run once only. Do not append flags, change paths after review, run in the background,
resume automatically, or retry under the same authorization.

## What the runner does

- Verifies all locked corpus/support hashes, model content hash, dependency versions,
  40/20 counts, family disjointness, held-out registry state, renderer identity, and
  runtime precision/device.
- Evaluates the immutable base on all locked suites before any training call: 20
  meta-trainer items, 10 arithmetic capability-retention items, and 15 safety probes.
- Stops if baseline evidence is not `SCORED`.
- Runs full SFT for `max_steps=120`, LR `1e-5`, batch 1, accumulation 1, seed
  `20260920`, save every 40 steps, at most two periodic checkpoints.
- Evaluates the accepted candidate on all three suites through the identical evaluator
  path and fails closed on any missing/invalid suite evidence.
- Runs every evaluator phase in a separate measured OS process with the same offline,
  filesystem-write, wall, CPU, memory and storage boundary used for training.
- Writes a combined result with `promotion_decision: null`.

The runner does not score the independent semantic rubric, run promotion logic,
publish, deploy, or schedule another cycle.

## Evidence inventory

Under the reviewed scratch root, retain:

- validation output captured by the operator;
- three baseline evaluator evidence files and raw outputs under
  `baseline_evaluation/<suite>/evaluation-<artifact_id>-<package_id>.json`;
- trainer evidence and its declared evidence hash;
- final candidate directory and content-identity hash;
- three candidate evaluator evidence files and raw outputs under the equivalent
  `candidate_evaluation/<suite>/` paths;
- `cycle_result.json`;
- OS resource-usage result;
- review-gate JSON;
- independent rubric sheet and reviewer identity/reference;
- retention comparison evidence;
- a manifest containing SHA-256, relative path, byte size, and disposition for every
  retained file.

Raw model weights and optimizer/checkpoint state stay outside git. Sanitised summaries
may be committed only after independent review confirms they contain no secrets,
private data, unsafe raw output, or misleading promotion claim.

## Independent rubric sheet

For each of the 20 ids, record the following for both baseline and candidate:

```json
{
  "example_id": "mtr-v2-heldout-0001",
  "semantic_family": "peft_lora_qlora_decisions",
  "artifact_blind_label": "A",
  "technical_conclusion": 0,
  "reasoning_and_qualification": 0,
  "evidence_discipline": 0,
  "uncertainty_refusal_boundary": 0,
  "critical_error": false,
  "rationale": ""
}
```

Each numeric field is `0` or `1`. The reviewer must score from the prompt, expected
answer, source locators, and raw output rather than the deterministic exact-match
flag alone. Resolve reviewer disagreement before applying thresholds, and retain both
original judgments if a second scorer is used.

## Stop checklist

Stop and preserve evidence immediately when any answer is yes:

- Did a locked hash, pin, renderer, device, precision, budget, path, or gate change?
- Did the trainer receive held-out content or a retention-suite item?
- Was baseline evidence invalid or incomplete?
- Did training return anything other than `ACCEPTED`?
- Did loss/gradient evidence become non-finite or clearly divergent?
- Did any resource ceiling or offline/filesystem boundary fail?
- Did candidate evaluation become invalid, contaminated, or renderer-mismatched?
- Did a critical factual/evidence/safety error occur?
- Did safety or capability retention cross the ADR's rejection thresholds?
- Is anyone proposing a retry, extra epoch, changed LR, second seed, promotion,
  publication, deployment, or next cycle without a new decision?

## Cleanup and rollback

There is no deployed state to roll back. The immutable base remains the only usable
reference. After evidence review:

- accepted-for-review candidate: retain under controlled local storage until a
  separate promotion decision;
- held candidate: quarantine with hashes and unresolved questions;
- rejected candidate: retain sanitised evidence, then delete or quarantine raw
  weights according to the independent review decision;
- always remove temporary caches/work directories only after their contents are
  represented in the evidence manifest.

Never delete or mutate the pinned base snapshot as part of candidate cleanup.
