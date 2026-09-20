# ADR-0011: MiniMind bounded pilot plan (drafted, not executed)

- Status: proposed
- Date: 2026-09-18

## Context

`docs/decisions/0010-minimind-trainer-adapter-v1.md` added
`MiniMindTrainerAdapter` (`src/codevolt_mdf/minimind_adapter.py`), the
framework's second real `TrainerAdapterV1` implementation, wrapping
[MiniMind](https://github.com/jingyaogong/minimind)'s
`trainer/train_full_sft.py` as a subprocess rather than an in-process
library call (the structural difference from `TRLTrainerAdapter` that
motivated building it as a second engine at all). That ADR authorised
the adapter and its 37 contract-conformance tests only — it explicitly
did not authorise any live training run: `adapter.train()` remains
"fully implemented against the real MiniMind CLI but ... never been
exercised by any run in this repository" (`docs/REAL_ADAPTERS.md`).

Issue #46 asks for the same discipline ADR-0006 applied to the first
bounded real pilot (TRL/SmolLM2-135M) — a specific, reviewable,
versioned configuration for one bounded MiniMind pilot — **without**
authorising its execution, and **without** copying ADR-0006's resource
numbers verbatim, because MiniMind's execution shape is genuinely
different: a child `subprocess.Popen` running an external CLI script,
not an in-process `SFTTrainer` object this framework's own process
calls directly. This ADR is that configuration. Like ADR-0006, its own
`Status` is `proposed`, not `accepted`: accepting a pilot-*execution*
decision is not this ADR's call to make alone. It requires the same
gated sequence named in "Gating" below.

This ADR was drafted by directly inspecting
`src/codevolt_mdf/minimind_adapter.py`'s actual budget-enforcement and
subprocess-construction code, and the real pinned MiniMind commit's
own `trainer/train_full_sft.py` argument parser
(`cc312c1cc614bc371cd85dcbcbc1d3ba1590f364`,
`raw.githubusercontent.com/jingyaogong/minimind/<pin>/trainer/train_full_sft.py`),
not assumed from the adapter's docstrings alone. That inspection
surfaced three concrete CLI-argument mismatches between what the
adapter constructs and what the pinned script actually accepts — see
"Known adapter/script CLI incompatibilities" below. This is exactly
the kind of gap a drafting pass is supposed to surface before anyone
asks for a live-execution security review, and it is reported here
honestly rather than smoothed over.

## Decision (the bounded configuration this ADR proposes, if approved)

### Model

One fixed, small MiniMind-native architecture — proposed: the
`minimind2-small` reference configuration documented in MiniMind's own
`README_en.md` model-configuration table (`d_model=512`,
`n_layers=8`, `kv_heads=2`, `q_heads=8`, `len_vocab=6400`, dense,
~26M parameters), constructed via `MiniMindConfig(hidden_size=512,
num_hidden_layers=8, use_moe=False)` exactly as
`trainer/train_full_sft.py` itself builds `lm_config` from
`--hidden_size`/`--num_hidden_layers`/`--use_moe`. This is smaller
than ADR-0006's `SmolLM2-135M` (26M vs 135M dense parameters), a
deliberate choice: MiniMind's own architecture is the point of this
adapter (proving contract-conformance against a from-scratch engine,
not against a third-party checkpoint), so the model should be
MiniMind-native rather than an imported HF checkpoint repurposed for
a different training path.

Two provenance options exist for the starting checkpoint, and this
ADR does not resolve which one a locked pilot spec should use — that
is exactly the kind of open parameter the pilot-specific security
review is expected to scrutinise:

1. **Randomly initialised, `--from_weight none`** — MiniMind's own
   `train_full_sft.py` supports this directly ("为none则不基于任何权重训练" —
   "if none, do not train based on any weight"): the script
   initialises fresh weights and trains from scratch. This has the
   simplest provenance story (nothing downloaded, nothing to verify
   against a third-party hash — `model_hash` becomes the hash of the
   adapter's own locally-generated initial-state file, computed and
   pinned *before* `prepare()` is called), at the cost of not
   exercising the adapter's `model_path`/`model_hash` verification
   path against externally-sourced weights the way ADR-0006's pilot
   did.
2. **A locally pre-trained MiniMind pretrain checkpoint**
   (`out/pretrain_512.pth`, produced by this repository's own
   offline, bounded run of MiniMind's `trainer/train_pretrain.py`
   against a tiny synthetic corpus) — closer to MiniMind's own
   intended SFT workflow (fine-tuning an existing pretrain checkpoint,
   not training a from-scratch base in the same run), but adds a
   second bounded subprocess run (pretraining) this ADR would also
   need to specify and gate, which materially increases scope beyond
   "one bounded pilot."

This ADR's own recommendation, for the security reviewer to accept or
reject: **option 1 (random init, `--from_weight none`)**, because it
keeps the pilot to exactly one bounded subprocess invocation, matches
this ADR's "Concurrency" section below, and sidesteps a second
provenance chain (an unofficial local pretrain artifact) that would
itself need its own hash-verification and dataset-governance
paperwork. Whichever option is locked, `TrainingInputs.model_hash`
must be the SHA-256 (via `minimind_adapter._hash_path_identity`) of
the actual local file/directory content found at
`training_params["model_path"]` before training starts — no name or
config string is trusted on its own, identical posture to ADR-0006's
"Model" section.

### Dataset

Synthetic, newly constructed for this pilot, per
`docs/DATA_GOVERNANCE.md` and `TrainingInputs.validate()`'s existing
`contamination_checked=True`/non-empty-`dataset_licence` requirements
(adapter-independent, unchanged by this ADR). Distinct in both
*format* and *content* from every existing package this repository
has registered:

- **Format**: MiniMind's `SFTDataset`
  (`dataset/lm_dataset.py`) expects JSONL records shaped as
  `{"conversations": [{"role": "user", "content": ...}, {"role":
  "assistant", "content": ...}]}` — MiniMind's own multi-turn chat
  schema, not TRL's flat prompt/completion pairs ADR-0006's pilot
  used. A pilot dataset built in this shape is a new, previously
  unexercised path through this framework's data-governance tooling,
  not a re-export of ADR-0006's dataset into a new file.
- **Content**: a fixed-format single-turn unit-conversion task (e.g.
  `user: "Convert 7 kilometres to metres."` /
  `assistant: "7 kilometres = 7000 metres."`), covering a small,
  enumerable set of SI-prefix conversions (km<->m, kg<->g, l<->ml,
  etc.), deterministically generated (fixed seed) analogous to
  ADR-0006's arithmetic generator but a different task family — not
  arithmetic (ADR-0006's `examples/pilot-adr0006`), not safety-probe
  refusal/PII/harmful-instruction content (ADR-0008's
  `wpb-safety-probes-v1`), and not any `evaluator_contract.py` test
  fixture.
- **Package id**: proposed `adr0011-pilot-heldout-v1` for the held-out
  split (distinct from `adr0006-pilot-heldout-v1` and
  `wpb-safety-probes-v1`, the two existing registered package ids
  found by inspecting `examples/` and `docs/decisions/` at drafting
  time) and `adr0011-pilot-train-v1` for the train split's own
  registry ids. As with ADR-0006, train/held-out numeric or
  categorical pairs must be drawn from disjoint pools so held-out is
  unseen *content*, not merely an unseen id over already-trained
  conversions, verified programmatically at generation time before
  registration.
- Reusing any existing held-out set (ADR-0006's arithmetic pairs,
  ADR-0008's safety probes, or any `evaluator_contract.py` fixture) as
  either split is explicitly disallowed, and would be caught by
  `HeldOutExclusionRegistry.check_held_out_not_trained` if attempted
  after registration, identical enforcement to ADR-0006.

### Concurrency

Exactly one concurrent run, unchanged in kind from ADR-0006's own
"Concurrency" section. No parallel pilot runs, no second pilot started
before the first reaches a terminal `TrainingStatus` and its evidence
is recorded. Caller-level discipline, not enforced by
`MiniMindTrainerAdapter` or `run_trainer_contract` themselves (the
module docstring is explicit: "Exactly one concurrent run (enforced by
the caller, not this module ... see the ADR, same as the TRL
adapter)").

### Resource limits (`ResourceBudget`, explicit values, justified for MiniMind's subprocess/CLI shape)

MiniMind's execution shape differs from TRL's in ways that change what
these numbers should be, not just what they happen to equal — TRL's
`SFTTrainer` runs in-process, inside the same Python interpreter and
memory space `run_trainer_contract` itself runs in; MiniMind's
`train()` spawns a **child OS process** running an external script via
`subprocess.Popen`, polled every 0.5s, with **coarse cancellation**
(`terminate()`, escalating to `kill()` after a 10s grace period, per
`minimind_adapter.train()`'s own polling loop — no cooperative
in-training-loop callback hook exists the way TRL's `TrainerCallback`
provides). That shape changes the resource picture in three concrete
ways this table accounts for, not copies from ADR-0006:

| Field | Proposed value | Rationale |
|---|---|---|
| `max_wall_seconds` | `900` (15 min) | Tighter than ADR-0006's `1800`, justified by a genuinely smaller job: a 26M-parameter model (vs. 135M) over a handful of synthetic conversion examples for one epoch, with no HF Hub network round-trip to pad for (MiniMind's own script never touches a hub in offline mode) and no TRL/`SFTConfig` preprocessing overhead. 15 minutes leaves generous headroom over an expected low-single-digit-minute run (by direct analogy to ADR-0006's own inference-based timing estimate for a smaller model on the same host class) while still being caught quickly if stuck. Must additionally account for up to ~10 extra seconds of `terminate()`→`kill()` grace period on cancellation (see "Filesystem"/kill criteria below) — the enforced wall-clock ceiling should be read as inclusive of that shutdown tail, not exclusive of it. |
| `max_cpu_seconds` | `1800` (30 min, allows some multi-core parallelism above wall-clock) | Half of ADR-0006's `3600`: a 26M dense model does roughly 5x less compute per forward/backward pass than the 135M model ADR-0006 budgeted for, and OS-level CPU-second measurement (v1.1 enforcement) covers the **parent process's own view of the child subprocess's resource usage** via the same OS-level accounting `docs/decisions/0003-...` established, not adapter self-reporting — the adapter's own `resource_usage.cpu_seconds=0.0` field is explicitly a placeholder `run_trainer_contract` overwrites, identical posture to the TRL adapter. |
| `max_memory_mb` | `4096` (4 GB) | Half of ADR-0006's `8192`, justified proportionally by model size (26M vs 135M parameters) plus a small fixed allowance for the extra process boundary: unlike TRL's in-process run, MiniMind's memory footprint includes a second independent Python/PyTorch interpreter (the child subprocess re-imports `torch`, `datasets`, etc. from scratch — `minimind_adapter.train()`'s `subprocess.Popen` does not share the parent's already-loaded modules), a real but bounded fixed cost (order of a few hundred MB for interpreter + library import) this table's headroom over the bare model-compute estimate is sized to absorb. |
| `max_gpu_count` | `0` | Unchanged from ADR-0006: this host class has no CUDA GPU (`Local-LLM-Feasibility/docs/HARDWARE_DECISION_GATE.md`); `minimind_adapter._gpu_count_used()` calls `torch.cuda.device_count()` identically to the TRL adapter's own GPU-usage measurement, so the same "0, not an MPS-specific count this project does not measure" reasoning applies unchanged. |
| `max_storage_mb` | `512` (0.5 GB) | A quarter of ADR-0006's `2048`: MiniMind's own checkpoint format (`torch.save({k: v.half().cpu() ...})`, per `train_full_sft.py`'s `train_epoch`) stores fp16 weights only — no optimizer/scheduler state bundled into the same file the way some frameworks do — and a 26M-parameter fp16 checkpoint is on the order of 50MB, an order of magnitude smaller than ADR-0006's 135M-parameter artifact. 512MB leaves ample headroom for the training-evidence JSON, `subprocess_output.log`, and more than one `save_interval` checkpoint write without the adapter's own `cleanup()` needing to run mid-pilot to stay under budget. |
| `network_policy` | `"offline"` | Unchanged: already the only policy `MiniMindTrainerAdapter.prepare()` accepts (`RejectedInputError` otherwise); the adapter additionally forces `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` inside the child subprocess's environment as defence in depth, even though MiniMind's own SFT path has no required Hub dependency for local-file training. |
| `training_params["epochs"]` | `1` | **Not** `max_steps` — see "Known adapter/script CLI incompatibilities" below: the pinned commit's `train_full_sft.py` argument parser has no `--max_steps` flag at all, only `--epochs`. This table proposes the adapter-enforced stopping bound MiniMind's own CLI actually supports, against a deliberately tiny synthetic dataset (on the order of tens of examples, mirroring ADR-0006's 40-record train split), so one epoch is itself a small, bounded amount of work, not an open-ended run. |

These are proposed starting values for review, not final until the
pilot-specific security review (see "Gating") signs off on them
specifically, identical posture to ADR-0006's own disclaimer.

### Known adapter/script CLI incompatibilities (found during this ADR's drafting)

Direct inspection of `minimind_adapter._build_subprocess_args` against
the pinned commit's actual `trainer/train_full_sft.py`
`argparse.ArgumentParser` (which calls plain `parser.parse_args()`,
not `parse_known_args()`, so unrecognised flags are a hard error, not
silently ignored) surfaced three concrete mismatches that would cause
`adapter.train()` to fail immediately on invocation, before any
compute is spent, if exercised as currently coded:

1. **`--out_dir` vs. `--save_dir`.** The adapter constructs
   `["--out_dir", str(checkpoint_dir)]`; the pinned script's parser
   defines `--save_dir` (default `"../out"`), not `--out_dir`.
2. **`--model_path` does not exist.** The adapter conditionally
   constructs `["--model_path", str(params["model_path"])]` when that
   param is set; the pinned script has no `--model_path` flag at all
   — it selects a starting checkpoint via `--from_weight` (a name
   string, default `"pretrain"`, or the literal string `"none"` for
   random initialisation) combined with a fixed
   `{save_weight}_{hidden_size}.pth`-style filename convention inside
   `--save_dir`/`--from_resume`'s checkpoint-loading path
   (`init_model`/`lm_checkpoint` in `trainer/trainer_utils.py`), not
   an arbitrary caller-supplied path.
3. **`--max_steps` does not exist.** The adapter conditionally
   constructs `["--max_steps", str(params["max_steps"])]`; the pinned
   script's parser has no `--max_steps` flag — only `--epochs`. This
   ADR's own resource-limit table above already routes around this by
   proposing `epochs=1` rather than `max_steps`, but the adapter code
   itself still needs the `max_steps` code path fixed or removed
   before that choice is actually exercisable end-to-end.

This ADR does not fix `minimind_adapter.py` — that is out of scope
for a docs-only drafting work package, and fixing it would itself be
new adapter code needing its own review, not something this ADR's own
authorship should quietly bundle in. It records these three gaps as
an explicit **blocking pre-condition**: a separate, narrowly-scoped
adapter-code PR (fixing `_build_subprocess_args` to emit `--save_dir`,
a `--from_weight`/naming-convention-compatible checkpoint-selection
path instead of `--model_path`, and `--epochs` instead of
`--max_steps`, with its own updated conformance tests) must land and
pass review before this ADR's locked pilot configuration is
executable at all — this is squarely inside the scope of what "Maya's
pilot-specific live-execution security review" (see "Gating") should
require resolved before clearing execution, not a detail to discover
during a live run.

**Tracking note (updated):** this defect was tracked as **issue #47**
and is now **fixed and merged** via PR #49 — `_build_subprocess_args`
was corrected to emit `--save_dir`, `--from_weight`, and `--epochs`
(dropping the nonexistent `--out_dir`/`--model_path`/`--max_steps`
flags), the existing mocked-subprocess conformance test was updated,
and a new static regression guard
(`tests/test_minimind_trainer_cli_flags.py`) now `ast.parse`s a
vendored copy of the real pinned `trainer/train_full_sft.py` and
asserts every flag the adapter can ever construct is a subset of the
real script's actual flag set — independently re-verified against the
real pinned MiniMind script, not merely re-asserted against the
adapter's own prior assumptions. This precondition is therefore
**satisfied**: the adapter's CLI-argument construction is no longer
the blocker this ADR flagged during drafting. This does **not**
change anything else about "Gating" below — Maya's pilot-specific
live-execution security review of the actual running subprocess, and
separate owner authorisation to schedule the pilot, remain required
and outstanding, in that order, exactly as before.

### Filesystem

`ResourceBudget.filesystem_root` set to a dedicated, empty, per-pilot
scratch directory (not the repository checkout, not any other
project's data directory). `MiniMindTrainerAdapter.work_dir` points at
the same root; `_run_dir(run_id)` creates a per-run subdirectory
(`checkpoints/`, then `final/` after `_collect_final_artifact`
relocates MiniMind's own checkpoint output). Cleaned up via
`cleanup()` (already implemented, `shutil.rmtree(run_dir,
ignore_errors=True)`) after the pilot's evidence is copied out — same
contract as the TRL adapter's `cleanup()`, unchanged by this ADR. One
MiniMind-specific filesystem note: unlike TRL's in-process
`Trainer.save_model`, MiniMind's script writes checkpoint files
directly into whatever directory its own `--save_dir`/`--out_dir` flag
(see "Known adapter/script CLI incompatibilities" above — this must
resolve to `--save_dir` for the file to land where the adapter expects
it) points at; `_collect_final_artifact` only works correctly once
that flag name is fixed, another concrete way the blocking
pre-condition above is not cosmetic.

### Independent evaluation wiring

Identical requirement in kind to ADR-0006's own "Independent
evaluation wiring" section, reused rather than redesigned: the pilot's
trained artifact must be scored through
`evaluator_contract.run_evaluator_contract` against a `HeldOutSet`
constructed and registered
(`HeldOutExclusionRegistry.register_package_held_out`) *before* the
pilot's training data is finalised, using the `adr0011-pilot-heldout-v1`
package id proposed above — distinct from the pilot's own
`adr0011-pilot-train-v1` training-data package id — so the
bidirectional contamination check in `held_out_registry.py` has
something real to check against. The already-merged, already-reviewed
`HFLocalCausalLMEvaluatorAdapter` (`src/codevolt_mdf/hf_local_evaluator_adapter.py`,
PR referenced in `docs/REAL_ADAPTERS.md`) is a plausible scoring
adapter for this pilot's task if its exact-match task-type mode is
pointed at a MiniMind-produced artifact loadable by
`AutoModelForCausalLM`-compatible tooling — MiniMind's own README
states its architecture is aligned with the `Qwen3`/`Qwen3-MoE`
ecosystem, which is a reasonable basis for that compatibility claim,
but this ADR does not verify it directly and does not decide the
evaluator adapter choice; that remains separate, not-yet-authorised
follow-up work, identical posture to ADR-0006's own disclaimer on this
point. This section records the *wiring requirement*, not a claim that
it has been exercised for a MiniMind artifact specifically.

### Kill criteria (same class as ADR-0006, restated for this pilot's subprocess shape)

Stop on: missing provenance, unbounded resources, trainer access to
held-out data, unverifiable artifacts, unsafe network/filesystem
access, or any path that treats completion as improvement. Concretely
for this pilot:

- A `TrainingOutput.status` other than `ACCEPTED` — including
  `INTERRUPTED` from a non-zero subprocess exit code
  (`minimind_adapter.train()`'s explicit
  `MiniMindSubprocessError`/`process.returncode != 0` branch, a
  failure mode the in-process TRL adapter has no direct analogue for)
  — halts the pilot.
- An evidence-hash mismatch (`TamperDetectedError` at the contract
  runner level, unchanged mechanism from ADR-0006).
- A `ContaminationDetectedError` from the evaluator contract.
- A resource budget violation on any of the five `ResourceBudget`
  dimensions in the table above, OS-measured for `wall_seconds`/
  `cpu_seconds`/`memory_mb_peak` (the process-isolation enforcement
  `docs/decisions/0003-...` established, unchanged by this ADR —
  MiniMind's child subprocess is inside the same OS-level enforcement
  boundary, not a separate trust domain).
- **MiniMind-specific**: a cancellation that does not stop the
  subprocess within its `terminate()`→`kill()` grace window (the
  adapter's own escalation already handles this correctly by
  design — `process.kill()` after a 10s `TimeoutExpired` — but the
  pilot-specific security review should specifically exercise this
  path against the real child process, not merely trust the mocked
  `subprocess.Popen` unit tests that currently cover it).

Each of the above independently halts the pilot and is recorded as
evidence, not silently retried or reinterpreted as success —
unchanged posture from ADR-0006.

## Gating (this ADR authorises none of it)

This ADR does not authorise the pilot to run. Before execution, in
order:

1. **ADR drafted** — this document, plus the separately delivered
   `MiniMindTrainerAdapter`/`evaluator_contract.py`/
   `held_out_registry.py`/`regression_check.py` components, satisfies
   the contract-and-configuration half of this work package. The
   blocking pre-condition named above (fixing the three
   `_build_subprocess_args` CLI mismatches, tracked as **issue #47**)
   is now **satisfied**: issue #47 is fixed and merged via **PR #49**,
   with its own updated conformance tests and a new static
   flag-introspection regression guard, and the adapter's CLI flags
   are now independently verified correct against the real pinned
   MiniMind script. A real, registered `adr0011-pilot-heldout-v1`
   held-out set still remains to be built when the pilot is scheduled
   — this ADR records that remaining requirement, not a completed
   artifact, identical posture to ADR-0006 at the same stage.
2. **Maya's pilot-specific live-execution security review** — sandbox,
   egress, secrets, artifact hashes, supply-chain identity (including
   re-verifying the `MINIMIND_PINNED_COMMIT` pin and its Apache-2.0
   licence are still current), and safe-stop behaviour of the *actual
   live MiniMind subprocess*, not the adapter code review PR #40
   already completed. This is a distinct review scope, and it must
   additionally cover the three CLI-mismatch fixes named above as part
   of its own scope, since those fixes are themselves new code this
   ADR does not pre-clear.
3. **Owner authorisation** to schedule the pilot itself, separate from
   the standing continual-training-framework authorisation that covers
   drafting this ADR and building the evaluation harness.

No training engine is invoked, no pilot is scheduled, and no resource
is reserved as a result of this ADR. `MiniMindTrainerAdapter.train()`
remains untested by any run in this repository, exactly as ADR-0010
left it.

## Consequences

- CodeVolt MDF now has a reviewable, versioned decision record for a
  bounded MiniMind pilot's exact proposed configuration, mirroring
  ADR-0006's discipline for the framework's second real engine — the
  next reviewer (Maya, for the pilot-specific review; the owner, for
  scheduling) has concrete numbers, a concrete wiring requirement, and
  a concrete list of adapter-code gaps to review against, not prose to
  interpret.
- This ADR's proposed values (model checkpoint provenance option,
  dataset shape, resource limits) are a starting point for that
  review, not immutable — the pilot-specific security review is
  expected to challenge or tighten them, exactly as ADR-0006's own
  "Consequences" section anticipated for its pilot.
- Nothing in this ADR changes `CONTRACT_VERSION`, the MiniMind pin
  (`MINIMIND_PINNED_COMMIT`), or any adapter code path. It is a
  planning document only — the CLI-mismatch fixes it identifies are
  named as required follow-up work, not performed here.
- This ADR is also the precondition named by
  `docs/CONTINUAL_IMPROVEMENT_LOOP_DESIGN.md`'s round-1 discussion: a
  companion design doc for how a *second*, later-authorised round
  would consume this pilot's evidence, drafted alongside this ADR per
  issue #46's Part 2 scope. That design doc does not authorise any
  round, including this one.

## Pilot execution result (2026-09-20, REAL, EXECUTED)

### Run identity

- `run_id` / `artifact_id`: `artifact-adr0011-pilot-20260920`.
- Date: 2026-09-20.
- MiniMind pinned commit: `cc312c1cc614bc371cd85dcbcbc1d3ba1590f364`.
- Reproduction entrypoint: `examples/pilot-adr0011/run_pilot.py`.

### TrainingOutput (real `MiniMindTrainerAdapter.train()` call)

| Field | Value |
|---|---|
| `status` | `accepted` |
| `reason` | `minimind train_full_sft.py subprocess completed` |
| `artifact_id` | `artifact-adr0011-pilot-20260920` |
| `resource_usage.wall_seconds` | `65.29` (budget: `900`) |
| `resource_usage.cpu_seconds` | `75.59` (budget: `1800`) |
| `resource_usage.memory_mb_peak` | `1456.6` (budget: `4096`) |
| `resource_usage.storage_mb_used` | `483.98` (budget: `512`) |
| `resource_usage.gpu_count_used` | `0` |
| Final training loss | `2.0751` (1 epoch, 40 examples) |

All five resource dimensions inside budget; no kill criterion
triggered. Sanitised evidence:
`examples/pilot-adr0011/evidence/pilot-result.sanitised.json`.

### Conversion caveat (evaluation-only, does not change what was trained)

MiniMind's `train_full_sft.py` produces a native MiniMind checkpoint
with no native Hugging Face format. To score it through the existing
`HFLocalCausalLMEvaluatorAdapter`, it was converted to a
shape-compatible HF `Qwen3ForCausalLM` checkpoint for evaluation only —
this conversion step happens after training completes and does not
alter what was actually trained or how it was trained. A real bug was
found and fixed during this conversion: the `Qwen3Config` used for the
converted checkpoint needs `num_key_value_heads=4`, not `2` as
initially assumed, confirmed by inspecting the raw trained tensor
shapes directly (`(256, 512)` = `4 x 64`, not `2 x 128`).

### EvaluationOutput (real `HFLocalCausalLMEvaluatorAdapter`, real held-out set)

Scored via `run_evaluator_contract` against `HeldOutSet` package
`adr0011-pilot-heldout-v1` (10 held-out examples), with
`HeldOutExclusionRegistry.check_held_out_not_trained` confirming zero
contamination before scoring.

| Field | Value |
|---|---|
| `status` | `scored` |
| `aggregate_score` | `0.0` (0/10 correct) |
| `evidence_hash` (independently re-verified with `shasum -a 256`) | `dd4dab1a594a253b1300b8f117670c0a8209d657a20aa1b9cdbcea622ac53271` |

All 10/10 held-out examples (`adr0011-pilot-heldout-v1`) scored
`correct=False` / `score=0.0`. Sanitised evidence:
`examples/pilot-adr0011/evidence/pilot-result.sanitised.json`.

### Root cause of the 0.0 score (confirmed via a real experiment)

The trained model emits the end-of-sequence token as the literal first
generated token for held-out unit-conversion prompts, regardless of
`max_new_tokens` (tested at both `8` and `30`) — this reproduces
identically via the evaluator's own generation path and via a
standalone script, ruling out an evaluator-harness bug. This was
further confirmed to be prompt-dependent, not a blanket generation
failure: a different sanity-check prompt against the exact same
model/parameters produced non-empty (though still visibly degenerate)
output. This is consistent with a legitimate artifact of severe
undertraining (40 examples, 1 epoch) rather than an adapter defect.

### What this result does and does not establish

This is a single bounded exploratory pilot demonstrating that the
propose -> train -> evaluate pipeline runs end-to-end for the MiniMind
adapter, from a real subprocess training run through a real,
contamination-checked evaluation against a real held-out set. 40
examples and 1 epoch are not enough to produce a usable model, so
`aggregate_score=0.0` is an expected, informative, and honestly
reported outcome of severe undertraining — not a pipeline failure, not
an adapter bug, and not evidence the wiring is broken (training
accepted within all five resource budgets; evaluation scored all 10
examples with zero contamination). It is not a capability claim, not a
production readiness claim, and not a promotion decision for the
trained artifact. It does not authorise any further MiniMind pilot,
larger run, or production use — a repeat or expanded pilot requires
repeating the full three-gate process (ADR, Maya's pilot-specific
live-execution security review, owner authorisation) from scratch, the
same as this one did.

### Review follow-up

Training evidence hash and the checkpoint conversion script were added
to the committed record following independent review. Specifically:
`examples/pilot-adr0011/evidence/training-evidence.sanitised.json`
(sanitised evidence for the `TrainingOutput` above, including the
`evidence_hash` computed by `minimind_adapter.py`'s `_write_evidence`)
and `examples/pilot-adr0011/convert_checkpoint.py` (the conversion
script referenced in "Conversion caveat" above, including the
`num_key_value_heads=4` fix) were both added to this pull request in
response to Maya-CodeVolt's review feedback.
