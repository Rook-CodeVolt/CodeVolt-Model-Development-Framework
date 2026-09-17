# ADR-0005: TrainerAdapterV1 for TRL (real engine, contract + tests only)

- Status: accepted
- Date: 2026-09-17

## Context

`docs/TRAINER_ADAPTER_CONTRACT.md`'s "Real bounded pilot plan" (added by
ADR-0002, unchanged since) documents but explicitly does not implement
the one bounded real-engine integration issue #7 asks for. That plan
names TRL or Unsloth as candidates and defers the specific engine choice
to "a separate follow-up ADR, not decided here." This ADR makes that
choice for TRL and implements the adapter side of the pilot plan, as a
new, separately gated work package (owner-confirmed 2026-09-17): the
adapter, its contract tests, and this ADR are authorised; **the real
pilot run itself is not** and remains gated exactly as the pilot plan
already requires (independent evaluation against a hidden held-out set,
and Maya's independent security review of the real engine specifically —
sandbox, egress, secrets, artifact hashes, supply-chain identity,
safe-stop). Nothing in this ADR or its PR authorises that pilot run.

TRL (`huggingface/trl`, PyPI package `trl`) was chosen over Unsloth for
this first real-engine slot because it depends only on
`transformers`/`datasets`/`accelerate` (all already pinned and installed
in this repository's `.venv` per `docs/INFRA_SOFTWARE_BASELINE.md`-style
provenance tracking), has no CUDA-only hard dependency for the bounded
CPU-only SFT path this ADR scopes to, and its `SFTTrainer`/`SFTConfig`
API directly maps onto `TrainerAdapterV1`'s `prepare`/`train`/`cleanup`
shape without a wrapper layer. Unsloth remains a candidate for a future,
separately gated adapter; choosing it would need its own ADR revisiting
this decision, not an extension of this one.

## Decision

Add `src/codevolt_mdf/trl_adapter.py` (`TRLTrainerAdapter`,
`contract_version = "1.1.0"`) implementing `TrainerAdapterV1` against
`CONTRACT_VERSION = "1.1.0"`, and `tests/test_trl_adapter.py` (28
conformance tests). Declare the optional `trl-adapter` extra in
`pyproject.toml` (`trl>=0.20.0,<0.25.0`, plus the transformers/datasets/
accelerate floors TRL itself requires) so the adapter's real dependency
is opt-in, not a new hard dependency of `codevolt-model-development-framework`
itself — consistent with `docs/GOVERNANCE.md`'s "optional engines must
not become core dependencies without an accepted architecture decision"
(this ADR is that acceptance, scoped to optional-extra status only).

### UpstreamRequirement version bounds

```python
UpstreamRequirement(
    engine_name="trl",
    min_version="0.20.0",
    max_version="0.24.0",
    installed_version=<detected at adapter construction>,
)
```

- **Lower bound `0.20.0`**: the oldest release in this project's
  supported range that still targets the modern `SFTTrainer`/`SFTConfig`
  shape this adapter's `train()` calls (`output_dir`, `max_steps`,
  `save_steps`, `save_strategy`, `learning_rate`,
  `per_device_train_batch_size`, `seed`, `report_to`, `logging_steps` on
  `SFTConfig`; `model`, `args`, `train_dataset`, `peft_config`,
  `callbacks` on `SFTTrainer`). Not independently re-verified against
  every release back to 0.20.0 in this work package — the floor is a
  documented choice, not a claim every intermediate version was
  installed and exercised.
- **Upper bound `0.24.0`**: the last TRL release that still supports
  Python 3.9 (`Requires-Python: >=3.9`; confirmed by downloading and
  inspecting `trl-0.24.0-py3-none-any.whl`'s `METADATA` directly from
  PyPI). TRL 0.25.0 raises its own floor to `Requires-Python: >=3.10`
  (confirmed the same way against `trl-0.25.1-py3-none-any.whl`), which
  is incompatible with this project's own `pyproject.toml`
  `requires-python = ">=3.9"` floor. Widening this bound past 0.24.0
  requires either (a) raising this project's own Python floor to >=3.10
  first (a separate, broader decision this ADR does not make), or (b) a
  new ADR if TRL ever backports a >=3.9-compatible release above 0.24.0
  (no evidence such a release exists or is planned).
- TRL's own `1.0.0` release (2026-03-31) is a breaking rearchitecture
  around a shared `_BaseTrainer` (per TRL's own `MIGRATION.md`); this
  bound does not reach it, so that migration is explicitly out of scope
  for this adapter as written.
- `_detect_installed_trl_version()` returns `"0.0.0"` (not an exception)
  when `trl` is not importable, so `UpstreamRequirement.is_compatible()`
  fails closed (rejects) through the contract runner's existing
  unconditional pre-`prepare()` check, rather than the adapter itself
  needing special-case handling for "not installed."

### Scope: one training method, offline-only, hash-verified provenance

- **Exactly one training method**: TRL's `SFTTrainer` (supervised
  fine-tuning). No PPO/GRPO/DPO/KTO/other RL trainers, no distributed or
  multi-node training, no vLLM integration. A future adapter or a
  revision of this one would need its own ADR to add any of those.
  Optional LoRA (`peft`) is supported but off by default
  (`training_params["use_lora"]`); `prepare()` raises
  `RejectedInputError` if requested without `peft` importable, rather
  than silently falling back to full fine-tuning.
- **Fully offline, enforced twice**: `prepare()` raises
  `RejectedInputError` unless `budget.network_policy == "offline"`
  (stricter than the contract requires, matching the fake adapter's own
  posture and `docs/THREAT_MODEL.md`'s "avoid network access by
  default"). `train()` additionally sets `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1` inside the isolated child process as defence
  in depth on top of `process_isolation.py`'s existing Python-level
  socket wrapper — this adapter never resolves or downloads a model or
  dataset from a hub itself; both `model_path`/`dataset_path` in
  `training_params` must already be local, pre-verified paths.
- **Hash-verified provenance, not name-trusted**: `prepare()` computes
  `_hash_path_identity()` over the actual content at `model_path`/
  `dataset_path` (a single file's SHA-256, or a sorted-manifest SHA-256
  over every regular file for a directory checkpoint) and raises
  `RejectedInputError` on any mismatch against `TrainingInputs.model_hash`/
  `dataset_hash` — a declared identity is checked against content, not
  trusted from a name or a prior claim. This closes the same class of
  gap `TrainingInputs.model_hash`/`dataset_hash` exist to close at the
  contract level, at the adapter's own boundary.
- **Adapter-enforced step bound**: `training_params["max_steps"]` is
  required and must be a positive int; `prepare()` rejects its absence
  or a non-positive value. This is deliberately in addition to (not a
  replacement for) the contract's own `budget.max_wall_seconds`/
  `max_cpu_seconds` OS-measured enforcement — a real engine should not
  rely solely on being killed from outside to stop.
- **Checkpoint/resume**: cooperative cancellation (checked via a
  `TrainerCallback.on_step_end` hook against `cancel_token`) triggers
  `trainer.save_model()` into the run's checkpoint directory and returns
  `TrainingStatus.INTERRUPTED` with a populated `CheckpointHandle` whose
  `state_hash` is `_hash_path_identity()` over that directory. Resuming
  independently re-verifies that hash before calling
  `trainer.train(resume_from_checkpoint=...)`, raising `InvalidInputError`
  on mismatch — mirroring the fake adapter's own tamper-on-resume
  behaviour (`docs/TRAINER_ADAPTER_CONTRACT.md`, "Safe-halt / checkpoint
  / resume").
- **Resource usage self-report is a placeholder for wall/cpu/memory**:
  `train()` returns `ResourceUsage(wall_seconds=0.0, cpu_seconds=0.0,
  memory_mb_peak=0.0, ...)` for those three fields deliberately, because
  `run_trainer_contract` (v1.1, ADR-0003) always overwrites them with
  OS-measured figures before the budget check runs — self-reporting a
  real number there would be misleading busywork the contract already
  discards. `storage_mb_used` (computed from the run directory's actual
  size) and `gpu_count_used` (via `torch.cuda.device_count()` when
  `torch` is importable, else `0`) remain genuinely adapter-self-reported,
  consistent with `docs/TRAINER_ADAPTER_CONTRACT.md`'s "Resource budget
  enforcement" section (GPU count is not OS-measured by this project;
  that remains separate follow-up work for real GPU hardware).
- **Concurrency**: exactly one concurrent run, same as the pilot plan
  already requires. This adapter does not itself enforce concurrency
  (neither does the fake adapter, nor `run_trainer_contract`) — that
  remains a caller-level discipline the pilot's own run harness must
  enforce when the pilot is eventually authorised, not something this
  ADR claims the adapter guarantees on its own.

### Reuse of fake-adapter-style contract testing, and its explicit limit

`tests/test_trl_adapter.py` (28 tests) reuses the same pattern
`tests/test_trainer_contract.py` established against the fake adapter —
build `TrainingInputs`/`ResourceBudget`, call either `adapter.prepare()`
directly or the full `run_trainer_contract()`, assert the resulting
status/exception — for every scenario the contract allows proving
**without an actual training run**:

- upstream contract-version and engine-version compatibility (both
  below-min and above-max);
- `prepare()` accepting well-formed local inputs (asserted by absence of
  an exception, never followed by a `train()` call);
- offline-network-policy enforcement, both at `prepare()` directly and
  through the full contract runner;
- missing/invalid `training_params` (`model_path`, `dataset_path`,
  `max_steps` absent, zero, or negative);
- nonexistent local paths;
- model/dataset hash mismatch (provenance/tamper), both at `prepare()`
  directly and through the full contract runner (a genuinely tampered
  file, hashed, then corrupted, then run through
  `run_trainer_contract()` end-to-end and shown to be `REJECTED` before
  any training work could start);
- LoRA requested without `peft` importable;
- the `_hash_path_identity()` helper's determinism, content-sensitivity,
  and directory-manifest behaviour, independent of the adapter;
- `cleanup()` idempotency, both empty and non-empty;
- the adapter-independent `TrainingInputs.validate()` precondition
  (contamination-checked, hash shape) still applying to TRL-adapter
  inputs, and the full contract runner never reaching `prepare()` at all
  for a bad hash shape.

**Explicit, disclosed limit — this pattern is NOT reused for every
scenario `test_trainer_contract.py` covers against the fake adapter**:
success, timeout, cancellation, resource-overrun, checkpoint/resume
completion, and evidence-hash tamper detection all require an actual
call into `adapter.train()` — which for this adapter means a real
(however small) TRL `SFTTrainer.train()` call against a real local
model/dataset. This work package does not execute that call anywhere:
not in a test, not in CI, not interactively. `train()` is fully
implemented (lazy-imported `trl`/`transformers`/`datasets`/`torch`, real
`SFTConfig`/`SFTTrainer` construction, a `TrainerCallback` for
cooperative cancellation, checkpoint save/verify/resume, evidence
write), reviewed, and left for the separately gated real pilot to
exercise for the first time — consistent with the instruction that "no
real pilot run" happens in this work package. Two tests
(`test_installed_trl_version_is_within_declared_bounds`,
`test_trl_api_surface_matches_adapter_expectations`) additionally
require `trl` importable (via `pytest.importorskip("trl")`) to check the
installed version against the declared bound and the real
`SFTConfig`/`SFTTrainer` signatures still expose what `train()` calls —
these are import/introspection checks, not training runs.

### Evidence

- `python -m pytest` (repository `.venv`, Python 3.9.6, `trl==0.24.0`
  installed via the new `trl-adapter` extra): **69 passed** (41
  pre-existing + 28 new in `tests/test_trl_adapter.py`), 0 failed, 0
  skipped.
- The same suite run in an independently created, trl-**not**-installed
  Python 3.11 venv (`./local-evidence/agent-work/notrl-venv`, `pip install -e
  ".[dev]"` only): **64 passed, 5 skipped** — the 5 skips are exactly
  the tests that require `trl` importable
  (`pytest.importorskip("trl")`), confirming the module and its tests
  degrade gracefully rather than hard-failing when the optional real
  engine is absent, and that no test accidentally depends on `trl` being
  present without declaring that dependency.
- `python -m ruff check .`: **all checks passed** (repository `.venv`).
- `trl==0.24.0`'s `Requires-Python: >=3.9` and `trl==0.25.1`'s
  `Requires-Python: >=3.10` were confirmed by downloading each wheel
  from PyPI with `pip download --no-deps` and reading `METADATA`
  directly (not from prose/changelog text), the same evidence standard
  `docs/TRAINER_ADAPTER_CONTRACT.md` expects of a real adapter's
  provenance claims.
- No training happened as a result of this work: `adapter.train()` is
  never called by any test, script, or CI step added or touched by this
  PR.

### 8-vs-9 baseline test-count note (flagged, not corrected here)

While reconstructing the exact pre-existing baseline test count for this
ADR's evidence section, PR #9's own description text says "27 passed (16
pre-existing + 11 new)" and separately states the new
`tests/test_trainer_contract.py` file has "8 conformance cases"; the
file as merged (commit `1b8a9ea`) actually contains **11** `def test_`
functions (matching the "+ 11 new" arithmetic, not the "8 conformance
cases" prose, which undercounts by not including
`test_contract_version_is_declared`,
`test_training_inputs_validate_rejects_uncontaminated_dataset`, and one
of the checkpoint-resume pair). This is a pre-existing PR-description
wording inconsistency unrelated to this ADR's own change, noted here
because it was directly observed while establishing this ADR's own
baseline-count evidence; no repository file or test was changed to
"fix" it, per `AGENTS.md`'s "do not silently change ... in response to a
finding." Reported to the requesting owner directly; corrected on the
GitHub side only if and when the owner directs it.

## Consequences

- CodeVolt MDF now has one real, reviewable (not yet run) engine adapter
  alongside the deterministic fake adapter, closing the gap
  `docs/TRAINER_ADAPTER_CONTRACT.md`'s pilot plan step "(a) an ADR
  records the chosen engine and its `UpstreamRequirement` bounds" asked
  for. Steps (b) (independent evaluation against a hidden held-out set)
  and (c) (Maya's security review of the real engine specifically) are
  **not** satisfied by this ADR and remain the explicit blocking
  conditions before any pilot run, unchanged from ADR-0002.
- The `trl-adapter` extra keeps `trl`/`transformers`/`datasets`/
  `accelerate` optional, not a hard dependency of
  `codevolt-model-development-framework` itself — an environment that
  never installs the extra still runs the full existing test suite
  unchanged (64 passed, 5 gracefully skipped, confirmed above).
- Future contract changes remain diffable per ADR-0002's discipline; this
  ADR does not change `CONTRACT_VERSION` (still `1.1.0`) or
  `SUPPORTED_CONTRACT_VERSIONS` — it adds a new adapter against the
  existing contract, not a contract change.
- `core.py`'s manifest-driven `run_experiment` flow is still not wired to
  `TrainerAdapterV1` (unchanged from ADR-0002); this adapter is callable
  directly via `run_trainer_contract()`, not yet through the CLI/manifest
  path.
- Widening the TRL version bound past `0.24.0`, adding a second training
  method (PPO/GRPO/DPO/...), enabling distributed training, or running
  the real pilot are all separate, not-yet-authorised follow-ups, each
  needing its own gated decision — this ADR authorises none of them.
