# ADR-0010: TrainerAdapterV1 for MiniMind (second real engine, contract + tests only)

- Status: accepted
- Date: 2026-09-18

## Context

`docs/decisions/0005-trl-trainer-adapter-v1.md` added the framework's
first real (non-fake) `TrainerAdapterV1` implementation, wrapping TRL's
`SFTTrainer`. That ADR's own "Consequences" section left the door open
for a second real engine adapter as a separate, not-yet-authorised
follow-up. Issue #36 records the owner's decision to build that second
adapter, wrapping [MiniMind](https://github.com/jingyaogong/minimind)
(Apache-2.0, from-scratch native PyTorch, no TRL/PEFT dependency) —
deliberately a different *shape* of engine than TRL, not a second
wrapper around the same one, so the framework's `TrainerAdapterV1`
contract gets exercised against a genuinely different integration
pattern (subprocess/CLI vs. in-process library call).

As with ADR-0005, this ADR authorises the adapter, its contract tests,
and this record as a work package. **It does not authorise a real
training pilot.** Any live MiniMind training run remains a separately
gated decision, exactly as ADR-0005 established for TRL and as issue #36
scopes this work package.

## Decision

Add `src/codevolt_mdf/minimind_adapter.py` (`MiniMindTrainerAdapter`,
`contract_version = "1.1.0"`) implementing `TrainerAdapterV1` against
`CONTRACT_VERSION = "1.1.0"`, and `tests/test_minimind_adapter.py` (37
conformance tests). No new `pyproject.toml` dependency is declared:
MiniMind is not a PyPI package and this adapter never imports it — it
shells out to the MiniMind checkout's own `trainer/train_full_sft.py`
via `subprocess`, using only Python's standard library (`subprocess`,
`hashlib`, `json`, `shutil`, `os`, `pathlib`) plus the framework's own
`trainer_contract` module.

### Pin

```python
MINIMIND_REPO_URL = "https://github.com/jingyaogong/minimind"
MINIMIND_PINNED_COMMIT = "cc312c1cc614bc371cd85dcbcbc1d3ba1590f364"
MINIMIND_PINNED_BRANCH = "master"
```

Re-verified current for this PR via `git ls-remote
https://github.com/jingyaogong/minimind master`, which returned exactly
`cc312c1cc614bc371cd85dcbcbc1d3ba1590f364` at the time this ADR was
written — the pin had not moved since it was first identified as a
candidate. Licence re-confirmed the same way: GitHub's repository
licence API reports `apache-2.0` for `jingyaogong/minimind`, and
`LICENSE` at the pinned commit is the standard Apache License 2.0 text
(fetched directly from
`raw.githubusercontent.com/jingyaogong/minimind/<pinned-commit>/LICENSE`).
Re-pinning to a newer commit requires re-verifying that
`trainer/train_full_sft.py` still exposes the CLI flags this adapter's
`train()` constructs (`_build_subprocess_args`), the same evidence bar
ADR-0005 applies to its own TRL version pin.

### Pin verification: subprocess `git rev-parse`, not an importable version

This is the adapter's one structural deviation from
`TRLTrainerAdapter`, and the reason it needed its own ADR rather than
being folded into ADR-0005 as "another engine, same pattern":

- TRL is a PyPI package with an importable `trl.__version__`.
  `TRLTrainerAdapter._detect_installed_trl_version()` imports `trl` and
  reads that attribute; `UpstreamRequirement.is_compatible()` compares
  it against a semantic-version range (an exact pin, per ADR-0005's own
  amendment) using the contract's built-in numeric-tuple version
  comparison.
- MiniMind has **no PyPI package, no `__init__.py`-level version
  string, and no importable Python API at all** — it is a script
  repository, invoked as `python3 trainer/train_full_sft.py --flags...`.
  There is nothing to `import` and no `__version__` attribute to read.
  Treating "installed_version" as a git commit SHA and feeding it
  through the *same* numeric-tuple version comparison
  `UpstreamRequirement.is_compatible()` uses for TRL would be
  incorrect: `_version_tuple()` extracts digit runs from
  dot-separated segments (designed for `"0.24.0"`-style strings), and a
  40-character hex SHA has no such structure — comparing two SHAs with
  that logic would produce a meaningless ordering, not a meaningful
  compatibility check.
- Instead, `MiniMindTrainerAdapter.prepare()` runs `git rev-parse HEAD`
  (via `subprocess.run`, 10-second timeout, `check=False`) against the
  caller-supplied `training_params["minimind_repo_path"]` and requires
  **exact string equality** with `MINIMIND_PINNED_COMMIT`. This is a
  narrower, stricter check than a version range: there is exactly one
  admissible commit, verified by direct inspection of the actual
  checkout the caller intends to run against — not a trusted label, a
  declared version string, or an installed-package metadata field, all
  of which could describe a checkout without the adapter ever looking
  at it. `_detect_minimind_checkout_commit()` returns `""` (never
  raises) for a missing path, a non-git directory, or any subprocess
  failure, mirroring `_detect_installed_trl_version()`'s fail-closed
  `"0.0.0"` sentinel shape — a value that can never accidentally equal
  a real pin.
- Because there is no meaningful "installed version" to compare through
  the *contract runner's* own pre-`prepare()` upstream gate
  (`run_trainer_contract`'s unconditional `if not
  adapter.upstream.is_compatible(): raise RejectedInputError(...)`,
  which *does* still exist and still runs unconditionally before this
  adapter's own logic), this adapter's `UpstreamRequirement` declares
  `min_version = max_version = installed_version =
  MINIMIND_PINNED_COMMIT` at construction time — trivially
  self-compatible by construction, so that generic gate passes and the
  adapter's *own* `prepare()`-level `git rev-parse` check is the one
  that actually does the pin verification against the real checkout.
  This is a deliberate reinterpretation of what `UpstreamRequirement`
  means for an adapter with no external version registry to query: it
  becomes documentation of the pin plus a placeholder that satisfies
  the `TrainerAdapterV1` Protocol's declared-attribute shape, not the
  live compatibility check. Every test that wants to exercise "checkout
  matches/doesn't match the pin" calls `adapter.prepare()` directly (or
  monkeypatches `adapter.upstream` separately when going through
  `run_trainer_contract()`), never relies on the contract runner's
  generic upstream gate to catch a pin mismatch.
- A future adapter for a *different* script-repo engine (no importable
  version) should reuse this same git-rev-parse-against-checkout
  pattern rather than reinvent it; a future adapter for a *packaged*
  engine (importable, has `__version__`) should follow ADR-0005's
  pattern instead. Both are now precedented in this repository.

### Scope: one training method, subprocess-based, offline-only, hash-verified provenance

- **Exactly one training method**: MiniMind's full-parameter SFT script,
  `trainer/train_full_sft.py`. MiniMind's repository also ships
  pretraining, LoRA, RLHF/DPO, and distillation scripts; none of those
  are wired into this adapter. A future adapter revision would need its
  own ADR to add any of them, matching ADR-0005's equivalent scope
  limit for TRL's RL trainers.
- **Subprocess, not in-process**: `train()` builds an argv list
  (`_build_subprocess_args`) and runs it via `subprocess.Popen` with the
  MiniMind checkout as `cwd`, polling `cancel_token` on a short interval
  and escalating `terminate()` → `kill()` on cooperative cancellation.
  This is the adapter's other significant structural difference from
  TRL: there is no in-process `TrainerCallback` hook to attach to
  (MiniMind's script has none), so cancellation cooperation happens at
  the OS-process level instead of inside the training loop. The
  contract runner's own OS-level process-isolation SIGKILL backstop
  (ADR-0003) remains the primary enforcement mechanism for both
  adapters, unchanged by this difference.
- **Fully offline, enforced twice**: `prepare()` raises
  `RejectedInputError` unless `budget.network_policy == "offline"`,
  identical to the TRL adapter's posture. `train()` additionally sets
  `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` in the subprocess
  environment as defence in depth, even though MiniMind's local-file SFT
  path has no required Hugging Face Hub dependency — matching the TRL
  adapter's belt-and-braces approach rather than assuming MiniMind's
  script never touches any HF-ecosystem code path.
- **Hash-verified provenance, not name-trusted**: identical
  `_hash_path_identity()` algorithm and posture to
  `trl_adapter.py`'s — `prepare()` computes the actual content hash of
  `model_path`/`dataset_path` and raises `RejectedInputError` on any
  mismatch against `TrainingInputs.model_hash`/`dataset_hash`.
- **Adapter-enforced stopping bound**: `training_params` must include a
  positive int `max_steps` *or* a positive int `epochs` — MiniMind's own
  `train_full_sft.py` exposes `--epochs` natively rather than a step
  count, so this adapter accepts either, but `prepare()` rejects the
  absence of both. This is deliberately in addition to (not a
  replacement for) the contract's own OS-measured wall/CPU budget
  enforcement, mirroring the rationale in ADR-0005's equivalent
  `max_steps` requirement.
- **Checkpoint/resume**: on cooperative cancellation, whatever MiniMind
  itself had already flushed to `--save_dir` at its own `--save_interval`
  cadence is hashed and returned as a `CheckpointHandle`. This is
  weaker than the TRL adapter's checkpoint story: TRL's in-process hook
  can force a clean `save_state()` at the exact moment of cancellation;
  MiniMind's subprocess can only be asked to stop and then whatever it
  had already written on disk is what gets preserved — there is no
  atomic "save now" MiniMind exposes. This is disclosed explicitly here
  rather than implied to be equivalent to the TRL adapter's guarantee.
- **Resource usage self-report is a placeholder for wall/cpu/memory**,
  identical rationale to ADR-0005: `run_trainer_contract` (v1.1,
  ADR-0003) always overwrites those three fields with OS-measured
  figures. `storage_mb_used` and `gpu_count_used` remain genuinely
  adapter-self-reported.
- **Concurrency**: exactly one concurrent run, same caller-level
  discipline as the TRL adapter and the pilot plan (ADR-0002/ADR-0006)
  already require; this adapter does not itself enforce concurrency.

### Reuse of fake-adapter-style contract testing, and its explicit limit

`tests/test_minimind_adapter.py` (37 tests) reuses the pattern
established by `tests/test_trainer_contract.py` and
`tests/test_trl_adapter.py` — build `TrainingInputs`/`ResourceBudget`,
call either `adapter.prepare()` directly or the full
`run_trainer_contract()`, assert the resulting status/exception — for
every scenario provable **without an actual MiniMind subprocess
training run**:

- upstream/contract-version gating (the generic pre-`prepare()` check);
- the pin-verification `git rev-parse` path itself: matching commit,
  mismatched commit, missing checkout, non-git directory, pinned commit
  present but missing the expected `trainer/train_full_sft.py` layout;
- offline-network-policy enforcement, both at `prepare()` directly and
  through the full contract runner;
- missing/invalid `training_params` (`minimind_repo_path`, `model_path`,
  `dataset_path` absent; neither `max_steps` nor `epochs` provided;
  `epochs`-only accepted as valid);
- nonexistent local paths;
- model/dataset hash mismatch (provenance/tamper), both at `prepare()`
  directly and through the full contract runner;
- `train()`'s subprocess argument construction, non-zero-exit handling,
  and cancellation-triggered `terminate()` — all against a **mocked**
  `subprocess.Popen`, never a real `Popen` call, let alone a real
  MiniMind process;
- the `_hash_path_identity()` helper's determinism and
  content-sensitivity;
- `cleanup()` idempotency;
- the adapter-independent `TrainingInputs.validate()` precondition still
  applying to MiniMind-adapter inputs.

Every MiniMind "checkout" used by these tests is a throwaway local git
repository created inside `tmp_path` by real `git init`/`git commit`
calls against a fake `trainer/train_full_sft.py` stub file — never a
clone of the real MiniMind repository, and no test in this file performs
any network access. Because a freshly created local commit can never
equal the real 40-character pin, tests that need to exercise the
"pin matches" path monkeypatch the module-level `MINIMIND_PINNED_COMMIT`
down to the fixture repo's own real (freshly created) `HEAD` commit,
rather than trying to forge or fetch a commit whose SHA equals the real
pin.

**Explicit, disclosed limit — mirroring ADR-0005's own**: this pattern
is not reused for scenarios that would require an actual MiniMind
subprocess training run to complete — success against a real training
loop, a real timeout, a real resource-overrun, or an evidence-hash
tamper check against genuinely trained output. `train()` is fully
implemented (real `subprocess.Popen` invocation, cancellation polling,
evidence write, checkpoint collection) and reviewed, but this work
package never calls it against a real MiniMind checkout — not in a
test, not in CI, not interactively. That remains for a separately
gated real pilot to exercise for the first time, exactly as ADR-0005
established for the TRL adapter.

### Evidence

- Pin currency: `git ls-remote https://github.com/jingyaogong/minimind
  master` → `cc312c1cc614bc371cd85dcbcbc1d3ba1590f364` (matches
  `MINIMIND_PINNED_COMMIT` exactly; re-run at PR time, not reused from
  prior research without re-verification).
- Licence: GitHub repository-licence API for `jingyaogong/minimind`
  reports `{"key": "apache-2.0", "name": "Apache License 2.0", ...}`;
  `LICENSE` fetched directly from
  `raw.githubusercontent.com/jingyaogong/minimind/<pinned-commit>/LICENSE`
  is the standard Apache License 2.0 text.
- `python -m pytest` (fresh clone, fresh `.venv`, Python 3.9.6, `pip
  install -e ".[dev]"` only — no MiniMind checkout, no `trl`, no `torch`
  installed): **226 passed, 25 skipped**, 0 failed. The 37 new tests in
  `tests/test_minimind_adapter.py` are all in the 226 passed; the 25
  skips are pre-existing, unrelated to this change (tests requiring
  `trl`/`torch` importable via `pytest.importorskip`, none of which this
  work package touches).
- `python -m ruff check .`: **All checks passed!**
- No training happened as a result of this work: `adapter.train()` is
  never called against a real MiniMind checkout by any test, script, or
  CI step added or touched by this PR — every `train()`-adjacent test
  patches `subprocess.Popen`.

## Consequences

- CodeVolt MDF now has two real, reviewable (neither yet run) engine
  adapters alongside the deterministic fake adapter: `TRLTrainerAdapter`
  (in-process library call, importable-version pin) and
  `MiniMindTrainerAdapter` (subprocess/CLI call, git-commit pin). The
  contract (`TrainerAdapterV1`) has now been proven to accommodate both
  integration shapes without a contract-version bump.
- No new hard or optional dependency is added to `pyproject.toml` —
  this adapter's only external dependency (a MiniMind git checkout) is
  supplied by the caller at run time via `training_params`, not
  installed by this package.
- Real pilot execution remains blocked for both adapters, unchanged from
  ADR-0005/ADR-0006: independent evaluation against a hidden held-out
  set and Maya's engine-specific live-execution security review are
  still required before any real training run, for MiniMind exactly as
  for TRL. This ADR does not authorise scheduling a MiniMind pilot.
- Adding a third training method to either adapter (e.g. MiniMind's
  LoRA/RLHF scripts, or a distributed-training path for either engine),
  or widening the MiniMind pin past `cc312c1cc614bc371cd85dcbcbc1d3ba1590f364`,
  are separate, not-yet-authorised follow-ups, each needing their own
  gated decision.
