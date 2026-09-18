"""TrainerAdapterV1 for MiniMind (github.com/jingyaogong/minimind).

This is the second bounded **real** engine integration, authorised as a
separately gated work package per
``docs/decisions/0010-minimind-trainer-adapter-v1.md`` (issue #36),
alongside the first (TRL, ``docs/decisions/0005-trl-trainer-adapter-v1.md``).

Scope, deliberately narrow, mirroring the TRL adapter's own posture:

- Exactly one training method: MiniMind's full-parameter supervised
  fine-tuning script, ``trainer/train_full_sft.py``, invoked as a
  subprocess. No pretraining, no LoRA/RLHF/distillation scripts MiniMind
  also ships, no distributed/multi-node training.
- Exactly one concurrent run (enforced by the caller, not this module --
  see the ADR, same as the TRL adapter).
- Fully offline: both the MiniMind checkout and the dataset must already
  exist as verified local paths before ``prepare()`` is called. This
  adapter never clones or downloads anything and never talks to a
  model/dataset hub, regardless of ``budget.network_policy`` -- it
  additionally forces common HF offline env vars inside the child
  subprocess as defence in depth (see ``train()``), even though MiniMind
  itself has no required HF Hub dependency for local-file training.
- **No importable version to check.** Unlike TRL (a PyPI package with
  ``trl.__version__``), MiniMind is a script repository with no package
  boundary and no version string. This adapter's ``UpstreamRequirement``
  therefore does not compare semantic versions at all: instead,
  ``prepare()`` runs ``git rev-parse HEAD`` against the caller-supplied
  MiniMind checkout directory and requires it to exactly equal the
  pinned commit SHA this adapter declares. This is a deliberate,
  documented deviation from the TRL adapter's importable-``__version__``
  check -- see the ADR's "Pin verification: subprocess `git rev-parse`,
  not an importable version" section for the full rationale.
- No pinned pilot model is chosen by this module. ``model_revision`` is
  an opaque identifier the caller assigns; ``model_hash`` is verified
  against the *actual* content found at
  ``training_params["model_path"]`` before any training work starts, so
  provenance is checked by content, not by a trusted name -- identical
  posture to the TRL adapter.
- ``train()`` is fully implemented to invoke the real MiniMind CLI via
  ``subprocess.run`` but is **not exercised by any run in this work
  package** -- no real training happens here. See
  ``docs/decisions/0010-minimind-trainer-adapter-v1.md`` ("What is NOT
  covered by this work package") for the exact list of contract
  scenarios (success/timeout/cancellation/resource-overrun/checkpoint-
  resume/tamper) that remain untested against this adapter until a
  separately gated real pilot runs.

This module performs no imports of MiniMind's own Python modules (there
is nothing importable to import -- it is invoked purely as a subprocess),
so it stays importable and runnable in every environment, including one
where no MiniMind checkout exists at all. Every test in
``tests/test_minimind_adapter.py`` uses a local git-repo fixture and/or a
mocked ``subprocess.run`` -- never a real MiniMind checkout, never a real
subprocess training invocation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .trainer_contract import (
    CancellationToken,
    CheckpointError,
    CheckpointHandle,
    InvalidInputError,
    RejectedInputError,
    ResourceBudget,
    ResourceUsage,
    TrainingInputs,
    TrainingOutput,
    TrainingStatus,
    UpstreamRequirement,
)

# --------------------------------------------------------------------------
# Upstream compatibility pin.
#
# MiniMind (github.com/jingyaogong/minimind) has no PyPI package and no
# importable ``__version__`` -- it is a script repository. This adapter
# therefore pins an exact commit SHA on the ``master`` branch rather than
# a semantic-version range, and verifies it via ``git rev-parse HEAD``
# against the caller-supplied checkout directory at ``prepare()`` time
# (see ``_detect_minimind_checkout_commit`` below and
# docs/decisions/0010-minimind-trainer-adapter-v1.md, "Pin verification").
# Re-pinning to a newer commit requires re-verifying that
# ``trainer/train_full_sft.py`` still exposes the CLI flags this adapter's
# ``train()`` constructs (see ``_build_subprocess_args``), the same
# evidence bar the TRL adapter applies to its own version pin.
# --------------------------------------------------------------------------
MINIMIND_REPO_URL = "https://github.com/jingyaogong/minimind"
MINIMIND_PINNED_COMMIT = "cc312c1cc614bc371cd85dcbcbc1d3ba1590f364"
MINIMIND_PINNED_BRANCH = "master"

_HASH_MANIFEST_EXCLUDE_NAMES = {".DS_Store", "__pycache__"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_path_identity(path: Path) -> str:
    """Deterministically hash a file or directory's content identity.

    Identical algorithm and rationale to ``trl_adapter._hash_path_identity``:
    a single file is hashed directly; a directory is hashed as a sorted
    JSON manifest of ``(relative_posix_path, size_bytes, sha256)`` for
    every regular file under it (hidden/cache noise excluded), so the
    same directory contents always produce the same hash regardless of
    traversal order.
    """
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise RejectedInputError(f"path {path} does not exist (neither file nor directory)")
    entries: list[tuple[str, int, str]] = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.name in _HASH_MANIFEST_EXCLUDE_NAMES:
            continue
        rel = candidate.relative_to(path).as_posix()
        entries.append((rel, candidate.stat().st_size, _sha256_file(candidate)))
    manifest = json.dumps(entries, sort_keys=True).encode("utf-8")
    return hashlib.sha256(manifest).hexdigest()


def _directory_size_mb(path: Path) -> float:
    total = 0
    if path.is_file():
        return path.stat().st_size / (1024.0 * 1024.0)
    for dirpath, _dirnames, filenames in os.walk(path):
        for filename in filenames:
            candidate = Path(dirpath) / filename
            try:
                total += candidate.stat().st_size
            except OSError:
                continue
    return total / (1024.0 * 1024.0)


def _detect_minimind_checkout_commit(minimind_repo_path: Path) -> str:
    """Return the exact commit SHA checked out at ``minimind_repo_path``.

    Returns ``""`` (never raises) if the path is not a directory, is not
    a git repository, or the ``git`` subprocess fails for any reason --
    deliberately mirroring ``trl_adapter._detect_installed_trl_version``'s
    fail-closed shape (a sentinel that can never accidentally equal a
    real pin, rather than propagating an unhandled exception from inside
    a compatibility check).
    """
    if not minimind_repo_path.is_dir():
        return ""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(minimind_repo_path),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


@dataclass
class MiniMindTrainerAdapter:
    """Real ``TrainerAdapterV1`` implementation backed by MiniMind's SFT CLI.

    ``work_dir`` is the root directory this adapter creates per-run
    working directories under (evidence, checkpoints, final artifacts).

    Required ``inputs.training_params`` keys (validated in ``prepare()``,
    never trusted from ``TrainingInputs.validate()`` since those are
    adapter-independent):

    - ``minimind_repo_path`` (str): local filesystem path to a MiniMind
      git checkout whose ``HEAD`` must equal ``MINIMIND_PINNED_COMMIT``.
      Never a URL -- this adapter does not clone or download MiniMind
      itself.
    - ``model_path`` (str): local filesystem path to a pre-verified model
      checkpoint (file or directory) MiniMind's SFT script can load.
    - ``dataset_path`` (str): local filesystem path to a pre-verified,
      already-admitted SFT dataset in MiniMind's expected JSONL
      conversation format.
    - ``max_steps`` (int, > 0) OR ``epochs`` (int, > 0): MiniMind's own
      ``train_full_sft.py`` exposes ``--epochs`` natively rather than a
      step count; this adapter accepts either but requires at least one
      of them so an adapter-enforced stopping bound always exists,
      independent of the contract's wall-clock budget (mirroring the
      TRL adapter's ``max_steps`` requirement).

    Optional ``training_params`` keys: ``learning_rate`` (float,
    default ``5e-4``, MiniMind's own script default),
    ``batch_size`` (int, default ``1``), ``save_interval`` (int,
    default ``100`` -- how often MiniMind writes a checkpoint file,
    passed through as ``--save_interval``), ``python_executable``
    (str, default ``"python3"`` -- the interpreter used to invoke the
    subprocess, so a caller can point at a specific venv without this
    adapter needing to manage one itself).
    """

    work_dir: Path
    name: str = "minimind-sft-adapter-v1"
    contract_version: str = "1.1.0"
    upstream: UpstreamRequirement = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if self.upstream is None:
            # MiniMind has no semantic version; both bounds are pinned to
            # the same commit SHA string and `installed_version` starts
            # empty ("no checkout known yet"). is_compatible() therefore
            # is trivially False until prepare() re-derives it from the
            # caller-supplied checkout -- see prepare()'s own
            # git-rev-parse check below, which is the real gate. This
            # UpstreamRequirement exists to satisfy the TrainerAdapterV1
            # Protocol's declared-attribute shape and to carry a
            # human-readable pin, not to drive run_trainer_contract's
            # pre-prepare() compatibility gate the way it does for TRL.
            self.upstream = UpstreamRequirement(
                engine_name="minimind",
                min_version=MINIMIND_PINNED_COMMIT,
                max_version=MINIMIND_PINNED_COMMIT,
                installed_version=MINIMIND_PINNED_COMMIT,
            )

    # -- contract methods ---------------------------------------------

    def prepare(self, inputs: TrainingInputs, budget: ResourceBudget) -> None:
        """Validate inputs/budget/policy. Must not perform training work.

        Deliberately performs no MiniMind Python import (there is none)
        and only a cheap, bounded ``git rev-parse`` subprocess call to
        verify the pinned commit -- no network access, no training work.
        """
        if budget.network_policy != "offline":
            raise RejectedInputError(
                "minimind-sft-adapter-v1 requires an offline network policy: the "
                "MiniMind checkout, model, and dataset must already be verified "
                "local paths before this adapter runs; it never clones or "
                "downloads anything itself"
            )

        params = inputs.params
        minimind_repo_raw = params.get("minimind_repo_path")
        model_path_raw = params.get("model_path")
        dataset_path_raw = params.get("dataset_path")
        if not minimind_repo_raw or not isinstance(minimind_repo_raw, str):
            raise InvalidInputError(
                "training_params['minimind_repo_path'] is required and must be a "
                "non-empty string"
            )
        if not model_path_raw or not isinstance(model_path_raw, str):
            raise InvalidInputError(
                "training_params['model_path'] is required and must be a non-empty string"
            )
        if not dataset_path_raw or not isinstance(dataset_path_raw, str):
            raise InvalidInputError(
                "training_params['dataset_path'] is required and must be a non-empty string"
            )

        max_steps = params.get("max_steps")
        epochs = params.get("epochs")
        max_steps_ok = isinstance(max_steps, int) and not isinstance(max_steps, bool) and max_steps > 0
        epochs_ok = isinstance(epochs, int) and not isinstance(epochs, bool) and epochs > 0
        if not max_steps_ok and not epochs_ok:
            raise InvalidInputError(
                "training_params must include a positive int 'max_steps' or a "
                "positive int 'epochs' (this adapter refuses to train without an "
                "adapter-enforced stopping bound)"
            )

        minimind_repo_path = Path(minimind_repo_raw)
        if not minimind_repo_path.exists():
            raise RejectedInputError(f"minimind_repo_path {minimind_repo_path} does not exist locally")

        actual_commit = _detect_minimind_checkout_commit(minimind_repo_path)
        if not actual_commit:
            raise RejectedInputError(
                f"minimind_repo_path {minimind_repo_path} is not a readable git "
                "checkout ('git rev-parse HEAD' failed); this adapter verifies "
                "the MiniMind pin by inspecting the caller's checkout directly, "
                "since MiniMind has no importable version to check"
            )
        if actual_commit != MINIMIND_PINNED_COMMIT:
            raise RejectedInputError(
                f"minimind_repo_path {minimind_repo_path} is checked out at commit "
                f"{actual_commit}, but this adapter is pinned to "
                f"{MINIMIND_PINNED_COMMIT}; refusing to train against an "
                "unverified MiniMind revision"
            )

        script_path = minimind_repo_path / "trainer" / "train_full_sft.py"
        if not script_path.exists():
            raise RejectedInputError(
                f"expected MiniMind training script {script_path} does not exist "
                "in this checkout (pinned commit verified, but the expected "
                "layout is missing -- checkout may be corrupt or sparse)"
            )

        model_path = Path(model_path_raw)
        dataset_path = Path(dataset_path_raw)
        if not model_path.exists():
            raise RejectedInputError(f"model_path {model_path} does not exist locally")
        if not dataset_path.exists():
            raise RejectedInputError(f"dataset_path {dataset_path} does not exist locally")

        actual_model_hash = _hash_path_identity(model_path)
        if actual_model_hash != inputs.model_hash:
            raise RejectedInputError(
                f"model_hash mismatch: declared {inputs.model_hash}, "
                f"actual content at {model_path} hashes to {actual_model_hash} "
                "(refusing to train against an unverified/tampered checkpoint)"
            )
        actual_dataset_hash = _hash_path_identity(dataset_path)
        if actual_dataset_hash != inputs.dataset_hash:
            raise RejectedInputError(
                f"dataset_hash mismatch: declared {inputs.dataset_hash}, "
                f"actual content at {dataset_path} hashes to {actual_dataset_hash} "
                "(refusing to train against an unverified/tampered dataset)"
            )

    def train(
        self,
        inputs: TrainingInputs,
        budget: ResourceBudget,
        cancel_token: CancellationToken,
        resume_from: CheckpointHandle | None = None,
    ) -> TrainingOutput:
        """Run (or resume) a bounded MiniMind SFT subprocess run.

        NOT exercised by this work package -- implemented against the
        real ``trainer/train_full_sft.py`` CLI (invoked via
        ``subprocess.run``, never imported), so the code is complete and
        reviewable, but no test or CI run in this repository calls this
        method. See the module docstring and
        ``docs/decisions/0010-minimind-trainer-adapter-v1.md`` for
        exactly what remains unexercised and why.

        Unlike TRL's in-process ``SFTTrainer``, MiniMind's own cancellation
        story is coarse: the child subprocess is asked to stop by polling
        ``cancel_token`` on a short interval and terminating the
        subprocess (``Popen.terminate()``, escalating to ``kill()``) if it
        is set, rather than a cooperative per-step callback hook inside
        the training loop itself (MiniMind's script has no such hook to
        attach to). The contract runner's own OS-level process-isolation
        SIGKILL backstop (``docs/decisions/0003-...``) remains the primary
        enforcement mechanism for both adapters; this is adapter-level
        best-effort cooperation on top of it, same role the TRL adapter's
        ``TrainerCallback`` plays, just via process signals instead of an
        in-process callback because there is no in-process hook available.
        """
        # Defence in depth: force offline mode inside this subprocess
        # regardless of ambient environment, on top of process_isolation's
        # own environment-allowlist stripping and prepare()'s network_policy
        # check above. MiniMind has no required Hub dependency for local
        # training, but these are harmless to set and match the TRL
        # adapter's posture if any optional HF-ecosystem code path is
        # reached (e.g. a tokenizer helper).
        env = dict(os.environ)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

        params = inputs.params
        minimind_repo_path = Path(params["minimind_repo_path"])
        run_dir = self._run_dir(inputs.run_id)
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        script_path = minimind_repo_path / "trainer" / "train_full_sft.py"
        python_executable = params.get("python_executable", "python3")
        args = self._build_subprocess_args(
            python_executable=python_executable,
            script_path=script_path,
            params=params,
            checkpoint_dir=checkpoint_dir,
            seed=inputs.seed,
        )

        # Run cwd: a per-run "shadow trainer directory" inside this run's
        # OWN work_dir, not <repo>/trainer directly, so the two relative
        # paths trainer/train_full_sft.py's real code hardcodes against its
        # cwd -- init_model()'s tokenizer_path='../model' default and
        # train_epoch()'s second, CLI-flag-independent
        # lm_checkpoint(..., save_dir='../checkpoints') call -- both resolve
        # to locations this adapter controls, not to the checkout itself.
        #
        # Why this matters (found and empirically verified fixing issue
        # #46/ADR-0011's Maya-reported secondary defect, internal tracking item
        #): once the cwd is corrected to <repo>/trainer, MiniMind's
        # own train_epoch() unconditionally also calls
        # lm_checkpoint(..., save_dir='../checkpoints') at least once per
        # epoch (`step == iters` always triggers it) -- a second checkpoint
        # write path with no CLI flag, entirely independent of --save_dir,
        # and NOT contained by process_isolation._pin_filesystem_root's
        # open()/os.open() write guard: that guard only patches builtins in
        # the immediate process_isolation child interpreter, and this write
        # happens inside a *separate OS subprocess* (MiniMind's own Python
        # interpreter, spawned via subprocess.Popen) whose own builtins/os
        # module were never patched. Direct reproduction against a genuinely
        # writable pinned checkout confirmed the escape is real, silent, and
        # uncaught: the checkpoint (including raw model + optimizer state)
        # was written to <repo>/checkpoints/ with TrainingOutput.status
        # still ACCEPTED and no exception raised anywhere -- this is worse
        # than the "no real exfiltration path exists" read from a read-only
        # checkout's incidental PermissionError, and must not be relied on
        # as the containment mechanism.
        #
        # This shadow directory closes the gap structurally, independent of
        # whether the caller's checkout happens to be read-only. It MUST use
        # real directory copies of trainer/ and model/ (never symlinks):
        # POSIX resolves ".." against the real parent directory of the
        # target a symlink points at, not against the symlink's own
        # location -- a symlinked shadow_trainer_dir would still make
        # '../checkpoints' resolve back into the real checkout's parent,
        # completely defeating the containment. This was verified directly:
        # a symlink-based version of this fix still wrote checkpoints into
        # the real checkout. trainer/ and model/ are both small
        # (script/tokenizer files, not model weights -- a few hundred KB
        # combined for the pinned commit), so copying them fresh per run is
        # cheap. Reads inside the shadow trainer/ dir (e.g. `from
        # trainer.trainer_utils import ...`, `sys.path.append('..')`
        # patterns) still work identically because the copy is
        # byte-for-byte identical Python source, not a stub.
        # NOTE (live-execution verification): the shadow copy
        # must include every sibling package trainer/train_full_sft.py's
        # own module-level imports resolve relative to the checkout root
        # -- not just trainer/ and model/. A real subprocess run against
        # the pinned commit's trainer/train_full_sft.py failed with
        # ModuleNotFoundError: No module named 'dataset' until dataset/
        # (containing lm_dataset.py, the source of SFTDataset/
        # PretrainDataset/etc.) was copied alongside trainer/ and model/
        # -- train_full_sft.py's own import block reads
        # ``from dataset.lm_dataset import SFTDataset``, a sibling-package
        # import resolved against cwd, exactly like ``from model...`` and
        # ``from trainer.trainer_utils import ...``. If a future re-pin
        # adds further top-level sibling packages to train_full_sft.py's
        # (or trainer_utils.py's) import block, they must be added here
        # too -- this is not auto-discovered.
        shadow_dir = run_dir / "mm_shadow"
        shadow_trainer_dir = shadow_dir / "trainer"
        shadow_model_dir = shadow_dir / "model"
        shadow_dataset_dir = shadow_dir / "dataset"
        shadow_checkpoints_dir = shadow_dir / "checkpoints"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        shadow_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        if not shadow_trainer_dir.exists():
            shutil.copytree(minimind_repo_path / "trainer", shadow_trainer_dir)
        if not shadow_model_dir.exists() and (minimind_repo_path / "model").exists():
            shutil.copytree(minimind_repo_path / "model", shadow_model_dir)
        if not shadow_dataset_dir.exists() and (minimind_repo_path / "dataset").exists():
            shutil.copytree(minimind_repo_path / "dataset", shadow_dataset_dir)

        process = subprocess.Popen(
            args,
            cwd=str(shadow_trainer_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        cancelled = False
        poll_interval = 0.5
        while True:
            try:
                process.wait(timeout=poll_interval)
                break
            except subprocess.TimeoutExpired:
                if cancel_token.is_cancelled():
                    cancelled = True
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break

        stdout_text = process.stdout.read() if process.stdout else ""
        (run_dir / "subprocess_output.log").write_text(stdout_text)

        if cancelled:
            return self._safe_halt_output(inputs, checkpoint_dir)

        if process.returncode != 0:
            return TrainingOutput(
                status=TrainingStatus.INTERRUPTED,
                reason=(
                    f"minimind train_full_sft.py exited with code "
                    f"{process.returncode}; see {run_dir / 'subprocess_output.log'}"
                ),
                error_class="MiniMindSubprocessError",
            )

        final_dir = run_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        self._collect_final_artifact(checkpoint_dir, final_dir)
        evidence_locator, evidence_hash = self._write_evidence(inputs, final_dir, stdout_text)

        shutil.rmtree(checkpoint_dir, ignore_errors=True)

        usage = ResourceUsage(
            # wall/cpu/memory are overwritten by the contract runner with
            # OS-measured figures (see trainer_contract.run_trainer_contract);
            # these self-reported placeholders are never trusted for those
            # three dimensions, only for storage/gpu below -- identical
            # posture to the TRL adapter.
            wall_seconds=0.0,
            cpu_seconds=0.0,
            memory_mb_peak=0.0,
            gpu_count_used=self._gpu_count_used(),
            storage_mb_used=_directory_size_mb(run_dir),
        )
        return TrainingOutput(
            status=TrainingStatus.ACCEPTED,
            reason="minimind train_full_sft.py subprocess completed",
            artifact_id=f"artifact-{inputs.run_id}",
            evidence_locator=evidence_locator,
            evidence_hash=evidence_hash,
            resource_usage=usage,
        )

    def cleanup(self, run_id: str) -> None:
        run_dir = self.work_dir / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

    # -- helpers ---------------------------------------------------------

    def _run_dir(self, run_id: str) -> Path:
        run_dir = self.work_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _build_subprocess_args(
        self,
        *,
        python_executable: str,
        script_path: Path,
        params: dict[str, Any],
        checkpoint_dir: Path,
        seed: int,
    ) -> list[str]:
        """Build the real ``trainer/train_full_sft.py`` CLI invocation.

        Flag names below are verified directly against the pinned commit's
        argparse block (``MINIMIND_PINNED_COMMIT``, see
        ``tests/test_minimind_trainer_cli_flags.py`` for the static
        regression guard). Do not add a flag here without re-verifying it
        exists at the currently pinned commit -- see issue #47, where
        ``--out_dir``, ``--model_path``, and ``--max_steps`` were
        previously constructed despite not existing in the real script.

        Two of this adapter's own concepts have no exact 1:1 real flag:

        - ``model_path`` (a caller-verified local file path) is passed as
          ``--from_weight``, MiniMind's own "which weight to start
          training from" flag. Note this is a *name/prefix* MiniMind
          resolves against ``--save_dir`` internally
          (``{save_dir}/{from_weight}_{hidden_size}.pth``), not a literal
          path passed through verbatim -- a known adapter limitation
          tracked separately from this flag-name fix.
        - ``max_steps`` has no real MiniMind CLI equivalent (the script
          only supports ``--epochs``). It remains a valid adapter-level
          stopping-bound input (validated in ``prepare()``) but is not
          forwarded as a nonexistent CLI flag; when only ``max_steps`` is
          supplied (no ``epochs``), the adapter-enforced bound is
          upheld by the contract runner's wall-clock/cancellation
          machinery rather than a MiniMind subprocess flag.
        """
        args = [python_executable, str(script_path)]
        args += ["--save_dir", str(checkpoint_dir)]
        args += ["--data_path", str(params["dataset_path"])]
        args += ["--seed", str(seed)]
        if params.get("model_path"):
            args += ["--from_weight", str(params["model_path"])]
        if isinstance(params.get("epochs"), int):
            args += ["--epochs", str(params["epochs"])]
        args += ["--learning_rate", str(params.get("learning_rate", 5e-4))]
        args += ["--batch_size", str(params.get("batch_size", 1))]
        args += ["--save_interval", str(params.get("save_interval", 100))]
        return args

    def _gpu_count_used(self) -> int:
        try:
            import torch
        except ImportError:
            return 0
        try:
            return torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:  # noqa: BLE001 - GPU probing must never crash a run
            return 0

    def _verify_checkpoint(self, checkpoint: CheckpointHandle) -> None:
        state_path = Path(checkpoint.state_locator)
        if not state_path.exists():
            raise CheckpointError(f"checkpoint state_locator {state_path} does not exist")
        actual_hash = _hash_path_identity(state_path)
        if actual_hash != checkpoint.state_hash:
            raise InvalidInputError(
                f"checkpoint state hash mismatch on resume: declared {checkpoint.state_hash}, "
                f"actual {actual_hash} (tamper or corruption detected)"
            )

    def _collect_final_artifact(self, checkpoint_dir: Path, final_dir: Path) -> None:
        """Copy MiniMind's own checkpoint output into this run's final dir.

        MiniMind's ``train_full_sft.py`` writes its own checkpoint
        file(s) directly into ``--save_dir`` (no separate "final model"
        save step the way TRL's ``Trainer.save_model`` provides); this
        simply relocates whatever was produced into the run's
        ``final/`` directory so cleanup of ``checkpoint_dir`` below does
        not delete the deliverable.
        """
        if not checkpoint_dir.exists():
            return
        for item in checkpoint_dir.iterdir():
            destination = final_dir / item.name
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)

    def _safe_halt_output(self, inputs: TrainingInputs, checkpoint_dir: Path) -> TrainingOutput:
        """Persist whatever MiniMind wrote to ``checkpoint_dir`` after cancellation.

        Unlike the TRL adapter, MiniMind's script does not expose a
        cooperative in-loop hook, so "resumable state" here is exactly
        the last checkpoint file(s) MiniMind itself had flushed to disk
        at ``--save_interval`` before the subprocess was terminated --
        not a guaranteed atomic snapshot at the moment of cancellation.
        """
        if not any(checkpoint_dir.iterdir()) if checkpoint_dir.exists() else True:
            return TrainingOutput(
                status=TrainingStatus.INTERRUPTED,
                reason="cancelled before any checkpoint was written; nothing to resume",
            )
        state_hash = _hash_path_identity(checkpoint_dir)
        checkpoint = CheckpointHandle(
            checkpoint_id=f"ckpt-{inputs.run_id}",
            step=0,
            state_locator=str(checkpoint_dir),
            state_hash=state_hash,
        )
        return TrainingOutput(
            status=TrainingStatus.INTERRUPTED,
            reason="cancelled cooperatively; last MiniMind checkpoint on disk preserved",
            checkpoint=checkpoint,
        )

    def _write_evidence(
        self, inputs: TrainingInputs, final_dir: Path, stdout_text: str
    ) -> tuple[str, str]:
        run_dir = self._run_dir(inputs.run_id)
        evidence_path = run_dir / "evidence.json"
        payload = {
            "run_id": inputs.run_id,
            "model_revision": inputs.model_revision,
            "dataset_version": inputs.dataset_version,
            "seed": inputs.seed,
            "final_dir": str(final_dir),
            "minimind_pinned_commit": MINIMIND_PINNED_COMMIT,
            "subprocess_output_tail": stdout_text[-4000:],
        }
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        evidence_path.write_bytes(raw)
        return str(evidence_path), hashlib.sha256(raw).hexdigest()
