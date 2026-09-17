"""TrainerAdapterV1 for TRL (Hugging Face Transformer Reinforcement Learning).

This is the one bounded **real** engine integration documented (not
executed) by ``docs/TRAINER_ADAPTER_CONTRACT.md``'s "Real bounded pilot
plan" and authorised as a separately gated work package per
``docs/decisions/0005-trl-trainer-adapter-v1.md``.

Scope, deliberately narrow:

- Exactly one training method: supervised fine-tuning via TRL's
  ``SFTTrainer``/``SFTConfig``. No RL trainers (PPO/GRPO/DPO/...), no
  distributed/multi-node training, no vLLM integration.
- Exactly one concurrent run (enforced by the caller, not this module --
  see the ADR).
- Fully offline: both the model and the dataset must already exist as
  verified local paths before ``prepare()`` is called. This adapter never
  downloads anything and never talks to a model/dataset hub, regardless
  of ``budget.network_policy`` -- it additionally forces
  ``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE`` inside the isolated child
  process as defence in depth (see ``train()``).
- No pinned pilot model is chosen by this module. ``model_revision`` is
  an opaque identifier the caller assigns; ``model_hash`` is verified
  against the *actual* content found at ``training_params["model_path"]``
  before any training work starts, so provenance is checked by content,
  not by a trusted name.
- ``train()`` is fully implemented against real TRL/transformers APIs but
  is **not exercised by any run in this work package** -- no real
  training happens here. See ``docs/decisions/0005-trl-trainer-adapter-v1.md``
  ("What is NOT covered by this work package") for the exact list of
  contract scenarios (success/timeout/cancellation/resource-overrun/
  checkpoint-resume/tamper) that remain untested against this adapter
  until the separately gated real pilot runs.

Imports of ``trl``/``transformers``/``datasets``/``torch`` are lazy
(inside methods, not at module import time) so this module -- and every
test that only exercises ``prepare()``-level validation or the contract's
pre-``train()`` rejection paths -- stays importable and runnable even in
an environment where those (large, optional) dependencies are not
installed. Where a test genuinely needs TRL importable to make an
assertion about the real library (e.g. checking ``trl.__version__``
falls inside the declared compatibility bound), it uses
``pytest.importorskip("trl")`` rather than failing hard.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
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
# Upstream compatibility bounds.
#
# Chosen deliberately narrow and pinned to versions this repository has
# actually installed and import-checked (see
# docs/decisions/0005-trl-trainer-adapter-v1.md, "Evidence"). Upper bound
# stops at the last TRL release that still supports Python 3.9
# (this project's pyproject.toml floor, `requires-python = ">=3.9"`):
# TRL 0.25.0 raises its own floor to Python >=3.10. Raising this bound to
# track a newer TRL release requires either (a) a new ADR revisiting this
# bound once this project's own Python floor moves to >=3.10, or (b) a new
# ADR re-verifying a specific newer 0.9-series-compatible release against
# Python 3.9 if TRL ever backports one (unlikely upstream policy).
# --------------------------------------------------------------------------
TRL_MIN_VERSION = "0.20.0"
TRL_MAX_VERSION = "0.24.0"

_HASH_MANIFEST_EXCLUDE_NAMES = {".DS_Store", "__pycache__"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_path_identity(path: Path) -> str:
    """Deterministically hash a file or directory's content identity.

    A single file is hashed directly. A directory is hashed as a sorted
    JSON manifest of ``(relative_posix_path, size_bytes, sha256)`` for
    every regular file under it (hidden/cache noise excluded), so the
    same directory contents always produce the same hash regardless of
    traversal order, and any content change -- including a change to a
    file nobody thought to check -- changes the hash. This is an
    adapter-specific interpretation of ``TrainingInputs.model_hash`` /
    ``dataset_hash`` ("content hash ... checkpoint"/"dataset"): the
    generic contract does not mandate single-file hashing, and a real HF
    model checkpoint is normally several files (config, tokenizer,
    weight shards).
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


def _detect_installed_trl_version() -> str:
    """Return the installed ``trl`` version, or ``"0.0.0"`` if unimportable.

    ``"0.0.0"`` deliberately sorts below every real bound so
    ``UpstreamRequirement.is_compatible()`` fails closed (rejects) rather
    than raising an unhandled ``ImportError`` from inside the contract
    runner's compatibility check when TRL is simply not installed in the
    current environment -- a normal, expected state for most CI/dev
    environments that do not opt into the ``trl-adapter`` extra.
    """
    try:
        import trl
    except ImportError:
        return "0.0.0"
    return getattr(trl, "__version__", "0.0.0")


@dataclass
class TRLTrainerAdapter:
    """Real ``TrainerAdapterV1`` implementation backed by TRL's ``SFTTrainer``.

    ``work_dir`` is the root directory this adapter creates per-run
    working directories under (evidence, checkpoints, final artifacts).

    Required ``inputs.training_params`` keys (validated in ``prepare()``,
    never trusted from ``TrainingInputs.validate()`` since those are
    adapter-independent):

    - ``model_path`` (str): local filesystem path to a pre-verified model
      checkpoint (file or directory). Never a Hub id -- this adapter does
      not resolve or download models.
    - ``dataset_path`` (str): local filesystem path to a pre-verified,
      already-admitted (synthetic or explicitly reviewed) SFT dataset in
      a format ``datasets.load_dataset("json", ...)`` can read. Never a
      Hub id.
    - ``max_steps`` (int, > 0): hard upper bound on training steps. This
      adapter refuses to train without one -- an unbounded
      ``num_train_epochs``-only run has no adapter-enforced stopping
      point independent of the contract's wall-clock budget, and the
      bounded-pilot plan requires an explicit step bound as well as a
      resource budget.

    Optional ``training_params`` keys: ``learning_rate`` (float,
    default ``2e-5``), ``per_device_train_batch_size`` (int, default
    ``1``), ``save_steps`` (int, default ``max(1, max_steps // 4)`` --
    how often a resumable checkpoint is written), ``use_lora`` (bool,
    default ``False`` -- requires the optional ``peft`` package; raises
    ``RejectedInputError`` in ``prepare()`` if requested but ``peft`` is
    not importable, rather than silently falling back to full
    fine-tuning).
    """

    work_dir: Path
    name: str = "trl-sft-adapter-v1"
    contract_version: str = "1.1.0"
    upstream: UpstreamRequirement = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if self.upstream is None:
            self.upstream = UpstreamRequirement(
                engine_name="trl",
                min_version=TRL_MIN_VERSION,
                max_version=TRL_MAX_VERSION,
                installed_version=_detect_installed_trl_version(),
            )

    # -- contract methods ---------------------------------------------

    def prepare(self, inputs: TrainingInputs, budget: ResourceBudget) -> None:
        """Validate inputs/budget/policy. Must not perform training work.

        Deliberately does not import ``trl``/``transformers``/``torch``:
        every check here is filesystem- and metadata-level, so
        ``prepare()`` stays cheap and safe to call even when the real
        training dependencies are not installed and the run is destined
        to be rejected anyway (e.g. by the contract runner's upstream
        compatibility gate, which runs before ``prepare()``).
        """
        if budget.network_policy != "offline":
            raise RejectedInputError(
                "trl-sft-adapter-v1 requires an offline network policy: model and "
                "dataset must already be verified local paths before this adapter "
                "runs; it never downloads anything itself"
            )

        params = inputs.params
        model_path_raw = params.get("model_path")
        dataset_path_raw = params.get("dataset_path")
        if not model_path_raw or not isinstance(model_path_raw, str):
            raise InvalidInputError(
                "training_params['model_path'] is required and must be a non-empty string"
            )
        if not dataset_path_raw or not isinstance(dataset_path_raw, str):
            raise InvalidInputError(
                "training_params['dataset_path'] is required and must be a non-empty string"
            )
        max_steps = params.get("max_steps")
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise InvalidInputError(
                "training_params['max_steps'] is required and must be a positive int "
                "(this adapter refuses to train without an adapter-enforced step bound)"
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

        if params.get("use_lora"):
            try:
                import peft  # noqa: F401 - lazy, optional dependency
            except ImportError as exc:
                raise RejectedInputError(
                    "training_params['use_lora'] is True but the optional 'peft' "
                    "package is not importable in this environment"
                ) from exc

    def train(
        self,
        inputs: TrainingInputs,
        budget: ResourceBudget,
        cancel_token: CancellationToken,
        resume_from: CheckpointHandle | None = None,
    ) -> TrainingOutput:
        """Run (or resume) a bounded TRL SFT run. NOT exercised by this work package.

        Implemented against real TRL/transformers APIs (lazy-imported
        here, not at module scope) so the code is complete and
        reviewable, but no test or CI run in this repository calls this
        method -- see the module docstring and
        ``docs/decisions/0005-trl-trainer-adapter-v1.md`` for exactly
        what remains unexercised and why, and the bounded-pilot gating
        conditions that must clear first.
        """
        # Defence in depth: force offline mode inside this process
        # regardless of ambient environment, on top of process_isolation's
        # own environment-allowlist stripping and prepare()'s network_policy
        # check above.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from datasets import load_dataset
        from transformers import TrainerCallback, TrainerControl, TrainerState
        from trl import SFTConfig, SFTTrainer

        params = inputs.params
        run_dir = self._run_dir(inputs.run_id)
        checkpoint_dir = run_dir / "checkpoints"
        final_dir = run_dir / "final"

        max_steps = int(params["max_steps"])
        save_steps = int(params.get("save_steps", max(1, max_steps // 4)))

        dataset = load_dataset("json", data_files=params["dataset_path"], split="train")

        sft_config = SFTConfig(
            output_dir=str(checkpoint_dir),
            max_steps=max_steps,
            save_steps=save_steps,
            save_strategy="steps",
            learning_rate=float(params.get("learning_rate", 2e-5)),
            per_device_train_batch_size=int(params.get("per_device_train_batch_size", 1)),
            seed=inputs.seed,
            report_to=[],
            logging_steps=max(1, save_steps // 2),
        )

        cancelled = {"flag": False}

        class _CancellationCallback(TrainerCallback):  # type: ignore[misc]
            def on_step_end(
                self,
                args: Any,
                state: TrainerState,
                control: TrainerControl,
                **kwargs: Any,
            ) -> TrainerControl:
                if cancel_token.is_cancelled():
                    cancelled["flag"] = True
                    control.should_training_stop = True
                return control

        peft_config = None
        if params.get("use_lora"):
            from peft import LoraConfig

            lora_params = params.get("lora_params", {})
            peft_config = LoraConfig(**lora_params) if lora_params else LoraConfig()

        trainer = SFTTrainer(
            model=params["model_path"],
            args=sft_config,
            train_dataset=dataset,
            peft_config=peft_config,
            callbacks=[_CancellationCallback()],
        )

        resume_arg: str | None = None
        if resume_from is not None:
            self._verify_checkpoint(resume_from)
            resume_arg = resume_from.state_locator

        trainer.train(resume_from_checkpoint=resume_arg)

        if cancelled["flag"]:
            return self._safe_halt_output(inputs, trainer, checkpoint_dir)

        trainer.save_model(str(final_dir))
        evidence_locator, evidence_hash = self._write_evidence(inputs, trainer, final_dir)
        usage = ResourceUsage(
            # wall/cpu/memory are overwritten by the contract runner with
            # OS-measured figures (see trainer_contract.run_trainer_contract);
            # these self-reported placeholders are never trusted for those
            # three dimensions, only for storage/gpu below.
            wall_seconds=0.0,
            cpu_seconds=0.0,
            memory_mb_peak=0.0,
            gpu_count_used=self._gpu_count_used(),
            storage_mb_used=_directory_size_mb(run_dir),
        )
        return TrainingOutput(
            status=TrainingStatus.ACCEPTED,
            reason=f"trl SFT run completed {max_steps} max_steps",
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

    def _safe_halt_output(self, inputs: TrainingInputs, trainer: Any, checkpoint_dir: Path) -> TrainingOutput:
        trainer.save_model(str(checkpoint_dir))
        state_hash = _hash_path_identity(checkpoint_dir)
        step = int(getattr(trainer.state, "global_step", 0))
        checkpoint = CheckpointHandle(
            checkpoint_id=f"ckpt-{inputs.run_id}-{step}",
            step=step,
            state_locator=str(checkpoint_dir),
            state_hash=state_hash,
        )
        return TrainingOutput(
            status=TrainingStatus.INTERRUPTED,
            reason=f"cancelled cooperatively after step {step}; checkpoint saved",
            checkpoint=checkpoint,
        )

    def _write_evidence(self, inputs: TrainingInputs, trainer: Any, final_dir: Path) -> tuple[str, str]:
        run_dir = self._run_dir(inputs.run_id)
        evidence_path = run_dir / "evidence.json"
        log_history = list(getattr(trainer.state, "log_history", []))
        payload = {
            "run_id": inputs.run_id,
            "model_revision": inputs.model_revision,
            "dataset_version": inputs.dataset_version,
            "seed": inputs.seed,
            "final_dir": str(final_dir),
            "global_step": int(getattr(trainer.state, "global_step", 0)),
            "log_history": log_history,
        }
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        evidence_path.write_bytes(raw)
        return str(evidence_path), hashlib.sha256(raw).hexdigest()
