"""DPOTrainerAdapterV1 for TRL (Hugging Face Transformer Reinforcement Learning).

This is the DPO-capable sibling of ``codevolt_mdf.trl_adapter.TRLTrainerAdapter``,
proposed by ADR-0017 (`docs/decisions/ADR-0017-dpo-preference-refusal-axis.md`,
Decision item 1 and item 4) as the "new adapter code required" for a scoped
DPO/preference-training cycle on the confabulated-recipe refusal axis.

Design/dataset-shape proposal only. Like ``trl_adapter.py`` before its own
first real pilot, ``train()`` here is fully implemented against real
TRL/transformers APIs but is **not exercised by any run in this work
package** -- no real training happens here. ADR-0017 authorizes drafting
this code as a separately gated PR; it does not authorize merging or
executing it (see ADR-0017's "What this ADR does and does not authorize").

Reuses ``TRLTrainerAdapter``'s established contract patterns rather than
inventing new ones, per ADR-0017 Decision item 1 ("New adapter code in
``src/codevolt_mdf/trl_adapter.py`` (or a clearly-scoped sibling module
reusing its contract)"):

- Same lazy-import discipline (``trl``/``transformers``/``datasets``/``torch``
  imported inside methods, never at module scope) so this module and its
  ``prepare()``-level tests stay importable without the optional
  ``trl-adapter`` extra installed.
- Same ``model_hash``/offline-enforcement pattern (``_hash_path_identity``,
  ``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE`` forced inside the isolated
  child process) -- extended here to a SECOND path, the reference model,
  per safety finding (b)(3) on ADR-0017 ("The new DPO adapter code
  must apply the identical content-hash verification and offline
  enforcement to whatever path is loaded as the reference model").
- Same containment properties this project already relies on (isolated
  child process via ``process_isolation``, ``report_to=[]``, resource
  ceiling enforcement via ``trainer_contract.run_trainer_contract``) --
  nothing about DPO's two-model memory footprint weakens any of them; it
  is the caller's ``ResourceBudget`` (via a fresh, DPO-specific
  ``ResourceBudget`` review, per ADR-0017 Decision item 6) that must widen
  to fit two models, not this adapter's contract-conformance.

Deliberately narrow, matching ADR-0017's own scope:

- Exactly one training method: preference optimization via TRL's
  ``DPOTrainer``/``DPOConfig``. No other RL trainers (PPO/GRPO/...).
- Full-parameter DPO only, **not** LoRA+DPO -- ``use_lora`` is intentionally
  NOT a supported ``training_params`` key here (unlike ``TRLTrainerAdapter``),
  per ADR-0017 Decision item 1's explicit "keeps the cycle to exactly one
  new variable... avoids compounding an unproven new trainer path with the
  evaluator's already-identified PEFT-scoring gap."
- Exactly one concurrent run (enforced by the caller, same as
  ``TRLTrainerAdapter``).
- Fully offline for BOTH the policy and reference model: neither is ever
  resolved from a Hub id, and both are content-hash-verified against
  locally-declared paths before any training work starts.
- ``beta`` and ``reference_free`` are REQUIRED, explicit ``training_params``
  (no library-default fallback), per safety finding (b)(4) ("expose
  ``beta`` and ``reference_free`` as reviewed, justified configuration
  rather than library defaults"). ``prepare()`` raises ``InvalidInputError``
  if either is omitted -- this is a deliberate, stricter departure from
  ``TRLTrainerAdapter``'s optional-with-library-default pattern for
  ``learning_rate``, precisely because ADR-0017's own gate calls out that a
  bare default is not acceptable here.
"""

from __future__ import annotations

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
from .trl_adapter import (
    TRL_MAX_VERSION,
    TRL_MIN_VERSION,
    _detect_installed_trl_version,
    _directory_size_mb,
    _hash_path_identity,
)

_HASH_MANIFEST_EXCLUDE_NAMES = {".DS_Store", "__pycache__"}


@dataclass
class DPOTrainerAdapter:
    """Real ``TrainerAdapterV1`` implementation backed by TRL's ``DPOTrainer``.

    ``work_dir`` is the root directory this adapter creates per-run working
    directories under (evidence, checkpoints, final artifacts) -- same
    convention as ``TRLTrainerAdapter``.

    Required ``inputs.training_params`` keys (validated in ``prepare()``,
    never trusted from ``TrainingInputs.validate()`` since those are
    adapter-independent):

    - ``model_path`` (str): local filesystem path to a pre-verified POLICY
      model checkpoint (file or directory). Never a Hub id.
    - ``dataset_path`` (str): local filesystem path to a pre-verified,
      already-admitted preference-pair dataset in a format
      ``datasets.load_dataset("json", ...)`` can read, with ``prompt``,
      ``chosen``, and ``rejected`` string columns. Never a Hub id.
    - ``max_steps`` (int, > 0): hard upper bound on training steps. Same
      rationale as ``TRLTrainerAdapter``: this adapter refuses to train
      without an adapter-enforced step bound independent of the contract's
      wall-clock budget.
    - ``reference_model_path`` (str): local filesystem path to a
      pre-verified REFERENCE model checkpoint (file or directory). Never a
      Hub id. May be the same content as ``model_path`` (e.g. the same base
      checkpoint DPO starts from) but is verified as its own, independently
      hashed artifact -- see ``reference_model_hash`` below. This is new
      relative to ``TRLTrainerAdapter``, which trains exactly one model.
    - ``reference_model_hash`` (str, sha256 hex): content hash of the
      reference model checkpoint, verified against the actual content at
      ``reference_model_path`` before any training work starts, exactly
      the same discipline ``inputs.model_hash`` already gets for the policy
      model. An unverified or content-mismatched reference model is
      rejected the same way a tampered policy model already is.
    - ``beta`` (float, > 0): DPO's KL-regularization strength. REQUIRED,
      no default -- see module docstring. The security reviewer's gate: too low risks
      degenerate drift, too high risks reproducing the observed
      insensitivity this ADR exists to fix; the caller's execution-config
      document must state and justify this value explicitly.
    - ``reference_free`` (bool): whether DPO runs without a reference model
      at all (comparing the policy against itself pre-update instead).
      REQUIRED, no default -- same rationale as ``beta``. Note that setting
      this ``True`` does not make ``reference_model_path``/
      ``reference_model_hash`` optional in THIS adapter's contract: they
      are still required and still verified, so a reviewer can see exactly
      what reference artifact was prepared and available even if the
      eventual run configuration chooses not to use it, rather than the
      adapter silently skipping provenance verification based on a
      hyperparameter choice.

    Optional ``training_params`` keys: ``learning_rate`` (float, default
    ``5e-7`` -- TRL's own DPO-appropriate default, notably far lower than
    ``TRLTrainerAdapter``'s SFT default of ``2e-5``, since DPO gradients are
    typically much larger per step), ``per_device_train_batch_size`` (int,
    default ``1``), ``save_steps`` (int, default
    ``max(1, max_steps // 4)``), ``save_total_limit`` (int, default ``2``),
    ``max_prompt_length`` (int, default ``512``, TRL's own ``DPOConfig``
    default), ``max_length`` (int, default ``1024``, TRL's own ``DPOConfig``
    default).

    ``use_lora``/``lora_params`` are deliberately NOT supported keys here --
    see module docstring "full-parameter DPO only" scope note. A caller
    that passes ``use_lora`` gets no special handling; TRL's own
    ``DPOConfig``/``DPOTrainer`` simply never receive a ``peft_config``
    from this adapter's ``train()``.
    """

    work_dir: Path
    name: str = "trl-dpo-adapter-v1"
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

        Deliberately does not import ``trl``/``transformers``/``torch``,
        same rationale as ``TRLTrainerAdapter.prepare()``: every check here
        is filesystem- and metadata-level, so this stays cheap and safe to
        call even when the real training dependencies are not installed.
        """
        if budget.network_policy != "offline":
            raise RejectedInputError(
                "trl-dpo-adapter-v1 requires an offline network policy: policy model, "
                "reference model, and dataset must already be verified local paths "
                "before this adapter runs; it never downloads anything itself"
            )

        params = inputs.params
        model_path_raw = params.get("model_path")
        dataset_path_raw = params.get("dataset_path")
        reference_model_path_raw = params.get("reference_model_path")
        reference_model_hash = params.get("reference_model_hash")

        if not model_path_raw or not isinstance(model_path_raw, str):
            raise InvalidInputError(
                "training_params['model_path'] is required and must be a non-empty string"
            )
        if not dataset_path_raw or not isinstance(dataset_path_raw, str):
            raise InvalidInputError(
                "training_params['dataset_path'] is required and must be a non-empty string"
            )
        if not reference_model_path_raw or not isinstance(reference_model_path_raw, str):
            raise InvalidInputError(
                "training_params['reference_model_path'] is required and must be a "
                "non-empty string (this adapter always verifies a reference-model "
                "artifact's provenance, per ADR-0017 safety finding (b)(3), regardless "
                "of the eventual reference_free setting)"
            )
        if not reference_model_hash or not isinstance(reference_model_hash, str):
            raise InvalidInputError(
                "training_params['reference_model_hash'] is required and must be a "
                "non-empty sha256 hex string"
            )
        if not _looks_like_sha256(reference_model_hash):
            raise InvalidInputError(
                "training_params['reference_model_hash'] must be a 64-character hex "
                "sha256 digest"
            )

        max_steps = params.get("max_steps")
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise InvalidInputError(
                "training_params['max_steps'] is required and must be a positive int "
                "(this adapter refuses to train without an adapter-enforced step bound)"
            )

        beta = params.get("beta")
        if not isinstance(beta, (int, float)) or isinstance(beta, bool) or beta <= 0:
            raise InvalidInputError(
                "training_params['beta'] is required and must be a positive number "
                "(no library default is used here -- ADR-0017 safety finding (b)(4) "
                "requires an explicit, reviewed, justified beta value, not a silent "
                "default: too low risks degenerate drift, too high risks reproducing "
                "the insensitivity this ADR exists to fix)"
            )

        reference_free = params.get("reference_free")
        if not isinstance(reference_free, bool):
            raise InvalidInputError(
                "training_params['reference_free'] is required and must be a bool "
                "(no library default is used here -- ADR-0017 safety finding (b)(4) "
                "requires this to be an explicit, reviewed choice)"
            )

        if params.get("use_lora"):
            raise RejectedInputError(
                "trl-dpo-adapter-v1 does not support use_lora: ADR-0017 Decision item 1 "
                "scopes this adapter to full-parameter DPO only, deliberately not "
                "LoRA+DPO, to keep this cycle to exactly one new variable relative to "
                "the full-parameter SFT baseline and avoid compounding an unproven new "
                "trainer path with the evaluator's already-identified PEFT-scoring gap"
            )

        model_path = Path(model_path_raw)
        dataset_path = Path(dataset_path_raw)
        reference_model_path = Path(reference_model_path_raw)

        if not model_path.exists():
            raise RejectedInputError(f"model_path {model_path} does not exist locally")
        if not dataset_path.exists():
            raise RejectedInputError(f"dataset_path {dataset_path} does not exist locally")
        if not reference_model_path.exists():
            raise RejectedInputError(
                f"reference_model_path {reference_model_path} does not exist locally"
            )

        actual_model_hash = _hash_path_identity(model_path)
        if actual_model_hash != inputs.model_hash:
            raise RejectedInputError(
                f"model_hash mismatch: declared {inputs.model_hash}, "
                f"actual content at {model_path} hashes to {actual_model_hash} "
                "(refusing to train against an unverified/tampered policy checkpoint)"
            )
        actual_dataset_hash = _hash_path_identity(dataset_path)
        if actual_dataset_hash != inputs.dataset_hash:
            raise RejectedInputError(
                f"dataset_hash mismatch: declared {inputs.dataset_hash}, "
                f"actual content at {dataset_path} hashes to {actual_dataset_hash} "
                "(refusing to train against an unverified/tampered dataset)"
            )
        actual_reference_hash = _hash_path_identity(reference_model_path)
        if actual_reference_hash != reference_model_hash:
            raise RejectedInputError(
                f"reference_model_hash mismatch: declared {reference_model_hash}, "
                f"actual content at {reference_model_path} hashes to "
                f"{actual_reference_hash} (refusing to train against an "
                "unverified/tampered reference checkpoint -- ADR-0017 safety finding "
                "(b)(3) requires the same provenance discipline for the reference "
                "model that the policy model already gets)"
            )

        self._validate_preference_pair_schema(dataset_path)

    def _validate_preference_pair_schema(self, dataset_path: Path) -> None:
        """Cheap, filesystem-only schema check: every row has prompt/chosen/rejected.

        Deliberately does not use ``datasets.load_dataset`` (would require
        the optional ``datasets`` dependency importable just to run
        ``prepare()``); reads the JSONL directly since the on-disk shape is
        already known (this adapter's own documented dataset contract).
        Only checks the first row plus a full-file key-presence scan, not
        full content validation -- that is the preference-pair package's
        own ``validate_dataset.py`` job, not this adapter's.
        """
        if dataset_path.is_dir():
            # A directory dataset (e.g. Arrow/parquet shards) is out of
            # scope for this cheap check; datasets.load_dataset's own
            # error surfaces at train() time instead if malformed.
            return
        required_keys = {"prompt", "chosen", "rejected"}
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise InvalidInputError(
                        f"dataset_path line {line_number} is not valid JSON: {exc}"
                    ) from exc
                missing = required_keys - set(row)
                if missing:
                    raise InvalidInputError(
                        f"dataset_path line {line_number} is missing required "
                        f"preference-pair keys {sorted(missing)}; every row must have "
                        "'prompt', 'chosen', 'rejected' string fields"
                    )

    def train(
        self,
        inputs: TrainingInputs,
        budget: ResourceBudget,
        cancel_token: CancellationToken,
        resume_from: CheckpointHandle | None = None,
    ) -> TrainingOutput:
        """Run (or resume) a bounded TRL DPO run. NOT exercised by this work package.

        Implemented against real TRL/transformers APIs (lazy-imported here,
        not at module scope) so the code is complete and reviewable, but no
        test or CI run in this repository calls this method -- same
        boundary ``TRLTrainerAdapter.train()`` documents for SFT.
        """
        # Defence in depth: force offline mode inside this process
        # regardless of ambient environment, same as TRLTrainerAdapter.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        params = inputs.params
        run_dir = self._run_dir(inputs.run_id)
        checkpoint_dir = run_dir / "checkpoints"
        final_dir = run_dir / "final"

        # Same HF cache-redirection rationale as TRLTrainerAdapter.train():
        # keep every cache/lock-file write under this run's own directory,
        # inside filesystem_root, rather than the ambient
        # ~/.cache/huggingface default. Must happen before
        # datasets/transformers/huggingface_hub are imported below.
        hf_cache_dir = run_dir / "hf_cache"
        hf_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_cache_dir))
        os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache_dir / "datasets"))
        os.environ.setdefault("HF_HUB_CACHE", str(hf_cache_dir / "hub"))

        from datasets import load_dataset
        from transformers import TrainerCallback, TrainerControl, TrainerState
        from trl import DPOConfig, DPOTrainer

        max_steps = int(params["max_steps"])
        save_steps = int(params.get("save_steps", max(1, max_steps // 4)))

        dataset = load_dataset("json", data_files=params["dataset_path"], split="train")

        dpo_config = DPOConfig(
            output_dir=str(checkpoint_dir),
            max_steps=max_steps,
            save_steps=save_steps,
            save_strategy="steps",
            # Same storage-growth rationale as TRLTrainerAdapter: bound
            # checkpoint accumulation generically, not just for this run's
            # numbers (PR #18 Finding 2, re-applied here).
            save_total_limit=int(params.get("save_total_limit", 2)),
            learning_rate=float(params.get("learning_rate", 5e-7)),
            per_device_train_batch_size=int(params.get("per_device_train_batch_size", 1)),
            seed=inputs.seed,
            report_to=[],
            logging_steps=max(1, save_steps // 2),
            beta=float(params["beta"]),
            reference_free=bool(params["reference_free"]),
            max_prompt_length=int(params.get("max_prompt_length", 512)),
            max_length=int(params.get("max_length", 1024)),
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

        # Reference model: passed by verified local path, same offline
        # posture as the policy model. When reference_free is True, TRL's
        # own DPOTrainer still accepts (and does not require) a ref_model
        # argument -- passing the verified path here regardless of
        # reference_free means this adapter's provenance verification
        # above is never dead code contingent on a hyperparameter choice
        # (see class docstring's reference_free note).
        trainer = DPOTrainer(
            model=params["model_path"],
            ref_model=params["reference_model_path"],
            args=dpo_config,
            train_dataset=dataset,
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

        # Prune the periodic-checkpoint directory once the run has
        # ACCEPTED, same rationale/discipline as TRLTrainerAdapter.train().
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

        usage = ResourceUsage(
            # wall/cpu/memory are overwritten by the contract runner with
            # OS-measured figures; see trainer_contract.run_trainer_contract.
            wall_seconds=0.0,
            cpu_seconds=0.0,
            memory_mb_peak=0.0,
            gpu_count_used=self._gpu_count_used(),
            storage_mb_used=_directory_size_mb(run_dir),
        )
        return TrainingOutput(
            status=TrainingStatus.ACCEPTED,
            reason=f"trl DPO run completed {max_steps} max_steps (beta={params['beta']}, reference_free={params['reference_free']})",
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
        """Persist a genuinely resumable checkpoint after cooperative cancellation.

        Same rationale and same private-``Trainer``-method usage as
        ``TRLTrainerAdapter._safe_halt_output`` -- ``DPOTrainer`` inherits
        directly from ``transformers.Trainer`` (confirmed via
        ``DPOTrainer.__mro__`` during this ADR's preparation:
        ``[DPOTrainer, BaseTrainer, Trainer, object]``), so it exposes the
        same private checkpoint-machinery methods
        (``_save_optimizer_and_scheduler``, ``_save_scaler``,
        ``_save_rng_state``, ``save_state``) this adapter needs.
        """
        trainer.save_model(str(checkpoint_dir))
        trainer._save_optimizer_and_scheduler(str(checkpoint_dir))
        trainer._save_scaler(str(checkpoint_dir))
        trainer._save_rng_state(str(checkpoint_dir))
        trainer.save_state()
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
        import hashlib

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
            "beta": inputs.params.get("beta"),
            "reference_free": inputs.params.get("reference_free"),
            "reference_model_hash": inputs.params.get("reference_model_hash"),
        }
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        evidence_path.write_bytes(raw)
        return str(evidence_path), hashlib.sha256(raw).hexdigest()


def _looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())
