#!/usr/bin/env python3
"""ADR-0013 bounded pretrained meta-trainer cycle runner.

The default/check mode performs static, hash, dependency, renderer, and
TrainerAdapter.prepare() validation only. It never calls the trainer or evaluator.
Live baseline/training/post-evaluation requires ``--execute`` plus a review-gate
JSON created after the exact commit receives the approvals named in ADR-0013.

This script records measurements only. It never promotes, publishes, deploys, or
starts another cycle.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from held_out_eval import HeldOutExclusionRegistry

from codevolt_mdf.evaluator_contract import (
    EvaluationOutput,
    HeldOutExample,
    HeldOutSet,
    verify_evidence,
)
from codevolt_mdf.process_isolation import run_evaluator_in_isolated_process
from codevolt_mdf.trainer_contract import (
    ResourceBudget,
    TrainingInputs,
    TrainingOutput,
    run_trainer_contract,
)
from codevolt_mdf.trl_adapter import _hash_path_identity

MODEL_REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct"
    / f"snapshots/{MODEL_REVISION}"
)
EXPECTED_MODEL_HASH = "43752b3f39894c0122d9a94f3b4e64ad2d76e43d25c2a08aa360ad17a1a0145c"
EXPECTED_RENDERER_ID = (
    "chat_template:sha256="
    "551557e5be16b6241465fab68eb9958e8e953a47ff00bb24e40ab38e76ee8aed;"
    "add_generation_prompt=true;tokenize=false;retokenize_return_tensors=pt"
)

TRAIN_PATH = HERE / "train.jsonl"
HELD_OUT_PATH = HERE / "held_out.json"
MANIFEST_PATH = HERE / "semantic_family_manifest.json"
REGISTRY_INPUT_PATH = HERE / "held_out_exclusion_registry.json"
APPROVAL_ALLOWED_SIGNERS_PATH = HERE / "approval_allowed_signers"
ARITHMETIC_PATH = REPO_ROOT / "examples/pilot-adr0006/held_out.json"
ARITHMETIC_REGISTRY_PATH = REPO_ROOT / "examples/pilot-adr0006/held_out_exclusion_registry.json"
SAFETY_PATH = REPO_ROOT / "examples/safety-probes-wpb/held_out.json"
SAFETY_REGISTRY_PATH = (
    REPO_ROOT / "examples/safety-probes-wpb/held_out_exclusion_registry.json"
)
EXPECTED_FILE_HASHES = {
    "train.jsonl": "d325539d695615499b4c1d1636672d1aea14b4099de5ba3e128625cc1d3f2ab3",
    "held_out.json": "6d96170aa13704bda37a9863ed1688e0e90ffaae7b65e201ad5d3b07796441d0",
    "semantic_family_manifest.json": (
        "a97a1038d6874c12de16ca1da4383837fa82684911c5290a709224a5a8363dba"
    ),
    "held_out_exclusion_registry.json": (
        "39e966df25dad4000a91beb2f093921b9454eaac35fa4b5959afb92ddb8e6e6c"
    ),
    "REPRESENTATIVE_EXAMPLES.json": (
        "85616ba5ae95f56bad7244603dea8e8465f4d86ae485c29d0ba1abf2636866e8"
    ),
    "SOURCE_MAP.md": "f0f550d7e0c5625f2b3ea8fc51b9c120b493d444c88b2f1367c18f61ebd0b5fb",
    "DATASET_CARD.md": "2f57be14b5050d675d3aa4f85a5a94d5ac02c4643fa4f6958bc1fda245c047ac",
    "VALIDATION_REPORT.json": (
        "4e5d002b08e42fab48ef8b5d3e2a28ce527748369f772a049c44460365ab9618"
    ),
}
EXPECTED_VERSIONS = {
    "trl": "0.24.0",
    "transformers": "4.56.1",
    "datasets": "3.0.0",
    "accelerate": "1.4.0",
    "torch": "2.8.0",
}
EVALUATION_SUITES = {
    "meta_trainer": {
        "path": HELD_OUT_PATH,
        "file_hash": EXPECTED_FILE_HASHES["held_out.json"],
        "package_id": "pilot-metatrainer-v2-heldout",
        "count": 20,
        "task_type": "exact_match",
        "registry_path": REGISTRY_INPUT_PATH,
        "registry_hash": EXPECTED_FILE_HASHES["held_out_exclusion_registry.json"],
    },
    "capability_retention": {
        "path": ARITHMETIC_PATH,
        "file_hash": "2709d81627bca4b05d080a6cd978d7ac1cf1c8250fbf83c433bc35c0bb544d19",
        "package_id": "adr0006-pilot-heldout-v1",
        "count": 10,
        "task_type": "exact_match",
        "registry_path": ARITHMETIC_REGISTRY_PATH,
        "registry_hash": "9ca151fd2cfffd445560e5516967ca61b12f36547f1ddb6f8157e8eb1d77e218",
    },
    "safety": {
        "path": SAFETY_PATH,
        "file_hash": "c7292d307b5033699934aa4548877a740cefc10708c07d2ae92cce40c633548f",
        "package_id": "wpb-safety-probes-v1",
        "count": 15,
        "task_type": "safety_probe",
        "registry_path": SAFETY_REGISTRY_PATH,
        "registry_hash": "099f004caf998bf6d25a61077be8cf2c32807d3d73fb080142d910dde1ce24fa",
    },
}
APPROVAL_NAMESPACE = "codevolt-adr0013"
APPROVAL_ROLES = {
    "security": ("security-reviewer", "approved"),
    "owner": ("project-owner", "authorized"),
    "dataset": ("dataset-rights-reviewer", "admitted"),
}
SEED = 20260920
MAX_STEPS = 120
LEARNING_RATE = 1e-5
BATCH_SIZE = 1
SAVE_STEPS = 40
SAVE_TOTAL_LIMIT = 2
MAX_NEW_TOKENS = 160
RUN_ID = "adr0013-metatrainer-sft-20260920"
HOST_CONTAINMENT_SCOPE = "complete-cycle-host-containment-v1"
HOST_CONTAINMENT_ENV = "CODEVOLT_ADR0013_HOST_CONTAINMENT"
APPROVED_ROOT = Path("./local-evidence/adr0013")
APPROVED_EVIDENCE_ROOT = APPROVED_ROOT / "evidence"
APPROVED_REVIEW_GATE_PATH = APPROVED_EVIDENCE_ROOT / "adr0013-review-gate.json"
APPROVED_SCRATCH_ROOT = APPROVED_ROOT / "scratch"
APPROVED_VALIDATION_SCRATCH = APPROVED_SCRATCH_ROOT / "adr0013-check"
APPROVED_EXECUTION_SCRATCH = APPROVED_SCRATCH_ROOT / RUN_ID
MINIMUM_FREE_BYTES = 2 * 1024 * 1024 * 1024


def _seatbelt_literal(value: str) -> str:
    """Return a quoted Seatbelt profile string literal."""
    return json.dumps(value)


def _host_containment_profile(scratch_root: Path) -> str:
    """Build the macOS Seatbelt profile for the complete ADR-0013 cycle.

    The profile allows ordinary reads and process execution because Python, MPS, the
    pinned model cache, and installed packages live outside the run directory. It
    denies all network operations and all filesystem writes except beneath the exact
    reviewed scratch root plus writes to the kernel-discarded ``/dev/null`` device.
    Seatbelt restrictions are inherited by subprocesses and native extensions.
    """
    root = str(scratch_root.resolve())
    return "\n".join(
        [
            "(version 1)",
            "(allow default)",
            "(deny network*)",
            "(deny file-write*)",
            f"(allow file-write* (subpath {_seatbelt_literal(root)}))",
            '(allow file-write* (literal "/dev/null"))',
        ]
    )


def _assert_no_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.exists() and stat.S_ISLNK(current.lstat().st_mode):
            raise SystemExit(f"reviewed path contains a symlink component: {current}")


def _prepare_reviewed_directory(path: Path) -> Path:
    if not path.is_absolute():
        raise SystemExit(f"reviewed path must be absolute: {path}")
    _assert_no_symlink_components(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False, mode=0o700)
    _assert_no_symlink_components(path)
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise SystemExit(f"reviewed path does not resolve exactly: {path} -> {resolved}")
    info = path.stat()
    if info.st_uid != os.getuid():
        raise SystemExit(f"reviewed path is not owned by uid {os.getuid()}: {path}")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise SystemExit(f"reviewed path permits group/world access: {path}")
    free_bytes = os.statvfs(path).f_bavail * os.statvfs(path).f_frsize
    if free_bytes < MINIMUM_FREE_BYTES:
        raise SystemExit(
            f"reviewed path has {free_bytes} free bytes; {MINIMUM_FREE_BYTES} required"
        )
    return resolved


def _verify_host_containment(scratch_root: Path) -> dict[str, Any]:
    """Prove the active process cannot bypass Seatbelt via native/subprocess paths."""
    marker = os.environ.get(HOST_CONTAINMENT_ENV)
    expected_marker = hashlib.sha256(
        _host_containment_profile(scratch_root).encode("utf-8")
    ).hexdigest()
    if marker != expected_marker:
        raise SystemExit("complete-cycle host containment marker is missing or stale")

    canary = scratch_root.parent / ".adr0013-host-containment-canary"
    if canary.exists():
        raise SystemExit(f"host-containment canary path already exists: {canary}")

    try:
        fd = os.open(canary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        if exc.errno not in {errno.EACCES, errno.EPERM}:
            raise SystemExit(f"unexpected direct-write containment result: {exc}") from exc
    else:
        os.close(fd)
        canary.unlink(missing_ok=True)
        raise SystemExit("host containment failed: direct native write escaped scratch root")

    touch = subprocess.run(
        ["/usr/bin/touch", str(canary)], capture_output=True, check=False, text=True
    )
    if touch.returncode == 0 or canary.exists():
        canary.unlink(missing_ok=True)
        raise SystemExit("host containment failed: subprocess write escaped scratch root")

    nested = subprocess.run(
        [
            "/usr/bin/sandbox-exec",
            "-p",
            "(version 1) (allow default)",
            "/usr/bin/touch",
            str(canary),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if nested.returncode == 0 or canary.exists():
        canary.unlink(missing_ok=True)
        raise SystemExit("host containment failed: nested sandbox attempted to relax policy")

    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.settimeout(0.1)
            probe.connect(("127.0.0.1", 9))
        finally:
            probe.close()
    except OSError as exc:
        if exc.errno not in {errno.EACCES, errno.EPERM}:
            raise SystemExit(f"network containment is not OS-enforced: {exc}") from exc
    else:
        raise SystemExit("host containment failed: network connection was permitted")

    return {
        "scope": HOST_CONTAINMENT_SCOPE,
        "platform": "macOS Seatbelt via /usr/bin/sandbox-exec",
        "network": "deny network* (offline for runner and descendants)",
        "writes": f"only {scratch_root} and /dev/null",
        "subprocess_escape_probe": "blocked",
        "nested_sandbox_relaxation_probe": "blocked",
        "native_network_probe": "blocked",
    }


def _run_under_host_containment(scratch_root: Path) -> int:
    sandbox_exec = Path("/usr/bin/sandbox-exec")
    if sys.platform != "darwin" or not sandbox_exec.is_file():
        raise SystemExit("ADR-0013 execution requires macOS /usr/bin/sandbox-exec")
    profile = _host_containment_profile(scratch_root)
    environment = {
        key: os.environ[key]
        for key in ("PATH", "PYTHONPATH", "HOME", "LANG", "LC_ALL")
        if key in os.environ
    }
    environment.update(
        {
            HOST_CONTAINMENT_ENV: hashlib.sha256(profile.encode("utf-8")).hexdigest(),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": str(scratch_root / "tmp"),
        }
    )
    (scratch_root / "tmp").mkdir(mode=0o700, exist_ok=True)
    proc = subprocess.run(
        [
            str(sandbox_exec),
            "-p",
            profile,
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
        env=environment,
        check=False,
    )
    return proc.returncode


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_train_records() -> list[dict[str, Any]]:
    return [json.loads(line) for line in TRAIN_PATH.read_text(encoding="utf-8").splitlines()]


def _load_held_out_records() -> list[dict[str, Any]]:
    payload = json.loads(HELD_OUT_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit("held_out.json must remain a top-level list")
    return payload


def _held_out_set(records: list[dict[str, Any]]) -> HeldOutSet:
    return HeldOutSet.create(
        "pilot-metatrainer-v2-heldout",
        [
            HeldOutExample(
                example_id=item["example_id"],
                input=item["messages"][0]["content"],
                expected=item["messages"][1]["content"],
                metadata=(("semantic_family", item["semantic_family"]),),
            )
            for item in records
        ],
    )


def _load_suite(name: str) -> HeldOutSet:
    spec = EVALUATION_SUITES[name]
    path = Path(spec["path"])
    registry_path = Path(spec["registry_path"])
    if _sha256(path) != spec["file_hash"]:
        raise SystemExit(f"{name} suite hash mismatch")
    if _sha256(registry_path) != spec["registry_hash"]:
        raise SystemExit(f"{name} registry hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("examples")
    package_id = spec["package_id"] if isinstance(payload, list) else payload.get("package_id")
    if package_id != spec["package_id"] or not isinstance(records, list):
        raise SystemExit(f"{name} suite package/schema mismatch")
    if len(records) != spec["count"]:
        raise SystemExit(f"{name} suite expected {spec['count']} records")
    examples = []
    for item in records:
        metadata = dict(item.get("metadata", {}))
        actual_task_type = metadata.get("task_type", "exact_match")
        if actual_task_type != spec["task_type"]:
            raise SystemExit(
                f"{name} suite scoring metadata mismatch for {item.get('example_id')!r}"
            )
        if name == "meta_trainer":
            input_value = item["messages"][0]["content"]
            expected_value = item["messages"][1]["content"]
            metadata["semantic_family"] = item["semantic_family"]
        else:
            input_value = item["input"]
            expected_value = item["expected"]
        examples.append(
            HeldOutExample(
                example_id=item["example_id"],
                input=input_value,
                expected=expected_value,
                metadata=tuple(sorted(metadata.items())),
            )
        )
    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    registered_ids = set(
        registry_payload.get("package_held_out_ids", {}).get(spec["package_id"], [])
    )
    if not registered_ids and name == "meta_trainer":
        registered_ids = set(registry_payload.get("held_out_example_ids", []))
    actual_ids = {item.example_id for item in examples}
    if registered_ids != actual_ids:
        raise SystemExit(f"{name} suite registry ids do not match the locked suite")
    return HeldOutSet.create(spec["package_id"], examples)


def _evaluation_suites(
    train_records: list[dict[str, Any]],
) -> tuple[dict[str, HeldOutSet], HeldOutExclusionRegistry]:
    suites = {name: _load_suite(name) for name in EVALUATION_SUITES}
    registry = HeldOutExclusionRegistry()
    train_ids = [item["example_id"] for item in train_records]
    registry.register_package_train("pilot-metatrainer-v2-train", train_ids)
    seen: set[str] = set(train_ids)
    for name, held_out in suites.items():
        overlap = seen & set(held_out.example_ids)
        if overlap:
            raise SystemExit(f"{name} suite overlaps training/another suite: {sorted(overlap)}")
        registry.register_package_held_out(held_out.package_id, held_out.example_ids)
        seen.update(held_out.example_ids)
    return suites, registry


def _version_map() -> dict[str, str]:
    import accelerate
    import datasets
    import torch
    import transformers
    import trl

    return {
        "trl": trl.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "accelerate": accelerate.__version__,
        "torch": torch.__version__,
    }


def _runtime_config() -> dict[str, Any]:
    from trl.trainer.sft_config import SFTConfig

    config = SFTConfig(
        output_dir=str(HERE / "scratch" / "config-probe"),
        max_steps=MAX_STEPS,
        save_steps=SAVE_STEPS,
        save_strategy="steps",
        save_total_limit=SAVE_TOTAL_LIMIT,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        seed=SEED,
        report_to=[],
        logging_steps=20,
    )
    return {
        "device": str(config.device),
        "bf16": bool(config.bf16),
        "fp16": bool(config.fp16),
        "use_cpu": bool(config.use_cpu),
        "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
    }


def _renderer_id() -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    payload = json.dumps(
        {
            "chat_template": tokenizer.chat_template,
            "options": {
                "add_generation_prompt": True,
                "tokenize": False,
                "retokenize_return_tensors": "pt",
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return (
        f"chat_template:sha256={digest};"
        "add_generation_prompt=true;tokenize=false;retokenize_return_tensors=pt"
    )


def _build_inputs(dataset_licence: str) -> TrainingInputs:
    return TrainingInputs.create(
        model_revision=f"{MODEL_REPO}@{MODEL_REVISION}",
        model_hash=EXPECTED_MODEL_HASH,
        dataset_version=f"pilot-metatrainer-v2@{EXPECTED_FILE_HASHES['train.jsonl']}",
        dataset_hash=EXPECTED_FILE_HASHES["train.jsonl"],
        seed=SEED,
        training_params={
            "model_path": str(MODEL_PATH),
            "dataset_path": str(TRAIN_PATH),
            "max_steps": MAX_STEPS,
            "learning_rate": LEARNING_RATE,
            "per_device_train_batch_size": BATCH_SIZE,
            "save_steps": SAVE_STEPS,
            "save_total_limit": SAVE_TOTAL_LIMIT,
            "use_lora": False,
        },
        dataset_licence=dataset_licence,
        contamination_checked=True,
        run_id=RUN_ID,
    )


def _budget(scratch_root: Path) -> ResourceBudget:
    return ResourceBudget(
        max_wall_seconds=1800,
        max_cpu_seconds=3600,
        max_memory_mb=8192,
        max_gpu_count=0,
        max_storage_mb=1024,
        network_policy="offline",
        filesystem_root=str(scratch_root),
    )


def validate_plan(scratch_root: Path) -> dict[str, Any]:
    from codevolt_mdf.trl_adapter import TRLTrainerAdapter

    file_hashes = {name: _sha256(HERE / name) for name in EXPECTED_FILE_HASHES}
    if file_hashes != EXPECTED_FILE_HASHES:
        raise SystemExit(f"dataset/support hash mismatch: {file_hashes}")
    if _hash_path_identity(MODEL_PATH) != EXPECTED_MODEL_HASH:
        raise SystemExit("model snapshot content hash mismatch")
    versions = _version_map()
    if versions != EXPECTED_VERSIONS:
        raise SystemExit(f"dependency pin mismatch: {versions}")
    runtime_config = _runtime_config()
    expected_runtime = {
        "device": "mps",
        "bf16": True,
        "fp16": False,
        "use_cpu": False,
        "gradient_accumulation_steps": 1,
    }
    if runtime_config != expected_runtime:
        raise SystemExit(f"runtime precision/device mismatch: {runtime_config}")
    renderer_id = _renderer_id()
    if renderer_id != EXPECTED_RENDERER_ID:
        raise SystemExit(f"chat renderer identity mismatch: {renderer_id}")

    train_records = _load_train_records()
    held_out_records = _load_held_out_records()
    if len(train_records) != 40 or len(held_out_records) != 20:
        raise SystemExit("expected exactly 40 train and 20 held-out records")
    train_ids = {item["example_id"] for item in train_records}
    held_out_ids = {item["example_id"] for item in held_out_records}
    if train_ids & held_out_ids:
        raise SystemExit("train/held-out example-id overlap detected")
    train_families = {item["semantic_family"] for item in train_records}
    held_out_families = {item["semantic_family"] for item in held_out_records}
    if train_families & held_out_families:
        raise SystemExit("train/held-out semantic-family overlap detected")

    suites, registry = _evaluation_suites(train_records)
    held_out = suites["meta_trainer"]
    for name, suite in suites.items():
        if registry.check_held_out_not_trained(suite.example_ids):
            raise SystemExit(f"{name} registry contamination check failed")

    inputs = _build_inputs("DRY-VALIDATION-ONLY-NOT-A-DATA-ADMISSION")
    inputs.validate()
    adapter = TRLTrainerAdapter(work_dir=scratch_root / "trainer_work")
    adapter.prepare(inputs, _budget(scratch_root))
    return {
        "status": "PASS",
        "training_called": False,
        "model_hash": EXPECTED_MODEL_HASH,
        "dataset_hashes": file_hashes,
        "versions": versions,
        "runtime_config": runtime_config,
        "renderer_id": renderer_id,
        "train_count": 40,
        "held_out_count": 20,
        "held_out_contract_hash": held_out.dataset_hash,
        "execution_blockers": [
            "role-specific public signer keys are not yet admitted in the trust root",
            "Maya exact-candidate live-execution approval",
            "owner confirmation for this one run",
        ],
    }


def _verify_signed_approval(
    role: str,
    approval: dict[str, Any],
    *,
    head: str,
    dataset_licence: str,
) -> dict[str, Any]:
    principal, decision = APPROVAL_ROLES[role]
    required = {"document", "signature", "document_sha256"}
    if set(approval) != required:
        raise SystemExit(f"{role} approval must contain exactly {sorted(required)}")
    document_path = Path(approval["document"])
    signature_path = Path(approval["signature"])
    if not document_path.is_file() or not signature_path.is_file():
        raise SystemExit(f"{role} approval document/signature is missing")
    if _sha256(document_path) != approval["document_sha256"]:
        raise SystemExit(f"{role} approval document hash mismatch")
    document = json.loads(document_path.read_text(encoding="utf-8"))
    required_document = {
        "schema_version",
        "role",
        "approver_id",
        "decision",
        "implementation_sha",
        "dataset_hash",
        "dataset_licence",
        "run_id",
        "scope",
    }
    if set(document) != required_document:
        raise SystemExit(f"{role} approval document schema mismatch")
    expected = {
        "schema_version": 1,
        "role": role,
        "approver_id": principal,
        "decision": decision,
        "implementation_sha": head,
        "dataset_hash": EXPECTED_FILE_HASHES["train.jsonl"],
        "dataset_licence": dataset_licence,
        "run_id": RUN_ID,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise SystemExit(f"{role} approval field {key!r} is not exact")
    scope = document["scope"]
    if not isinstance(scope, list) or not scope or not all(
        isinstance(item, str) and item.strip() for item in scope
    ):
        raise SystemExit(f"{role} approval scope must be a non-empty string list")
    required_security_scopes = {
        "evaluator-process-containment-v1",
        HOST_CONTAINMENT_SCOPE,
    }
    if role == "security" and not required_security_scopes.issubset(scope):
        missing = sorted(required_security_scopes - set(scope))
        raise SystemExit(f"security approval is missing required scope tokens: {missing}")
    proc = subprocess.run(
        [
            "ssh-keygen",
            "-Y",
            "verify",
            "-f",
            str(APPROVAL_ALLOWED_SIGNERS_PATH),
            "-I",
            principal,
            "-n",
            APPROVAL_NAMESPACE,
            "-s",
            str(signature_path),
        ],
        input=document_path.read_bytes(),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"{role} approval signature is not trusted or valid")
    return document


def _load_gate(path: Path) -> dict[str, Any]:
    gate = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "implementation_sha", "dataset_hash", "dataset_licence", "approvals"}
    if set(gate) != required or gate.get("schema_version") != 1:
        raise SystemExit("review gate schema mismatch")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if gate["implementation_sha"] != head:
        raise SystemExit(
            f"gate approves {gate['implementation_sha']}, but checkout HEAD is {head}"
        )
    if gate["dataset_hash"] != EXPECTED_FILE_HASHES["train.jsonl"]:
        raise SystemExit("gate dataset decision is not bound to the locked training dataset")
    dataset_licence = gate["dataset_licence"]
    if not isinstance(dataset_licence, str) or not dataset_licence.strip():
        raise SystemExit("gate dataset_licence must be an exact non-empty admitted value")
    approvals = gate["approvals"]
    if not isinstance(approvals, dict) or set(approvals) != set(APPROVAL_ROLES):
        raise SystemExit("review gate requires exact security, owner, and dataset approvals")
    trusted_signer_lines = [
        line
        for line in APPROVAL_ALLOWED_SIGNERS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not trusted_signer_lines:
        raise SystemExit("approval trust root has no admitted signer keys; execution remains blocked")
    verified = {
        role: _verify_signed_approval(
            role,
            approvals[role],
            head=head,
            dataset_licence=dataset_licence,
        )
        for role in APPROVAL_ROLES
    }
    return {**gate, "verified_approvals": verified}


def _evaluate(
    artifact_id: str,
    artifact_path: Path,
    held_out: HeldOutSet,
    registry: HeldOutExclusionRegistry,
    work_dir: Path,
    scratch_root: Path,
) -> tuple[EvaluationOutput, dict[str, Any]]:
    from codevolt_mdf.hf_local_evaluator_adapter import HFLocalCausalLMEvaluatorAdapter

    evaluator = HFLocalCausalLMEvaluatorAdapter(
        max_new_tokens=MAX_NEW_TOKENS,
        expected_model_hash=_hash_path_identity(artifact_path),
        use_chat_template=True,
    )
    output, error, measured = run_evaluator_in_isolated_process(
        adapter=evaluator,
        artifact_id=artifact_id,
        artifact_locator=str(artifact_path),
        held_out=held_out,
        registry=registry,
        work_dir=work_dir,
        budget=_budget(scratch_root),
    )
    if measured.killed_for_timeout:
        raise SystemExit(
            f"evaluator exceeded max_wall_seconds={_budget(scratch_root).max_wall_seconds}"
        )
    if measured.killed_for_overrun:
        raise SystemExit("evaluator exceeded CPU or memory resource budget")
    if error is not None or output is None:
        raise SystemExit(f"evaluator containment failed: {error}")
    if output.status.value != "scored":
        raise SystemExit(f"evaluator returned {output.status.value}: {output.reason}")
    if not output.evidence_locator or not output.evidence_hash:
        raise SystemExit("evaluator omitted evidence locator/hash")
    if not verify_evidence(output.evidence_locator, output.evidence_hash):
        raise SystemExit("evaluator evidence is missing or hash-invalid")
    if any(f"renderer={EXPECTED_RENDERER_ID}" not in item.raw_output for item in output.results):
        raise SystemExit("evaluator output omitted or changed the locked renderer identity")
    usage = {
        "wall_seconds": measured.wall_seconds,
        "cpu_seconds": measured.cpu_seconds,
        "memory_mb_peak": measured.memory_mb_peak,
        "storage_mb_used": measured.storage_mb_used,
    }
    return output, usage


def _evaluation_dict(output: EvaluationOutput) -> dict[str, Any]:
    return {
        "status": output.status.value,
        "reason": output.reason,
        "artifact_id": output.artifact_id,
        "package_id": output.package_id,
        "aggregate_score": output.aggregate_score,
        "evidence_locator": output.evidence_locator,
        "evidence_hash": output.evidence_hash,
        "error_class": output.error_class,
        "contaminated_ids": list(output.contaminated_ids),
        "results": [
            {
                "example_id": item.example_id,
                "correct": item.correct,
                "score": item.score,
                "raw_output": item.raw_output,
            }
            for item in output.results
        ],
    }


def _training_dict(output: TrainingOutput) -> dict[str, Any]:
    usage = output.resource_usage
    return {
        "status": output.status.value,
        "reason": output.reason,
        "artifact_id": output.artifact_id,
        "evidence_locator": output.evidence_locator,
        "evidence_hash": output.evidence_hash,
        "error_class": output.error_class,
        "resource_usage": None
        if usage is None
        else {
            "wall_seconds": usage.wall_seconds,
            "cpu_seconds": usage.cpu_seconds,
            "memory_mb_peak": usage.memory_mb_peak,
            "gpu_count_used": usage.gpu_count_used,
            "storage_mb_used": usage.storage_mb_used,
        },
    }


def execute_cycle(
    scratch_root: Path,
    gate_path: Path,
    host_containment: dict[str, Any],
) -> int:
    from codevolt_mdf.trl_adapter import TRLTrainerAdapter

    gate = _load_gate(gate_path)
    validation = {**validate_plan(scratch_root), "host_containment": host_containment}
    train_records = _load_train_records()
    suites, registry = _evaluation_suites(train_records)

    baseline: dict[str, dict[str, Any]] = {}
    for suite_name, held_out in suites.items():
        output, usage = _evaluate(
            f"adr0013-base-{suite_name}",
            MODEL_PATH,
            held_out,
            registry,
            scratch_root / "baseline_evaluation" / suite_name,
            scratch_root,
        )
        baseline[suite_name] = {**_evaluation_dict(output), "resource_usage": usage}

    inputs = _build_inputs(gate["dataset_licence"])
    adapter = TRLTrainerAdapter(work_dir=scratch_root / "trainer_work")
    training = run_trainer_contract(adapter, inputs, _budget(scratch_root))
    result: dict[str, Any] = {
        "gate": gate,
        "validation": validation,
        "baseline": baseline,
        "training": _training_dict(training),
        "candidate": None,
        "promotion_decision": None,
    }
    if training.status.value != "accepted" or training.evidence_locator is None:
        _write_result(scratch_root, result)
        return 1

    final_dir = Path(training.evidence_locator).parent / "final"
    if not final_dir.is_dir():
        raise SystemExit(f"accepted training output has no final artifact at {final_dir}")
    candidate: dict[str, dict[str, Any]] = {}
    for suite_name, held_out in suites.items():
        output, usage = _evaluate(
            f"{training.artifact_id or RUN_ID}-{suite_name}",
            final_dir,
            held_out,
            registry,
            scratch_root / "candidate_evaluation" / suite_name,
            scratch_root,
        )
        candidate[suite_name] = {**_evaluation_dict(output), "resource_usage": usage}
    result["candidate"] = candidate
    _write_result(scratch_root, result)
    return 0


def _write_result(scratch_root: Path, result: dict[str, Any]) -> None:
    path = scratch_root / "cycle_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--review-gate", type=Path)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=HERE / "scratch" / RUN_ID,
    )
    args = parser.parse_args()
    if not args.execute:
        args.scratch_root.mkdir(parents=True, exist_ok=True)
        print(json.dumps(validate_plan(args.scratch_root), indent=2, sort_keys=True))
        return 0
    if args.review_gate is None:
        raise SystemExit("--execute requires --review-gate")
    if args.scratch_root != APPROVED_EXECUTION_SCRATCH:
        raise SystemExit(
            f"--execute requires exact reviewed scratch root {APPROVED_EXECUTION_SCRATCH}"
        )
    if args.review_gate != APPROVED_REVIEW_GATE_PATH:
        raise SystemExit(
            f"--execute requires exact reviewed gate path {APPROVED_REVIEW_GATE_PATH}"
        )
    _prepare_reviewed_directory(APPROVED_ROOT)
    _prepare_reviewed_directory(APPROVED_EVIDENCE_ROOT)
    _prepare_reviewed_directory(APPROVED_SCRATCH_ROOT)
    scratch_root = _prepare_reviewed_directory(args.scratch_root)
    if HOST_CONTAINMENT_ENV not in os.environ:
        return _run_under_host_containment(scratch_root)
    host_containment = _verify_host_containment(scratch_root)
    return execute_cycle(scratch_root, args.review_gate, host_containment)


if __name__ == "__main__":
    raise SystemExit(main())
