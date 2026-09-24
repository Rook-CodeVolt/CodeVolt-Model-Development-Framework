# ADR-0018-execution-config: hyperparameter and gate selection for the first governed DPO run (confabulated-recipe refusal axis, per ADR-0017)

- Status: execution-config note, not itself an authorizing ADR — records the
  reasoning behind the proposed first `DPOTrainerAdapter`
  (`src/codevolt_mdf/dpo_adapter.py`, PR #98) run against the ADR-0017
  preference-pair package (`examples/pilot-metatrainer-v3-dpo/`, PR #98).
  Training remains blocked until every gate in this record clears, per this
  project's standing ADR-0013/0014/0015/0016 execution-gating convention.
  **This document authorizes no training run.** No `--execute` runner script
  exists yet for this cycle; writing one is a separate follow-up this
  document does not perform (see "What this document does and does not
  authorize").
- Date: 2026-09-23
- Tracking: (this drafting task, commissioned by
  Rook). Upstream: ADR-0017 (`docs/decisions/ADR-0017-dpo-preference-refusal-axis.md`,
  PR #97, squash commit `8a0c4309cf510cfd0f7028ba019add921c769804`), PR #98
  (`src/codevolt_mdf/dpo_adapter.py` + `examples/pilot-metatrainer-v3-dpo/`,
  squash commit `2bb55c066013e1ddbf3cf416e562723d68228565`), both independently
  re-verified against the current `main` during this document's drafting (see
  "Independent re-verification performed for this document" below).
- Scope: this note follows the same pattern as
  `docs/decisions/ADR-0015-execution-config.md` and
  `docs/decisions/ADR-0016-execution-config.md` — a hyperparameter/config
  record for a fresh execution-config ADR, gated by the same three signer
  roles (`security-reviewer`, `dataset-rights-reviewer`, `project-owner`) under a fresh,
  never-reused approval namespace. Unlike ADR-0015/0016 (mechanical
  adaptations of an already-existing runner script to new corpus content),
  this is the **first** execution-config document for a **new trainer path**
  (`DPOTrainerAdapter`, not `TRLTrainerAdapter`) and a **new dataset shape**
  (`prompt/chosen/rejected` triples, not `messages` lists), so every
  parameter below is justified from first principles against this project's
  own evidence rather than carried forward by default.

## Independent re-verification performed for this document

Every fact below was re-checked directly against the current `main` branch
and this host's real files during this document's drafting, not accepted
from ADR-0017's or PR #98's own prose:

- ADR-0017 (PR #97) and PR #98 are both `MERGED` on `main`
  (`gh pr list --state all`), at the exact commits named above (`git log
  --oneline` on `main`).
- Base-model content hash re-computed independently via a standalone
  re-implementation of `_hash_path_identity` (`trl_adapter.py`'s directory
  manifest hash: sorted `(relative_path, size, sha256)` tuples, then sha256
  of the JSON manifest) against the real, on-disk
  `~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac/`
  directory: **`43752b3f39894c0122d9a94f3b4e64ad2d76e43d25c2a08aa360ad17a1a0145c`**,
  matching `EXPECTED_MODEL_HASH` in every prior runner
  (`run_bounded_cycle_adr0014.py`/`adr0015.py`/`adr0016.py`) exactly.
- Dataset file hashes re-computed independently via `sha256`/direct Python
  hashing against the real, on-disk
  `examples/pilot-metatrainer-v3-dpo/` files (command used:
  `hashlib.sha256(path.read_bytes()).hexdigest()` on each file, and
  `shasum -a 256` cross-check on `preference_pairs.jsonl`):
  - `preference_pairs.jsonl`:
    `e6c1d20884eda08aac354a49ba4514ec44adfe565bb70e03bd9526a3f8da5b33`
    (matches `VALIDATION_REPORT.json`'s recorded hash exactly)
  - `generate_pairs.py`:
    `9a8d63cb44d30cb03ac9b4ae5e9bc23a12e7e914df4470bde8c7b338cafed054`
  - `validate_dataset.py`:
    `8b352dd666d4cf42a101833af4bec775f47ae3be5d4e611c16f317b246216144`
  - `DATASET_CARD.md`:
    `d9e371a3a59a78a1e5c0691ae207a9327c6a064e815d1730a701ceccc97dadab`
  - `VALIDATION_REPORT.json` (as committed on `main`):
    `68059d3f5c7b558b5821de66aa620df17024448aee6e22689fd2dd81678967fd`
- `examples/pilot-metatrainer-v3-dpo/validate_dataset.py` was re-run fresh
  against the real, on-disk package during this document's drafting (not
  trusted from the committed `VALIDATION_REPORT.json` alone): result
  `{"status": "PASS", "checks": 14, "counts": {"total": 22, "refusal": 16,
  "counter": 6, "counter_share": 0.2727}}`, identical to the committed
  report's counts and check statuses. **Counter-direction share, computed,
  not asserted: 6/22 = 0.2727 (27.27%), above Maya's >=20% over-refusal-guard
  minimum from ADR-0017.**
- Held-out contamination re-verified as part of the same re-run: zero exact
  reuse hits, zero near-paraphrase hits against `mtr-v2-heldout-0013`, and a
  fresh manual `grep`-equivalent check confirmed no record's
  `semantic_family` or free text mentions the forbidden
  `synthetic_data_tradeoffs` family label or name. `mtr-v2-heldout-0013`
  itself is confirmed still present and unmodified in
  `examples/pilot-metatrainer-v3/held_out.json` (76-character prompt text,
  matching the validator's own registry-drift check).
- `DPOTrainerAdapter.prepare()`/`train()` (`src/codevolt_mdf/dpo_adapter.py`)
  re-read in full: confirmed it requires `model_path`, `dataset_path`,
  `reference_model_path`, `reference_model_hash`, `max_steps`, `beta`, and
  `reference_free` all explicitly (no bare defaults for `beta`/
  `reference_free`, matching Maya's ADR-0017 gate (b)(4)); confirmed it
  rejects `use_lora`; confirmed it independently content-hashes and verifies
  the reference-model path exactly as the policy model, per Maya's gate
  (b)(3).
- `trl==0.24.0`'s real, installed `DPOConfig` defaults were re-confirmed by
  direct `inspect.signature()` introspection of the exact pinned environment
  (`./.venv-adr0014-test`,
  `trl==0.24.0`, `transformers==4.56.1`, `datasets==3.0.0`,
  `accelerate==1.4.0`, `torch==2.8.0` — all five pins match this project's
  `pyproject.toml` `trl-adapter` extra exactly): `beta=0.1`,
  `reference_free=False`, `max_prompt_length=512`, `max_length=1024`,
  `learning_rate=1e-6`.
- `src/codevolt_mdf/hf_local_evaluator_adapter.py` re-searched fresh:
  confirmed zero matches for `PeftModel`/`merge_and_unload`/`import peft`/
  `from peft` anywhere in the module, matching ADR-0017's own cited
  evaluator-gap finding.
- `.github/workflows/ci.yml` re-read: CI matrix is `["3.9", "3.11", "3.13"]`,
  running `ruff check` and `pytest` — this document's own PR must pass all
  three, unchanged.

## What this ADR proposes

If and only if every gate below clears (via a separately written, gated
runner script this document does not itself contain — see "What this
document does and does not authorize"), run exactly one bounded DPO cycle:
policy model = reference model = the immutable `SmolLM2-135M-Instruct` base
checkpoint (full-parameter DPO starting from the untrained base, mirroring
every prior real cycle ADR-0013 through ADR-0016 starting fresh from the same
base rather than chaining off a previously-trained candidate), trained
against the 22-record ADR-0017 preference-pair package, then evaluated
through the existing, unmodified `hf_local_evaluator_adapter.py` and the same
independent manual rubric methodology already
established.

### 1. Model identity, hash parity, and offline loading

- **Policy model** and **reference model** are the same immutable checkpoint:
  `HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f77366fa6b3b4b768ec3050bf629380bac`,
  content hash `43752b3f39894c0122d9a94f3b4e64ad2d76e43d25c2a08aa360ad17a1a0145c`
  (independently re-verified above), at
  `~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac/`.
  This is a deliberate choice, not an oversight: DPO's reference model is
  standardly the frozen pre-update policy, and every prior real cycle on this
  project (ADR-0013 through ADR-0016) started full-parameter training fresh
  from this same base checkpoint rather than continuing from a previously
  trained candidate, so this cycle preserves that same "one clean starting
  point" property. `model_path` and `reference_model_path` are the identical
  filesystem path; `reference_model_hash` is still independently verified by
  `DPOTrainerAdapter.prepare()` against the actual on-disk content at that
  path, per its own contract (it does not special-case identical paths).
- **Hash-parity check the adapter enforces:** `DPOTrainerAdapter.prepare()`
  computes `_hash_path_identity()` (the same directory-manifest sha256
  scheme `trl_adapter.py` uses for the policy model) independently against
  both `model_path` and `reference_model_path`, and raises
  `RejectedInputError` on any mismatch against the declared
  `inputs.model_hash` / `training_params["reference_model_hash"]`
  respectively — the same tamper-detection discipline the policy model
  already had, now applied twice.
- **Offline-only loading:** `DPOTrainerAdapter.prepare()` raises
  `RejectedInputError` if `budget.network_policy != "offline"`, and
  `DPOTrainerAdapter.train()` force-sets `HF_HUB_OFFLINE=1` /
  `TRANSFORMERS_OFFLINE=1` inside the isolated child process as defence in
  depth, identical to `TRLTrainerAdapter`'s existing posture, now covering
  both the policy and reference model loads.

### 2. Dataset: exact identity, per-file hashes, counter-direction share, contamination

- **Path:** `examples/pilot-metatrainer-v3-dpo/preference_pairs.jsonl`
  (22 records, all `split: "train"` — this package proposes no held-out
  preference pairs, per its own `DATASET_CARD.md`).
- **Per-file sha256** (independently recomputed for this document, command:
  `hashlib.sha256(Path(f).read_bytes()).hexdigest()` per file, cross-checked
  with `shasum -a 256` on the primary data file):

  | File | SHA-256 |
  |---|---|
  | `preference_pairs.jsonl` | `e6c1d20884eda08aac354a49ba4514ec44adfe565bb70e03bd9526a3f8da5b33` |
  | `generate_pairs.py` | `9a8d63cb44d30cb03ac9b4ae5e9bc23a12e7e914df4470bde8c7b338cafed054` |
  | `validate_dataset.py` | `8b352dd666d4cf42a101833af4bec775f47ae3be5d4e611c16f317b246216144` |
  | `DATASET_CARD.md` | `d9e371a3a59a78a1e5c0691ae207a9327c6a064e815d1730a701ceccc97dadab` |
  | `VALIDATION_REPORT.json` (as committed) | `68059d3f5c7b558b5821de66aa620df17024448aee6e22689fd2dd81678967fd` |

  Only `preference_pairs.jsonl`'s hash is load-bearing for `prepare()`'s
  `dataset_hash` check (the other four are support files); it is reported
  here in full for reviewer convenience, matching this project's own
  `EXPECTED_FILE_HASHES`-table convention from `run_bounded_cycle_adr0016.py`.
- **Counter-direction share, computed not asserted:** 6 counter-direction /
  22 total = **0.2727 (27.27%)**, re-verified by re-running
  `validate_dataset.py` fresh against the real on-disk file during this
  document's drafting (`counts.counter_share: 0.2727`, `checks: 14`, all
  `PASS`) — above Maya's ADR-0017 gate (b)(2) minimum of >=20%.
- **Contamination check proving no held-out prompt is reused:**
  `validate_dataset.py`'s three programmatic checks (re-run fresh, not
  trusted from the committed report): (a) normalized-string exact-match
  against all 48 real held-out prompts in
  `examples/pilot-metatrainer-v3/held_out.json` — zero hits; (b) a
  >=60%-token-overlap near-paraphrase heuristic specifically against
  `mtr-v2-heldout-0013`'s own 76-character prompt text — zero hits; (c) a
  full-text scan for the forbidden `synthetic_data_tradeoffs` family label or
  name anywhere in the package — zero hits. `mtr-v2-heldout-0013` itself is
  confirmed still present, unmodified, and un-reused in the real held-out
  registry. This is a token-overlap heuristic, not a semantic-similarity
  model (the validator's own stated limitation, re-confirmed by reading its
  source) — it structurally proves the absence of literal/near-literal reuse,
  not subtler semantic adjacency; an independent reviewer's own judgment on
  that narrower residual risk is still required before `--execute`, exactly
  as `pilot-metatrainer-v3/validate_dataset.py`'s own `semantic_family_disjoint`
  check states about itself.

### 3. Every DPO hyperparameter, explicit, with beta given a real justification

| Parameter | Value | Library/adapter default | Justification |
|---|---|---|---|
| `beta` | **0.3** | `0.1` (TRL `DPOConfig` default, confirmed by introspection above) | **Not a bare default, per Maya's ADR-0017 gate (b)(4).** The original DPO paper's own tested range is `0.1`-`0.5`; lower `beta` permits larger per-step divergence from the reference policy (more aggressive optimization, less KL regularization pressure), higher `beta` constrains divergence more tightly per update. This is the **first** DPO run on this project, against a **22-record** preference-pair package (the smallest labeled-signal dataset of any real cycle to date; ADR-0013's rejected 40-example full-SFT corpus and its degenerate-repetition-loop failure remain this project's own cautionary evidence for what happens under too little regularization pressure relative to update strength on a small dataset for this exact 135M model). Choosing a value from the upper-middle of the paper's own tested range, rather than TRL's lower-end library default, directly targets Maya's named risk (b)(4) ("too low a beta risks the same kind of degenerate drift ADR-0013's full-SFT run showed") without moving so high that it risks the opposite named risk (reproducing the ADR-0015/0016 insensitivity this whole DPO cycle exists to fix) — `0.3` is the paper's own tested midpoint-to-upper value, not an untested extreme in either direction. |
| `reference_free` | **`False`** | `False` (matches default) | Explicit, not inherited silently: a real frozen reference model (the same base checkpoint, hash-verified per item 1 above) is used for KL-anchoring, which is the entire point of choosing DPO over a from-scratch/reference-free variant for this narrow-axis correction — `reference_free=True` would compare the policy only against itself pre-update, discarding the reference-provenance verification work item 1 already requires regardless of this flag (per the adapter's own documented design: `reference_model_path`/`reference_model_hash` stay required even when `reference_free=True`). |
| LoRA vs full | **Full-parameter DPO** | N/A (`use_lora` unsupported) | Per ADR-0017 Decision item 1: keeps this cycle to exactly one new variable (the training objective) relative to every prior full-parameter cycle (ADR-0013 through ADR-0016), and avoids compounding an unproven new trainer path with the evaluator's known PEFT-scoring gap (see item 5 below). `DPOTrainerAdapter.prepare()` itself rejects `use_lora` with `RejectedInputError`, so this is enforced by code, not by convention alone. |
| `learning_rate` | **`5e-7`** (adapter/library default) | `5e-7` is `DPOTrainerAdapter`'s own documented default (TRL's own `DPOConfig` library default is `1e-6`; the adapter's module docstring states its chosen default of `5e-7` explicitly, "notably far lower than `TRLTrainerAdapter`'s SFT default of `2e-5`, since DPO gradients are typically much larger per step") | Kept at the adapter's own stated default: unlike SFT's `1e-5` (ADR-0013, rejected after a real degenerate-drift failure) -> `5e-6` (ADR-0014, corrected with direct evidence), there is no real prior DPO run on this project yet to motivate a deviation from the adapter author's own documented, reasoned default in either direction. If this first real run shows degenerate drift or under-training at this rate, that becomes the evidence basis for a corrected value in a fresh ADR, exactly as ADR-0013 -> ADR-0014's own precedent. |
| `max_steps` | **22** | none (required, no default) | One full epoch over the 22-record preference-pair train split, `per_device_train_batch_size=1`, no gradient accumulation — the same "one training step per training example, one epoch per corpus size" invariant ADR-0014 established as safe and ADR-0015/0016 each re-confirmed at larger corpus sizes, applied here at DPO's much smaller dataset size. No evidence exists yet to motivate multi-epoch training on a preference-pair dataset this small, and the project's own established discipline (`0014-corrected-bounded-metatrainer-cycle.md`) treats multi-pass training over a small corpus as the specific mechanism that produced ADR-0013's rejected degenerate-repetition failure — that caution applies at least as strongly here, on a 3.7x-smaller record count than ADR-0013's already-small 40. |
| `per_device_train_batch_size` | **1** | `1` (matches default) | Unchanged from every prior cycle; one training step consumes exactly one preference pair, so "steps" and "epochs" relate 1:1 to record count, same as ADR-0013 through ADR-0016. |
| `save_steps` | **11** | `max(1, max_steps // 4)` = 5 (adapter default) | Overridden from the adapter's own default to match this project's own established convention instead: 50% of `max_steps` (one checkpoint at the run's midpoint plus the final checkpoint), the same fraction ADR-0014/0015/0016 each used (`round(0.5 * 22) = 11`), for continuity of checkpoint-cadence convention across this project's cycles rather than adopting the adapter's own newer, unrelated default fraction. |
| `save_total_limit` | **2** | `2` (matches default) | Unchanged from every prior cycle. |
| `max_prompt_length` | **512** (adapter/library default) | `512` (matches default) | The real ADR-0017 package's longest `prompt` field is 182 characters (~40-50 tokens); 512 tokens is generous headroom without truncation risk, and no evidence motivates a different value. |
| `max_length` | **1024** (adapter/library default) | `1024` (matches default) | The real package's longest single record's combined `prompt`+`chosen` length is 262+154 characters (`dpo-refusal-0004`, well under any plausible token-count near 1024); kept at default, same rationale as `max_prompt_length`. |
| `seed` | **20260923** | N/A | Fresh, run-specific seed distinct from every prior cycle's seed (`20260920` for ADR-0013, `20260922` reused by ADR-0014/0015/0016), per this project's own "fresh run id/seed per new proposal, never reuse a prior cycle's identity" convention (`0014-corrected-bounded-metatrainer-cycle.md`, "Rationale: fresh run id, approval namespace, and host paths"). |
| `trl` version | **`0.24.0`** (exact pin, unchanged) | — | Unchanged from every prior cycle's `EXPECTED_VERSIONS`; independently re-confirmed installed and importable in the exact pinned environment during this document's drafting (`transformers==4.56.1`, `datasets==3.0.0`, `accelerate==1.4.0`, `torch==2.8.0` — all five match `pyproject.toml`'s `trl-adapter` extra and every prior runner's `EXPECTED_VERSIONS` table exactly). |

### 4. Promotion/failure criteria, stated in advance

This cycle is measurement-only, exactly like ADR-0013 through ADR-0016 — no
promotion path exists in any runner on this project, and this document
proposes none.

**Primary target-item measurement:** `mtr-v2-heldout-0013`'s raw candidate
output, scored via the existing `meta_trainer` suite (48 held-out items,
automated `exact_match` — already established by
`docs/decisions/ADR-0015-outcome.md` as structurally near-zero-information
for this task shape, reported for completeness only, not as the primary
signal) **and** a fresh independent manual rubric review of the DPO
candidate on the same fixed 20-item set (`mtr-v2-heldout-0001`..`0020`) used
by every prior rubric review, scored by the same
4-axis 0/1 standard, performed by someone distinct from whoever executes the
run.

**Binding secondary criterion (my own recommendation from the PR #97/
review, carried forward explicitly as this ADR's stated failure condition,
per this task's instruction):** the `uncertainty_refusal_boundary` rubric
axis must not regress below its last real measured value. The only real
measured value for this axis is **ADR-0015's: 5/20 items = 0.25/1.0 axis
mean** (independently recomputed for this document from
the earlier rubric reviewer's own artifact,
(a reviewer's local evidence archive),
`part2_adr0015_candidate_rubric_scored_on_shared_20_item_set` items, summing
the `uncertainty_refusal_boundary` field across all 20 items: baseline 2/20 =
0.10, ADR-0013 3/20 = 0.15, ADR-0014 4/20 = 0.20, ADR-0015 5/20 = 0.25).
**ADR-0016 was never independently rubric-scored** — its own outcome record
(`docs/decisions/ADR-0016-outcome.md`, "ADR-0016's real result, in detail")
states explicitly: "No new independent manual rubric review of the ADR-0016
candidate was commissioned or performed as part of this cycle." This is a
real gap this document cannot paper over: the task's framing ("must not
regress vs the ADR-0015/0016 baseline") cannot be checked against an
ADR-0016 number that was never measured. This document's binding baseline is
therefore **ADR-0015's real 0.25/1.0**, the most recent real measurement of
this axis, and this gap (no ADR-0016 rubric baseline exists) is recorded here
as something this document could not verify because the underlying
measurement was never taken, not because it was overlooked.

**Target confabulation-refusal improvement on held-out items:** the DPO
candidate's rubric score on `mtr-v2-heldout-0013` specifically
(`technical_conclusion`/`reasoning_and_qualification`/`evidence_discipline`/
`uncertainty_refusal_boundary`, scored against the same reference answer "No.
Numeric recipes must be supported by cited evidence...") should move away
from ADR-0014/0015/0016's shared pattern of confidently endorsing or
inventing the confabulated recipe, toward a correct, hedged "No" — this is
the specific behavior the entire ADR-0015 -> ADR-0016 -> ADR-0017 chain has
been aimed at since the ADR-0015-regression root-cause diagnosis.

**What counts as a negative outcome (stated in advance, per this project's
own honesty discipline):** any of the following, individually, is a negative
result that must be reported as such, not rounded into a partial success:

1. `mtr-v2-heldout-0013`'s candidate output remains a confident endorsement
   of the confabulation (byte-identical or materially equivalent to ADR-0015/
   ADR-0016's "Yes... if it sounds plausible" pattern) — per ADR-0017's own
   "Consequences" section, this would be materially stronger evidence for a
   structural DPO/method-scale limitation on this specific item than exists
   today, not evidence the cycle "mostly worked."
2. The `uncertainty_refusal_boundary` rubric axis regresses below `0.25/1.0`
   (ADR-0015's real baseline) — this is the calibration-collapse/
   over-refusal risk Maya's ADR-0017 gate (b)(2) named, and the 27.27%
   counter-direction share was this package's concrete, and only,
   mitigation for it; if the axis regresses anyway, that is direct evidence
   the counter-direction pairs did not prevent the failure mode they were
   built to guard against, and must be reported as such rather than folded
   into "still a net improvement."
3. Any new critical error (per the same 8-category taxonomy
    already used) appears anywhere in the 20-item
   rubric-reviewed set that was not present in ADR-0015's candidate.
4. Training does not reach `TrainingStatus.ACCEPTED` (any resource-budget
   overrun, cancellation, or adapter-level rejection).

### 5. Evaluation path

**No evaluator code change is required or proposed by this document.**
`DPOTrainerAdapter.train()` calls `trainer.save_model(str(final_dir))` on a
full-parameter-trained model (`DPOTrainer` inherits directly from
`transformers.Trainer` via `BaseTrainer`, confirmed by the adapter's own
docstring reading `DPOTrainer.__mro__`), producing the exact same
`AutoModelForCausalLM`-compatible checkpoint-directory shape
`hf_local_evaluator_adapter.py` already loads for every prior SFT candidate
(ADR-0013 through ADR-0016) via `AutoModelForCausalLM.from_pretrained(str(path),
dtype=torch.float32)`. Because ADR-0017 Decision item 1 scopes this cycle to
**full-parameter DPO only, not LoRA+DPO**, the evaluator's independently
confirmed PEFT/adapter-merge gap (zero `PeftModel`/`merge_and_unload`/
`import peft` matches anywhere in `hf_local_evaluator_adapter.py`, re-verified
fresh for this document) is not exercised by this run and remains exactly
what ADR-0017 already stated: real, unresolved, and out of scope here. If a
future cycle wants to compare against a LoRA+DPO variant, that PEFT-scoring
gap becomes a real prerequisite and would need its own separately gated PR —
this document does not need or propose that work, and none should be bundled
into this document's own follow-on runner PR.

### 6. Resource/runtime envelope and rollback

- **`max_wall_seconds`:** unchanged at `1800` (ADR-0013 through ADR-0016's
  value). 22 steps is 5.2x fewer than ADR-0016's 115, so this remains
  generous headroom; no evidence motivates a change.
- **`max_cpu_seconds`:** unchanged at `3600`, same rationale.
- **`max_memory_mb`:** raised from ADR-0016's `16384` to **`20480`** (+4096MB,
  +25%). This is a fresh review, not a copy of ADR-0015/0016's numbers, per
  ADR-0017 Decision item 6's explicit instruction. Reasoning: DPO holds
  **two** full 135M-parameter model copies resident simultaneously (policy +
  frozen reference), where every prior real cycle held exactly one. The
  reference model adds a second copy's static weight-memory footprint (the
  real on-disk `model.safetensors` blob is 256.6MB, so a second resident copy
  in this rough size class) plus a second forward pass per training step (to
  compute reference log-probabilities), which adds transient activation
  memory beyond weight-residency alone — while optimizer state and gradients
  (the dominant memory cost under AdamW full-parameter training, per
  ADR-0015's own memory diagnosis) apply only to the policy
  model and are unchanged from the single-model SFT case. ADR-0016's real
  measured single-model peak was `11,604MB` under the `16384MB` ceiling
  (`4,780MB`/~41% headroom). A flat `+4096MB` increase is a conservative,
  explicitly-reasoned estimate for the second model's weight-plus-activation
  delta given the *specific, already-documented* MPS caching-allocator
  high-water-mark growth pattern ADR-0015's own real overshoot exhibited — this is an estimate pending the real first execution's own
  measurement, not a guarantee; if the real run overshoots this ceiling, that
  becomes its own fresh, evidence-based correction (a new PR raising the
  ceiling with the real measured peak cited), exactly as ADR-0015's own
  `8192 -> 16384` correction (PR #89) was handled, not a silent retry.
- **`max_storage_mb`:** unchanged at `1024`. The reference model reuses the
  same on-disk path as the policy model (see item 1) rather than a duplicated
  copy, so no additional static storage is required beyond what SFT runs
  already budgeted; `DPOTrainerAdapter`'s own `hf_cache_dir` redirection
  pattern is identical to `TRLTrainerAdapter`'s.
- **`network_policy`:** `offline`, enforced by `DPOTrainerAdapter.prepare()`
  itself (raises `RejectedInputError` otherwise), unchanged in mechanism from
  every prior cycle.
- **Concurrency:** exactly one run, enforced by the caller, unchanged.
- **Containment:** the same macOS Seatbelt complete-cycle host-containment
  profile and OS-level resource enforcement every prior real cycle used,
  re-scoped only to a fresh reviewed root
  (`./local-evidence/adr0018`, mirroring
  ADR-0013/0014/0015/0016's own reviewed-path pattern) — this document does
  not propose any change to the containment mechanism itself.
- **Rollback:** identical in mechanism to every prior cycle — the candidate
  artifact is written under an isolated, reviewed scratch root; rollback is
  deletion/quarantine of that directory with no production change; this
  document, like every predecessor, grants no promotion, deployment, or
  automatic retry authority. Nothing is promoted automatically regardless of
  outcome.

### 7. Gate blocks (all three UNSIGNED — this document signs none of them)

```
GATE 1 — SECURITY (Maya)
  Role: security-reviewer
  Scope required: evaluator-process-containment-v1, complete-cycle-host-containment-v1
  Decision: UNSIGNED — pending Maya's independent review of this exact
    document (config identity, beta/KL-strength justification per her own
    ADR-0017 gate (b)(4), reference-model provenance handling per her own
    gate (b)(3), sandbox/offline posture, and the resource-budget increase
    above) at this exact commit SHA. This document explicitly discloses
    that its drafting author (Marcus profile) also authored ADR-0017's own
    training-methodology assessment and PR #98's adapter/dataset code, so
    this Gate 1 review must come from Maya independently, not be inferred
    from any prior review of related but distinct artifacts.

GATE 2 — DATASET-RIGHTS (Maya)
  Role: dataset-rights-reviewer
  Decision: UNSIGNED — pending independent confirmation that the
    `examples/pilot-metatrainer-v3-dpo/` package (already PR #98-merged, not
    modified by this document) remains admissible for this specific proposed
    training use under this document's exact hyperparameter/config identity,
    per this project's standing practice of never treating a prior
    admission's sign-off as automatically covering a new proposed run.

GATE 3 — OWNER (Rook)
  Role: project-owner
  Decision: UNSIGNED — pending Rook's confirmation of this document's exact
    hyperparameter choices, resource-budget increase, and the explicit
    ADR-0016-rubric-baseline gap named in "Promotion/failure criteria" above,
    and authorization to proceed to writing (not executing) a gated runner
    script implementing this configuration.
```

No secret may be placed in any gate document. This document authorizes no
publication, deployment, promotion, repeated scheduling, continuous
unattended training, new credentials, or new egress — identical in kind to
every prior cycle's grant of authority (none beyond, eventually, one bounded
cycle, and not even that until a runner script and its own fresh gate
signatures exist).

## What this document does and does not authorize

- **This document authorizes no training run.** It is a hyperparameter and
  gate-config record only — the same category as
  `docs/decisions/ADR-0015-execution-config.md` and
  `docs/decisions/ADR-0016-execution-config.md`, neither of which itself
  contained an authorizing runner script either; each was followed by its
  own separate PR adding the actual `run_bounded_cycle_adrXXXX.py` script
  under its own review. **This document does not add a runner script.**
  Writing `examples/pilot-metatrainer-v2/run_bounded_cycle_adr0018.py` (or an
  equivalently-scoped new script for the DPO adapter) implementing exactly
  the configuration above is a required, separate follow-up PR, gated the
  same way every prior runner PR was, before any of the three gates above
  can be meaningfully signed against a concrete, reviewable script rather
  than this narrative document alone.
- It does not reinterpret or retroactively edit ADR-0013 through ADR-0017 or
  any of their outcome records; each remains its own authoritative,
  evidence-preserved record.
- It does not modify `src/codevolt_mdf/dpo_adapter.py`,
  `examples/pilot-metatrainer-v3-dpo/`, or any of their locked hashes — all
  are already-merged, unmodified inputs to this document.
- Repository admission of this document does not by itself authorize the
  eventual live cycle; per item 7 above, three fresh signed approvals bound
  to the exact runner-script SHA (once written) are required first, and none
  exist yet.

## Documentation impact and maintenance triggers

This decision adds `docs/decisions/ADR-0018-dpo-execution-config.md` (this
file). It does not modify `docs/decisions/ADR-0017-dpo-preference-refusal-axis.md`,
`src/codevolt_mdf/dpo_adapter.py`, `examples/pilot-metatrainer-v3-dpo/`, or
any prior ADR's own text or locked hash.

Per this project's standing practice (`docs/decisions/0015-metatrainer-corpus-v3.md`'s
and PR #91's precedent of pairing an outcome/decision record with a
case-study corpus-addition proposal in the same PR), this same PR also adds
`examples/metatrainer-corpus-addition-dpo-method-switch-lesson/` — a
DRAFT, NOT-YET-REVIEWED case-study proposal recording the ADR-0016 ->
ADR-0017 lesson (two independently constructed SFT corpus-rewrite attempts
at fixing one specific calibrated-refusal item both converged on the
identical wrong output; the project then switched training mechanism to
preference training rather than attempting a third rewrite). That file does
not modify `examples/pilot-metatrainer-v2/train.jsonl`/`held_out.json`,
`examples/pilot-metatrainer-v3/`, or `examples/pilot-metatrainer-v3-dpo/`,
and is not admitted to any training-eligible corpus by this PR — see its own
`DATASET_CARD.md` for the required independent-audit and review path before
any future admission.

Re-open or supersede this document before any run if any of the following
changes:

- model repo/revision/files/hash/licence;
- dataset/support file, split, source, admission, or hash, or the
  counter-direction share drops back below 20%;
- `DPOTrainerAdapter`'s implementation or contract version;
- Python or dependency pin (`trl`/`transformers`/`datasets`/`accelerate`/
  `torch`);
- renderer/template/decoding/rubric/threshold;
- device, precision, host class, budget, filesystem, network, or evidence
  path;
- review-gate schema or authority boundary;
- any of `beta`/`reference_free`/`max_steps`/`learning_rate`/`seed` above
  needs further adjustment — that is a new ADR, not an edit to this one, per
  this project's "no in-run sweep, no unrecorded retry" discipline
  (`0013-bounded-pretrained-metatrainer-cycle.md`).

After the one run (once a runner script exists and executes), append only
verified result/evidence references to a companion outcome record,
`ADR-0018-outcome.md`, mirroring `ADR-0015-outcome.md`/`ADR-0016-outcome.md`.
Do not rewrite this document as if predicted results had occurred.
