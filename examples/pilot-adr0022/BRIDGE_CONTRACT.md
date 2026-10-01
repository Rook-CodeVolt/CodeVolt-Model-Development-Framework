# ADR-0022 LLF → MDF per-entry execution contract

The execute path runs exactly one admitted entry/model/host combination and the complete reviewed ADR-0022 item set per process. Live execution remains gated and requires independent Maya review.

## Invocation

```text
python examples/pilot-adr0022/run_adr0022_baseline_eval.py \
  --execute \
  --entry-id <entry> \
  --model-id <reference|candidate|third_model> \
  --host-id <host> \
  --item-set-id pilot-adr0022-heldout-v1 \
  --merged-registry <current-merged-registry.json> \
  --review-gate <review-gate.json> \
  --scratch-root <reviewed-scratch-root> \
  --wall-clock-limit-seconds <positive-number> \
  --cost-inputs-json <cost-inputs.json>
```

`--cost-inputs-json` must name a direct regular file containing a JSON object. `--merged-registry` must name a direct, non-symlink regular file containing the current merged `HeldOutExclusionRegistry` snapshot. Dry-run accepts an optional single `--model-id` and never scores.

Admitted mappings are fixed:

| entry_id | model_id | host_id |
| --- | --- | --- |
| `meta_trainer_adr0022_smol135m_test1` | `reference` | `test1_cv_test1` |
| `meta_trainer_adr0022_smol135m_test1` | `candidate` | `test1_cv_test1` |
| `meta_trainer_adr0022_baseline` | `third_model` | `test2_evo_x3_102` |

All other combinations fail closed.

## Fixed item set

Every call evaluates all 56 IDs in source order: C1=24, C2=12, C4=20. Byte-level SHA-256 pins are:

- C1: `5494f655b8798b92cf643bb950a2b6f9022d6e564e3eba920e25663af64ca2a1`
- C2: `5926e4bbed5b9732d6a5b6d2ec47b18a9810a5535a8823003e1277d42ffd0c62`
- C4: `f3971c50b0f6fd0cd44a393ce61beac30180ed66cbd29f3442fcbd6310003278`

The execution-time merged-registry check is additional to the gate-bound split-time registry. MDF reads, hashes and parses the same bytes immediately before starting the isolated evaluator; requires the package registration to equal the exact 56 IDs; rejects any intersection with any package's train IDs; and verifies file identity plus digest again before evidence finalisation.

## Output

Successful stdout is one JSON locator object:

```json
{"result_json":"/absolute/path/adr0022_result.json","result_sha256":"/absolute/path/adr0022_result.json.sha256"}
```

The result preserves the existing ledger fields (`per_model`, `third_model_queue_admission_floor`, `isolation`, `max_wall_seconds`, `max_memory_mb`) while `per_model` contains exactly one selected-model key. It also records `runner`, explicit `outcome` booleans, invocation identity and cost inputs, the contamination verdict, and evidence paths. The third-model floor is `null` for reference/candidate calls; for `third_model` it includes the three capability booleans, numeric `capabilities_passed`, and `overall_sufficient_signal` (threshold: at least 2 capabilities).

Missing/malformed/mutated registry state, overlap, wrong mapping, changed item bytes/order, bad gate or artifact identity, resource failure, evaluator failure, or an incomplete/reordered report exits nonzero and never records accepted/evaluated as true. A pre-score contamination refusal writes a refusal artifact under the reviewed model scratch directory with `outcome.status="refused"`, both booleans false, no scores, and the structured contamination verdict.
