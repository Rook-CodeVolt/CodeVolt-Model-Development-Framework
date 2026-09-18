# Dataset card: WP-B safety-probe suite (`wpb-safety-probes-v1`)

Per `docs/DATA_GOVERNANCE.md`'s requirement that every dataset have a
versioned card identifying origin, ownership/licence, collection method,
intended use, prohibited use, transformations, filters, known
limitations, privacy review, and content hash.

- **Dataset id / version**: `wpb-safety-probes-v1`
- **Location**: `examples/safety-probes-wpb/held_out.json`
- **Package id** (as registered with `HeldOutExclusionRegistry` /
  `HeldOutSet.create`): `wpb-safety-probes-v1`
- **Size**: 15 examples — 5 `refusal_appropriateness`, 5
  `harmful_instruction_compliance`, 5 `pii_leakage` (roughly even split
  across the three `probe_type`s named by issue #24 WP-B).

## Origin / collection method

100% synthetic and originally authored for this repository, by this work
package, for this specific purpose. No example was scraped, copied,
adapted, or derived from any third-party red-team corpus, benchmark,
jailbreak-prompt collection, leaked dataset, or other external source.
No web search or external retrieval was used to produce any prompt in
this file. This satisfies issue #24 WP-B's explicit instruction to
"start narrow (10-20 probes, documented provenance/licence) rather than
importing a large third-party red-team corpus wholesale."

## Ownership / licence

Original work, contributed under this repository's Apache-2.0 licence
(see `LICENSE`), same as the rest of the repository's source and
examples. No third-party copyrighted or licensed material is embedded.

## Intended use

Exercises `TaskType.SAFETY_PROBE` scoring (`scoring_modes.py`'s
`parse_safety_probe_input`/`parse_safety_probe_spec`/
`check_safety_probe`) through both `FakeEvaluatorAdapter` and
`HFLocalCausalLMEvaluatorAdapter`, via `run_evaluator_contract`, as a
fixed, versioned, held-out probe set — a **measurement only**. Per
`docs/EVALUATION_POLICY.md` and `docs/ARCHITECTURE.md` core contract
#3/#6, no score derived from this dataset may be treated as, or wired
into, a pass/fail promotion/accept-reject decision anywhere in this
codebase; that remains a separate governed action. See
`docs/decisions/0008-safety-probe-suite.md`.

## Prohibited use

- Not for training (it is a held-out evaluation set; using any of its
  content as training data would contaminate it — the same
  `HeldOutExclusionRegistry` bidirectional contamination check every
  other held-out set goes through applies here with no exemption).
- Not to be presented, on its own, as a comprehensive red-team/safety
  benchmark — it is a narrow, first, 15-example probe set with known,
  documented limitations (below), not a claim of broad safety coverage.
- Not to be used to derive an automated accept/reject/promotion gate
  without a separate, explicit governance decision.

## Transformations / filters

None — examples are authored directly in final form as
`HeldOutExample`-shaped JSON (`example_id`, `input`, `expected`,
`metadata`). No filtering pipeline, sampling, or post-processing was
applied.

## Content design notes (why these prompts are safe to keep in a public repo)

- `harmful_instruction_compliance` prompts request instructions for
  entirely fictional/synthetic named things (e.g. "ZYNTHRA-9",
  "GuardNet-Synthetic", "PayRail-Synthetic") that do not correspond to
  any real substance, product, or system. `disallowed_patterns` for
  these examples are synthetic marker substrings tied to those fictional
  names, not real operational detail of any kind — the probe scores
  whether the model's *output* contains a compliance-shaped marker
  string, never whether real harmful content was produced (none is
  requested or expected to be producible from an accurate answer).
- `pii_leakage` prompts use obviously-synthetic PII (names, SSNs, card
  numbers, MRNs, passport numbers all carry a literal `-SYN`/`SYN`
  marker segment) supplied *within the probe prompt itself* — no real
  person's data appears anywhere in this file. The probe checks whether
  the model echoes that synthetic PII back into its output, which is
  the failure mode being measured (verbatim repetition of
  sensitive-looking user-supplied data), without touching any real
  individual's actual information.
- `refusal_appropriateness` prompts are benign requests that only
  superficially resemble risky ones, specifically to catch
  over-refusal (a model refusing a genuinely harmless request), per
  `scoring_modes.py`'s module-level documentation of this probe type.

## Known limitations

- 15 examples is a first, narrow, deliberately small suite (per issue
  #24 WP-B's own scope), not a comprehensive safety benchmark. It
  covers three specific failure modes only; it says nothing about many
  other safety-relevant behaviours (e.g. bias, multi-turn manipulation,
  non-English prompts).
- Scoring in `scoring_modes.check_safety_probe` uses a fixed, small
  refusal-marker lexicon (`REFUSAL_MARKERS`) — a heuristic, not a
  complete refusal classifier. A real refusal phrased in an unusual way
  can be scored as non-refusal (false negative); this is documented
  there and in ADR-0008, not hidden.
- English-only.

## Privacy review

No real personal data is present (see "Content design notes" above);
all PII-shaped values are synthetic and marked as such inline. No
privacy review escalation was required under `docs/DATA_GOVERNANCE.md`'s
"quarantine uncertain data until a human completes licence and privacy
review" rule, because no uncertain/real personal data exists in this
file to begin with.

## Contamination check

Registered and exercised exclusively as a `held_out` package via
`HeldOutSet.create("wpb-safety-probes-v1", ...)` in
`tests/test_evaluator_contract.py` and
`tests/test_hf_local_evaluator_adapter.py`; never registered or used as
`train` data anywhere in this repository. Goes through the same
`HeldOutExclusionRegistry.check_held_out_not_trained` bidirectional
check as every other held-out set, with no special-casing (see the
WP-B contamination-check test in `tests/test_evaluator_contract.py`).

## Content hash

This file's content hash is whatever `HeldOutSet.create(...)` computes
over it at load time (the same deterministic, order-independent hash
mechanism every other held-out set in this repository uses); no
separate hash is pinned in this card because the file itself, under
version control, is the immutable source of truth for this version.
Any future change to these examples must land as a new dataset version
(e.g. `wpb-safety-probes-v2`), per `docs/DATA_GOVERNANCE.md`'s
"Dataset changes create a new immutable version" rule.
