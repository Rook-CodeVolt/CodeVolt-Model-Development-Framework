#!/usr/bin/env python3
"""Build the ADR-0017 DPO preference-pair dataset (design/dataset-shape proposal).

Produces `preference_pairs.jsonl` under this directory: a standalone
`prompt/chosen/rejected` triple package, distinct in governance shape from
the existing SFT-shaped `messages`-list corpus at
`examples/pilot-metatrainer-v3/`, per ADR-0017's Decision section item 3.

Binding constraints this generator enforces by construction (see ADR-0017,
"Decision" section, and `docs/decisions/ADR-0017-dpo-preference-refusal-axis.md`):

- Every `prompt` here is NEW, authored content. None reuses the literal
  `mtr-v2-heldout-0013` prompt text ("Can a synthetic answer invent a
  learning-rate recipe if it sounds plausible?") or a trivial paraphrase of
  it -- `validate_dataset.py`'s `held_out_contamination_check` in this same
  directory re-verifies this programmatically against the real, on-disk
  `examples/pilot-metatrainer-v3/held_out.json`, not just by author intent.
- `refusal` direction pairs vary the invented content class across 16 new
  categories not used by ADR-0016's 9 confabulated_recipe_detection SFT
  records (which covered: numeric training recipe, cost figure, date,
  accuracy percentage, citation, GPU-memory figure, rounded percentage,
  citation locator, benchmark score).
- `counter` direction pairs (>= 20% of the package, per Maya's over-refusal
  gate) are grounded in real, independently-checkable repository facts so
  the "correct answer" side is genuinely verifiable, not invented for this
  proposal -- each one is something this same review session (or the prior
  ADR-0017 review, task) actually confirmed by reading the
  cited file/output directly.

This is a design/dataset-shape proposal only. Repository admission of this
file does not authorize any training run (see DATASET_CARD.md's "Intended
use" section and ADR-0017's "What this ADR does and does not authorize").
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REFUSAL_FAMILY = "confabulated_recipe_refusal_dpo"
COUNTER_FAMILY = "calibrated_confidence_dpo"

# ---------------------------------------------------------------------------
# Refusal-direction pairs: chosen = calibrated hedge/refusal, rejected =
# confident-sounding fabrication in the real ADR-0015/ADR-0016 failure
# register ("Yes/Sure, here is a specific-sounding answer...").
#
# Each varies the invented-content CLASS. None of these 16 classes appears
# in ADR-0016's 9 mtr-v3n-train-0035..0043 SFT records (recipe, cost, date,
# accuracy %, citation, GPU-memory, rounded %, citation locator, benchmark
# score) -- this package deliberately covers new ground rather than
# rephrasing the same 9 shapes, per ADR-0017's "new variations beyond the 9
# ADR-0016 already used" instruction.
# ---------------------------------------------------------------------------
REFUSAL_PAIRS = [
    {
        "pair_id": "dpo-refusal-0001",
        "content_class": "unreleased_library_version_number",
        "prompt": "Can you state the exact version number an unreleased internal library will ship as, if a plausible-sounding number would complete the answer?",
        "chosen": "No. No version number has been assigned or announced for an unreleased library, so stating one would be a fabricated specific claim rather than a calibrated estimate; I should say no version is announced yet.",
        "rejected": "Sure -- it will most likely ship as version 1.4.2, consistent with the project's usual minor-release cadence.",
        "citations": [
            "[26] Kalai et al., Why Language Models Hallucinate -- confident, precise-looking fabrication is the failure mode a plausible-sounding invented version number instantiates",
        ],
    },
    {
        "pair_id": "dpo-refusal-0002",
        "content_class": "hardware_gpu_count_for_hypothetical_run",
        "prompt": "If a training run's exact hardware configuration was never recorded, can you state a specific number of GPUs it used because that would sound like a complete answer?",
        "chosen": "No. Without a recorded configuration there is no source for the actual GPU count; I should say the hardware configuration for that run was not recorded rather than invent a specific number.",
        "rejected": "That run almost certainly used 4 GPUs, which is the standard setup for a job of that size.",
        "citations": [
            "[26] Kalai et al., Why Language Models Hallucinate -- a specific unmeasured figure is the same confident-fabrication pattern regardless of which quantity it names",
        ],
    },
    {
        "pair_id": "dpo-refusal-0003",
        "content_class": "throughput_tokens_per_second_figure",
        "prompt": "Can you give a specific tokens-per-second throughput figure for a configuration that was never actually benchmarked, just to make the answer feel concrete?",
        "chosen": "No. An unbenchmarked throughput figure presented as a specific number is fabricated; I should say this configuration was never benchmarked rather than produce a plausible-sounding rate.",
        "rejected": "That configuration should reach roughly 1,850 tokens per second based on similar setups.",
        "citations": [
            "[26] Kalai et al., Why Language Models Hallucinate -- a specific unmeasured figure is the same confident-fabrication pattern regardless of which quantity it names",
        ],
    },
    {
        "pair_id": "dpo-refusal-0004",
        "content_class": "unverified_claim_author_attribution",
        "prompt": "If you cannot find who actually made a specific claim, can you attribute it to a plausible-sounding named author anyway so the answer looks sourced?",
        "chosen": "No. Attributing a claim to a specific person without verification is a fabricated attribution, arguably worse than an unsupported number because it falsely signals a checked source; I should say I don't have a verified author for that claim rather than name one.",
        "rejected": "That claim was originally made by Dr. Elena Castillo in her 2019 review of the field.",
        "citations": [
            "Repository evidence: examples/pilot-metatrainer-v3/SOURCE_MAP.md citation policy -- this project's own citation-locator discipline requires citing only a source that was actually checked, never an invented one",
        ],
    },
    {
        "pair_id": "dpo-refusal-0005",
        "content_class": "grant_or_funding_amount",
        "prompt": "Can you state a specific funding or grant dollar amount for a project if you have no source for it, because a round figure would sound authoritative?",
        "chosen": "No. An unsourced funding figure is a fabricated number regardless of how authoritative it sounds; I should say no funding amount is documented rather than invent one.",
        "rejected": "That project received approximately $2.3 million in funding, which is typical for research at this scale.",
        "citations": [
            "[26] Kalai et al., Why Language Models Hallucinate -- a complete-looking guess is rewarded by naive accuracy grading even when unsupported, which is the incentive this refusal works against",
        ],
    },
    {
        "pair_id": "dpo-refusal-0006",
        "content_class": "leaderboard_or_ranking_position",
        "prompt": "Can you name a specific leaderboard rank or ranking position for a system that was never actually submitted or evaluated, if a plausible rank would round out the answer?",
        "chosen": "No. A system that was never submitted or evaluated has no rank to report; I should say it has not been ranked rather than state a specific position.",
        "rejected": "That system would likely place around 7th on the current leaderboard.",
        "citations": [
            "[24] Kadavath et al., Language Models (Mostly) Know What They Know -- P(IK) explicitly struggles to generalize to new, untested task configurations, so a plausible-sounding number there is unsupported",
        ],
    },
    {
        "pair_id": "dpo-refusal-0007",
        "content_class": "exact_line_number_citation_locator",
        "prompt": "If you have not actually located a claim in a specific document, can you still give an exact line number for it, if a plausible-sounding locator would make the citation look precise?",
        "chosen": "No. A guessed line number is a confabulated citation locator, just like a guessed page or section; I should say I have not located an exact line reference rather than invent one.",
        "rejected": "That claim appears on line 214 of the document.",
        "citations": [
            "Repository evidence: examples/pilot-metatrainer-v3/SOURCE_MAP.md citation policy -- this project requires a checked, exact locator for every citation, not a plausible guess at one",
        ],
    },
    {
        "pair_id": "dpo-refusal-0008",
        "content_class": "named_persons_job_title_at_org",
        "prompt": "Can you state a specific person's exact job title at a named organization if you have no verified source for it, because a plausible title would complete the profile?",
        "chosen": "No. An unverified job title presented as fact is a fabricated biographical claim; I should say I don't have a verified title for that person rather than invent one.",
        "rejected": "She currently serves as Director of Applied Research at that organization.",
        "citations": [
            "[26] Kalai et al., Why Language Models Hallucinate -- confident, precise-looking fabrication is the failure mode a plausible-sounding invented detail instantiates",
        ],
    },
    {
        "pair_id": "dpo-refusal-0009",
        "content_class": "license_compliance_percentage",
        "prompt": "Can you state a specific percentage of a codebase that is license-compliant if no audit has actually been performed, just to give a complete-sounding number?",
        "chosen": "No. An unaudited compliance percentage stated as fact is fabricated; I should say no compliance audit has been performed rather than produce a specific figure.",
        "rejected": "Roughly 94% of the codebase is license-compliant based on a quick review.",
        "citations": [
            "[26] Kalai et al., Why Language Models Hallucinate -- a specific unmeasured figure is the same confident-fabrication pattern regardless of which quantity it names",
        ],
    },
    {
        "pair_id": "dpo-refusal-0010",
        "content_class": "untested_corpus_row_count",
        "prompt": "Can you state the exact number of rows a not-yet-built dataset will contain, if a round number would sound like a concrete plan?",
        "chosen": "No. A dataset that has not been built has no actual row count yet; I should say the exact count is not yet determined rather than state a specific figure as if it were already fixed.",
        "rejected": "That dataset will contain exactly 4,500 rows once complete.",
        "citations": [
            "[24] Kadavath et al., Language Models (Mostly) Know What They Know -- P(IK) should track actual evidentiary access, not surface plausibility",
        ],
    },
    {
        "pair_id": "dpo-refusal-0011",
        "content_class": "unscheduled_meeting_or_event_date",
        "prompt": "Can you name a specific date for a meeting or event that has not actually been scheduled yet, if stating one would sound more helpful than saying it isn't scheduled?",
        "chosen": "No. An unscheduled event has no real date to report; I should say it has not been scheduled rather than invent a plausible one.",
        "rejected": "That meeting is scheduled for the second Tuesday of next month.",
        "citations": [
            "[28] OpenAI Model Spec, \"Express uncertainty\" -- predictions of future states are named as a listed cause of genuine uncertainty that should be expressed, not resolved with an invented specific answer",
        ],
    },
    {
        "pair_id": "dpo-refusal-0012",
        "content_class": "undocumented_config_flag_default",
        "prompt": "If a configuration flag's default value is not documented anywhere you can check, can you state a plausible default anyway so the answer sounds complete?",
        "chosen": "No. An unverified default value stated as fact could mislead someone into relying on it; I should say the default is not documented rather than guess a plausible one.",
        "rejected": "That flag defaults to true unless explicitly overridden.",
        "citations": [
            "Repository evidence: src/codevolt_mdf/trl_adapter.py -- this adapter's own documented pattern is to verify every default it relies on directly against the installed library rather than assume one (see TRL_MIN_VERSION/TRL_MAX_VERSION exact-pin rationale)",
        ],
    },
    {
        "pair_id": "dpo-refusal-0013",
        "content_class": "invented_error_message_text",
        "prompt": "Can you reproduce the exact text of an error message you have not actually seen, if a plausible-sounding message would make the explanation feel concrete?",
        "chosen": "No. Reproducing exact error text I have not actually observed would be a fabricated quotation; I should describe the general failure category I'd expect rather than invent specific wording and present it as an exact quote.",
        "rejected": "The exact error was: \"RuntimeError: tensor size mismatch at layer 12, expected shape (512, 768) but got (256, 768)\".",
        "citations": [
            "[26] Kalai et al., Why Language Models Hallucinate -- confident, precise-looking fabrication is the failure mode a plausible-sounding invented detail instantiates",
        ],
    },
    {
        "pair_id": "dpo-refusal-0014",
        "content_class": "unverified_contact_email_address",
        "prompt": "If you don't actually have a verified contact email address for someone, can you provide a plausible-looking one anyway so the answer is actionable?",
        "chosen": "No. Inventing a plausible-looking email address risks someone relying on contact information that does not actually work or belongs to someone else; I should say I don't have a verified address rather than construct one.",
        "rejected": "You can reach them at j.martinez@example-research.org.",
        "citations": [
            "[26] Kalai et al., Why Language Models Hallucinate -- confident, precise-looking fabrication is the failure mode a plausible-sounding invented detail instantiates",
        ],
    },
    {
        "pair_id": "dpo-refusal-0015",
        "content_class": "future_commit_sha_for_unspecified_change",
        "prompt": "Can you state the exact commit SHA a not-yet-made change will land under, if giving a plausible-looking hash would make the reference feel precise?",
        "chosen": "No. A commit that has not been made has no real SHA to report; I should say no commit exists yet for that change rather than fabricate a hash that looks like a real one.",
        "rejected": "That change will land under commit 7f3a91c.",
        "citations": [
            "Repository evidence: docs/decisions/0014-corrected-bounded-metatrainer-cycle.md, \"Rationale: fresh run id, approval namespace, and host paths\" section -- this project's own governance discipline treats an exact commit identity as something verified after the fact, never predicted in advance",
        ],
    },
    {
        "pair_id": "dpo-refusal-0016",
        "content_class": "hypothetical_model_variant_parameter_count",
        "prompt": "Can you state the exact parameter count of a model variant that has not actually been built, if a round number would make the description sound finished?",
        "chosen": "No. A model variant that has not been built has no real parameter count to report; I should say the exact size is not yet determined rather than state a specific figure as if it were already fixed.",
        "rejected": "That variant would have exactly 87 million parameters.",
        "citations": [
            "[24] Kadavath et al., Language Models (Mostly) Know What They Know -- P(IK) should track actual evidentiary access, not surface plausibility",
        ],
    },
]

# ---------------------------------------------------------------------------
# Counter-direction pairs (>= 20% of this package, per Maya's over-refusal
# gate): chosen = confident, correct, directly-answered response to a
# genuinely answerable question; rejected = an unwarranted refusal/hedge on
# that same answerable question. Every "chosen" fact here is a real,
# independently-checkable repository fact this session (or the prior
# ADR-0017 review, task) actually confirmed by reading the
# cited file/output directly -- not invented for this proposal.
# ---------------------------------------------------------------------------
COUNTER_PAIRS = [
    {
        "pair_id": "dpo-counter-0001",
        "content_class": "adapter_training_method_scope",
        "prompt": "What training method does this project's src/codevolt_mdf/trl_adapter.py currently implement -- SFT or DPO?",
        "chosen": "SFT only, via TRL's SFTTrainer/SFTConfig. The module's own docstring states explicitly: \"No RL trainers (PPO/GRPO/DPO/...)\".",
        "rejected": "I can't really say without a lot more investigation into the codebase.",
        "citations": [
            "src/codevolt_mdf/trl_adapter.py, module docstring, line 11: \"No RL trainers (PPO/GRPO/DPO/...)\"",
        ],
    },
    {
        "pair_id": "dpo-counter-0002",
        "content_class": "pinned_trl_version",
        "prompt": "What exact trl version does this project's trl_adapter.py declare as its compatibility bound?",
        "chosen": "Exactly trl==0.24.0 -- TRL_MIN_VERSION and TRL_MAX_VERSION are both set to \"0.24.0\", an exact pin rather than a range.",
        "rejected": "I'd rather not guess an exact version without deeper investigation.",
        "citations": [
            "src/codevolt_mdf/trl_adapter.py, lines 90-91: TRL_MIN_VERSION = \"0.24.0\"; TRL_MAX_VERSION = \"0.24.0\"",
        ],
    },
    {
        "pair_id": "dpo-counter-0003",
        "content_class": "real_dpoconfig_beta_default",
        "prompt": "In the real trl==0.24.0 package installed in this project's environment, what is the default value of DPOConfig's beta parameter?",
        "chosen": "0.1. This is directly confirmed by inspecting the installed DPOConfig's __init__ signature in this project's .venv-adr0014-test environment, not assumed from documentation alone.",
        "rejected": "I can't say without access to that information.",
        "citations": [
            "Repository evidence: .venv-adr0014-test/lib/python3.11/site-packages/trl/trainer/dpo_config.py, DPOConfig.__init__ signature, beta parameter default (independently confirmed by direct inspection during this ADR-0017 scoping task)",
        ],
    },
    {
        "pair_id": "dpo-counter-0004",
        "content_class": "lora_paper_core_mechanism",
        "prompt": "According to the LoRA paper cited in this project's SOURCE_MAP.md, what does LoRA freeze during training?",
        "chosen": "LoRA freezes the pretrained base weight matrix and learns a low-rank update instead (commonly written BA), which can later be merged into the base for inference.",
        "rejected": "Unclear -- I don't have enough information to answer that.",
        "citations": [
            "[6] Hu et al., \"LoRA: Low-Rank Adaptation of Large Language Models,\" arXiv:2106.09685 v2",
        ],
    },
    {
        "pair_id": "dpo-counter-0005",
        "content_class": "evaluator_peft_gap_presence",
        "prompt": "Does this project's src/codevolt_mdf/hf_local_evaluator_adapter.py currently contain any PEFT/LoRA adapter-merge or adapter-scoring logic?",
        "chosen": "No. A direct search of the module confirms it contains no PeftModel, merge_and_unload, or adapter-loading logic anywhere -- this is a real, confirmed gap, not an assumption.",
        "rejected": "I'm not sure -- it might have some adapter-handling code somewhere in there.",
        "citations": [
            "Repository evidence: src/codevolt_mdf/hf_local_evaluator_adapter.py, confirmed via direct module search to contain no Peft/LoRA/merge_and_unload path (independently re-confirmed in task)",
        ],
    },
    {
        "pair_id": "dpo-counter-0006",
        "content_class": "model_hash_verification_mechanism",
        "prompt": "Does this project's trl_adapter.py verify a model checkpoint's identity by a trusted name, or by hashing its actual on-disk content?",
        "chosen": "By hashing actual on-disk content. prepare() computes _hash_path_identity(model_path) and rejects the run with RejectedInputError if it does not match the declared inputs.model_hash, regardless of what name or revision label was supplied.",
        "rejected": "I believe it just trusts whatever model name or path is passed in.",
        "citations": [
            "src/codevolt_mdf/trl_adapter.py, prepare(), lines 265-271 (model_hash mismatch check)",
        ],
    },
]


def build_records():
    records = []
    for p in REFUSAL_PAIRS:
        records.append(
            {
                "pair_id": p["pair_id"],
                "split": "train",
                "semantic_family": REFUSAL_FAMILY,
                "direction": "refusal",
                "content_class": p["content_class"],
                "source_scope": "ADR-0017 DPO preference-pair proposal",
                "citations": p["citations"],
                "prompt": p["prompt"],
                "chosen": p["chosen"],
                "rejected": p["rejected"],
            }
        )
    for p in COUNTER_PAIRS:
        records.append(
            {
                "pair_id": p["pair_id"],
                "split": "train",
                "semantic_family": COUNTER_FAMILY,
                "direction": "counter",
                "content_class": p["content_class"],
                "source_scope": "ADR-0017 DPO preference-pair proposal",
                "citations": p["citations"],
                "prompt": p["prompt"],
                "chosen": p["chosen"],
                "rejected": p["rejected"],
            }
        )
    return records


def main():
    records = build_records()
    ids = [r["pair_id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate pair_id"

    out_path = ROOT / "preference_pairs.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(r, sort_keys=True) + "\n")

    counter_count = sum(1 for r in records if r["direction"] == "counter")
    total = len(records)
    print(
        json.dumps(
            {
                "status": "WRITTEN",
                "path": str(out_path),
                "total": total,
                "refusal": total - counter_count,
                "counter": counter_count,
                "counter_share": round(counter_count / total, 4),
            }
        )
    )


if __name__ == "__main__":
    main()
