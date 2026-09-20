#!/usr/bin/env python3
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent

FAMILIES = {
    "zero_score_diagnostic_ladder": {"split": "train", "scope": "Seed-rich A, SS4.2"},
    "sft_objective_masking_format": {"split": "train", "scope": "Seed-rich A, SS1.1"},
    "qualified_hyperparameter_sweeps": {"split": "train", "scope": "Seed-rich A, SS1.2; Seed-rich B, SS5"},
    "peft_lora_qlora_decisions": {"split": "held_out", "scope": "Seed-rich A, SS2"},
    "contamination_split_controls": {"split": "train", "scope": "Seed-rich A, SS3.2"},
    "limited_data_evaluation": {"split": "train", "scope": "Seed-rich B, SS4.1"},
    "synthetic_data_tradeoffs": {"split": "held_out", "scope": "Seed-rich B, SS3.3"},
    "catastrophic_forgetting_retention": {"split": "held_out", "scope": "Seed-rich A, SS1.3"},
}

ROWS = []
def add(family, prompt, answer, citations):
    ROWS.append((family, prompt, answer, citations))

# TRAIN FAMILY 1: zero-score diagnostic ladder (8)
add("zero_score_diagnostic_ladder",
    "A 40-example, one-epoch run reports aggregate_score=0.0. What conclusion is justified before inspecting anything else?",
    "Treat the run as a pipeline smoke test, not an efficacy verdict. A literal zero can be plausible after limited exposure, but it can also reflect missing targets, no meaningful updates, malformed generations, or evaluator failure. The result is inconclusive until the training and evaluation path is checked.",
    ["Clara synthesis, SS4.2"])
add("zero_score_diagnostic_ladder",
    "The dashboard shows 0.0, but missing or NaN results might be converted to zero. What is the first diagnostic question?",
    "Determine whether 0.0 means zero successful items or whether missing, NaN, or unparsed values were coerced to zero. This separates an actual all-item failure from an aggregation or parsing failure before changing training settings.",
    ["Clara synthesis, SS4.2"])
add("zero_score_diagnostic_ladder",
    "How can you test whether a zero-scoring evaluator can recognize a correct answer?",
    "Run the evaluator on a hand-written known-good output and also run the unchanged base model through the same path. If the known-good output is rejected, repair the evaluator or parser before interpreting model quality.",
    ["Clara synthesis, SS4.2"])
add("zero_score_diagnostic_ladder",
    "A training job completed without error, yet every evaluation item failed. What data inspection should happen before a hyperparameter sweep?",
    "Inspect rendered training records and target-token masks, then count supervised tokens. A successful process exit does not prove that assistant or completion tokens were actually trained.",
    ["[1] TRL v0.21 SFT Trainer — Computing the loss; Train on assistant messages only", "Clara synthesis, SS4.2"])
add("zero_score_diagnostic_ladder",
    "What evidence shows that a nominal training run performed real optimization?",
    "Confirm that at least one optimizer update occurred and that trained parameters or the checkpoint changed. Example count and epoch count alone do not establish the number of updates.",
    ["[1] TRL v0.21 SFT Trainer — logged optimizer steps", "Clara synthesis, SS4.2"])
add("zero_score_diagnostic_ladder",
    "Which logged traces are useful when investigating a zero downstream score?",
    "Inspect per-step training loss, target-token accuracy, learning rate, and gradient norm. Flat learning signals, non-finite gradients, or an ineffective schedule point to different failure modes.",
    ["[1] TRL v0.21 SFT Trainer — logged loss, token accuracy, learning rate, and gradient norm", "[2] Axolotl Training Stability & Debugging — plateau and instability diagnostics", "Clara synthesis, SS4.2"])
add("zero_score_diagnostic_ladder",
    "Why generate on several training prompts during a zero-score investigation?",
    "Failure even on training prompts suggests a pipeline, masking, update, or learning-rate problem. Success on training prompts with held-out failure instead suggests memorization or insufficient diversity.",
    ["Clara synthesis, SS4.2"])
add("zero_score_diagnostic_ladder",
    "Does a 40-example, one-epoch score of zero prove that 40 examples can never be useful?",
    "No. The evidence supports no universal sample-count threshold or categorical conclusion from that run alone. Task scope, base model, token exposure, update count, formatting, and evaluator behavior remain relevant.",
    ["Clara synthesis, SS4.2", "Clara synthesis, SS5"])

# TRAIN FAMILY 2: SFT objective, masking, and format (8)
add("sft_objective_masking_format",
    "What objective does supervised fine-tuning optimize in the cited TRL guidance?",
    "SFT minimizes token-level next-token cross-entropy over selected supervised tokens. Padding is masked, and supported schemas can restrict loss to completion or assistant tokens.",
    ["[1] TRL v0.21 SFT Trainer — Computing the loss"])
add("sft_objective_masking_format",
    "When is completion-only loss useful for prompt-completion data?",
    "It is useful when the prompt supplies context and the desired signal is the completion. Exclude prompt tokens, retain completion tokens, and inspect the rendered sample to verify the mask.",
    ["[1] TRL v0.21 SFT Trainer — completion-only loss", "Clara synthesis, SS1.1"])
add("sft_objective_masking_format",
    "What prerequisite is needed for assistant-only loss on conversational data?",
    "The chat template must expose an assistant-token mask that identifies assistant messages. Without a correct mask, the intended assistant-only supervision may not be applied.",
    ["[1] TRL v0.21 SFT Trainer — Train on assistant messages only"])
add("sft_objective_masking_format",
    "Why must training and inference use the same tokenizer and chat template?",
    "A mismatch changes the serialized conversation. Keep tokenizer, chat template, role tokens, and EOS policy aligned so the learned response boundary is the one used at inference.",
    ["[1] TRL v0.21 SFT Trainer — chat templates and EOS alignment", "Clara synthesis, SS1.1"])
add("sft_objective_masking_format",
    "Loss decreases, but prompt text rather than answer text was supervised. Is the loss evidence of task learning?",
    "No. Loss can decrease on the wrong tokens. Inspect the rendered sample and token mask to confirm that intended assistant or completion tokens receive supervision.",
    ["[1] TRL v0.21 SFT Trainer — completion/assistant-only loss", "Clara synthesis, SS1.1"])
add("sft_objective_masking_format",
    "How should max_length be chosen for an SFT corpus?",
    "Measure the sequence-length distribution first. A limit that is too small discards evidence, while an unnecessarily large limit wastes memory and padding; choose from observed lengths and task retention needs.",
    ["Clara synthesis, SS1.1"])
add("sft_objective_masking_format",
    "What is the benefit and caution of TRL packing?",
    "Packing can group examples to use sequence capacity efficiently. The wrapped strategy can break sequence continuity, so verify that its behavior fits the intended data and attention semantics.",
    ["[5] TRL Reducing Memory Usage — packing and wrapped packing"])
add("sft_objective_masking_format",
    "Why inspect tokenized examples instead of trusting only source JSON?",
    "The model trains on rendered tokens, not source objects. Token inspection can reveal truncated answers, missing assistant tokens, wrong delimiters, or supervision assigned to unintended text.",
    ["[1] TRL v0.21 SFT Trainer — loss masking and chat templates", "[2] Axolotl Training Stability & Debugging — inspect tokenized samples", "Clara synthesis, SS1.1"])

# TRAIN FAMILY 3: qualified hyperparameter sweeps (8)
add("qualified_hyperparameter_sweeps",
    "Give a defensible full-SFT learning-rate region from Clara's evidence pack.",
    "Use roughly 1e-5 to 5e-5 only as a validation-driven sweep region. Axolotl calls it typical, a Hugging Face tutorial uses 2e-5, and MiniMind's cited script defaults to 1e-5; none is a universal optimum.",
    ["[2] Axolotl Training Stability & Debugging — full fine-tuning 1e-5 to 5e-5", "[3] Hugging Face Transformers Fine-tuning — tutorial uses 2e-5", "[11] MiniMind train_full_sft.py — default 1e-5", "Clara synthesis, SS1.2"])
add("qualified_hyperparameter_sweeps",
    "Give a defensible LoRA or QLoRA learning-rate region from Clara's evidence pack.",
    "Use roughly 1e-4 to 3e-4 as sweep anchors, not a law. TRL notes about 1e-4 for adapters, Axolotl gives 1e-4 to 3e-4, and Unsloth recommends 2e-4 as a starting point.",
    ["[1] TRL v0.21 SFT Trainer — Train adapters with PEFT", "[2] Axolotl Training Stability & Debugging — LoRA 1e-4 to 3e-4", "[4] Unsloth LoRA Hyperparameters Guide — 2e-4 starting point", "Clara synthesis, SS1.2"])
add("qualified_hyperparameter_sweeps",
    "How should the cited 1-3 epoch guidance be stated?",
    "State it as an initial experiment range. Unsloth recommends 1-3 for instruction datasets and warns about diminishing returns or overfitting beyond that; the cited Hugging Face tutorial uses 3 and MiniMind defaults to 2. Validation should decide.",
    ["[3] Hugging Face Transformers Fine-tuning — tutorial uses 3 epochs", "[4] Unsloth LoRA Hyperparameters Guide — 1 to 3 epochs", "[11] MiniMind train_full_sft.py — default 2 epochs", "Clara synthesis, SS1.2"])
add("qualified_hyperparameter_sweeps",
    "How is effective batch size calculated in the cited guidance?",
    "Effective batch size is micro-batch size multiplied by gradient-accumulation steps and device count. Report all components rather than only one batch field.",
    ["[2] Axolotl Training Stability & Debugging — effective batch formula"])
add("qualified_hyperparameter_sweeps",
    "Why can a large effective batch be poor for a 40-example experiment?",
    "It can leave only a handful of optimizer updates. Keep micro-batches small enough to produce observable updates, log them, and compare controlled settings; this is diagnostic guidance, not a universal batch rule.",
    ["Clara synthesis, SS1.2"])
add("qualified_hyperparameter_sweeps",
    "What is the single correct LR for a 30M-parameter model on 40 examples?",
    "No retrieved source establishes one from model size and example count alone. Validate the pipeline, then compare a small LR and epoch grid on development data while reporting tokens, updates, and held-out behavior.",
    ["Clara synthesis, SS5"])
add("qualified_hyperparameter_sweeps",
    "What full-SFT sweep anchors does Clara recommend for the baseline?",
    "Use about 1e-5, 2e-5, and 5e-5 with 1-3 epochs and validation-based selection. They are engineering starting points assembled from framework guidance, not guaranteed optima.",
    ["Clara synthesis, SS5", "[2] Axolotl Training Stability & Debugging — full fine-tuning 1e-5 to 5e-5", "[3] Hugging Face Transformers Fine-tuning — tutorial uses 2e-5 and 3 epochs", "[4] Unsloth LoRA Hyperparameters Guide — 1 to 3 epochs", "[11] MiniMind train_full_sft.py — default 1e-5 and 2 epochs"])
add("qualified_hyperparameter_sweeps",
    "Frameworks recommend different adapter learning rates. Does that identify one source as wrong?",
    "No. TRL, Axolotl, Unsloth, and Axolotl's quickstart offer overlapping but non-identical recipes. Treat variation as evidence for a bounded sweep rather than false consensus on one number.",
    ["[1] TRL v0.21 SFT Trainer", "[2] Axolotl Training Stability & Debugging", "[4] Unsloth LoRA Hyperparameters Guide", "[23] Axolotl Quickstart", "Clara synthesis, SS1.2"])

# TRAIN FAMILY 4: PEFT, LoRA, QLoRA decisions (8)
add("peft_lora_qlora_decisions",
    "What does PEFT mean in this evidence pack?",
    "PEFT is a family of methods that adapts a pretrained model by training a small subset of parameters or added parameters rather than updating every base weight.",
    ["[6] LoRA paper — frozen pretrained weights and low-rank adaptation", "Clara synthesis, SS2"])
add("peft_lora_qlora_decisions",
    "Describe the core LoRA update without importing large-model memory ratios.",
    "LoRA freezes a pretrained weight matrix and learns a low-rank update, commonly written BA, whose rank is much smaller than the original dimensions. The update can be merged into the base for inference.",
    ["[6] LoRA paper — abstract and Section 4.1"])
add("peft_lora_qlora_decisions",
    "Can the LoRA paper's 10,000x trainable-parameter and roughly 3x memory figures be promised for a 30M model?",
    "No. Those figures come from the paper's GPT-3 175B setting and must not be projected onto a tiny model. Measure the actual implementation instead.",
    ["[6] LoRA paper — GPT-3 175B experimental setting", "Clara synthesis, SS2"])
add("peft_lora_qlora_decisions",
    "What is QLoRA's core training arrangement?",
    "QLoRA keeps the pretrained base frozen and quantized to 4-bit while backpropagating into LoRA adapters. Its named mechanisms include NF4, double quantization, and paged optimizers.",
    ["[7] QLoRA paper — abstract and Sections 1-3"])
add("peft_lora_qlora_decisions",
    "Does fitting a 65B model on one 48GB GPU prove a memory ratio for a 26M model?",
    "No. The result is bounded to the QLoRA paper's tested setting. It demonstrates a large-model memory technique, not a ratio that can be extrapolated to MiniMind-scale models.",
    ["[7] QLoRA paper — 65B model on one 48GB GPU", "Clara synthesis, SS2"])
add("peft_lora_qlora_decisions",
    "When is full SFT reasonable to benchmark?",
    "Benchmark it when the model fits comfortably, the desired shift is broad, and adapters may lack capacity. This is Clara's qualified decision guidance, not a direct claim that full SFT always wins on small models.",
    ["Clara synthesis, SS2"])
add("peft_lora_qlora_decisions",
    "When is LoRA a reasonable default candidate?",
    "Use it for narrow or evolving domains, reversible experiments, several domain adapters, or when retaining an untouched base matters. For tiny models, also benchmark full SFT rather than assuming PEFT wins.",
    ["Clara synthesis, SS2", "[6] LoRA paper — frozen base and low-rank adapters"])
add("peft_lora_qlora_decisions",
    "When should QLoRA be chosen according to Clara's guidance?",
    "Choose it when the base does not fit at 16-bit. It addresses a memory constraint and is not inherently a quality upgrade; if a 26M-135M model already fits, quantization complexity may be unnecessary.",
    ["Clara synthesis, SS2", "[7] QLoRA paper — quantized frozen base with LoRA adapters"])

# TRAIN FAMILY 5: contamination and split controls (8)
add("contamination_split_controls",
    "When should train, development, and test manifests be established relative to paraphrase generation?",
    "Establish immutable manifests before generating paraphrases, Q&A, or synthetic expansions. Otherwise related derivatives can be assigned independently and leak across splits.",
    ["Clara synthesis, SS3.2"])
add("contamination_split_controls",
    "Why split by source document or topic family instead of by row?",
    "Group-wise splitting keeps chunks, paraphrases, and adjacent scenarios from one source or concept family on one side. Row-level random splitting can place near-duplicates into training and evaluation.",
    ["Clara synthesis, SS3.2", "[16] Google ML Crash Course — separate train/validation/test data and avoid duplicates"])
add("contamination_split_controls",
    "What exact-overlap rule is directly supported by Google's cited guidance?",
    "Validation and test examples must not duplicate training examples. Exact duplicate removal is necessary, although it does not catch paraphrases by itself.",
    ["[16] Google ML Crash Course — Dividing the original dataset"])
add("contamination_split_controls",
    "Why are exact hashes insufficient as the only contamination check?",
    "They miss normalized variants and paraphrases. Clara recommends normalized-text, token-n-gram, and semantic-similarity scans plus an overlap ledger; that operational prescription is her synthesis, not a claim that Google specifies those mechanisms.",
    ["Clara synthesis, SS3.2", "[16] Google ML Crash Course — exact duplicate separation"])
add("contamination_split_controls",
    "Should every fuzzy match be deleted automatically?",
    "No. Aggressive fuzzy filtering can remove legitimate recurring technical language. Review context and record the disposition rather than treating similarity alone as proof of leakage.",
    ["Clara synthesis, SS3.2"])
add("contamination_split_controls",
    "What should happen if hidden evaluation answers or grader rationales are used for training?",
    "Retire and replace the affected test set before reporting new results. Repeated tuning against evaluation material wears out its independence.",
    ["[16] Google ML Crash Course — repeated use wears out validation/test data", "Clara synthesis, SS3.2"])
add("contamination_split_controls",
    "Why deduplicate within training as well as across splits?",
    "Within-training duplicates can increase memorized emission and distort effective data. Lee et al. found duplicates and train/validation overlap in all four NLP datasets studied, and deduplication improved evaluation validity in their experiments.",
    ["[17] Lee et al., Deduplicating Training Data Makes Language Models Better"])
add("contamination_split_controls",
    "Two records share common technical terms but describe unrelated source families. Is that alone a split failure?",
    "No. Common technical language can recur legitimately. Inspect source, scenario, and paraphrase lineage; decide leakage from semantic relationship, not wording alone.",
    ["Clara synthesis, SS3.2"])

# TRAIN FAMILY 6: limited-data evaluation (8)
add("limited_data_evaluation",
    "What pipeline gates should precede interpretation of task scores?",
    "Confirm that examples parse, supervised target-token count is positive, optimizer updates are nonzero, losses and gradients are finite, the checkpoint changed, and inference uses the expected template and EOS behavior.",
    ["Clara synthesis, SS4.1", "[1] TRL v0.21 SFT Trainer — metrics and masking"])
add("limited_data_evaluation",
    "How should learning diagnostics compare a base and tuned model?",
    "Compare base and tuned train/development loss and target-token accuracy at checkpoints, then evaluate both on the same task items. Pairing controls prompt differences and exposes the tuning delta.",
    ["Clara synthesis, SS4.1", "[1] TRL v0.21 SFT Trainer — loss and token-accuracy metrics"])
add("limited_data_evaluation",
    "When should exact checks be preferred over an open-ended judge?",
    "Use deterministic exact or schema checks for structured tasks. Add rubric-based human or calibrated-judge scoring only when the target requires semantic judgment.",
    ["[20] OpenAI Evaluation Best Practices — task-specific evaluation and constrained judgments", "Clara synthesis, SS4.1"])
add("limited_data_evaluation",
    "What makes an evaluation production-shaped?",
    "Use task-specific prompts and outputs resembling intended use, and compare fixed alternatives with clear criteria rather than vague open-ended impressions.",
    ["[20] OpenAI Evaluation Best Practices", "Clara synthesis, SS4.1"])
add("limited_data_evaluation",
    "What challenge slices does Clara recommend for the bounded trainer model?",
    "Include unseen phrasings, contradictory premises, insufficient-evidence cases, wrong-tool choices, LR/epoch troubleshooting, contamination questions, and out-of-scope requests. This list is Clara's synthesis.",
    ["Clara synthesis, SS4.1"])
add("limited_data_evaluation",
    "How should uncertainty be reported for a small held-out set?",
    "Report numerator, denominator, and per-item paired deltas, not only an aggregate. Repeat stochastic generation and, when feasible, training seeds so variability is visible.",
    ["Clara synthesis, SS4.1"])
add("limited_data_evaluation",
    "What does tinyBenchmarks support, and what does it not support?",
    "It supports carefully curated 100-item subsets estimating specific large-benchmark scores within about 2% on average in the paper's IRT- and prior-model-informed setting. It does not justify any arbitrary 100 questions as universally accurate.",
    ["[21] tinyBenchmarks paper — benchmark-specific 100-item subsets using IRT and previous-model data", "Clara synthesis, SS4.1"])
add("limited_data_evaluation",
    "Why pin the evaluation harness, task revision, and prompt?",
    "Harness fixes can change results. The cited release corrected few-shot leakage, answer-filter errors, and standard-error aggregation, so record revisions, tokenizer/template, decoding settings, and raw outputs.",
    ["[22] lm-evaluation-harness v0.4.13 release notes", "Clara synthesis, SS4.1"])

# HELD-OUT FAMILY 1: synthetic data trade-offs (6), wholly withheld
add("synthetic_data_tradeoffs",
    "What positive result does Self-Instruct demonstrate without implying a tiny-model sample threshold?",
    "Self-Instruct expanded 175 human-written seed tasks into about 52K instructions and 82K instances while filtering invalid, repeated, and overly similar generations. It demonstrates controlled synthetic expansion, not that any count is sufficient for a small model.",
    ["[18] Self-Instruct paper — Sections 1-2", "Clara synthesis, SS3.3"])
add("synthetic_data_tradeoffs",
    "Does the model-collapse paper show that every synthetic dataset is harmful?",
    "No. It studies indiscriminate recursive training on model-generated data and finds progressive loss of distribution tails. It does not establish that a curated one-generation mixture necessarily collapses.",
    ["[19] Nature model-collapse paper — abstract and Definition 2.1", "Clara synthesis, SS3.3"])
add("synthetic_data_tradeoffs",
    "A synthetic explanation is fluent but adds factual details absent from its retrieved source. What should the reviewer do?",
    "Reject or revise the unsupported details. Synthetic items must preserve their source claims and receive claim-level verification against retrieved sources; fluent wording is not evidence that a new factual claim is grounded.",
    ["Clara synthesis, SS3.3"])
add("synthetic_data_tradeoffs",
    "A generated example passes factual review. May its synthetic-origin label now be removed?",
    "No. Factual verification does not change the item's origin. Keep every synthetic item labeled so generated material remains distinguishable from human-authored or primary-source anchors throughout corpus review and use.",
    ["Clara synthesis, SS3.3"])
add("synthetic_data_tradeoffs",
    "Can a synthetic answer invent a learning-rate recipe if it sounds plausible?",
    "No. Numeric recipes must be supported by cited evidence and retain their qualifications. Plausibility is not a substitute for a source-backed anchor.",
    ["Clara synthesis, SS3.3"])
add("synthetic_data_tradeoffs",
    "Name the main synthetic-data risks Clara identifies.",
    "They are generator bias, factual errors, reduced diversity, hidden benchmark leakage, and recursive self-training effects. Their presence and severity should be measured rather than assumed.",
    ["Clara synthesis, SS3.3", "[19] Nature model-collapse paper — recursive-training risk"])

# HELD-OUT FAMILY 2: catastrophic forgetting and retention (6), wholly withheld
add("catastrophic_forgetting_retention",
    "What pattern indicates catastrophic forgetting or capability regression?",
    "The target task improves while pre-existing general or instruction-following behavior declines. Evaluate both the new capability and retained base behavior rather than treating target-task gain as sufficient.",
    ["Clara synthesis, SS1.3"])
add("catastrophic_forgetting_retention",
    "Why does full SFT carry a different regression risk from LoRA?",
    "Full SFT changes all model weights, while LoRA freezes the base and learns low-rank updates. Freezing can reduce risk and make the adapter reversible, but it does not guarantee served behavior cannot regress.",
    ["[6] LoRA paper — frozen base weights and low-rank updates", "Clara synthesis, SS1.3", "Clara synthesis, SS2 — reversible experiments"])
add("catastrophic_forgetting_retention",
    "Does using LoRA eliminate the need for retention testing?",
    "No. LoRA confines parameter changes to adapters, but the combined model can still change outputs. Keep a frozen base-regression suite and compare base versus adapted behavior.",
    ["[6] LoRA paper — frozen base and learned adapters", "Clara synthesis, SS1.3"])
add("catastrophic_forgetting_retention",
    "What is the purpose of a frozen base-regression suite?",
    "It supplies a stable reference for pre-existing instruction following, basic language behavior, safety boundaries, and prior capabilities. Clara recommends it as a control; its exact composition should match deployment scope.",
    ["Clara synthesis, SS1.3", "Clara synthesis, SS4.1 — capability-retention suite composition"])
add("catastrophic_forgetting_retention",
    "Target-task accuracy rose, but basic instruction following worsened. Should the checkpoint be promoted?",
    "Not on target-task gain alone. Treat the retention failure as material evidence against promotion until the trade-off is evaluated against predeclared criteria.",
    ["Clara synthesis, SS1.3", "Clara synthesis, SS5"])
add("catastrophic_forgetting_retention",
    "What is an evidence-qualified promotion rule for a tuned checkpoint?",
    "Require a predeclared target-task gain, no material base-regression failure, and a paired improvement larger than observed run or evaluation variability. This is Clara's recommended gate, not a universal external standard.",
    ["Clara synthesis, SS5"])


def build_records():
    counters = {"train": 0, "held_out": 0}
    records = []
    for family, prompt, answer, citations in ROWS:
        split = FAMILIES[family]["split"]
        counters[split] += 1
        prefix = "mtr-v2-train" if split == "train" else "mtr-v2-heldout"
        example_id = f"{prefix}-{counters[split]:04d}"
        source_scope = FAMILIES[family]["scope"]
        if example_id == "mtr-v2-heldout-0020":
            source_scope = "Seed-rich B, SS5"
        records.append({
            "example_id": example_id,
            "split": split,
            "semantic_family": family,
            "source_scope": source_scope,
            "citations": citations,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
        })
    return records


def main():
    records = build_records()
    assert len(records) == 60
    train = [r for r in records if r["split"] == "train"]
    held = [r for r in records if r["split"] == "held_out"]
    assert len(train) == 40 and len(held) == 20

    with (OUT / "train.jsonl").open("w", encoding="utf-8") as f:
        for record in train:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    (OUT / "held_out.json").write_text(json.dumps(held, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = []
    for family, metadata in FAMILIES.items():
        ids = [r["example_id"] for r in records if r["semantic_family"] == family]
        manifest.append({
            "semantic_family": family,
            "split": metadata["split"],
            "source_scope": metadata["scope"],
            "count": len(ids),
            "example_ids": ids,
        })
    (OUT / "semantic_family_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    registry = {
        "schema_version": "2.0",
        "split_rule": "Every semantic_family was assigned wholly to one split before examples were authored.",
        "train_families": [m for m in manifest if m["split"] == "train"],
        "held_out_families": [m for m in manifest if m["split"] == "held_out"],
        "held_out_example_ids": [r["example_id"] for r in held],
    }
    (OUT / "held_out_exclusion_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    representative_ids = [
        "mtr-v2-train-0001",
        "mtr-v2-train-0012",
        "mtr-v2-train-0026",
        "mtr-v2-heldout-0002",
        "mtr-v2-heldout-0017",
    ]
    representatives = [next(r for r in records if r["example_id"] == rid) for rid in representative_ids]
    (OUT / "REPRESENTATIVE_EXAMPLES.json").write_text(json.dumps(representatives, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"total": len(records), "train": len(train), "held_out": len(held), "families": len(manifest)}))


if __name__ == "__main__":
    main()
