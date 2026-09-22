# Source map — meta-trainer corpus v3

This corpus reuses sources [1]-[23] verbatim from `examples/pilot-metatrainer-v2/SOURCE_MAP.md`
(Clara's 2026-09-20 research bibliography) for continuity of the
established, already-independently-audited source classes, and adds sources [24]-[30], each
directly fetched and read in full (abstract/body text, not paraphrased from a secondary summary)
during the drafting of this corpus on 2026-09-22. No source below was reproduced verbatim in any
corpus record; every record is a synthetic Q&A pair that cites the specific claim the source
supports.

## Carried forward from corpus v2 (unchanged; see pilot-metatrainer-v2/SOURCE_MAP.md for full list)

[1] Hugging Face, "SFT Trainer," TRL v0.21.0 documentation. https://huggingface.co/docs/trl/v0.21.0/sft_trainer

[2] Axolotl, "Training Stability & Debugging." https://docs.axolotl.ai/docs/training_stability.html

[3] Hugging Face, "Fine-tuning," Transformers documentation. https://huggingface.co/docs/transformers/main/en/training

[4] Unsloth, "LoRA fine-tuning Hyperparameters Guide." https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide

[5] Hugging Face, "Reducing Memory Usage," TRL documentation. https://huggingface.co/docs/trl/en/reducing_memory_usage

[6] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," arXiv:2106.09685 v2. https://arxiv.org/pdf/2106.09685

[7] Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," arXiv:2305.14314. https://ar5iv.labs.arxiv.org/html/2305.14314

[11] Jingyao Gong, MiniMind `trainer/train_full_sft.py`, commit `3f1a7cc25b19a861cd1bd6ed313be526b9ecdaf8`. https://github.com/jingyaogong/minimind/blob/3f1a7cc2/trainer/train_full_sft.py

[16] Google, "Datasets: Dividing the original dataset," ML Crash Course. https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets

[17] Lee et al., "Deduplicating Training Data Makes Language Models Better," arXiv:2107.06499 v2. https://arxiv.org/html/2107.06499v2

[18] Wang et al., "Self-Instruct: Aligning Language Models with Self-Generated Instructions," ACL 2023 / arXiv:2212.10560. https://arxiv.org/html/2212.10560

[19] Shumailov et al., "AI models collapse when trained on recursively generated data," Nature 631. https://doi.org/10.1038/s41586-024-07566-y

[20] OpenAI, "Evaluation best practices." https://developers.openai.com/api/docs/guides/evaluation-best-practices

[21] Polo et al., "tinyBenchmarks: evaluating LLMs with fewer examples," arXiv:2402.14992. https://arxiv.org/html/2402.14992v1

[22] EleutherAI, `lm-evaluation-harness` v0.4.13 release notes. https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.13

[23] Axolotl, "Quickstart." https://docs.axolotl.ai/docs/getting-started

(Sources [8], [9], [10], [12], [13], [14], [15] from corpus v2's bibliography are not cited by
any v3 record and are omitted here for brevity; see the v2 file if needed.)

## New sources added for corpus v3 (fetched and read directly, 2026-09-22)

[24] Kadavath et al., "Language Models (Mostly) Know What They Know," arXiv:2207.05221 (Anthropic).
https://arxiv.org/abs/2207.05221
Verified claims used: larger models are well-calibrated on multiple-choice/true-false questions
given the right format; models can be trained to predict "P(IK)" (probability the model "knows"
the answer) without reference to a specific proposed answer; P(IK) partially generalizes across
tasks but the paper explicitly states models "struggle with calibration of P(IK) on new tasks."

[25] Lin, Hilton, and Evans, "TruthfulQA: Measuring How Models Mimic Human Falsehoods,"
arXiv:2109.07958. https://arxiv.org/abs/2109.07958
Verified claims used: 817-question, 38-category benchmark; best tested model was truthful on 58%
of questions vs. 94% human performance; larger models were generally *less* truthful in this
benchmark (on the multiple-choice task, the 6B-parameter GPT-J model was 12% less truthful than
its 125M-parameter counterpart; the paper's separate generation-task finding is that the largest
GPT-Neo/J model is 17% less truthful than a model 60x smaller, with no specific model size named
for that comparison), which the paper contrasts explicitly with other NLP tasks where performance
improves with model size; the paper's term "imitative falsehoods" for plausible answers learned
from imitating human text rather than truthfully reasoned.

[26] Kalai, Nachum, Vempala, and Zhang, "Why Language Models Hallucinate," arXiv:2509.04664
(OpenAI). https://arxiv.org/abs/2509.04664 and https://openai.com/index/why-language-models-hallucinate/
Verified claims used: standard training/evaluation procedures reward guessing over acknowledging
uncertainty; a worked reported comparison table (gpt-5-thinking-mini: 52% abstention / 22%
accuracy / 26% error; OpenAI o4-mini: 1% abstention / 24% accuracy / 75% error) showing a model
that abstains far more often can have a much lower error rate for a similar accuracy rate; the
explicit outcome ranking "accurate responses, errors, and abstentions" with errors treated as
worse than abstentions; the birthday-guessing illustration (a random guess has some non-zero
chance of being right, "I don't know" guarantees zero points under naive accuracy grading, and
grading on accuracy alone rewards the guesser on average); the claim that pretraining sees only
positive examples of fluent text with no true/false labels, so some hallucination is a natural
statistical consequence, not a mysterious glitch; the claim that a good hallucination eval alone
does not fix the incentive problem because it competes against many traditional accuracy-graded
evals that still reward guessing.

[27] Manakul, Liusie, and Gales, "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for
Generative Large Language Models," arXiv:2303.08896. https://arxiv.org/abs/2303.08896
Verified claims used: a sampling-based approach that fact-checks black-box model outputs without
an external database or access to output probabilities; the core idea that if a model has real
knowledge of a concept, independently sampled responses tend to be mutually consistent, while
hallucinated content tends to diverge and contradict across samples; evaluated on GPT-3-generated
WikiBio passages with human factuality annotation; the method's own comparison is against
grey-box (probability-requiring) baselines.

[28] OpenAI, "Model Spec" (2025-02-12 revision), section "Express uncertainty" (Guideline
authority). https://model-spec.openai.com/2025-02-12.html#express_uncertainty
Verified claims used (exact text read in full): the guideline's outcome ranking "confident right
answer > hedged right answer > no answer > hedged wrong answer > confident wrong answer"; the
listed causes of uncertainty (knowledge/reasoning limitations, outdated information, ambiguous
user intent, inherent world limitations such as subjective or private matters, and predictions of
future states); the instruction to express uncertainty by default in natural conversational
language rather than quantified percentages unless requested; the example phrases given verbatim
in the spec: "I don't know", "I'm not sure", "I was unable to solve ...", "I think", "I believe",
"It might be", "If I understand what you mean", "If my calculations are correct", "If my sources
are correct", "If my information is up to date".

[29] Hugging Face, "Exact Match" metric card, `evaluate` library.
https://huggingface.co/spaces/evaluate-metric/exact_match (README.md)
Verified claims used (exact text read in full): "A given predicted string's exact match score is
1 if it is the exact same as its reference string, and is 0 otherwise"; the metric card's own
worked example scoring "Happy Birthday!" against reference "Happy New Year!" as 0; the
aggregate score is the mean of the per-example binary scores; optional normalization parameters
(`ignore_case`, `ignore_punctuation`, `ignore_numbers`, `regexes_to_ignore`) exist but do not
change the underlying full-string-equality (or, in variant implementations, full-string
containment) comparison unit.

[30] OpenAI, "Findings from a pilot Anthropic-OpenAI alignment evaluation exercise."
https://openai.com/index/openai-anthropic-safety-evaluation/
Verified claims used: on the cited hallucination evaluations, "Claude models had an extremely
high rate of refusals—as much as 70%," which the post states "shows these models are aware of
their uncertainty" while noting "the high refusal rate limits utility"; the contrasted finding
that "OpenAI o3 and OpenAI o4-mini show lower refusal rates with higher hallucination rates" in
the same tool-restricted setting; the description of the "Person Hallucination Test (v4)," which
"allows the model to explicitly refuse to answer when uncertainty is too high"; and the
description of "SimpleQA No Browse (v1)" as a fact-seeking short-answer benchmark answered from
internal knowledge only, with the stated overall pattern that the higher-refusing models produced
fewer hallucinated errors but also fewer overall correct answers than the lower-refusing models.

## Citation policy (unchanged in kind from corpus v2, extended in label)

Direct factual claims use a numbered source `[N]` where that source directly supports the
statement. Cross-source or generalizing conclusions that no single numbered source states on its
own are labeled `Marcus synthesis, MSx` and resolved against `SYNTHESIS_NOTES.md` in this same
directory (the v3 analogue of Clara's `SSX.X` locators) rather than being misattributed to one
topically related source. `Marcus synthesis` labels are used only where the corresponding
`SYNTHESIS_NOTES.md` entry itself names the specific numbered source(s) the synthesis step draws
on and states the added inferential step explicitly, so the chain from claim to source is always
traceable.

No licensing determination was made for any source in this file.
