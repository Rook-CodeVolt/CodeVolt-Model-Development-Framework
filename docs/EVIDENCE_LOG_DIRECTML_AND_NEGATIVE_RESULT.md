# Evidence log: a real DirectML bug and a well-evidenced negative result from real GPU training

This page exists for the same reason as the rest of this evidence log: to
let a cold reader check, from committed evidence files rather than a
summary written after the fact, whether this project's "evidence before
claims" principle (see [README](../README.md#principles)) held up on real,
non-trivial work — not just on the deterministic demo.

Until now, the only end-to-end evidence trail in this repository came from
[issue #7's trainer-adapter pilot](EVIDENCE_LOG.md), which trained a tiny
model on a synthetic arithmetic task specifically because it was small and
easy to verify — useful for proving the contract works, but not a
demanding test of it. This page documents something different: an
independent internal team used this project's own methodology (bounded
experiments, predeclared thresholds, and independent review before any
result is accepted) to run a real, harder training investigation on real
GPU hardware. Sometimes "dogfooding" is used as shorthand for this —
a team using its own tool for its own real work, rather than only
testing it on toy cases. That real investigation surfaced two things
worth recording here: a genuine, previously-unreported upstream bug in a
third-party training library, and an honest negative result reached only
after independent review closed off every plausible alternative
explanation. Both outcomes are evidence that the process works on work
that actually mattered, not just on a demo built to make it look like it
works.

Both findings below come from real LoRA (Low-Rank Adaptation — a
parameter-efficient fine-tuning method) fine-tuning runs on real
consumer AI-accelerator hardware (an AMD integrated GPU, driven through
Microsoft's DirectML backend for PyTorch, since no supported ROCm
training path existed for that hardware at the time). Nothing here is a
production or capability claim about any model; both are process and
methodology findings.

## Finding 1: a real, reproducible, silently-wrong-gradient bug in `torch-directml`

**What happened.** During a controlled experiment that used a manual
per-token weighted loss (to up-weight a specific "cold start" token
during training), the training run completed without any error, produced
a plausible-looking, steadily decreasing loss curve, and reported a
successful result — but the model's trainable LoRA adapter weights never
actually changed. Diagnostic inspection after the run found all 120
LoRA `B`-matrix tensors still held their untouched zero-initialised
values after 444 real optimizer steps, and the training loss had in fact
been flat (~9.5) for all 6 epochs rather than decreasing as the logged
curve suggested.

**Root cause.** The manual loss used `torch.nn.functional.cross_entropy`
with `reduction='none'` (i.e. "give me the per-token loss values, don't
average them for me — I'll combine them myself"). On this build of
`torch-directml`, that specific code path builds a forward pass and an
autograd graph that both look completely normal, but its backward pass
silently produces a zero gradient. No error, warning, or NaN is raised —
the run simply trains nothing while reporting success.

**Minimal reproduction.** A standalone script
(`diag_gradflow_repro.py`, referenced in the source project's
`DECISIONS.md` entry D-008/adapter-v2) runs a single real
forward/backward pass three ways on identical inputs and compares the
resulting gradient norm on the same tensor:

| Loss computation | Gradient norm after backward() |
|---|---|
| `F.cross_entropy(reduction='none')` (fused kernel) | **0.0** — silently wrong |
| Manual `log_softmax` + `gather` (unfused, mathematically equivalent) | 0.0534 |
| Hugging Face Transformers' internal mean-reduction loss (control) | 0.0699 |

The manual unfused path and the HF-internal control both produce a real,
nonzero gradient on the exact same batch, model, and device. Only the
fused `reduction='none'` kernel on this DirectML build returns zero.
This isolates the bug to that one specific fused-kernel/reduction-mode
combination on `torch-directml`, not to the surrounding loss-weighting
logic, the model, or the dataset.

**Environment.**
- `torch==2.4.1+cpu`
- `torch-directml==0.2.5.dev240914`
- `transformers==4.46.3`, `peft==0.20.0`, `accelerate==1.15.0`
- AMD integrated GPU (Radeon 8060S-class iGPU), Windows, DirectML backend

**Why this matters for anyone else using `torch-directml` for training.**
Because the forward pass, the autograd graph construction, and the
reported loss curve all look entirely normal, this failure mode is
invisible unless you specifically check that your trainable parameters
actually changed after training — checking the loss curve alone is not
enough. A model can appear to train successfully, produce a smooth
decreasing loss curve, and complete without any error, while learning
nothing at all. This is the kind of bug that is easy to ship past
review by accident, because every visible signal looks fine.

**What we did about it, and what we didn't claim.** The immediate fix
was mechanical, not just a workaround for one run: rewrite the specific
loss computation to avoid the fused `reduction='none'` path (using
manual `log_softmax` + `gather` instead), and add a hard pre-flight
check before any full training run — one real forward/backward pass on
a real batch, asserting the gradient norm on a known trainable tensor is
non-zero, hard-aborting the run otherwise. That pre-flight check is now
a standing safeguard against this exact failure recurring undetected;
later rounds of the same investigation confirmed it stayed in place and
caught nothing further, i.e. it worked as a guard, not just a one-time
fix. We have not filed this with the `torch-directml` maintainers yet —
see the companion upstream report drafted alongside this document for
that next step. This project's own contribution is limited to: finding
the bug, root-causing it to the specific fused-kernel/reduction
combination, building a minimal reproduction, and fixing it locally; it
does not extend to any claim about the wider scope of the bug across
other operations or DirectML versions.

## Finding 2: a well-evidenced negative result, reached by closing real confounds instead of accepting the first negative answer

**The task.** A separate, unrelated line of work attempted to LoRA
fine-tune small open-weight language models to extract structured JSON
records from short text inputs (a bounded, well-defined
information-extraction task — not a general capability claim about
either model). Success was measured against a predeclared, fixed
threshold (0.85 exact full-record match rate on a held-out set that was
never used in training) decided *before* any run, specifically so the
bar could not be quietly moved to fit whatever result showed up.

**What was actually tried, over eight real training rounds:**

1. A 135M-parameter base model, plain prompt format, unweighted loss —
   collapsed into repeating the prompt template verbatim instead of
   producing JSON.
2. The same model with a per-token loss reweighting fix targeting the
   specific failure point (this is where Finding 1's DirectML bug was
   first discovered and fixed mid-investigation) — no improvement.
3. The same model with scheduled sampling (a training-time technique
   meant to reduce the gap between what the model sees during training
   and what it faces at generation time) — still complete collapse, but
   with a distinct diagnostic detail: every held-out failure was now an
   outright parse failure (no structured output produced at all) rather
   than a structured-but-wrong answer.
4. Format-priming — a small fixed set of worked examples prepended to
   every prompt, ending right at the point where the model should start
   writing JSON — produced the first non-zero signal in the whole
   investigation: 3 of 148 held-out records parsed successfully (versus
   zero in every prior round), though far too few to clear the
   predeclared threshold or count as a working fix.
5. A hard JSON-prefix handed directly to the model (removing even the
   decision of *when* to start writing JSON) — a different, but again
   total, collapse pattern, distinct from format-priming's partial
   signal.
6. Schema simplification (from 6 output fields down to 2) — collapse
   became *more* total, not less: every single held-out record failed
   to parse.
7. A 4-combination sweep of the LoRA fine-tuning hyperparameters
   (adapter rank, scaling factor, and learning rate) on the same base
   model — zero change in outcome across all four combinations.
8. The same task and harness run on a second, roughly 10x larger
   (1.5B-parameter) base model, first with a bare/completion-style
   prompt (also complete collapse), then correctly formatted with its
   native instruction-following chat template (see below for why this
   step existed at all) — still complete collapse, with the same
   qualitative failure pattern (short repeating token loops).

In every single round, the *training* loss collapsed to near-zero —
the model fit its training data fine. The failure was entirely in
generalisation to unseen inputs: on held-out data, every configuration
produced degenerate, repetitive, non-JSON output instead of the
extraction task.

**Why this is a genuine negative result and not just "we gave up
after one bad run."** The key discipline here is what happened *between*
those rounds. After round 6 (the larger model, bare prompt), an
independent reviewer — someone other than whoever ran the experiment,
checking the raw evidence files directly rather than trusting the
written summary — found a real, previously unflagged problem: the
1.5B model is specifically fine-tuned to expect its own conversational
chat-template formatting, and round 6 had prompted it with a plain
completion-style string instead. Instruction-tuned models are known to
behave unpredictably, including via exactly this kind of repetitive
collapse, when given prompts outside the format they were tuned for.
That meant the round-6 result could not yet support the conclusion
being drawn from it ("this isn't a model-capacity problem") — it might
just have been a prompt-formatting artifact. The reviewer named one
specific, cheap, decisive follow-up experiment to close that gap rather
than accepting the ambiguous result: re-run the identical setup with
the model's correct native chat template. Round 7 did exactly that and
produced the same collapse, closing the gap the reviewer had identified.
An earlier LoRA-hyperparameter gap raised by the same review process
was closed the same way, by round 5's sweep, before either was accepted
as closing the investigation.

**The conclusion.** With model scale (a 10x parameter-count difference),
the LoRA fine-tuning hyperparameters, prompt/schema reformulation, and
correct instruction-following prompt formatting all tested and ruled
out as explanations, the investigation concluded that this specific
combination of task, dataset, and evaluation harness — not either
model's underlying capacity — is what produced the total, repeated
collapse. That is a materially different and more useful conclusion
than "this model can't do this task," and it was only reached because
an independent reviewer twice pushed back on a too-early version of that
conclusion and named the exact cheap experiment needed to actually earn
it, rather than the investigation accepting its first negative-looking
result at face value.

**What this negative result does and does not establish.** It does not
establish that the underlying task (structured extraction into JSON) is
unsolvable by small models in general — only that this specific dataset,
schema, and evaluation harness, as actually built and tested across
eight rounds, did not work with either model tried. It does not
establish anything about larger models, different datasets, or
different training approaches (e.g. full fine-tuning rather than
LoRA), which were named as open questions rather than tested. It is a
negative result about one bounded piece of work, recorded so the next
attempt does not have to rediscover the same eight rounds' worth of
ruled-out explanations from scratch.

## Why both of these are being recorded here

This project's stated principle is to record rejected experiments as
useful knowledge, not just accepted ones (see the
[README's principles](../README.md#principles)). Both findings above are
exactly that: a real bug, root-caused and fixed rather than
silently worked around, and a real negative result, reached through
genuine independent review rather than asserted after one convenient
run. Neither is a claim about model capability, production readiness,
or this framework's own completeness — they are evidence that the
review discipline this project asks for produces real findings when
applied to real, harder work, not only to a demo built to exercise the
contract.
