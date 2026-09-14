# Training-loop reliability

This document collects known failure classes that a training loop can hit
silently — no exception, a plausible-looking loss curve, but no real
learning — and the mitigation patterns this project recommends in
response. It is a home for hardware/software-backend-specific caveats and
for the general engineering practice that protects against the whole
class, independent of any one specific cause.

## Failure class: silent zero-gradient training

**Symptom:** a training run completes without error, and its logged loss
curve looks plausible (flat or slowly moving, no `NaN`, no explosion), but
the run never actually updates its trainable parameters — gradients into
those parameters are exactly zero for the entire run. Because the failure
produces no exception and no obviously wrong metric, it is only detectable
by inspecting parameter state directly (e.g. comparing checkpoint tensor
values/norms before and after training), which most run harnesses do not
do by default.

This is dangerous specifically because it is silent: a run in this state
reports success, consumes real compute, and can be mistaken for a
legitimately hard-to-fit dataset or a bad hyperparameter choice rather
than a computation that never trained anything.

### Known-bad case: DirectML + `F.cross_entropy(reduction='none')` with a custom downstream reduction

Observed on `torch-directml` 0.2.5.dev240914 (`torch` 2.4.1+cpu,
`transformers` 4.46.3, `peft` 0.20.0, AMD Radeon 8060S iGPU / Ryzen AI
Max+ 395, Windows). Calling `torch.nn.functional.cross_entropy(logits,
labels, reduction='none')` and then applying a manual downstream reduction
(for example a per-token weighted mean, as used for curriculum/cold-start
loss weighting) produces:

- a forward-pass loss value that matches a manual `log_softmax` + `gather`
  NLL reimplementation on the identical batch to several decimal places;
- a normal-looking autograd graph (`NllLossBackward0`);
- but an exactly-zero backward gradient (a real, non-`None`, all-zero
  tensor) for every upstream trainable parameter.

A concrete instance of this: a 6-epoch LoRA fine-tune completed with no
exception and a flat-but-plausible loss curve; post-hoc inspection showed
every LoRA `B` tensor was still at its zero-initialised value after 444
optimizer steps. Re-running with only the loss computation changed
(manual `log_softmax`+`gather` instead of the fused `reduction='none'`
call) produced real, diversified parameter updates and a genuine loss
collapse on identical data and hyperparameters — isolating the fused
kernel path as the cause.

This is understood as a DirectML backward-kernel defect specific to that
fused op combination, not a general claim about DirectML. It is filed
upstream at [microsoft/DirectML#739](https://github.com/microsoft/DirectML/issues/739),
which contains a minimal reproduction. Full internal evidence, including
the isolated single-batch repro and the before/after parameter-norm
comparison, is linked from
[issue #10](https://github.com/Rook-CodeVolt/CodeVolt-Model-Development-Framework/issues/10)
of this repository.

**Mitigation for this specific case:** on the DirectML backend, when a
custom per-token or otherwise non-default reduction is needed, compute the
per-token negative log-likelihood manually with `log_softmax(...).gather(...)`
rather than `F.cross_entropy(..., reduction='none')` followed by a custom
reduction. The forward values are equivalent; only the fused kernel's
backward path is implicated.

### General mitigation: a hard pre-flight gradient-flow check

The DirectML case above is one instance of a broader class: any custom
loss computation — a fused kernel with an unexpected backward
implementation, a detached intermediate tensor, a masked reduction that
zeroes out its own gradient path, a mixed-precision cast that silently
drops gradient — can produce this same "trains without error, gradients
are zero" failure. Relying on a human or agent to manually diff parameter
norms after every run does not scale, and by the time someone notices a
loss curve looks *suspiciously* flat, real compute has already been
spent.

The recommended standing practice for any custom loss computation, on any
backend:

1. Before the full training loop starts, run one real forward/backward
   pass on one real batch from the actual training data pipeline (not a
   synthetic smoke-test batch — it must exercise the same code path
   production training will use).
2. After that backward pass, assert that at least one trainable
   parameter's `grad.norm() > 1e-9`.
3. If the assertion fails, hard-abort the run before any optimizer step is
   taken. Do not log a warning and continue — a run that starts anyway
   produces exactly the misleading "completed successfully" signal this
   check exists to prevent.

This check is cheap (one batch, run once per training job) and backend-
and loss-function-agnostic: it does not require knowing in advance which
op, kernel, or backend combination might silently fail to produce
gradients. Treat it as a standing precondition for any training loop that
uses a custom loss computation, in the same spirit as the immutable-input
validation this project already requires elsewhere (see
`docs/TRAINER_ADAPTER_CONTRACT.md`'s `TrainingInputs.validate()` for the
analogous pattern applied to input provenance rather than gradient flow).
