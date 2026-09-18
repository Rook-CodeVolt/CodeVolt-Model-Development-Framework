# held-out-eval

A small, standalone, **trainer-agnostic** library for one specific
problem: making sure your held-out (test/eval) data never accidentally
becomes training data, and vice versa -- *by construction*, checked at
registration time, not discovered after the fact by diffing files.

## The problem this solves

Benchmark contamination -- an eval set's examples leaking into a
model's training data -- is a well-documented, industry-wide problem.
Most tooling detects it *after the fact*: you train first, then run an
n-gram or embedding-overlap check against the training corpus and hope
it catches what leaked in. That approach has two structural weaknesses:

1. It depends on the trainer disclosing (or you being able to inspect)
   exactly what was trained on, which is often not true in practice.
2. It only catches contamination that's already happened. By the time
   you find it, the model has already been trained on tainted data and
   any reported eval score is compromised.

`HeldOutExclusionRegistry` instead makes contamination structurally
hard to introduce in the first place: you register a package's train
ids and held-out ids explicitly, and any attempt to register an id as
held-out that's already registered as train **anywhere** (or vice
versa) raises immediately, before training or evaluation proceeds. It
checks **both directions** and tracks contributions **per package**
(so a legitimate re-split of your own data doesn't get flagged as
self-contamination, and a stale id removed from your own split
correctly stops being flagged for anyone else either).

This project (CodeVolt-Model-Development-Framework) originally built
this class for its own internal evaluator contract, after a real
incident (issue #11) where 67 ids from one dataset package's train
split were silently present in a later package's held-out split,
undetected until a bidirectional check was added. That check is the
whole of what this package ships. It has no dependency on this
project's trainer adapters, evaluator contract, CLI, or anything else
in the wider framework -- import it and use it next to **any**
training setup.

## Install

```bash
pip install held-out-eval
```

Zero runtime dependencies: it imports only `json`, `dataclasses`,
`pathlib`, and `collections.abc` from the Python standard library.

## Usage: alongside a plain PyTorch training loop

This example has nothing to do with any specific trainer or framework
-- swap the two-line "pretend training loop" for TRL, Axolotl,
MiniMind, a raw `for` loop over `torch.optim`, or anything else. The
registry doesn't know or care what trained the model; it only tracks
which ids were used as train data and which were used as held-out data.

```python
import torch
from held_out_eval import HeldOutExclusionRegistry, ContaminationError

registry = HeldOutExclusionRegistry()

# 1. Register your held-out (eval) set BEFORE training starts.
held_out_ids = [f"eval-{i}" for i in range(20)]
registry.register_package_held_out("my-experiment", held_out_ids)

# 2. Build your actual training batch however you like. This is a
#    stand-in for "call your trainer" -- it could be TRL's
#    SFTTrainer.train(), a raw PyTorch loop, or anything else.
train_ids = [f"train-{i}" for i in range(1000)]
model = torch.nn.Linear(4, 4)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for step in range(3):  # pretend training loop
    optimizer.zero_grad()
    loss = model(torch.randn(4)).sum()
    loss.backward()
    optimizer.step()

# 3. Register what was actually trained on. If any id here was already
#    claimed as held-out (by this or a different package), this raises
#    ContaminationError instead of silently letting it through.
try:
    registry.register_package_train("my-experiment", train_ids)
except ContaminationError as exc:
    raise SystemExit(f"contamination detected before scoring: {exc}")

# 4. Before scoring the trained model against held_out_ids, re-check:
#    did any of them end up registered as train data anywhere?
leaked = registry.check_held_out_not_trained(held_out_ids)
assert not leaked, f"held-out ids leaked into training: {leaked}"
print("clean: no contamination between train and held-out ids")
```

Run it (after `pip install held-out-eval torch`): prints `clean: no
contamination between train and held-out ids`. Change `train_ids` to
include `"eval-0"` and re-run to see it raise `ContaminationError`
instead.

A fuller, standalone copy of this example (no torch import required,
using a plain Python "trainer" stand-in) lives at
[`examples/plain_training_loop.py`](examples/plain_training_loop.py)
in this directory, and is what this package's own clean-venv adoption
test actually executes.

## API

- `HeldOutExclusionRegistry` -- the registry. `register_package_train`,
  `register_package_held_out`, `all_train_ids`, `all_held_out_ids`,
  `check_held_out_not_trained`, `to_dict`/`from_dict`, `save`/`load`.
- `ContaminationError` -- raised by the two `register_*` methods when a
  registration would create train/held-out contamination.
- `HeldOutRegistryError` -- base class for both.

See `src/held_out_eval/registry.py` for full docstrings on every
method, including the exact bidirectional-check semantics.

## Relationship to CodeVolt-Model-Development-Framework

This package is extracted from, and still used by, that framework's
own evaluator adapters (`hf_local_evaluator_adapter.py`,
`fake_evaluator_adapter.py`) -- they now depend on this package rather
than duplicating the class. That framework's `TrainerAdapterV1` /
`EvaluatorAdapterV1` contracts, CLI, and other tooling are a much
larger, opinionated system for a specific set of trainer/evaluator
adapters; this package is not that, and does not require it. See
[`docs/decisions/0012-extract-held-out-eval-standalone-package.md`](https://github.com/Rook-CodeVolt/CodeVolt-Model-Development-Framework/blob/main/docs/decisions/0012-extract-held-out-eval-standalone-package.md)
in that repository for the extraction rationale.

## What this package does *not* claim

- It does not claim any external project has adopted it yet. This
  extraction makes independent adoption *possible*; it is not evidence
  that adoption has happened.
- It does not detect contamination that occurred before you started
  using it (e.g. a pretrained base model's own pretraining corpus). It
  only prevents contamination in ids you explicitly register through
  it going forward.
- It has no opinion on data formats, model architectures, or training
  methods -- it tracks opaque string ids only.

## License

Apache-2.0, matching the parent repository.
