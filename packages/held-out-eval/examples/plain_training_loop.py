"""Standalone example: held-out-eval used alongside a plain training loop.

Deliberately has nothing to do with any specific trainer, framework, or
this repository's own adapters -- it defines a trivial `train()`
function to stand in for "call whatever trainer you actually use" (TRL,
Axolotl, MiniMind, raw PyTorch/JAX, or anything else). The only thing
this script demonstrates is: register a held-out set before training,
then verify after training that nothing in it leaked into what was
actually trained on.

This is the script the clean-venv adoption test in this repository
actually runs, with only `held-out-eval` installed (no `torch`, no
`codevolt-mdf`) -- see this repository's
docs/decisions/0012-extract-held-out-eval-standalone-package.md.

Run: python examples/plain_training_loop.py
"""

from __future__ import annotations

from held_out_eval import ContaminationError, HeldOutExclusionRegistry


def train(train_ids: list[str]) -> None:
    """Stand-in for an arbitrary training loop from any trainer.

    Replace this with TRL's SFTTrainer.train(), a raw
    torch.optim training loop, an Axolotl CLI invocation, or
    anything else -- held-out-eval does not know or care.
    """
    print(f"(pretend) training on {len(train_ids)} examples")


def main() -> None:
    registry = HeldOutExclusionRegistry()

    # 1. Register the held-out (eval) set BEFORE training starts.
    held_out_ids = [f"eval-{i}" for i in range(20)]
    registry.register_package_held_out("plain-loop-demo", held_out_ids)

    # 2. Train however you like. Here: ids that do NOT overlap
    #    held_out_ids, so this run should be clean.
    train_ids = [f"train-{i}" for i in range(200)]
    train(train_ids)

    # 3. Register what was actually trained on. Raises ContaminationError
    #    instead of silently succeeding if any id was already claimed as
    #    held-out by this or a different package.
    try:
        registry.register_package_train("plain-loop-demo", train_ids)
    except ContaminationError as exc:
        raise SystemExit(f"contamination detected: {exc}") from exc

    # 4. Post-training check: did any held-out id end up in train data
    #    anywhere in the registry?
    leaked = registry.check_held_out_not_trained(held_out_ids)
    if leaked:
        raise SystemExit(f"held-out ids leaked into training: {sorted(leaked)}")

    print("clean: no contamination between train and held-out ids")


if __name__ == "__main__":
    main()
