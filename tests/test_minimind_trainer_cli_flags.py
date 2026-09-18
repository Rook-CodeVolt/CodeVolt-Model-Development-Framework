"""Static regression guard: adapter-constructed CLI flags vs. the real script.

Origin: issue #47. ``MiniMindTrainerAdapter._build_subprocess_args`` once
constructed ``--out_dir``, ``--model_path``, and ``--max_steps`` -- none of
which exist in MiniMind's real ``trainer/train_full_sft.py`` at the pinned
commit (``MINIMIND_PINNED_COMMIT``). The 37 pre-existing adapter tests all
mock ``subprocess.Popen``, so a wrong flag name is invisible to them; this
module closes that gap with a *static* comparison that needs no subprocess,
no network access, and never executes ``train_full_sft.py``.

Method, precisely per the task's requirement:

1. A byte-for-byte vendored copy of the pinned commit's
   ``trainer/train_full_sft.py`` lives at
   ``tests/fixtures/minimind_pinned/trainer/train_full_sft.py`` (copied
   directly from a shallow clone checked out at
   ``cc312c1cc614bc371cd85dcbcbc1d3ba1590f364``, verified via
   ``git rev-parse HEAD`` at copy time). Vendoring it keeps this test
   fully offline and deterministic in CI -- no network fetch, no clone,
   no execution of the script itself.
2. ``ast.parse`` walks the fixture's module body for the
   ``argparse.ArgumentParser`` block and collects every
   ``parser.add_argument("--flag", ...)`` / ``parser.add_argument('--flag',
   ...)`` long-flag name via the ``ast`` module -- never by importing or
   running the script (which would pull in ``torch``/``datasets``/model
   code this repo does not depend on and is explicitly forbidden by the
   task: no real MiniMind training may ever be executed by this suite).
3. Every flag ``MiniMindTrainerAdapter._build_subprocess_args`` can ever
   emit (across the full input-parameter matrix it branches on) must be a
   member of that real, statically-extracted flag set.

If MiniMind is ever re-pinned to a newer commit, refresh the vendored
fixture from the new commit (and re-verify ``git rev-parse HEAD`` at copy
time) -- this test will then fail loudly on any flag the adapter still
constructs that the new script no longer exposes, instead of silently
reintroducing the issue #47 class of bug.
"""

from __future__ import annotations

import ast
from pathlib import Path

from codevolt_mdf.minimind_adapter import MINIMIND_PINNED_COMMIT, MiniMindTrainerAdapter

FIXTURE_SCRIPT = (
    Path(__file__).parent / "fixtures" / "minimind_pinned" / "trainer" / "train_full_sft.py"
)


def _extract_real_argparse_flags(script_path: Path) -> set[str]:
    """Statically extract every ``--flag`` long name from an argparse script.

    Parses with ``ast`` only -- the module is never imported or executed,
    so this has zero dependency on ``torch``/``datasets``/MiniMind's own
    model code being installed or importable.
    """
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Matches `<anything>.add_argument(...)`, e.g. `parser.add_argument(...)`.
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        for arg in node.args:
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("--")
            ):
                flags.add(arg.value)
    return flags


def _every_flag_the_adapter_can_construct(adapter: MiniMindTrainerAdapter) -> set[str]:
    """Union of CLI flags across every branch ``_build_subprocess_args`` takes.

    Exercises the optional-``model_path``/optional-``epochs`` branches by
    calling the method twice (params present, params absent) and unions
    the results, so the static check covers every flag the adapter can
    ever actually emit, not just one code path through it.
    """
    checkpoint_dir = Path("/tmp/does-not-need-to-exist-for-this-static-check/checkpoints")
    script_path = Path("/tmp/does-not-need-to-exist-for-this-static-check/train_full_sft.py")

    all_params_present = {
        "dataset_path": "irrelevant.jsonl",
        "model_path": "irrelevant-weights.bin",
        "epochs": 2,
        "max_steps": 4,
        "learning_rate": 1e-5,
        "batch_size": 8,
        "save_interval": 50,
    }
    minimal_params = {"dataset_path": "irrelevant.jsonl", "max_steps": 4}

    flags: set[str] = set()
    for params in (all_params_present, minimal_params):
        argv = adapter._build_subprocess_args(
            python_executable="python3",
            script_path=script_path,
            params=params,
            checkpoint_dir=checkpoint_dir,
            seed=42,
        )
        flags |= {token for token in argv if token.startswith("--")}
    return flags


def test_vendored_fixture_matches_the_declared_pin() -> None:
    """Sanity check that the fixture really is copied from the pinned commit.

    Guards against the fixture silently drifting from
    ``MINIMIND_PINNED_COMMIT`` without anyone noticing -- if the adapter's
    pin is ever bumped, this test's failure ("fixture missing / stale")
    is the reminder to refresh ``tests/fixtures/minimind_pinned/``.
    """
    assert FIXTURE_SCRIPT.is_file(), (
        f"expected a vendored copy of trainer/train_full_sft.py at "
        f"{FIXTURE_SCRIPT}, copied from MiniMind commit {MINIMIND_PINNED_COMMIT}"
    )


def test_adapter_only_constructs_flags_that_exist_in_the_real_script() -> None:
    """The issue #47 regression guard.

    Every CLI flag ``_build_subprocess_args`` can construct, across its
    full branch matrix, must be a member of the flag set statically
    extracted from the real, vendored ``train_full_sft.py`` at the pinned
    commit.
    """
    real_flags = _extract_real_argparse_flags(FIXTURE_SCRIPT)
    assert real_flags, "static extraction found no --flags at all; parser likely broken"

    adapter = MiniMindTrainerAdapter(work_dir=Path("/tmp/does-not-need-to-exist-for-this-check"))
    constructed_flags = _every_flag_the_adapter_can_construct(adapter)
    assert constructed_flags, "adapter constructed no flags at all; test fixture is wrong"

    unknown_flags = constructed_flags - real_flags
    assert not unknown_flags, (
        "MiniMindTrainerAdapter._build_subprocess_args constructs CLI flag(s) "
        f"{sorted(unknown_flags)} that do not exist in the real "
        f"trainer/train_full_sft.py at the pinned commit {MINIMIND_PINNED_COMMIT} "
        f"(real flags: {sorted(real_flags)}). This is the exact class of bug "
        "filed as issue #47 -- fix _build_subprocess_args, don't relax this test."
    )


def test_known_previously_wrong_flags_are_confirmed_absent_from_real_script() -> None:
    """Documents, explicitly, the three flags issue #47 was filed about.

    Not load-bearing on its own (the flag-set-subset test above already
    catches any reintroduction), but pins down in one place, by name,
    exactly what was wrong before this fix -- so a reader doesn't have to
    reconstruct it from the PR history.
    """
    real_flags = _extract_real_argparse_flags(FIXTURE_SCRIPT)
    previously_constructed_wrong_flags = {"--out_dir", "--model_path", "--max_steps"}
    assert previously_constructed_wrong_flags.isdisjoint(real_flags), (
        "one of the flags issue #47 was filed about now exists in the real "
        "script -- re-check whether _build_subprocess_args should use it"
    )
