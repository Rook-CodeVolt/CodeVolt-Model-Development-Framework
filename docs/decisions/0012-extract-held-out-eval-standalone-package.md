# ADR-0012: Extract HeldOutExclusionRegistry as a standalone, trainer-agnostic package

- Status: accepted
- Date: 2026-09-18

## Context

`docs/PROGRESSION.md`'s core judgment call, based on an external research
pass, is that this framework should not compete with TRL, Axolotl,
Unsloth, or LLaMA-Factory on training-method breadth — those projects
already cover that ground well. The framework's actual, defensible
point of difference is **contamination-resistant evaluation by
construction**: `HeldOutExclusionRegistry` (previously
`src/codevolt_mdf/held_out_registry.py`) registers a held-out set and
excludes it from training data *before* an experiment runs, rather
than checking for overlap after the fact — a stronger guarantee than
`lm-evaluation-harness`'s post-hoc n-gram decontamination utility, and
a capability none of promptfoo, DeepEval, or RAGAS attempt at all.

That module was, until this ADR, 194 lines with zero imports beyond
Python's standard library (`json`, `dataclasses`, `pathlib`,
`collections.abc`) — genuinely independent of `trainer_contract.py`,
the evaluator contract classes, and the `codevolt-mdf` CLI. But it
only shipped bundled inside this repository's ~6,000-line framework.
A team already training with MiniMind, TRL, or Axolotl directly had no
way to adopt just the contamination check without also adopting this
framework's adapter architecture, provenance hashing, and CLI — an
unreasonable ask for a capability that owes nothing to any of that
machinery.

Put plainly: the framework's own strategy document says the
differentiator is evaluation trust, not training tooling, but the
packaging contradicted that strategy by making evaluation trust
inseparable from a specific, opinionated training framework.

## Decision

Extract `HeldOutExclusionRegistry` into `packages/held-out-eval/`, a
separate, independently pip-installable Python package (`held-out-eval`
on PyPI naming conventions, importable as `held_out_eval`), with:

- its own `pyproject.toml`, `README.md`, `LICENSE` (Apache-2.0, matching
  the parent repository), and test suite (`packages/held-out-eval/tests/`,
  moved verbatim from `tests/test_held_out_registry.py` with no
  behavioral change);
- zero dependency on `codevolt_mdf`, any trainer adapter, or any
  evaluator contract class — it depends on nothing but the Python
  standard library;
- a runnable example (`examples/plain_training_loop.py`) showing the
  registry used next to a plain PyTorch training loop that has nothing
  to do with this framework or any specific trainer, to make the
  trainer-agnostic claim concrete rather than assumed.

The main framework's evaluator contract
(`src/codevolt_mdf/evaluator_contract.py`) now imports
`HeldOutExclusionRegistry` from the new package instead of a local
module. This is a dependency-direction change only: the framework now
depends on the standalone package, not the other way around. No public
contract, method signature, or behavior of `EvaluatorAdapterV1`, any
trainer adapter, or the `codevolt-mdf` CLI changed.

## What this decision does and does not establish

**Does establish**, verified directly, not assumed:

- The package builds as a real wheel/sdist (`python3 -m build`).
- Installing *only* that wheel into a brand-new virtual environment
  that has never seen `codevolt-mdf` succeeds, and the package's own
  16 tests pass against that installed wheel.
- `examples/plain_training_loop.py` runs end-to-end in that same clean
  environment, demonstrating registration of a held-out set before a
  (mocked) training loop and a post-training contamination check.
- `python3 -c "import held_out_eval; ...; assert 'codevolt_mdf' not in
  sys.modules"` passes in that clean environment — the package genuinely
  imports nothing from this framework.
- The full main-framework test suite (238 tests) and `ruff check .`
  remain green after the refactor — the extraction did not regress any
  existing adapter or contract.

**Does not establish:**

- That any external team has adopted, evaluated, or even seen this
  package. No outside usage exists yet. This ADR creates the
  *possibility* of adoption without a framework buy-in; it is not
  evidence that adoption will happen.
- That the full framework (adapters, CLI, provenance hashing) is
  deprecated, reduced in scope, or less important. The framework
  remains the reference integration that dogfoods this package end to
  end; the extraction adds a second, narrower distribution path
  alongside it, it does not replace the first.
- Any change to gating, authority, or contract surfaces elsewhere in
  the framework. `HeldOutExclusionRegistry`'s contamination-check
  semantics are unchanged — this ADR is a packaging and distribution
  decision, not a capability or authority decision.

## Consequences

- A team using MiniMind, TRL, Axolotl, LLaMA-Factory, or a custom
  training loop can now `pip install held-out-eval` (once published)
  and get contamination-resistant held-out set management without
  adopting any part of this framework's adapter architecture.
- This framework's own evaluator adapters continue to use the same
  registry, now via an external dependency rather than a local module
  — no duplication, no behavioral drift risk between "the framework's
  copy" and "the standalone copy," because there is only one copy.
- Future evaluation-suite work (issue #24 WP-B/WP-C style capability
  extensions) should default to extending the standalone package when
  the capability in question is genuinely trainer-agnostic, and keep
  framework-specific integration code (adapters, CLI wiring) in the
  main package.
