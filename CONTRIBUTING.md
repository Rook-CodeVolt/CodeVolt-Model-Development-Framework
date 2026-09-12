# Contributing

Thank you for helping build trustworthy model-development infrastructure.

## Ways to contribute

- Propose or implement a trainer, evaluator, dataset, or learning-strategy adapter.
- Add reproducible experiments and independent evaluation suites.
- Improve contracts, documentation, privacy, safety, portability, or accessibility.
- Report failed approaches and negative results.

See [`docs/FEEDBACK.md`](docs/FEEDBACK.md) for how to choose between an issue form, Discussions, and private vulnerability reporting, and how a filed report is triaged and closed.

## Workflow

1. Open an issue for substantial features or contract changes.
2. Fork the repository and create a focused branch.
3. Add implementation, tests, documentation, provenance, and licence information.
4. Run `python -m pytest` and `python -m ruff check .`.
5. Open a pull request using the template.

Do not submit private, personal, customer, restricted, or unlicensed training data. Examples must be synthetic, appropriately licensed, or accompanied by documented permission.

## Evidence standard

An improvement claim must identify its immutable baseline, manifest, dataset provenance, environment, measurements, acceptance criteria, limitations, and reproduction steps. Missing evidence is an invalid result. A successful training process alone is not proof of improvement.

## Adapter expectations

Adapters should fail closed, pin upstream compatibility, respect declared budgets, avoid network access by default during evaluation, and never promote their own output. Optional engines must not become core dependencies without an accepted architecture decision.

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
