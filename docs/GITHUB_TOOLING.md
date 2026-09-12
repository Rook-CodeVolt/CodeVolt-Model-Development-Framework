# GitHub tooling

The repository uses native GitHub features where they improve trust without adding unnecessary process.

## Configured in the repository

- **GitHub Actions:** test and lint pull requests and pushes to `main`.
- **Dependabot:** proposes monthly updates for Actions and Python dependencies.
- **Issue forms and pull-request template:** request reproducible evidence and provenance. See [`docs/FEEDBACK.md`](FEEDBACK.md) for the full submission and triage lifecycle.
- **CODEOWNERS:** makes review ownership explicit.
- **Citation metadata:** helps researchers and downstream projects cite releases.

## Recommended repository settings

These settings are enabled separately in GitHub rather than through committed code:

- **Discussions** for questions, ideas, experiment reports, and community decisions.
- **Private vulnerability reporting** for responsible security disclosure.
- **Repository rules for `main`:** require a pull request, passing CI, resolved conversations, and no force pushes. Apply these after the initial bootstrap commit.
- **Projects** when the issue backlog becomes large enough to need roadmap views.
- **Releases and generated release notes** beginning with the first stable public demonstration.
- **Environments** if future workflows publish packages or models; require approval before production publication.

GitHub-hosted runners are appropriate for contract tests, linting, and small deterministic evaluations. Model training should run on declared external hardware or self-hosted runners with strict labels, budgets, isolation, and approval gates. Pull requests from forks must never receive training-machine secrets.

## Features intentionally deferred

- CodeQL/code scanning is disabled while this repository remains private because no authorized GitHub Code Security or Advanced Security entitlement is available. Ordinary CI does not provide CodeQL or equivalent SAST coverage. Re-enable only after explicit owner authorization and live GitHub API confirmation that code scanning is enabled.
- GitHub Pages until the documentation outgrows the README and `docs/` directory.
- Package publication until the Python API is stable enough for downstream users.
- Self-hosted GPU runners until threat modelling, isolation, concurrency, and spend controls are implemented.
- Automated model publication until promotion and rollback governance exists.
