# Governance

CodeVolt maintains the project and its evidence standard. Governance will evolve as the contributor community grows.

## Decision process

- Small compatible changes use ordinary pull requests.
- New contracts, dependencies, engines, or governance changes require an architecture decision record in `docs/decisions/`.
- Claims of improvement require reproducible evidence and comparison with a declared baseline.
- Maintainers may accept, reject, or request revision; rejection rationale should be recorded.

## Capability lifecycle

```text
proposed -> researched -> experimental -> independently evaluated -> supported
                                                    `-----------> rejected
```

“Rejected” records are retained because they prevent repeated mistakes. Supported does not mean suitable for every model, dataset, jurisdiction, or hardware target.

## AI-assisted contributions

AI assistance is welcome and should be disclosed when material. Human contributors remain accountable for licences, tests, security, claims, and submitted content. Generated output is not evidence by itself.

## Re-review after a doc-only stale-review dismissal

GitHub's `dismiss_stale_reviews` branch-protection setting dismisses an approved review on any push to the pull request branch, including a rebase that only re-resolves a merge conflict in documentation. It cannot distinguish "the reviewed diff changed" from "an unrelated doc merge touched a status line." Left unmitigated, this forces a full independent re-review even when zero `src/` or `tests/` bytes changed since the prior approval — pure cost with no additional safety.

This convention gives reviewers and rebase authors a fast, auditable path for that specific case. It does not change the independent-review requirement itself: every substantive change to code, data, or contracts still requires a full independent review from a different account before merge, as described above and in `AGENTS.md`.

**When a review is dismissed solely by a doc-only rebase conflict** (for example, resolving status text in `docs/ROADMAP.md` or `docs/REAL_ADAPTERS.md` against a new `main`), the re-reviewer should:

1. Diff the new HEAD against the last-reviewed commit SHA, restricted to `src/` and `tests/` paths only, e.g. `git diff <last-reviewed-sha>..<new-head-sha> -- src/ tests/`.
2. If that diff is empty, record in the new review that `src/` and `tests/` are byte-identical to the commit that received the prior approved review, cite both SHAs, and approve on that basis. Do not re-derive a full review from scratch — the substantive change has already been independently reviewed and has not moved.
3. If that diff is **not** empty — any `src/` or `tests/` byte changed, even incidentally — treat it as a normal review: perform a full independent review of the changed diff before approving.

This fast path applies only to re-review after a stale-review dismissal. It never substitutes for the first independent review of a change, and it never applies when the diff outside `docs/` is nonempty.

**Recommendation for rebase authors:** when resolving a doc-only conflict during a rebase or merge of `main` into an already-approved pull request, keep that resolution in a single, clearly separate commit (e.g. `docs: resolve rebase conflict in ROADMAP.md/REAL_ADAPTERS.md status text`) rather than folding it into other changes. A single isolated commit makes step 1 above fast and unambiguous to verify, both for a human reviewer and for an agent.

## Maintainer authority

Maintainers protect project integrity, contributor safety, and evidence quality. Security response may temporarily override the normal change process. See `SECURITY.md` for private reporting.
