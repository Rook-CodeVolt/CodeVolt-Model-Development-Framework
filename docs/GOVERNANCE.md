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

## Maintainer authority

Maintainers protect project integrity, contributor safety, and evidence quality. Security response may temporarily override the normal change process. See `SECURITY.md` for private reporting.
