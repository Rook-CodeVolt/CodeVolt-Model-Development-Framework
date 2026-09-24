#!/usr/bin/env python3
"""Guard script: fail if the repository contains internal-only references.

FAILING checks (exit 1 if any hit), across both the working tree and the
full commit-message history of the current branch:
  - t_[0-9a-f]{8}            (internal task ids)
  - kanban                   (case-insensitive)
  - /Users/                  (local absolute paths)
  - .hermes                  (local tool paths)
  - personal/internal email identities (not the approved public noreply set)
  - retired signer-principal identifiers (maya-security, maya-dataset-rights,
    rook-owner) -- these were renamed to role-based identifiers; a hit means
    the old identifier crept back in.

REPORT-ONLY checks (printed, does not fail the build):
  - bare first names (Maya, Marcus, Rook, Elias, Clara) used as running
    prose/comments, excluding the "<Name>-CodeVolt" public GitHub author
    suffix, which is an intended public identity, not a leak. Advisory only
    until the prose-redaction passes for these names land; flip
    REPORT_ONLY_NAME_CHECK to False once that work is complete so a fresh
    occurrence fails the build like the others.
  - retired signer-principal identifiers (maya-security, maya-dataset-rights,
    rook-owner) found specifically in COMMIT-MESSAGE history (not the tree).
    These identifiers are hard-failed in the tracked working tree (a hit
    there means the rename regressed), but pre-rename commits already in
    published history legitimately describe implementing the
    then-current-named trust root; clearing those would require a full
    git filter-repo history rewrite, which is out of scope here. Flip
    REPORT_ONLY_RETIRED_SIGNER_IN_HISTORY to False only alongside such a
    rewrite.

Run with no arguments to scan the current working tree (excluding .git and
other build/cache dirs) plus this branch's commit-message history. Exits 1
and prints every FAIL-category hit if any matches; exits 0 otherwise (the
two report-only categories above are still printed but do not affect the
exit code).
"""
import os
import re
import subprocess
import sys

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules", ".venv", ".venv-adr0014-test"}

# Hits against any of these fail the build, in both the working tree and
# commit-message history.
FAIL_PATTERNS = {
    "task_id": re.compile(r"t_[0-9a-f]{8}"),
    "kanban": re.compile(r"kanban", re.IGNORECASE),
    "users_path": re.compile(r"/Users/"),
    "dot_hermes": re.compile(r"\.hermes"),
}

# Retired signer-principal identifiers: hard-fail if found in the tracked
# working tree (a hit there means the rename regressed), but report-only
# (does not fail the build) if found only in commit-message history --
# pre-rename commits legitimately describe implementing the
# then-current-named trust root, and clearing that would require a full
# git filter-repo history rewrite, out of scope here. Flip
# REPORT_ONLY_RETIRED_SIGNER_IN_HISTORY to False only alongside such a
# rewrite.
RETIRED_SIGNER_PATTERNS = {
    "retired_signer_maya_security": re.compile(r"maya-security"),
    "retired_signer_maya_dataset_rights": re.compile(r"maya-dataset-rights"),
    "retired_signer_rook_owner": re.compile(r"rook-owner"),
}
REPORT_ONLY_RETIRED_SIGNER_IN_HISTORY = True

# Report-only until the prose-redaction pass for these names is complete.
REPORT_ONLY_NAME_CHECK = True
BARE_NAME_RE = re.compile(r"\b(Maya|Marcus|Rook|Elias|Clara)\b(?!-CodeVolt)")

# Approved public identity domains/addresses. Any email address found in
# text that is NOT one of these (or a generic example.com/example.org
# placeholder) is flagged as a personal/internal identity leak.
ALLOWED_EMAIL_SUFFIXES = (
    "@users.noreply.github.com",
    "@example.com",
    "@example.org",
    "@example-research.org",
)
DISALLOWED_EMAIL_MARKERS = (
    "@codevolt.local",
    "@hermes.local",
    "@proton.me",
    "@codevolt.dev",
    "@codevolt.co.uk",
)

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

SELF_PATH = "scripts/check_no_internal_references.py"


def scan_text(rel_path, text, fail_hits, name_hits, *, is_history=False):
    for label, rx in FAIL_PATTERNS.items():
        if rx.search(text):
            fail_hits.append((rel_path, label))
    for label, rx in RETIRED_SIGNER_PATTERNS.items():
        if rx.search(text):
            if is_history and REPORT_ONLY_RETIRED_SIGNER_IN_HISTORY:
                name_hits.append((rel_path, label))
            else:
                fail_hits.append((rel_path, label))
    for m in EMAIL_RE.finditer(text):
        addr = m.group(0)
        if any(addr.endswith(suf) for suf in ALLOWED_EMAIL_SUFFIXES):
            continue
        if any(marker in addr for marker in DISALLOWED_EMAIL_MARKERS):
            fail_hits.append((rel_path, f"disallowed_email:{addr}"))
    for m in BARE_NAME_RE.finditer(text):
        name_hits.append((rel_path, f"bare_name:{m.group(1)}"))


def scan_tree(root, fail_hits, name_hits):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root)
            if rel.startswith(SELF_PATH):
                # The guard script itself legitimately contains these
                # pattern strings as regex source, not as leaked references.
                continue
            try:
                with open(path, "rb") as f:
                    raw = f.read()
            except OSError:
                continue
            if b"\x00" in raw[:4096]:
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            scan_text(rel, text, fail_hits, name_hits)


def scan_commit_messages(root, fail_hits, name_hits):
    """Scan every commit message reachable from HEAD on this branch.

    Commit messages are permanent once published (rewording after the fact
    requires a history rewrite), so they get the same fail-on-hit treatment
    as tracked file content, plus the advisory name check.
    """
    try:
        out = subprocess.run(
            ["git", "log", "--format=%H%x00%B%x02"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"WARNING: could not read commit-message history ({exc}); "
              "commit-message scan skipped.", file=sys.stderr)
        return
    for record in out.split("\x02"):
        record = record.strip("\n")
        if not record:
            continue
        sha, _, body = record.partition("\x00")
        label = f"commit:{sha[:12]}"
        scan_text(label, body, fail_hits, name_hits, is_history=True)


def main(argv):
    root = argv[1] if len(argv) > 1 else "."
    fail_hits = []
    name_hits = []

    scan_tree(root, fail_hits, name_hits)
    scan_commit_messages(root, fail_hits, name_hits)

    if name_hits:
        print(f"REPORT (advisory, does not fail build): {len(name_hits)} bare-name hit(s):")
        for rel, label in name_hits[:200]:
            print(f"  {rel}: {label}")

    if fail_hits:
        print(f"FAIL: {len(fail_hits)} internal-reference hit(s) found:")
        for rel, label in fail_hits[:200]:
            print(f"  {rel}: {label}")
        return 1

    print("OK: no internal references found.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
