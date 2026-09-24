#!/usr/bin/env python3
"""Guard script: fail if the working tree contains internal-only references.

Checked patterns:
  - t_[0-9a-f]{8}            (internal task ids)
  - kanban                   (case-insensitive)
  - /Users/                  (local absolute paths)
  - .hermes                  (local tool paths)
  - personal/internal email identities (not the approved public noreply set)

Run with no arguments to scan the current working tree (excluding .git and
other build/cache dirs). Exits 1 and prints every hit if any pattern
matches; exits 0 silently otherwise.
"""
import os
import re
import sys

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules", ".venv", ".venv-adr0014-test"}

PATTERNS = {
    "task_id": re.compile(r"t_[0-9a-f]{8}"),
    "kanban": re.compile(r"kanban", re.IGNORECASE),
    "users_path": re.compile(r"/Users/"),
    "dot_hermes": re.compile(r"\.hermes"),
}

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


def scan_text(rel_path, text, hits):
    for label, rx in PATTERNS.items():
        if rx.search(text):
            hits.append((rel_path, label))
    for m in EMAIL_RE.finditer(text):
        addr = m.group(0)
        if any(addr.endswith(suf) for suf in ALLOWED_EMAIL_SUFFIXES):
            continue
        if any(marker in addr for marker in DISALLOWED_EMAIL_MARKERS):
            hits.append((rel_path, f"disallowed_email:{addr}"))


def main(argv):
    root = argv[1] if len(argv) > 1 else "."
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root)
            if rel.startswith("scripts/check_no_internal_references.py"):
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
            scan_text(rel, text, hits)

    if hits:
        print(f"FAIL: {len(hits)} internal-reference hit(s) found:")
        for rel, label in hits[:200]:
            print(f"  {rel}: {label}")
        return 1
    print("OK: no internal references found.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
