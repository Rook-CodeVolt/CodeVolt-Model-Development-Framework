"""Deterministic validation of feedback-related repository docs and forms.

These are structural checks, not GitHub API calls: they do not require
network access, a token, or live repository settings, and they never mutate
anything. They exist so a documentation or issue-form change that breaks the
feedback pathway (bad YAML, a dangling relative link, a form referencing a
label nobody defined) fails CI instead of being discovered by a contributor.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ISSUE_TEMPLATE_DIR = ROOT / ".github" / "ISSUE_TEMPLATE"

# Relative markdown links we intentionally do not resolve on disk because
# they point at GitHub-hosted, non-repository-file destinations.
EXTERNAL_LINK_PREFIXES = ("http://", "https://", "mailto:")

MARKDOWN_FILES = [
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "SUPPORT.md",
    ROOT / "SECURITY.md",
    *sorted((ROOT / "docs").glob("*.md")),
]

LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _issue_form_files():
    return sorted(p for p in ISSUE_TEMPLATE_DIR.glob("*.yml") if p.name != "config.yml")


def test_issue_template_config_is_valid_yaml_with_contact_links():
    config = yaml.safe_load((ISSUE_TEMPLATE_DIR / "config.yml").read_text())
    assert config["blank_issues_enabled"] is False
    for link in config["contact_links"]:
        assert link["url"].startswith("https://")
        assert link["about"]


def test_every_issue_form_is_valid_yaml_with_required_shape():
    for path in _issue_form_files():
        form = yaml.safe_load(path.read_text())
        assert form.get("name"), f"{path.name} is missing name"
        assert form.get("description"), f"{path.name} is missing description"
        assert form.get("labels"), f"{path.name} must declare at least one label"
        assert isinstance(form["body"], list) and form["body"], f"{path.name} has no body fields"
        for field in form["body"]:
            assert "type" in field, f"{path.name} has a body field without a type"


def test_agent_feedback_form_matches_the_agents_md_contract():
    form = yaml.safe_load((ISSUE_TEMPLATE_DIR / "agent-feedback.yml").read_text())
    assert "agent-feedback" in form["labels"]
    assert "triage" in form["labels"]
    safety = next(f for f in form["body"] if f.get("attributes", {}).get("label") == "Safety checks")
    checklist_text = " ".join(opt["label"] for opt in safety["attributes"]["options"])
    assert "existing report" in checklist_text
    assert "credentials" in checklist_text
    assert "private vulnerability reporting" in checklist_text


def test_capability_form_requires_an_evidence_plan():
    form = yaml.safe_load((ISSUE_TEMPLATE_DIR / "capability.yml").read_text())
    assert "capability" in form["labels"]
    assert "proposal" in form["labels"]
    labels = {f["attributes"]["label"] for f in form["body"]}
    assert "Evidence plan" in labels


def test_feedback_doc_exists_and_covers_required_topics():
    feedback_doc = ROOT / "docs" / "FEEDBACK.md"
    assert feedback_doc.exists()
    text = feedback_doc.read_text()
    required_topics = [
        "Channel selection",
        "Safe evidence",
        "Deduplication",
        "Triage lifecycle",
        "Security and private-report boundary",
        "Decision and closure loop",
        "training data",
    ]
    for topic in required_topics:
        assert topic in text, f"docs/FEEDBACK.md is missing coverage of: {topic}"


def test_markdown_relative_links_resolve_to_existing_files():
    broken = []
    for md_file in MARKDOWN_FILES:
        text = md_file.read_text()
        for target in LINK_PATTERN.findall(text):
            target = target.split(" ", 1)[0].strip("<>")
            if not target or target.startswith("#") or target.startswith(EXTERNAL_LINK_PREFIXES):
                continue
            path_part = target.split("#", 1)[0]
            resolved = (md_file.parent / path_part).resolve()
            if not resolved.exists():
                broken.append(f"{md_file.relative_to(ROOT)} -> {target}")
    assert not broken, "Broken relative links:\n" + "\n".join(broken)


def test_feedback_doc_is_linked_from_top_level_entry_points():
    for path in (ROOT / "README.md", ROOT / "SUPPORT.md", ROOT / "CONTRIBUTING.md", ROOT / "AGENTS.md"):
        assert "FEEDBACK.md" in path.read_text(), f"{path.name} should reference docs/FEEDBACK.md"
