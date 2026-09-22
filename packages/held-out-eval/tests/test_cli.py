"""Tests for the held-out-eval CLI (src/held_out_eval/cli.py).

Covers the acceptance criteria from issue #56: --help documents every
subcommand, the full register/check workflow works purely through the
CLI against a JSON file, exit codes are meaningful for CI use, and each
subcommand has a success-path and at least one failure/contamination
test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from held_out_eval.cli import EXIT_CONTAMINATION, EXIT_ERROR, EXIT_OK, main


def _write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


# -- --help / usage -----------------------------------------------------------


def test_help_documents_all_subcommands(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for subcommand in ("register-train", "register-held-out", "check", "show"):
        assert subcommand in out


def test_no_command_is_a_usage_error():
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code != 0


# -- register-train -------------------------------------------------------------


def test_register_train_success_writes_registry(tmp_path, capsys):
    ids_file = tmp_path / "train_ids.txt"
    _write_ids(ids_file, ["t1", "t2", "t3"])
    registry_path = tmp_path / "registry.json"

    rc = main(["--registry", str(registry_path), "register-train", "P1", str(ids_file)])

    assert rc == EXIT_OK
    assert registry_path.exists()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert payload["package_train_ids"] == {"P1": ["t1", "t2", "t3"]}
    assert "registered 3 train id(s)" in capsys.readouterr().out


def test_register_train_contamination_exits_nonzero_and_does_not_write(tmp_path, capsys):
    registry_path = tmp_path / "registry.json"
    held_out_ids = tmp_path / "held.txt"
    _write_ids(held_out_ids, ["shared-id"])
    assert main(["--registry", str(registry_path), "register-held-out", "P1", str(held_out_ids)]) == EXIT_OK

    train_ids = tmp_path / "train.txt"
    _write_ids(train_ids, ["shared-id", "other-id"])
    rc = main(["--registry", str(registry_path), "register-train", "P2", str(train_ids)])

    assert rc == EXIT_CONTAMINATION
    err = capsys.readouterr().err
    assert "contamination detected" in err
    # Registry file must reflect only the successful P1 registration,
    # never a partial write from the rejected P2 registration.
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert "P2" not in payload["package_train_ids"]


def test_register_train_missing_ids_file_is_usage_error(tmp_path, capsys):
    rc = main(
        [
            "--registry",
            str(tmp_path / "registry.json"),
            "register-train",
            "P1",
            str(tmp_path / "missing.txt"),
        ]
    )
    assert rc == EXIT_ERROR
    assert "not found" in capsys.readouterr().err


# -- register-held-out -----------------------------------------------------------


def test_register_held_out_success(tmp_path, capsys):
    ids_file = tmp_path / "held.txt"
    _write_ids(ids_file, ["h1", "h2"])
    registry_path = tmp_path / "registry.json"

    rc = main(["--registry", str(registry_path), "register-held-out", "P1", str(ids_file)])

    assert rc == EXIT_OK
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert payload["package_held_out_ids"] == {"P1": ["h1", "h2"]}
    assert "registered 2 held-out id(s)" in capsys.readouterr().out


def test_register_held_out_contamination(tmp_path, capsys):
    registry_path = tmp_path / "registry.json"
    train_ids = tmp_path / "train.txt"
    _write_ids(train_ids, ["shared-id"])
    assert main(["--registry", str(registry_path), "register-train", "P1", str(train_ids)]) == EXIT_OK

    held_ids = tmp_path / "held.txt"
    _write_ids(held_ids, ["shared-id"])
    rc = main(["--registry", str(registry_path), "register-held-out", "P2", str(held_ids)])

    assert rc == EXIT_CONTAMINATION
    assert "contamination detected" in capsys.readouterr().err


# -- check ------------------------------------------------------------------------


def test_check_clean_exits_zero(tmp_path, capsys):
    registry_path = tmp_path / "registry.json"
    train_ids = tmp_path / "train.txt"
    _write_ids(train_ids, ["t1"])
    assert main(["--registry", str(registry_path), "register-train", "P1", str(train_ids)]) == EXIT_OK

    check_ids = tmp_path / "check.txt"
    _write_ids(check_ids, ["h1", "h2"])
    rc = main(["--registry", str(registry_path), "check", str(check_ids)])

    assert rc == EXIT_OK
    assert "clean" in capsys.readouterr().out


def test_check_contaminated_exits_nonzero_and_lists_ids(tmp_path, capsys):
    registry_path = tmp_path / "registry.json"
    train_ids = tmp_path / "train.txt"
    _write_ids(train_ids, ["t1", "t2"])
    assert main(["--registry", str(registry_path), "register-train", "P1", str(train_ids)]) == EXIT_OK

    check_ids = tmp_path / "check.txt"
    _write_ids(check_ids, ["t1", "unrelated"])
    rc = main(["--registry", str(registry_path), "check", str(check_ids)])

    assert rc == EXIT_CONTAMINATION
    err = capsys.readouterr().err
    assert "t1" in err
    assert "unrelated" not in err


def test_check_against_missing_registry_file_treats_as_empty(tmp_path):
    registry_path = tmp_path / "does-not-exist.json"
    check_ids = tmp_path / "check.txt"
    _write_ids(check_ids, ["anything"])

    rc = main(["--registry", str(registry_path), "check", str(check_ids)])

    assert rc == EXIT_OK


# -- show -------------------------------------------------------------------------


def test_show_empty_registry(tmp_path, capsys):
    registry_path = tmp_path / "does-not-exist.json"
    rc = main(["--registry", str(registry_path), "show"])
    assert rc == EXIT_OK
    assert "empty registry" in capsys.readouterr().out


def test_show_lists_packages_and_totals(tmp_path, capsys):
    registry_path = tmp_path / "registry.json"
    train_ids = tmp_path / "train.txt"
    _write_ids(train_ids, ["t1", "t2"])
    held_ids = tmp_path / "held.txt"
    _write_ids(held_ids, ["h1"])
    assert main(["--registry", str(registry_path), "register-train", "P1", str(train_ids)]) == EXIT_OK
    assert main(["--registry", str(registry_path), "register-held-out", "P2", str(held_ids)]) == EXIT_OK

    rc = main(["--registry", str(registry_path), "show"])

    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert "P1: 2 train id(s), 0 held-out id(s)" in out
    assert "P2: 0 train id(s), 1 held-out id(s)" in out
    assert "totals: 2 train id(s), 1 held-out id(s) across 2 package(s)" in out


def test_show_unsupported_schema_version_is_usage_error(tmp_path, capsys):
    registry_path = tmp_path / "future.json"
    registry_path.write_text(
        json.dumps({"schema_version": 9999, "package_train_ids": {}, "package_held_out_ids": {}}),
        encoding="utf-8",
    )

    rc = main(["--registry", str(registry_path), "show"])

    assert rc == EXIT_ERROR
    assert "error loading registry" in capsys.readouterr().err


# -- ids-file parsing conventions --------------------------------------------------


def test_ids_file_ignores_blank_lines_and_comments(tmp_path):
    ids_file = tmp_path / "train.txt"
    ids_file.write_text("t1\n\n# a comment\nt2\n", encoding="utf-8")
    registry_path = tmp_path / "registry.json"

    rc = main(["--registry", str(registry_path), "register-train", "P1", str(ids_file)])

    assert rc == EXIT_OK
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert payload["package_train_ids"] == {"P1": ["t1", "t2"]}
