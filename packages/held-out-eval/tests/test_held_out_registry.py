"""Contract-conformance tests for HeldOutExclusionRegistry.

Directly tests the two design rules from ``docs/DATA_GOVERNANCE.md``
("Multi-package held-out exclusion registries: track both directions,
per package") and reproduces, in miniature, the exact contamination
shape issue #11 found: one package's train ids silently accepted as a
later package's held-out ids.
"""

from __future__ import annotations

import os

import held_out_eval.registry
import pytest
from held_out_eval import (
    ContaminationError,
    HeldOutExclusionRegistry,
    UnsupportedSchemaVersionError,
)

# 1. Basic registration and derived flat views -------------------------------


def test_register_train_then_all_train_ids_reflects_it():
    registry = HeldOutExclusionRegistry()
    registry.register_package_train("P1", ["t1", "t2", "t3"])
    assert registry.all_train_ids() == frozenset({"t1", "t2", "t3"})
    assert registry.all_held_out_ids() == frozenset()


def test_register_held_out_then_all_held_out_ids_reflects_it():
    registry = HeldOutExclusionRegistry()
    registry.register_package_held_out("P1", ["h1", "h2"])
    assert registry.all_held_out_ids() == frozenset({"h1", "h2"})
    assert registry.all_train_ids() == frozenset()


def test_multiple_packages_train_ids_union_correctly():
    registry = HeldOutExclusionRegistry()
    registry.register_package_train("P1", ["t1", "t2"])
    registry.register_package_train("P2", ["t3", "t4"])
    assert registry.all_train_ids() == frozenset({"t1", "t2", "t3", "t4"})


# 2. Bidirectional contamination checks (design rule 1) ----------------------
#
# The reverse direction (held-out leaking into a later train split) was
# already commonly checked before issue #11; these two tests prove BOTH
# directions are now enforced, not just the historically-checked one.


def test_registering_held_out_id_already_used_as_train_elsewhere_is_rejected():
    """This is exactly the direction issue #11's original registry missed:
    an id already used as TRAIN by an earlier package must never later
    be accepted as HELD-OUT by a later package."""
    registry = HeldOutExclusionRegistry()
    registry.register_package_train("P1", ["task-001", "task-002", "task-003"])

    with pytest.raises(ContaminationError) as exc_info:
        registry.register_package_held_out("P2", ["task-002", "task-999"])

    assert exc_info.value.direction == "held-out"
    assert exc_info.value.package_id == "P2"
    assert exc_info.value.contaminated_ids == frozenset({"task-002"})
    # The uncontaminated package remains untouched -- a rejected
    # registration must not partially apply.
    assert "P2" not in registry.package_held_out_ids


def test_registering_train_id_already_used_as_held_out_elsewhere_is_rejected():
    """The historically-checked direction: an id already used as
    HELD-OUT by an earlier package must never later be registered as
    TRAIN by a later package."""
    registry = HeldOutExclusionRegistry()
    registry.register_package_held_out("P1", ["task-010", "task-011"])

    with pytest.raises(ContaminationError) as exc_info:
        registry.register_package_train("P2", ["task-010", "task-020"])

    assert exc_info.value.direction == "train"
    assert exc_info.value.package_id == "P2"
    assert exc_info.value.contaminated_ids == frozenset({"task-010"})
    assert "P2" not in registry.package_train_ids


def test_issue_11_reproduction_67_id_shaped_contamination_is_caught():
    """Miniature reproduction of the issue #11 incident shape: P1's train
    split and P2's held-out split share ids. The bidirectional registry
    catches it at registration time instead of silently admitting it."""
    registry = HeldOutExclusionRegistry()
    p1_train = {f"task-{i:04d}" for i in range(200)}
    registry.register_package_train("P1", p1_train)

    p2_held_out_with_leak = {f"task-{i:04d}" for i in range(133, 217)}  # 67 overlapping ids (133-199)
    overlap = p1_train & p2_held_out_with_leak
    assert len(overlap) == 67  # sanity-check the reproduction shape itself

    with pytest.raises(ContaminationError) as exc_info:
        registry.register_package_held_out("P2", p2_held_out_with_leak)
    assert exc_info.value.contaminated_ids == overlap


# 3. A package may supersede its own prior contribution (design rule 2) ------


def test_package_can_reregister_its_own_train_ids_without_self_contamination():
    """A package re-splitting/correcting its own train ids must not be
    flagged as contaminating itself -- exclude_package logic must apply."""
    registry = HeldOutExclusionRegistry()
    registry.register_package_train("P1", ["t1", "t2", "t3"])
    # P1 corrects its own split, dropping t3 and adding t4. Must not raise.
    registry.register_package_train("P1", ["t1", "t2", "t4"])
    assert registry.all_train_ids() == frozenset({"t1", "t2", "t4"})


def test_re_registration_supersedes_only_its_own_contribution_not_a_flat_union():
    """Second design-rule regression test: after P1 re-splits to remove a
    previously-registered id, that id must be gone from the registry's
    flat view entirely -- not left behind forever in an ever-growing
    union (the second gap issue #11 found: a flat-union design keeps
    flagging against ids that no longer exist in any current package,
    punishing the fix)."""
    registry = HeldOutExclusionRegistry()
    registry.register_package_train("P1", ["stale-id", "kept-id"])
    assert "stale-id" in registry.all_train_ids()

    # P1 re-splits, removing "stale-id" entirely (it was found to be
    # contaminated and was excised from P1's own dataset).
    registry.register_package_train("P1", ["kept-id"])

    assert "stale-id" not in registry.all_train_ids()
    assert registry.all_train_ids() == frozenset({"kept-id"})

    # A later package must now be able to legitimately use "stale-id" as
    # held-out, since no current package claims it as train anymore.
    registry.register_package_held_out("P2", ["stale-id"])
    assert "stale-id" in registry.all_held_out_ids()


def test_reregistering_held_out_does_not_self_contaminate():
    registry = HeldOutExclusionRegistry()
    registry.register_package_held_out("P1", ["h1", "h2"])
    registry.register_package_held_out("P1", ["h1", "h3"])
    assert registry.all_held_out_ids() == frozenset({"h1", "h3"})


# 4. check_held_out_not_trained: the read-only precondition check used by
#    the evaluator contract at evaluation time (not only at split time) ----


def test_check_held_out_not_trained_returns_empty_when_clean():
    registry = HeldOutExclusionRegistry()
    registry.register_package_train("P1", ["t1"])
    registry.register_package_held_out("P2", ["h1", "h2"])
    assert registry.check_held_out_not_trained(["h1", "h2"]) == frozenset()


def test_check_held_out_not_trained_flags_ids_trained_by_any_package():
    registry = HeldOutExclusionRegistry()
    registry.register_package_train("P1", ["t1", "t2"])
    # Does not require prior registration as held-out anywhere -- this
    # is a re-check against the registry's CURRENT state, independent
    # of whatever was declared at split time.
    assert registry.check_held_out_not_trained(["t1", "unrelated"]) == frozenset({"t1"})


def test_check_held_out_not_trained_catches_contamination_registered_after_split():
    """Models the evaluator-contract's actual use: a held-out set was
    validly registered as held-out at split time, but a LATER package
    then (incorrectly) registers an overlapping id as train. The
    evaluator's own re-check at evaluation time must still catch it,
    not just the registry's own registration-time check."""
    registry = HeldOutExclusionRegistry()
    registry.register_package_held_out("P1", ["shared-id"])
    # A hypothetical bug/bypass elsewhere lets P2 register "shared-id" as
    # train despite P1's prior held-out claim; this call itself would be
    # rejected by register_package_train's own bidirectional check, so
    # simulate the registry ending up in this (invalid, unreachable via
    # this module's own API) state directly to prove the read-only
    # checker doesn't silently trust stale split-time state either.
    registry.package_train_ids["P2"] = frozenset({"shared-id"})
    assert registry.check_held_out_not_trained(["shared-id"]) == frozenset({"shared-id"})


# 5. Persistence round-trip ---------------------------------------------------


def test_to_dict_from_dict_round_trip():
    registry = HeldOutExclusionRegistry()
    registry.register_package_train("P1", ["t1", "t2"])
    registry.register_package_held_out("P2", ["h1"])
    restored = HeldOutExclusionRegistry.from_dict(registry.to_dict())
    assert restored.all_train_ids() == registry.all_train_ids()
    assert restored.all_held_out_ids() == registry.all_held_out_ids()
    assert restored.package_train_ids == registry.package_train_ids
    assert restored.package_held_out_ids == registry.package_held_out_ids


def test_save_load_round_trip(tmp_path):
    registry = HeldOutExclusionRegistry()
    registry.register_package_train("P1", ["t1"])
    registry.register_package_held_out("P2", ["h1", "h2"])
    path = tmp_path / "registry.json"
    registry.save(path)

    restored = HeldOutExclusionRegistry.load(path)
    assert restored.all_train_ids() == frozenset({"t1"})
    assert restored.all_held_out_ids() == frozenset({"h1", "h2"})


def test_load_nonexistent_path_returns_empty_registry(tmp_path):
    registry = HeldOutExclusionRegistry.load(tmp_path / "does-not-exist.json")
    assert registry.all_train_ids() == frozenset()
    assert registry.all_held_out_ids() == frozenset()


def test_saved_registry_is_per_package_not_a_flat_blob(tmp_path):
    """Verifies the on-disk shape itself is per-package (design rule 2),
    not a single flat array -- so the persisted format cannot silently
    regress back to the flat-union design issue #11 flagged."""
    import json

    registry = HeldOutExclusionRegistry()
    registry.register_package_train("P1", ["t1"])
    registry.register_package_held_out("P2", ["h1"])
    path = tmp_path / "registry.json"
    registry.save(path)

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == 1
    assert on_disk["package_train_ids"] == {"P1": ["t1"]}
    assert on_disk["package_held_out_ids"] == {"P2": ["h1"]}


# 6. Schema versioning (issue #53) --------------------------------------------


def test_to_dict_includes_current_schema_version():
    registry = HeldOutExclusionRegistry()
    registry.register_package_train("P1", ["t1"])
    payload = registry.to_dict()
    assert payload["schema_version"] == held_out_eval.registry.CURRENT_SCHEMA_VERSION


def test_from_dict_round_trips_with_schema_version():
    registry = HeldOutExclusionRegistry()
    registry.register_package_train("P1", ["t1", "t2"])
    registry.register_package_held_out("P2", ["h1"])
    payload = registry.to_dict()

    restored = HeldOutExclusionRegistry.from_dict(payload)
    assert restored.package_train_ids == registry.package_train_ids
    assert restored.package_held_out_ids == registry.package_held_out_ids


def test_from_dict_missing_schema_version_raises():
    payload = {"package_train_ids": {"P1": ["t1"]}, "package_held_out_ids": {}}
    with pytest.raises(UnsupportedSchemaVersionError):
        HeldOutExclusionRegistry.from_dict(payload)


def test_from_dict_unknown_schema_version_raises():
    payload = {
        "schema_version": 9999,
        "package_train_ids": {},
        "package_held_out_ids": {},
    }
    with pytest.raises(UnsupportedSchemaVersionError):
        HeldOutExclusionRegistry.from_dict(payload)


def test_load_file_with_unknown_schema_version_raises(tmp_path):
    import json

    path = tmp_path / "future.json"
    path.write_text(
        json.dumps({"schema_version": 9999, "package_train_ids": {}, "package_held_out_ids": {}}),
        encoding="utf-8",
    )
    with pytest.raises(UnsupportedSchemaVersionError):
        HeldOutExclusionRegistry.load(path)


# 7. save() atomicity (issue #54) ---------------------------------------------


def test_save_uses_atomic_replace(tmp_path, monkeypatch):
    """save() must go through os.replace(), not a direct in-place write,
    so an interruption mid-write can never leave a partial file at the
    target path."""
    import os as os_module

    registry = HeldOutExclusionRegistry()
    registry.register_package_train("P1", ["t1"])
    path = tmp_path / "registry.json"

    calls = []
    real_replace = os_module.replace

    def spy_replace(src, dst):
        calls.append((src, dst))
        return real_replace(src, dst)

    monkeypatch.setattr(os_module, "replace", spy_replace)
    registry.save(path)

    assert len(calls) == 1
    assert calls[0][1] == path
    assert path.exists()


def test_save_leaves_original_file_untouched_on_mid_write_failure(tmp_path, monkeypatch):
    """If writing the temp file fails partway through, the original file
    at the target path (if any) must remain fully intact -- never
    truncated or partially overwritten."""
    registry = HeldOutExclusionRegistry()
    registry.register_package_train("P1", ["t1"])
    path = tmp_path / "registry.json"

    original = HeldOutExclusionRegistry()
    original.register_package_train("ORIGINAL", ["orig1"])
    original.save(path)
    original_bytes = path.read_bytes()

    def broken_fsync(_fd):
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr(os, "fsync", broken_fsync)

    with pytest.raises(OSError):
        registry.save(path)

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_creates_parent_directories(tmp_path):
    registry = HeldOutExclusionRegistry()
    registry.register_package_train("P1", ["t1"])
    nested_path = tmp_path / "nested" / "dirs" / "registry.json"

    registry.save(nested_path)

    assert nested_path.exists()
    restored = HeldOutExclusionRegistry.load(nested_path)
    assert restored.all_train_ids() == frozenset({"t1"})
