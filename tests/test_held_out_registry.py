"""Contract-conformance tests for HeldOutExclusionRegistry.

Directly tests the two design rules from ``docs/DATA_GOVERNANCE.md``
("Multi-package held-out exclusion registries: track both directions,
per package") and reproduces, in miniature, the exact contamination
shape issue #11 found: one package's train ids silently accepted as a
later package's held-out ids.
"""

from __future__ import annotations

import pytest

from codevolt_mdf.held_out_registry import ContaminationError, HeldOutExclusionRegistry

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
    assert on_disk["package_train_ids"] == {"P1": ["t1"]}
    assert on_disk["package_held_out_ids"] == {"P2": ["h1"]}
