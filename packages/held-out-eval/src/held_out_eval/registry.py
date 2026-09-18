"""Bidirectional, per-package held-out exclusion registry.

Reference implementation of the design pattern documented in
``docs/DATA_GOVERNANCE.md`` ("Multi-package held-out exclusion
registries: track both directions, per package"), itself written up
from a real contamination incident
(`issue #11 <https://github.com/Rook-CodeVolt/CodeVolt-Model-Development-Framework/issues/11>`_).

The incident this closes, restated concretely: a held-out-exclusion
registry that only ever tracks "used as held-out by any package" and
excludes those ids from later packages' candidate pools catches only
one direction of contamination (an earlier package's held-out ids
leaking into a later package's train split). It misses the reverse,
equally serious direction: an id already used as **train** data by an
earlier package must never later be selected as **held-out** by a
later package, because the model may already have seen that item
during the earlier package's training -- which invalidates any
"held-out" claim about it. The incident behind this module found 67
ids from one package's train split were also present in a later
package's held-out split, undetected until a general train/held-out
contamination check ran in *both* directions.

This class originated inside the CodeVolt-Model-Development-Framework,
where it is used by that project's ``evaluator_contract.py``'s
``run_evaluator_contract`` to make this bidirectional check a real,
enforced precondition of every evaluation run -- not just a documented
pattern -- so "the trainer adapter never has access to the held-out
set" is backed by "and the held-out set was never actually used as
train data either," not merely declared. It now also ships standalone
as the ``held-out-eval`` package (this file), with zero dependency on
that framework or on any specific trainer/evaluator, precisely so it
can be used the same way next to any training loop -- see this
package's README for a trainer-agnostic usage example.

Design rules (both required; see the docstring above and
``docs/DATA_GOVERNANCE.md`` for the full rationale):

1. Track both directions symmetrically: ``train_used_ids`` and
   ``held_out_used_ids`` are each checked against the other's flat
   union before a new registration is admitted.
2. Store registry state per package (``package_train_ids``,
   ``package_held_out_ids``), never as a single flat ever-growing
   union, so a package's own re-registration supersedes only its own
   prior contribution rather than leaving stale ids in the registry
   forever after a legitimate re-split.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

#: Current on-disk schema version written by :meth:`HeldOutExclusionRegistry.to_dict`.
#: Bump this, and add the old value to ``SUPPORTED_SCHEMA_VERSIONS`` with an
#: explicit migration in :meth:`from_dict`, whenever the persisted shape changes.
CURRENT_SCHEMA_VERSION = 1

#: Schema versions this version of the library can load without migration.
SUPPORTED_SCHEMA_VERSIONS = frozenset({1})


class HeldOutRegistryError(Exception):
    """Base class for every error this module defines."""


class UnsupportedSchemaVersionError(HeldOutRegistryError):
    """A persisted registry file's ``schema_version`` is missing or unrecognized.

    Raised by :meth:`HeldOutExclusionRegistry.from_dict` rather than letting
    an old or newer file be silently misparsed as the current schema, or
    fail later with an opaque ``KeyError``/``TypeError``.
    """

    def __init__(self, found: object) -> None:
        self.found = found
        super().__init__(
            f"unsupported or missing schema_version {found!r} in persisted registry "
            f"payload; this version of held-out-eval supports "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}. Loading a file written by a "
            "materially different version of this library without an explicit "
            "migration is refused rather than silently misinterpreted."
        )


class ContaminationError(HeldOutRegistryError):
    """A registration would create train/held-out contamination.

    Raised when a package tries to register an id as held-out that is
    already registered as train by *any* package (including itself),
    or vice versa -- the exact bidirectional check issue #11's
    original one-directional registry did not perform.
    """

    def __init__(self, direction: str, package_id: str, contaminated_ids: frozenset[str]) -> None:
        self.direction = direction
        self.package_id = package_id
        self.contaminated_ids = contaminated_ids
        super().__init__(
            f"package {package_id!r} attempted to register {len(contaminated_ids)} id(s) as "
            f"{direction}, but they are already registered as the opposite usage by another "
            f"package's contribution to this registry: {sorted(contaminated_ids)[:10]}"
            + (" ..." if len(contaminated_ids) > 10 else "")
        )


@dataclass
class HeldOutExclusionRegistry:
    """Bidirectional, per-package train/held-out id registry.

    ``package_train_ids`` and ``package_held_out_ids`` are each keyed by
    an opaque ``package_id`` string; every derived flat view
    (``all_train_ids()``, ``all_held_out_ids()``) is computed from
    these, never stored as its own separate source of truth, so a
    package's own re-registration supersedes only its own prior
    contribution (design rule 2).
    """

    package_train_ids: dict[str, frozenset[str]] = field(default_factory=dict)
    package_held_out_ids: dict[str, frozenset[str]] = field(default_factory=dict)

    # -- derived flat views -------------------------------------------------

    def all_train_ids(self, *, exclude_package: str | None = None) -> frozenset[str]:
        """Union of every package's train ids.

        ``exclude_package`` omits that package's own contribution --
        used when checking a package's *own* new registration, so a
        package is allowed to re-register a superseding version of its
        own prior contribution without being flagged as contaminating
        itself.
        """
        result: set[str] = set()
        for package_id, ids in self.package_train_ids.items():
            if package_id == exclude_package:
                continue
            result.update(ids)
        return frozenset(result)

    def all_held_out_ids(self, *, exclude_package: str | None = None) -> frozenset[str]:
        result: set[str] = set()
        for package_id, ids in self.package_held_out_ids.items():
            if package_id == exclude_package:
                continue
            result.update(ids)
        return frozenset(result)

    # -- registration (design rule 1: bidirectional check on every call) ---

    def register_package_train(self, package_id: str, ids: Iterable[str]) -> None:
        """Register ``package_id``'s train ids, superseding its own prior contribution.

        Raises ``ContaminationError`` if any id is already registered as
        held-out by a *different* package (direction: an id already
        spent as someone else's held-out example must never become
        train data -- the historically-checked direction).
        """
        candidate = frozenset(str(i) for i in ids)
        existing_held_out_elsewhere = self.all_held_out_ids(exclude_package=package_id)
        collision = candidate & existing_held_out_elsewhere
        if collision:
            raise ContaminationError("train", package_id, collision)
        self.package_train_ids[package_id] = candidate

    def register_package_held_out(self, package_id: str, ids: Iterable[str]) -> None:
        """Register ``package_id``'s held-out ids, superseding its own prior contribution.

        Raises ``ContaminationError`` if any id is already registered as
        train data by a *different* package (direction: an id already
        used as someone else's train data must never become a held-out
        example -- the direction issue #11's original registry did not
        check, and the one that let 67 contaminated ids through
        undetected).
        """
        candidate = frozenset(str(i) for i in ids)
        existing_train_elsewhere = self.all_train_ids(exclude_package=package_id)
        collision = candidate & existing_train_elsewhere
        if collision:
            raise ContaminationError("held-out", package_id, collision)
        self.package_held_out_ids[package_id] = candidate

    # -- read-only precondition check (used by the evaluator contract) -----

    def check_held_out_not_trained(self, ids: Iterable[str]) -> frozenset[str]:
        """Return the subset of ``ids`` that are registered as train data anywhere.

        Non-mutating, read-only precondition check -- does not require
        ``ids`` to already be registered as held-out by any package.
        This is what ``evaluator_contract.run_evaluator_contract`` calls
        immediately before scoring: even if a held-out set was properly
        registered via ``register_package_held_out`` at split time, this
        independently re-checks at *evaluation* time against the
        registry's current state (which may have grown since), rather
        than trusting the split-time registration to still be valid.
        """
        candidate = frozenset(str(i) for i in ids)
        return candidate & self.all_train_ids()

    # -- persistence: per-package, not a flat blob --------------------------

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "package_train_ids": {k: sorted(v) for k, v in self.package_train_ids.items()},
            "package_held_out_ids": {k: sorted(v) for k, v in self.package_held_out_ids.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> HeldOutExclusionRegistry:
        version = payload.get("schema_version")
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise UnsupportedSchemaVersionError(version)
        train_ids = payload.get("package_train_ids", {})
        held_out_ids = payload.get("package_held_out_ids", {})
        assert isinstance(train_ids, dict) and isinstance(held_out_ids, dict)
        return cls(
            package_train_ids={k: frozenset(v) for k, v in train_ids.items()},
            package_held_out_ids={k: frozenset(v) for k, v in held_out_ids.items()},
        )

    def save(self, path: Path) -> None:
        """Write this registry to ``path``, atomically.

        Writes to a temporary file in the same directory, ``fsync``s it,
        then ``os.replace()``s it into place. A crash or interruption
        mid-write can never leave a partially-written file at ``path`` --
        either the previous file remains fully intact, or the new one is
        fully in place. This does NOT make concurrent writers to the same
        path safe: two processes calling ``save()`` on the same path at
        the same time can still race (last writer wins), it only protects
        against a single writer being interrupted mid-write. Coordinate
        external locking yourself if multiple processes may write the
        same path concurrently.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                tmp_file.write(payload)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: Path) -> HeldOutExclusionRegistry:
        path = Path(path)
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
