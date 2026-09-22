"""Minimal stdlib-only (argparse) command-line interface for held-out-eval.

Wraps :class:`~held_out_eval.registry.HeldOutExclusionRegistry` so the
entire register/check workflow is scriptable from bash/CI against a JSON
registry file on disk, with no Python glue code required. See the
"CLI usage" section of this package's README for a worked example.

Design notes
------------
- Stdlib-only: only ``argparse``, ``json`` (transitively, via the
  registry module), ``pathlib``, and ``sys`` are used here, consistent
  with this package's zero-dependency goal (see ``pyproject.toml``).
- Exit codes are deliberately meaningful for CI use:
    0 -- success / no contamination found
    1 -- contamination detected (a register-* call would create
         contamination, or `check` found ids registered as train data)
    2 -- usage or I/O error (missing ids file, unreadable or
         unsupported-schema registry file, etc.)
- Ids files are plain text, one id per line. Blank lines and lines
  starting with ``#`` are ignored, so a checked-in ids file can carry
  comments describing where a split came from.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .registry import ContaminationError, HeldOutExclusionRegistry, HeldOutRegistryError

#: Process exit codes. Meaningful and stable for use in CI pipelines'
#: exit-code checks -- see module docstring.
EXIT_OK = 0
EXIT_CONTAMINATION = 1
EXIT_ERROR = 2

DEFAULT_REGISTRY_PATH = Path("held_out_registry.json")


def _read_ids_file(path: Path) -> list[str]:
    """Read one id per line from ``path``.

    Blank lines and lines starting with ``#`` are ignored. Raises
    ``FileNotFoundError`` if ``path`` does not exist -- callers turn
    that into an ``EXIT_ERROR`` CLI exit rather than a traceback.
    """
    if not path.exists():
        raise FileNotFoundError(f"ids file not found: {path}")
    ids: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return ids


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="held-out-eval",
        description=(
            "Register and check train/held-out id sets against a "
            "HeldOutExclusionRegistry JSON file on disk, without writing "
            "any Python glue code. See the Python API "
            "(held_out_eval.HeldOutExclusionRegistry) for programmatic use."
        ),
    )
    parser.add_argument(
        "--registry",
        "-r",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        metavar="PATH",
        help=(
            "Path to the registry JSON file to read/write "
            f"(default: {DEFAULT_REGISTRY_PATH}). Created on first "
            "successful register-* call if it does not yet exist."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser(
        "register-train",
        help="Register a package's train ids; rejects ids already registered as held-out elsewhere.",
    )
    p_train.add_argument("package_id", help="Opaque id of the package registering these ids.")
    p_train.add_argument(
        "ids_file", type=Path, help="Text file, one id per line (# comments and blank lines allowed)."
    )

    p_held_out = subparsers.add_parser(
        "register-held-out",
        help="Register a package's held-out ids; rejects ids already registered as train elsewhere.",
    )
    p_held_out.add_argument("package_id", help="Opaque id of the package registering these ids.")
    p_held_out.add_argument(
        "ids_file", type=Path, help="Text file, one id per line (# comments and blank lines allowed)."
    )

    p_check = subparsers.add_parser(
        "check",
        help="Check whether any id in a file is registered as train data anywhere in the registry.",
    )
    p_check.add_argument(
        "ids_file", type=Path, help="Text file, one id per line (# comments and blank lines allowed)."
    )

    subparsers.add_parser(
        "show",
        help="Print a summary of every package's registered train/held-out id counts.",
    )

    return parser


def _load_registry(path: Path) -> HeldOutExclusionRegistry | None:
    """Load the registry at ``path``, or print an error and return ``None``.

    A missing file is not an error (``HeldOutExclusionRegistry.load``
    already treats it as an empty registry); this only intercepts
    genuine load failures such as an unsupported schema version or
    malformed JSON.
    """
    try:
        return HeldOutExclusionRegistry.load(path)
    except HeldOutRegistryError as exc:
        print(f"error loading registry {path}: {exc}", file=sys.stderr)
        return None
    except ValueError as exc:  # malformed JSON
        print(f"error loading registry {path}: invalid JSON ({exc})", file=sys.stderr)
        return None


def _cmd_register_train(args: argparse.Namespace) -> int:
    return _register(args, direction="train")


def _cmd_register_held_out(args: argparse.Namespace) -> int:
    return _register(args, direction="held-out")


def _register(args: argparse.Namespace, *, direction: str) -> int:
    try:
        ids = _read_ids_file(args.ids_file)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    registry = _load_registry(args.registry)
    if registry is None:
        return EXIT_ERROR

    try:
        if direction == "train":
            registry.register_package_train(args.package_id, ids)
        else:
            registry.register_package_held_out(args.package_id, ids)
    except ContaminationError as exc:
        print(f"contamination detected: {exc}", file=sys.stderr)
        return EXIT_CONTAMINATION

    registry.save(args.registry)
    print(
        f"registered {len(ids)} {direction} id(s) for package {args.package_id!r} in {args.registry}"
    )
    return EXIT_OK


def _cmd_check(args: argparse.Namespace) -> int:
    try:
        ids = _read_ids_file(args.ids_file)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    registry = _load_registry(args.registry)
    if registry is None:
        return EXIT_ERROR

    leaked = registry.check_held_out_not_trained(ids)
    if leaked:
        print(
            f"contamination: {len(leaked)} id(s) from {args.ids_file} "
            "are registered as train data:",
            file=sys.stderr,
        )
        for id_ in sorted(leaked):
            print(f"  {id_}", file=sys.stderr)
        return EXIT_CONTAMINATION

    print(
        f"clean: no ids from {args.ids_file} are registered as train data "
        f"({len(ids)} id(s) checked)"
    )
    return EXIT_OK


def _cmd_show(args: argparse.Namespace) -> int:
    registry = _load_registry(args.registry)
    if registry is None:
        return EXIT_ERROR

    packages = sorted(set(registry.package_train_ids) | set(registry.package_held_out_ids))
    if not packages:
        print(f"{args.registry}: empty registry (no packages registered)")
        return EXIT_OK

    print(f"{args.registry}:")
    for package_id in packages:
        train_count = len(registry.package_train_ids.get(package_id, frozenset()))
        held_out_count = len(registry.package_held_out_ids.get(package_id, frozenset()))
        print(f"  {package_id}: {train_count} train id(s), {held_out_count} held-out id(s)")
    print(
        f"totals: {len(registry.all_train_ids())} train id(s), "
        f"{len(registry.all_held_out_ids())} held-out id(s) across {len(packages)} package(s)"
    )
    return EXIT_OK


_COMMANDS = {
    "register-train": _cmd_register_train,
    "register-held-out": _cmd_register_held_out,
    "check": _cmd_check,
    "show": _cmd_show,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _COMMANDS[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
