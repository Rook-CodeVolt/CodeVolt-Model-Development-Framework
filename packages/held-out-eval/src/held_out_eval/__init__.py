"""held-out-eval: a standalone, trainer-agnostic contamination registry.

Public API re-exported here is the entire surface of this package:
``HeldOutExclusionRegistry``, ``HeldOutRegistryError``, and
``ContaminationError``. See ``registry.py`` for the full implementation
and design-rationale docstrings (unchanged from the original module
this package was extracted from,
``codevolt_mdf.held_out_registry`` in the CodeVolt-Model-Development-Framework
repository -- see that project's ``docs/decisions/0012-*.md`` for the
extraction rationale).

This package has zero dependency on any specific trainer, evaluator,
or ML framework, and zero dependency on codevolt-mdf. It only imports
the Python standard library (``json``, ``dataclasses``, ``pathlib``,
``collections.abc``). It is usable next to any training loop -- PyTorch,
TensorFlow, JAX, a plain shell script, or an entirely different
framework's trainer -- because it is decoupled from all of them.
"""

from __future__ import annotations

from .registry import (
    ContaminationError,
    HeldOutExclusionRegistry,
    HeldOutRegistryError,
    UnsupportedSchemaVersionError,
)

__all__ = [
    "ContaminationError",
    "HeldOutExclusionRegistry",
    "HeldOutRegistryError",
    "UnsupportedSchemaVersionError",
]

__version__ = "0.1.0"
