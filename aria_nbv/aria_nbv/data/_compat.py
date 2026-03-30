"""Compatibility helpers for binding legacy data modules to canonical owners."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


def alias_module(legacy_name: str, canonical_name: str) -> ModuleType:
    """Bind a legacy module path to its canonical module object."""

    module = importlib.import_module(canonical_name)
    sys.modules[legacy_name] = module
    return module
