"""Legacy compatibility wrapper for the canonical mesh-cache helpers.

The active implementation now lives in :mod:`aria_nbv.data_handling.mesh_cache`.
"""

from __future__ import annotations

from ..data_handling.mesh_cache import *  # noqa: F401,F403
from ..data_handling.mesh_cache import __all__ as canonical_all

__all__ = list(canonical_all)
