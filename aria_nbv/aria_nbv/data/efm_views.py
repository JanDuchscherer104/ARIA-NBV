"""Legacy compatibility wrapper for the canonical EFM/VIN view contracts.

The active implementation now lives in :mod:`aria_nbv.data_handling.efm_views`.
"""

from __future__ import annotations

from ..data_handling.efm_views import *  # noqa: F401,F403
from ..data_handling.efm_views import __all__ as canonical_all

__all__ = list(canonical_all)
