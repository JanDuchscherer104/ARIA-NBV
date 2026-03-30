"""Legacy compatibility wrapper for canonical offline-cache coverage helpers.

The active implementation now lives in
:mod:`aria_nbv.data_handling.cache_coverage`.
"""

from __future__ import annotations

from ..data_handling.cache_coverage import *  # noqa: F401,F403
from ..data_handling.cache_coverage import __all__ as canonical_all

__all__ = [*canonical_all]
