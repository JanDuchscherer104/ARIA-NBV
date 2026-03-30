"""Legacy compatibility wrapper for canonical VIN snippet helper functions.

The active implementation now lives in :mod:`aria_nbv.data_handling.vin_adapter`.
"""

from __future__ import annotations

from ..data_handling.vin_adapter import *  # noqa: F401,F403
from ..data_handling.vin_adapter import __all__ as canonical_all

__all__ = list(canonical_all)
