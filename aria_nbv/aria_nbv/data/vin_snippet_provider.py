"""Legacy compatibility wrapper for canonical VIN snippet providers.

The active implementation now lives in :mod:`aria_nbv.data_handling.vin_provider`.
"""

from __future__ import annotations

from ..data_handling.vin_provider import *  # noqa: F401,F403
from ..data_handling.vin_provider import __all__ as canonical_all

__all__ = list(canonical_all)
