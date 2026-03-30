"""Compatibility wrapper for the legacy VIN snippet cache runtime.

NBV_LEGACY_OFFLINE_CACHE_REMOVE_AFTER_FULL_MIGRATION:
This public module exists only to preserve older import paths. The actual
legacy implementation lives in ``aria_nbv.data_handling._legacy_vin_cache``.
"""

from __future__ import annotations

import sys

from . import _legacy_vin_cache as _impl

sys.modules[__name__] = _impl
