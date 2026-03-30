"""Compatibility wrapper for legacy offline-cache metadata helpers.

NBV_LEGACY_OFFLINE_CACHE_REMOVE_AFTER_FULL_MIGRATION:
This public module exists only to preserve older import paths. The actual
legacy implementation lives in
``aria_nbv.data_handling._legacy_offline_cache_store``.
"""

from __future__ import annotations

import sys

from . import _legacy_offline_cache_store as _impl

sys.modules[__name__] = _impl
