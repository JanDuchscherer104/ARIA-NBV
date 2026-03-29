"""Legacy compatibility re-exports for offline cache record types.

The canonical oracle/VIN cache record contracts now live in
``aria_nbv.data_handling.cache_contracts``. This legacy module remains only as
the historical import surface for ``aria_nbv.data`` call sites that have not
been migrated yet.
"""

from __future__ import annotations

from ..data_handling.cache_contracts import (
    OracleRriCacheEntry,
    OracleRriCacheMetadata,
    OracleRriCacheSample,
)

__all__ = [
    "OracleRriCacheEntry",
    "OracleRriCacheMetadata",
    "OracleRriCacheSample",
]
