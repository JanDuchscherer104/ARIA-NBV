"""Compatibility wrapper for VIN dataset-source configs.

The canonical source configs now live in ``aria_nbv.data_handling._vin_sources``
and cover only the online and immutable offline dataset paths. The legacy
oracle-cache-backed source config remains isolated in
``aria_nbv.data_handling._legacy_vin_source`` until the full cutover is done.
"""

from __future__ import annotations

from ._legacy_vin_source import (
    LegacyVinDatasetSourceConfig,
    VinOracleCacheDatasetConfig,
    VinOracleDatasetConfig,
)
from ._vin_sources import (
    VinOfflineSourceConfig,
    VinOracleOnlineDataset,
    VinOracleOnlineDatasetConfig,
)

VinDatasetSourceConfig = LegacyVinDatasetSourceConfig
"""Compatibility alias that still includes the legacy cached source branch."""

__all__ = [
    "VinDatasetSourceConfig",
    "VinOfflineSourceConfig",
    "VinOracleCacheDatasetConfig",
    "VinOracleDatasetConfig",
    "VinOracleOnlineDataset",
    "VinOracleOnlineDatasetConfig",
]
