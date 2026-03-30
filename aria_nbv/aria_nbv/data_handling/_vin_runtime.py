"""VIN-facing runtime types and helpers for ``aria_nbv.data_handling``.

This module centralizes the model-facing VIN runtime contracts:

- canonical EFM-to-VIN adaptation helpers,
- the ``VinOracleBatch`` training batch type,
- the base dataset protocol shared by Lightning, and
- the split-aware online VIN dataset configuration used for live oracle labels.
"""

from __future__ import annotations

from ._vin_sources import VinOracleOnlineDataset, VinOracleOnlineDatasetConfig
from .vin_adapter import (
    DEFAULT_VIN_SNIPPET_PAD_POINTS,
    build_vin_snippet_view,
    collapse_vin_points,
    empty_vin_snippet,
    pad_vin_points,
    vin_snippet_cache_config_hash,
)
from .vin_oracle_types import VinOracleBatch, VinOracleDatasetBase

VinOnlineDatasetConfig = VinOracleOnlineDatasetConfig
"""Stable alias for the online VIN dataset configuration."""

__all__ = [
    "DEFAULT_VIN_SNIPPET_PAD_POINTS",
    "VinOnlineDatasetConfig",
    "VinOracleBatch",
    "VinOracleDatasetBase",
    "VinOracleOnlineDataset",
    "VinOracleOnlineDatasetConfig",
    "build_vin_snippet_view",
    "collapse_vin_points",
    "empty_vin_snippet",
    "pad_vin_points",
    "vin_snippet_cache_config_hash",
]
