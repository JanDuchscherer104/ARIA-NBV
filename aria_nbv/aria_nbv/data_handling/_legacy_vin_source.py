"""Legacy oracle-cache-backed VIN source config for the training datamodule.

NBV_LEGACY_OFFLINE_CACHE_REMOVE_AFTER_FULL_MIGRATION:
This module isolates the last training-facing config branch that still depends
on the legacy oracle cache. Remove it once all consumers use the canonical
online or immutable offline dataset sources from ``_vin_sources``.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from ..configs import PathConfig
from ..utils import BaseConfig, Stage
from ._legacy_oracle_cache import OracleRriCacheDatasetConfig, OracleRriCacheVinDataset
from ._vin_sources import VinOfflineSourceConfig, VinOracleOnlineDatasetConfig


class VinOracleCacheDatasetConfig(BaseConfig):
    """Configuration for the legacy oracle-cache-backed VIN dataset source."""

    kind: Literal["offline_cache"] = "offline_cache"
    """Discriminator for the legacy offline cache dataset."""

    @property
    def target(self) -> type[OracleRriCacheVinDataset]:
        """Return the factory target for :meth:`BaseConfig.setup_target`."""
        return OracleRriCacheVinDataset

    paths: PathConfig = Field(default_factory=PathConfig)
    """Project path resolver."""

    cache: OracleRriCacheDatasetConfig = Field(default_factory=OracleRriCacheDatasetConfig)
    """Legacy oracle-cache dataset configuration with VIN-batch reads enabled."""

    train_split: Literal["train", "val", "all"] = "train"
    """Cache split to use for training."""

    val_split: Literal["train", "val", "all"] = "val"
    """Cache split to use for validation and testing."""

    def setup_target(self, *, split: Stage) -> OracleRriCacheVinDataset:  # type: ignore[override]
        """Instantiate the legacy cached VIN dataset for the requested split."""
        cache_split = self.train_split if split is Stage.TRAIN else self.val_split
        cache_cfg = self.cache.model_copy(deep=True)
        cache_cfg.paths = self.paths
        cache_cfg.cache.paths = self.paths
        cache_cfg.split = cache_split
        return OracleRriCacheVinDataset(cache_cfg)

    @property
    def is_map_style(self) -> bool:
        """Return whether this source yields a map-style dataset."""
        return True


LegacyVinDatasetSourceConfig: TypeAlias = Annotated[
    VinOracleOnlineDatasetConfig | VinOracleCacheDatasetConfig | VinOfflineSourceConfig,
    Field(discriminator="kind"),
]
"""Compatibility union that still includes the legacy cached training source."""

VinOracleDatasetConfig = LegacyVinDatasetSourceConfig
"""Legacy compatibility alias for the broader training source union."""

__all__ = [
    "LegacyVinDatasetSourceConfig",
    "VinOracleCacheDatasetConfig",
    "VinOracleDatasetConfig",
]
