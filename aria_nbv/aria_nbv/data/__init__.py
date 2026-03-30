"""Legacy compatibility exports backed by :mod:`aria_nbv.data_handling`."""

from .downloader import ASEDownloader, ASEDownloaderConfig
from .efm_dataset import AseEfmDataset, AseEfmDatasetConfig

# Backward compatibility: legacy name used in tests.
from .efm_views import (  # noqa: E402
    EfmCameraView,
    EfmGTView,
    EfmObbView,
    EfmPointsView,
    EfmSnippetView,
    EfmTrajectoryView,
    VinSnippetView,
)
from .metadata import ASEMetadata, SceneMetadata
from .offline_cache import (
    OracleRriCacheDataset,
    OracleRriCacheDatasetConfig,
    OracleRriCacheVinDataset,
    OracleRriCacheWriter,
    OracleRriCacheWriterConfig,
)
from .vin_oracle_datasets import VinOracleCacheDatasetConfig, VinOracleDatasetConfig, VinOracleOnlineDatasetConfig
from .vin_oracle_types import VinOracleBatch
from .vin_snippet_cache import (
    VinSnippetCacheConfig,
    VinSnippetCacheDataset,
    VinSnippetCacheDatasetConfig,
    VinSnippetCacheWriter,
    VinSnippetCacheWriterConfig,
)

__all__ = [
    "ASEMetadata",
    "SceneMetadata",
    "ASEDownloader",
    "ASEDownloaderConfig",
    "AseEfmDataset",
    "AseEfmDatasetConfig",
    "OracleRriCacheDataset",
    "OracleRriCacheDatasetConfig",
    "OracleRriCacheVinDataset",
    "OracleRriCacheWriter",
    "OracleRriCacheWriterConfig",
    "VinOracleBatch",
    "VinOracleCacheDatasetConfig",
    "VinOracleDatasetConfig",
    "VinOracleOnlineDatasetConfig",
    "VinSnippetCacheConfig",
    "VinSnippetCacheDataset",
    "VinSnippetCacheDatasetConfig",
    "VinSnippetCacheWriter",
    "VinSnippetCacheWriterConfig",
    "EfmCameraView",
    "EfmTrajectoryView",
    "EfmPointsView",
    "EfmObbView",
    "EfmGTView",
    "EfmSnippetView",
    "VinSnippetView",
]
