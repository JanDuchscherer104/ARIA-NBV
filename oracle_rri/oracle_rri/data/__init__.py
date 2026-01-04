"""ASE dataset handling - simplified."""

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
    OracleRriCacheAppender,
    OracleRriCacheAppenderConfig,
    OracleRriCacheDataset,
    OracleRriCacheDatasetConfig,
    OracleRriCacheWriter,
    OracleRriCacheWriterConfig,
)
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
    "OracleRriCacheAppender",
    "OracleRriCacheAppenderConfig",
    "OracleRriCacheDataset",
    "OracleRriCacheDatasetConfig",
    "OracleRriCacheWriter",
    "OracleRriCacheWriterConfig",
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
