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
    "EfmCameraView",
    "EfmTrajectoryView",
    "EfmPointsView",
    "EfmObbView",
    "EfmGTView",
    "EfmSnippetView",
]
