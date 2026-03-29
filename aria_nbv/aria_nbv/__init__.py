"""Oracle RRI package."""

from __future__ import annotations

# Ensure vendored dependencies are on sys.path when not installed as packages.
import importlib.util
import sys
from pathlib import Path

if importlib.util.find_spec("efm3d") is None:  # pragma: no cover - environment dependent
    vendor = Path(__file__).resolve().parents[1].parent / "external" / "efm3d"
    if vendor.exists():
        sys.path.append(str(vendor))
    else:  # pragma: no cover
        raise ModuleNotFoundError("efm3d not installed and vendor path missing")

from .data import (
    AseEfmDataset,
    AseEfmDatasetConfig,
    ASEMetadata,
    EfmCameraView,
    EfmGTView,
    EfmObbView,
    EfmPointsView,
    EfmSnippetView,
    EfmTrajectoryView,
    SceneMetadata,
)

__all__ = [
    "AseEfmDataset",
    "AseEfmDatasetConfig",
    "EfmCameraView",
    "EfmTrajectoryView",
    "EfmPointsView",
    "EfmObbView",
    "EfmGTView",
    "EfmSnippetView",
    "ASEMetadata",
    "SceneMetadata",
]
