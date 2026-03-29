"""Dataclasses for offline cache records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..pose_generation.types import CandidateSamplingResult
from ..rendering.candidate_depth_renderer import CandidateDepths
from ..rendering.candidate_pointclouds import CandidatePointClouds
from ..rri_metrics.types import RriResult
from ..vin.types import EvlBackboneOutput
from .efm_views import EfmSnippetView


@dataclass(slots=True)
class OracleRriCacheMetadata:
    """Top-level metadata for an oracle cache directory."""

    version: int
    created_at: str
    labeler_config: dict[str, Any]
    labeler_signature: str
    dataset_config: dict[str, Any] | None
    backbone_config: dict[str, Any] | None
    backbone_signature: str | None
    config_hash: str | None = None
    include_backbone: bool | None = None
    include_depths: bool | None = None
    include_pointclouds: bool | None = None
    num_samples: int | None = None


@dataclass(slots=True)
class OracleRriCacheEntry:
    """Single index entry describing a cached sample."""

    key: str
    scene_id: str
    snippet_id: str
    path: str


@dataclass(slots=True)
class OracleRriCacheSample:
    """Cached oracle outputs for a single snippet (no raw EFM data)."""

    key: str
    """Unique cache key derived from scene, snippet, and config hash."""

    scene_id: str
    """ASE scene identifier."""

    snippet_id: str
    """ASE snippet identifier."""

    candidates: CandidateSamplingResult
    """Candidate sampling results for this snippet."""

    depths: CandidateDepths
    """Rendered candidate depth maps and camera metadata."""

    candidate_pcs: CandidatePointClouds
    """Backprojected candidate point clouds (fused with semi-dense SLAM)."""

    rri: RriResult
    """Oracle RRI scores and diagnostics for the candidate set."""

    backbone_out: EvlBackboneOutput | None
    """Cached EVL backbone outputs for this snippet (if enabled)."""

    efm_snippet_view: EfmSnippetView | None = None
    """Optional raw EFM snippet view loaded on demand (if enabled)."""


__all__ = [
    "OracleRriCacheEntry",
    "OracleRriCacheMetadata",
    "OracleRriCacheSample",
]
