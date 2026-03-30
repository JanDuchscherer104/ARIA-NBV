"""Shared dataclasses used by the v2 oracle and VIN caches.

This module defines the serializable record types exchanged between the cache
writers, readers, and index helpers in :mod:`aria_nbv.data_handling`.

Contents:
- oracle cache metadata and index/sample records,
- VIN snippet cache metadata and index records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

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
    """Schema version for the oracle cache directory."""

    created_at: str
    """UTC timestamp describing when the cache metadata was written."""

    labeler_config: dict[str, Any]
    """Serialized oracle-labeler configuration snapshot."""

    labeler_signature: str
    """Stable signature used to identify the oracle-labeler implementation."""

    dataset_config: dict[str, Any] | None
    """Serialized raw-dataset configuration used to enumerate snippets."""

    backbone_config: dict[str, Any] | None
    """Serialized EVL backbone configuration snapshot when backbone caching is enabled."""

    backbone_signature: str | None
    """Stable signature used to identify the backbone implementation."""

    config_hash: str | None = None
    """Combined cache-configuration hash for compatibility checks."""

    include_backbone: bool | None = None
    """Whether backbone outputs were written into cached payloads."""

    include_depths: bool | None = None
    """Whether candidate depth maps were written into cached payloads."""

    include_pointclouds: bool | None = None
    """Whether candidate point clouds were written into cached payloads."""

    num_samples: int | None = None
    """Number of cached samples known when the metadata was last rewritten."""

    def to_json_payload(self) -> dict[str, Any]:
        """Serialize oracle-cache metadata into a JSON-ready payload."""

        return {
            "version": self.version,
            "created_at": self.created_at,
            "labeler_config": self.labeler_config,
            "labeler_signature": self.labeler_signature,
            "dataset_config": self.dataset_config,
            "backbone_config": self.backbone_config,
            "backbone_signature": self.backbone_signature,
            "config_hash": self.config_hash,
            "include_backbone": self.include_backbone,
            "include_depths": self.include_depths,
            "include_pointclouds": self.include_pointclouds,
            "num_samples": self.num_samples,
        }

    @classmethod
    def from_json_payload(cls, payload: dict[str, Any]) -> Self:
        """Construct oracle-cache metadata from a decoded JSON payload."""

        return cls(
            version=int(payload["version"]),
            created_at=str(payload["created_at"]),
            labeler_config=payload["labeler_config"],
            labeler_signature=payload["labeler_signature"],
            dataset_config=payload.get("dataset_config"),
            backbone_config=payload.get("backbone_config"),
            backbone_signature=payload.get("backbone_signature"),
            config_hash=payload.get("config_hash"),
            include_backbone=payload.get("include_backbone"),
            include_depths=payload.get("include_depths"),
            include_pointclouds=payload.get("include_pointclouds"),
            num_samples=payload.get("num_samples"),
        )


@dataclass(slots=True)
class OracleRriCacheEntry:
    """Single index entry describing a cached oracle sample."""

    key: str
    """Stable cache key for the sample payload."""

    scene_id: str
    """ASE scene identifier for the cached snippet."""

    snippet_id: str
    """ASE snippet identifier for the cached snippet."""

    path: str
    """Path to the payload file, relative to the cache directory."""


@dataclass(slots=True)
class OracleRriCacheSample:
    """Cached oracle outputs for a single snippet."""

    key: str
    """Stable cache key for the sample payload."""

    scene_id: str
    """ASE scene identifier for the cached snippet."""

    snippet_id: str
    """ASE snippet identifier for the cached snippet."""

    candidates: CandidateSamplingResult
    """Decoded candidate-pose sampling results for this snippet."""

    depths: CandidateDepths
    """Decoded candidate depth maps and camera metadata."""

    candidate_pcs: CandidatePointClouds
    """Decoded candidate point clouds derived from cached depths."""

    rri: RriResult
    """Decoded oracle RRI targets and diagnostics."""

    backbone_out: EvlBackboneOutput | None
    """Optional decoded EVL backbone outputs for this snippet."""

    efm_snippet_view: EfmSnippetView | None = None
    """Optional raw EFM snippet view loaded on demand by the reader."""


@dataclass(slots=True)
class VinSnippetCacheMetadata:
    """Top-level metadata for a VIN snippet cache directory."""

    version: int
    """Schema version for the VIN snippet cache directory."""

    created_at: str
    """UTC timestamp describing when the VIN metadata was written."""

    source_cache_dir: str | None
    """Source oracle-cache directory used to enumerate VIN snippets."""

    source_cache_hash: str | None
    """Config hash of the source oracle cache when it was built."""

    dataset_config: dict[str, Any] | None
    """Serialized raw-dataset configuration used for live snippet loading."""

    include_inv_dist_std: bool
    """Whether VIN points include inverse distance standard deviation."""

    include_obs_count: bool
    """Whether VIN points include per-point observation counts."""

    semidense_max_points: int | None
    """Optional collapse-time cap applied before padding cached VIN points."""

    pad_points: int | None
    """Fixed padded point count stored on disk for each cached snippet."""

    config_hash: str | None = None
    """Combined VIN-cache configuration hash for compatibility checks."""

    num_samples: int | None = None
    """Number of cached VIN snippets known when metadata was last rewritten."""

    def to_json_payload(self) -> dict[str, Any]:
        """Serialize VIN-cache metadata into a JSON-ready payload."""

        return {
            "version": self.version,
            "created_at": self.created_at,
            "source_cache_dir": self.source_cache_dir,
            "source_cache_hash": self.source_cache_hash,
            "dataset_config": self.dataset_config,
            "include_inv_dist_std": self.include_inv_dist_std,
            "include_obs_count": self.include_obs_count,
            "semidense_max_points": self.semidense_max_points,
            "pad_points": self.pad_points,
            "config_hash": self.config_hash,
            "num_samples": self.num_samples,
        }

    @classmethod
    def from_json_payload(cls, payload: dict[str, Any]) -> Self:
        """Construct VIN-cache metadata from a decoded JSON payload."""

        return cls(
            version=int(payload["version"]),
            created_at=str(payload["created_at"]),
            source_cache_dir=payload.get("source_cache_dir"),
            source_cache_hash=payload.get("source_cache_hash"),
            dataset_config=payload.get("dataset_config"),
            include_inv_dist_std=bool(payload.get("include_inv_dist_std", True)),
            include_obs_count=bool(payload.get("include_obs_count", False)),
            semidense_max_points=payload.get("semidense_max_points"),
            pad_points=payload.get("pad_points"),
            config_hash=payload.get("config_hash"),
            num_samples=payload.get("num_samples"),
        )


@dataclass(slots=True)
class VinSnippetCacheEntry:
    """Index entry describing a cached VIN snippet."""

    key: str
    """Stable cache key for the VIN snippet payload."""

    scene_id: str
    """ASE scene identifier for the cached VIN snippet."""

    snippet_id: str
    """ASE snippet identifier for the cached VIN snippet."""

    path: str
    """Path to the payload file, relative to the cache directory."""


__all__ = [
    "OracleRriCacheEntry",
    "OracleRriCacheMetadata",
    "OracleRriCacheSample",
    "VinSnippetCacheEntry",
    "VinSnippetCacheMetadata",
]
