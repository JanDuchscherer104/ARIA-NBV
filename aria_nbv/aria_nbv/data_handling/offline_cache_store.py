"""Metadata and naming helpers for offline cache artifacts.

This module owns oracle-cache config snapshots plus the compatibility
read/write helpers for oracle-cache metadata files. Low-level shared cache
helpers live in :mod:`aria_nbv.data_handling._cache_utils`, while the metadata
dataclass itself owns JSON encode/decode in :mod:`aria_nbv.data_handling.cache_contracts`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..pipelines.oracle_rri_labeler import OracleRriLabelerConfig
from ..utils import BaseConfig
from ..vin.backbone_evl import EvlBackboneConfig
from ._cache_utils import (
    extract_snippet_token,
    format_sample_key,
    json_signature,
    sanitize_token,
    unique_sample_path,
    utc_now_iso,
)
from .cache_contracts import OracleRriCacheMetadata
from .cache_index import load_json_metadata, write_json_metadata
from .efm_dataset import AseEfmDatasetConfig


def snapshot_config(cfg: BaseConfig, *, exclude: set[str] | dict[str, Any] | None = None) -> dict[str, Any]:
    """Snapshot a config using JSON-friendly model dumps."""
    return cfg.model_dump_cache(exclude=set(exclude or set()), exclude_none=True)


def snapshot_dataset_config(cfg: AseEfmDatasetConfig) -> dict[str, Any]:
    """Snapshot AseEfmDatasetConfig excluding large path lists."""
    return snapshot_config(cfg)


def _config_signature(payload: dict[str, Any]) -> str:
    """Compute a stable hash for a JSON-serializable config payload."""
    return json_signature(payload)


def _compute_config_hash(
    *,
    labeler_signature: str,
    backbone_signature: str | None,
    include_backbone: bool,
    include_depths: bool,
    include_pointclouds: bool,
) -> str:
    """Compute the compatibility hash for an oracle-cache directory."""
    payload = {
        "labeler": labeler_signature,
        "backbone": backbone_signature,
        "include_backbone": include_backbone,
        "include_depths": include_depths,
        "include_pointclouds": include_pointclouds,
    }
    return _config_signature(payload)


def _ensure_config_hash(
    meta: OracleRriCacheMetadata,
    *,
    include_backbone: bool | None = None,
    include_depths: bool | None = None,
    include_pointclouds: bool | None = None,
) -> str:
    """Return the metadata config hash, computing it when missing."""
    if meta.config_hash:
        return meta.config_hash
    use_backbone = meta.include_backbone if meta.include_backbone is not None else include_backbone
    use_depths = meta.include_depths if meta.include_depths is not None else include_depths
    use_pointclouds = meta.include_pointclouds if meta.include_pointclouds is not None else include_pointclouds
    config_hash = _compute_config_hash(
        labeler_signature=meta.labeler_signature,
        backbone_signature=meta.backbone_signature,
        include_backbone=bool(use_backbone),
        include_depths=bool(use_depths),
        include_pointclouds=bool(use_pointclouds),
    )
    meta.config_hash = config_hash
    return config_hash


def _write_metadata(path: Path, meta: OracleRriCacheMetadata) -> None:
    """Write oracle-cache metadata to disk as JSON."""
    write_json_metadata(path, meta.to_json_payload())


def _read_metadata(path: Path) -> OracleRriCacheMetadata:
    """Read oracle-cache metadata from disk."""
    return OracleRriCacheMetadata.from_json_payload(load_json_metadata(path))


def build_cache_metadata(
    *,
    labeler: OracleRriLabelerConfig,
    dataset: AseEfmDatasetConfig | None,
    backbone: EvlBackboneConfig | None,
    include_backbone: bool,
    include_depths: bool,
    include_pointclouds: bool,
    version: int,
) -> OracleRriCacheMetadata:
    """Construct cache metadata from config snapshots."""
    labeler_snapshot = snapshot_config(labeler)
    backbone_snapshot = snapshot_config(backbone) if backbone else None
    dataset_snapshot = snapshot_dataset_config(dataset) if dataset is not None else None
    meta = OracleRriCacheMetadata(
        version=version,
        created_at=utc_now_iso(),
        labeler_config=labeler_snapshot,
        labeler_signature=_config_signature(labeler_snapshot),
        dataset_config=dataset_snapshot,
        backbone_config=backbone_snapshot,
        backbone_signature=_config_signature(backbone_snapshot) if backbone_snapshot else None,
        include_backbone=include_backbone,
        include_depths=include_depths,
        include_pointclouds=include_pointclouds,
        num_samples=None,
    )
    meta.config_hash = _ensure_config_hash(
        meta,
        include_backbone=include_backbone,
        include_depths=include_depths,
        include_pointclouds=include_pointclouds,
    )
    return meta


compute_config_hash = _compute_config_hash
config_signature = _config_signature
ensure_config_hash = _ensure_config_hash
read_metadata = _read_metadata
write_metadata = _write_metadata

__all__ = [
    "build_cache_metadata",
    "compute_config_hash",
    "config_signature",
    "ensure_config_hash",
    "extract_snippet_token",
    "format_sample_key",
    "read_metadata",
    "sanitize_token",
    "snapshot_config",
    "snapshot_dataset_config",
    "unique_sample_path",
    "write_metadata",
]
