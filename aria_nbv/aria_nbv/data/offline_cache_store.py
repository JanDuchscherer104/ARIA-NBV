"""Metadata and naming helpers for offline cache artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..pipelines.oracle_rri_labeler import OracleRriLabelerConfig
from ..utils import BaseConfig
from ..vin.backbone_evl import EvlBackboneConfig
from .efm_dataset import AseEfmDatasetConfig
from .offline_cache_types import OracleRriCacheMetadata


def snapshot_config(cfg: BaseConfig, *, exclude: set[str] | dict[str, Any] | None = None) -> dict[str, Any]:
    """Snapshot a config using JSON-friendly model dumps."""
    return cfg.model_dump_cache(exclude=set(exclude or set()), exclude_none=True)


def snapshot_dataset_config(cfg: AseEfmDatasetConfig) -> dict[str, Any]:
    """Snapshot AseEfmDatasetConfig excluding large path lists."""
    return snapshot_config(cfg)


def _config_signature(payload: dict[str, Any]) -> str:
    serial = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(serial.encode("utf-8")).hexdigest()


def _compute_config_hash(
    *,
    labeler_signature: str,
    backbone_signature: str | None,
    include_backbone: bool,
    include_depths: bool,
    include_pointclouds: bool,
) -> str:
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
    payload = {
        "version": meta.version,
        "created_at": meta.created_at,
        "labeler_config": meta.labeler_config,
        "labeler_signature": meta.labeler_signature,
        "dataset_config": meta.dataset_config,
        "backbone_config": meta.backbone_config,
        "backbone_signature": meta.backbone_signature,
        "config_hash": meta.config_hash,
        "include_backbone": meta.include_backbone,
        "include_depths": meta.include_depths,
        "include_pointclouds": meta.include_pointclouds,
        "num_samples": meta.num_samples,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_metadata(path: Path) -> OracleRriCacheMetadata:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return OracleRriCacheMetadata(
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


def _sanitize_token(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", value).strip("_")


def _extract_snippet_token(snippet_id: str) -> str:
    match = re.search(r"(?:AtekDataSample|DataSample)_([0-9]+)$", snippet_id)
    if match:
        return match.group(1)
    return snippet_id


def _format_sample_key(scene_id: str, snippet_id: str, config_hash: str) -> str:
    scene_safe = _sanitize_token(scene_id)
    snippet_token = _extract_snippet_token(snippet_id)
    snippet_safe = _sanitize_token(snippet_token)
    key = f"ASE_NBV_SNIPPET_{scene_safe}_{snippet_safe}_{config_hash}"
    if len(key) <= 200:
        return key
    snippet_digest = hashlib.sha1(snippet_id.encode("utf-8")).hexdigest()[:12]
    return f"ASE_NBV_SNIPPET_{scene_safe}_{snippet_digest}_{config_hash}"


def _unique_sample_path(samples_dir: Path, base_key: str) -> tuple[str, Path]:
    path = samples_dir / f"{base_key}.pt"
    if not path.exists():
        return base_key, path
    while True:
        suffix = hashlib.sha1(f"{base_key}-{time.time_ns()}".encode()).hexdigest()[:6]
        key = f"{base_key}__{suffix}"
        path = samples_dir / f"{key}.pt"
        if not path.exists():
            return key, path


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
        created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
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


__all__ = [
    "_compute_config_hash",
    "_config_signature",
    "_ensure_config_hash",
    "_format_sample_key",
    "_read_metadata",
    "_unique_sample_path",
    "_write_metadata",
    "build_cache_metadata",
    "snapshot_config",
    "snapshot_dataset_config",
]
