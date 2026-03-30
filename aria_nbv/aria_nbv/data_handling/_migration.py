"""Legacy scan, conversion, and verification helpers for VIN offline migration.

NBV_LEGACY_OFFLINE_CACHE_REMOVE_AFTER_FULL_MIGRATION:
This module exists only to migrate and verify the legacy oracle/VIN cache
layout. Remove it after the full offline-store cutover is complete.

This module provides the reusable migration logic behind the temporary tools in
``.agents/workspace/data_handling_migration``. It understands the legacy
oracle-cache and VIN-snippet-cache layouts, converts them into prepared shard
rows for the new immutable VIN offline format, and verifies the migrated
dataset against the legacy sources.
"""

from __future__ import annotations

import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from efm3d.aria.pose import PoseTW

from ..configs import PathConfig
from ..pose_generation.types import CandidateSamplingResult
from ..rendering.candidate_depth_renderer import CandidateDepths
from ..rendering.candidate_pointclouds import CandidatePointClouds
from ..rri_metrics.types import RriResult
from ..utils import Console
from ..vin.types import EvlBackboneOutput
from ._legacy_offline_cache_store import _read_metadata as read_legacy_oracle_metadata
from ._legacy_oracle_cache import OracleRriCacheConfig
from ._legacy_vin_cache import VinSnippetCacheConfig, read_vin_snippet_cache_metadata
from ._offline_format import (
    VinOfflineIndexRecord,
    VinOfflineManifest,
    VinOfflineMaterializedBlocks,
)
from ._offline_store import (
    OFFLINE_DATASET_VERSION,
    VinOfflineStoreConfig,
    VinOfflineStoreReader,
)
from ._offline_writer import (
    PreparedVinOfflineSample,
    flush_prepared_samples_to_shard,
    prepare_vin_offline_sample,
)
from ._raw import EfmSnippetLoader, VinSnippetView
from ._vin_runtime import build_vin_snippet_view
from .cache_contracts import OracleRriCacheEntry, VinSnippetCacheEntry, VinSnippetCacheMetadata
from .cache_index import read_index, repair_oracle_split_indices, validate_oracle_split_indices


def _extract_snippet_token(snippet_id: str) -> str:
    """Extract the stable numeric snippet token when present."""

    match = re.search(r"(?:AtekDataSample|DataSample)_([0-9]+)$", snippet_id)
    return match.group(1) if match else snippet_id


@dataclass(slots=True)
class LegacyOfflineRecord:
    """One legacy oracle-cache sample paired with optional VIN-cache metadata."""

    entry: OracleRriCacheEntry
    """Legacy oracle-cache index entry."""

    split: str
    """Legacy split membership, either ``train`` or ``val``."""

    vin_entry: VinSnippetCacheEntry | None = None
    """Optional matching legacy VIN-cache entry."""


@dataclass(slots=True)
class LegacyOfflinePlan:
    """Legacy dataset scan result used for migration and verification."""

    oracle_cache_dir: Path
    """Legacy oracle-cache directory."""

    vin_cache_dir: Path | None
    """Optional legacy VIN-cache directory."""

    dataset_payload: dict[str, Any] | None
    """Raw ASE/EFM dataset config snapshot used for live VIN fallback."""

    records: list[LegacyOfflineRecord]
    """Ordered legacy records to migrate."""

    train_count: int
    """Number of train samples in the legacy cache."""

    val_count: int
    """Number of validation samples in the legacy cache."""

    missing_vin_pairs: list[tuple[str, str]]
    """Pairs missing from the legacy VIN cache and requiring live rebuild."""

    oracle_metadata: dict[str, Any]
    """Legacy oracle metadata snapshot."""

    vin_metadata: VinSnippetCacheMetadata | None
    """Optional legacy VIN metadata snapshot."""


def _normalize_legacy_selection(
    *,
    scene_ids: list[str] | tuple[str, ...] | None,
    split: str,
    max_records: int | None,
) -> tuple[set[str] | None, str, int | None]:
    """Normalize optional legacy-record selection filters."""

    normalized_split = str(split).strip().lower()
    if normalized_split not in {"all", "train", "val"}:
        raise ValueError("split must be one of: 'all', 'train', 'val'.")
    normalized_limit = None if max_records is None else int(max_records)
    if normalized_limit is not None and normalized_limit < 1:
        raise ValueError("max_records must be >= 1 when provided.")
    if not scene_ids:
        return None, normalized_split, normalized_limit
    normalized_scene_ids = {str(scene_id).strip() for scene_id in scene_ids if str(scene_id).strip()}
    return normalized_scene_ids or None, normalized_split, normalized_limit


def _select_legacy_entries(
    *,
    entries: list[OracleRriCacheEntry],
    split_by_key: dict[str, str],
    scene_ids: set[str] | None,
    split: str,
    max_records: int | None,
) -> tuple[list[OracleRriCacheEntry], int, int]:
    """Filter legacy oracle entries down to the requested migration subset."""

    selected: list[OracleRriCacheEntry] = []
    train_count = 0
    val_count = 0
    for entry in entries:
        entry_split = split_by_key.get(entry.key, "train")
        if split != "all" and entry_split != split:
            continue
        if scene_ids is not None and entry.scene_id not in scene_ids:
            continue
        selected.append(entry)
        if entry_split == "val":
            val_count += 1
        else:
            train_count += 1
        if max_records is not None and len(selected) >= max_records:
            break
    return selected, train_count, val_count


def scan_legacy_offline_data(
    *,
    oracle_cache: OracleRriCacheConfig,
    vin_cache: VinSnippetCacheConfig | None = None,
    scene_ids: list[str] | tuple[str, ...] | None = None,
    split: str = "all",
    max_records: int | None = None,
    train_val_split: float = 0.2,
    repair_splits: bool = False,
    console: Console | None = None,
) -> LegacyOfflinePlan:
    """Scan legacy oracle and VIN caches and build a migration plan.

    Args:
        oracle_cache: Legacy oracle-cache configuration.
        vin_cache: Optional legacy VIN-cache configuration.
        scene_ids: Optional scene-id allowlist used to select a migration subset.
        split: Optional legacy split selector: ``all``, ``train``, or ``val``.
        max_records: Optional cap applied after split and scene filtering.
        train_val_split: Validation fraction used only when legacy split files
            need explicit repair.
        repair_splits: Whether to repair missing or stale oracle split files.
        console: Optional logger.

    Returns:
        Normalized migration plan for the legacy caches.
    """

    console = console or Console.with_prefix("LegacyOfflineScan")
    selected_scene_ids, selected_split, selected_max_records = _normalize_legacy_selection(
        scene_ids=scene_ids,
        split=split,
        max_records=max_records,
    )
    base_entries = read_index(oracle_cache.index_path, entry_type=OracleRriCacheEntry)
    try:
        train_entries, val_entries = validate_oracle_split_indices(
            base_entries=base_entries,
            train_index_path=oracle_cache.train_index_path,
            val_index_path=oracle_cache.val_index_path,
        )
    except Exception:
        if not repair_splits:
            raise
        train_entries, val_entries = repair_oracle_split_indices(
            base_entries=base_entries,
            train_index_path=oracle_cache.train_index_path,
            val_index_path=oracle_cache.val_index_path,
            val_fraction=train_val_split,
            console=console,
        )

    split_by_key = {entry.key: "train" for entry in train_entries}
    split_by_key.update({entry.key: "val" for entry in val_entries})
    selected_entries, train_count, val_count = _select_legacy_entries(
        entries=base_entries,
        split_by_key=split_by_key,
        scene_ids=selected_scene_ids,
        split=selected_split,
        max_records=selected_max_records,
    )

    oracle_meta = read_legacy_oracle_metadata(oracle_cache.metadata_path)
    dataset_payload = dict(oracle_meta.dataset_config or {}) if oracle_meta.dataset_config is not None else None

    vin_entries_by_pair: dict[tuple[str, str], VinSnippetCacheEntry] = {}
    vin_meta: VinSnippetCacheMetadata | None = None
    if vin_cache is not None and vin_cache.index_path.exists():
        vin_entries = read_index(vin_cache.index_path, entry_type=VinSnippetCacheEntry, allow_missing=True)
        vin_entries_by_pair = {(entry.scene_id, entry.snippet_id): entry for entry in vin_entries}
        if vin_cache.metadata_path.exists():
            vin_meta = read_vin_snippet_cache_metadata(vin_cache.cache_dir)

    records: list[LegacyOfflineRecord] = []
    missing_vin_pairs: list[tuple[str, str]] = []
    for entry in selected_entries:
        pair = (entry.scene_id, entry.snippet_id)
        vin_entry = vin_entries_by_pair.get(pair)
        if vin_entry is None:
            vin_entry = vin_entries_by_pair.get((entry.scene_id, _extract_snippet_token(entry.snippet_id)))
        if vin_cache is not None and vin_entry is None:
            missing_vin_pairs.append(pair)
        records.append(
            LegacyOfflineRecord(
                entry=entry,
                split=split_by_key.get(entry.key, "train"),
                vin_entry=vin_entry,
            ),
        )

    return LegacyOfflinePlan(
        oracle_cache_dir=oracle_cache.cache_dir,
        vin_cache_dir=vin_cache.cache_dir if vin_cache is not None else None,
        dataset_payload=dataset_payload,
        records=records,
        train_count=train_count,
        val_count=val_count,
        missing_vin_pairs=missing_vin_pairs,
        oracle_metadata=asdict(oracle_meta),
        vin_metadata=vin_meta,
    )


def _collect_vin_cache_setting_mismatches(
    *,
    metadata: VinSnippetCacheMetadata,
    semidense_max_points: int | None,
    semidense_include_obs_count: bool,
    pad_points: int,
) -> list[str]:
    """Describe legacy VIN-cache settings that disagree with migration inputs."""

    mismatches: list[str] = []
    if metadata.pad_points != int(pad_points):
        mismatches.append(f"pad_points cache={metadata.pad_points} requested={int(pad_points)}")
    if metadata.semidense_max_points != semidense_max_points:
        mismatches.append(
            f"semidense_max_points cache={metadata.semidense_max_points} requested={semidense_max_points}",
        )
    if bool(metadata.include_obs_count) != bool(semidense_include_obs_count):
        mismatches.append(
            f"include_obs_count cache={bool(metadata.include_obs_count)} requested={bool(semidense_include_obs_count)}",
        )
    return mismatches


def _validate_legacy_vin_cache_reuse(
    *,
    plan: LegacyOfflinePlan,
    semidense_max_points: int | None,
    semidense_include_obs_count: bool,
    pad_points: int,
) -> None:
    """Fail fast when legacy VIN-cache reuse would produce inconsistent samples."""

    reused_rows = sum(1 for record in plan.records if record.vin_entry is not None)
    if reused_rows == 0:
        return

    partial_note = ""
    if plan.missing_vin_pairs:
        partial_note = (
            " The cache only partially covers the oracle cache, so cached and live-rebuilt VIN rows "
            "would otherwise be mixed."
        )

    if plan.vin_metadata is None:
        raise ValueError(
            "Legacy VIN cache reuse requires metadata to validate pad_points, semidense_max_points, "
            f"and include_obs_count.{partial_note} Omit vin_cache to rebuild all VIN snippets live, "
            "or restore the VIN cache metadata and retry."
        )

    mismatches = _collect_vin_cache_setting_mismatches(
        metadata=plan.vin_metadata,
        semidense_max_points=semidense_max_points,
        semidense_include_obs_count=semidense_include_obs_count,
        pad_points=pad_points,
    )
    if mismatches:
        raise ValueError(
            "Legacy VIN cache settings are incompatible with requested migration settings "
            f"({'; '.join(mismatches)}).{partial_note} Rerun migration with matching "
            "pad_points/semidense_max_points/include_obs_count, or omit vin_cache to rebuild all VIN snippets live."
        )


def _load_legacy_vin_snippet(
    *,
    record: LegacyOfflineRecord,
    vin_cache_dir: Path | None,
    dataset_payload: dict[str, Any] | None,
    loader: EfmSnippetLoader | None,
    semidense_max_points: int | None,
    semidense_include_obs_count: bool,
    pad_points: int,
) -> tuple[VinSnippetView, str | None, str | None, EfmSnippetLoader | None]:
    """Load or rebuild one VIN snippet for migration."""

    if record.vin_entry is not None and vin_cache_dir is not None:
        payload = torch.load(vin_cache_dir / record.vin_entry.path, map_location="cpu", weights_only=False)
        points_world = payload["points_world"].to(dtype=torch.float32)
        points_length = payload.get("points_length")
        if points_length is None:
            points_length = torch.tensor(
                [int(torch.isfinite(points_world[:, :3]).all(dim=-1).sum().item())],
                dtype=torch.int64,
            )
        t_world_rig = payload["t_world_rig"].to(dtype=torch.float32)
        snippet = VinSnippetView(
            points_world=points_world,
            lengths=points_length.reshape(-1).to(dtype=torch.int64),
            t_world_rig=PoseTW(t_world_rig),
        )
        return snippet, record.vin_entry.key, record.vin_entry.path, loader

    if dataset_payload is None:
        raise ValueError(
            "Legacy VIN cache missing and oracle metadata has no dataset_config for live rebuild.",
        )
    if loader is None:
        loader = EfmSnippetLoader(
            dataset_payload=dataset_payload,
            device="cpu",
            paths=PathConfig(),
            include_gt_mesh=False,
        )
    efm_snippet = loader.load(scene_id=record.entry.scene_id, snippet_id=record.entry.snippet_id)
    snippet = build_vin_snippet_view(
        efm_snippet,
        device=torch.device("cpu"),
        max_points=semidense_max_points,
        include_inv_dist_std=True,
        include_obs_count=semidense_include_obs_count,
        pad_points=pad_points,
    )
    return snippet, None, None, loader


def prepare_legacy_records(
    *,
    records: list[LegacyOfflineRecord],
    oracle_cache_dir: Path,
    vin_cache_dir: Path | None,
    dataset_payload: dict[str, Any] | None,
    max_candidates: int,
    include_backbone: bool,
    include_depths: bool,
    include_pointclouds: bool,
    semidense_max_points: int | None,
    semidense_include_obs_count: bool,
    pad_points: int,
) -> list[PreparedVinOfflineSample]:
    """Prepare legacy oracle/VIN entries for immutable shard materialization.

    Args:
        records: Ordered legacy records for one shard.
        oracle_cache_dir: Legacy oracle-cache directory.
        vin_cache_dir: Optional legacy VIN-cache directory.
        dataset_payload: Optional raw-dataset payload for live VIN rebuild.
        max_candidates: Candidate budget stored in fixed blocks.
        include_backbone: Whether to materialize backbone outputs.
        include_depths: Whether to materialize depth blocks.
        include_pointclouds: Whether to materialize candidate point clouds.
        semidense_max_points: Optional VIN collapse cap.
        semidense_include_obs_count: Whether VIN points include observation counts.
        pad_points: Stored VIN padding budget.

    Returns:
        Prepared offline rows ready for shard flushing.
    """

    prepared: list[PreparedVinOfflineSample] = []
    loader: EfmSnippetLoader | None = None
    for record in records:
        payload = torch.load(oracle_cache_dir / record.entry.path, map_location="cpu", weights_only=False)
        candidates = CandidateSamplingResult.from_serializable(payload["candidates"], device=None)
        depths = CandidateDepths.from_serializable(payload["depths"], device=torch.device("cpu"))
        rri = RriResult.from_serializable(payload["rri"], device=torch.device("cpu"))
        candidate_pcs = None
        if include_pointclouds and payload.get("candidate_pcs") is not None:
            candidate_pcs = CandidatePointClouds.from_serializable(
                payload["candidate_pcs"],
                device=torch.device("cpu"),
            )
        backbone_out = None
        if include_backbone and payload.get("backbone") is not None:
            backbone_out = EvlBackboneOutput.from_serializable(payload["backbone"], device=torch.device("cpu"))

        vin_snippet, legacy_vin_key, legacy_vin_path, loader = _load_legacy_vin_snippet(
            record=record,
            vin_cache_dir=vin_cache_dir,
            dataset_payload=dataset_payload,
            loader=loader,
            semidense_max_points=semidense_max_points,
            semidense_include_obs_count=semidense_include_obs_count,
            pad_points=pad_points,
        )
        prepared.append(
            prepare_vin_offline_sample(
                scene_id=record.entry.scene_id,
                snippet_id=record.entry.snippet_id,
                vin_snippet=vin_snippet,
                candidates=candidates,
                depths=depths,
                rri=rri,
                candidate_pcs=candidate_pcs,
                backbone_out=backbone_out,
                max_candidates=max_candidates,
                include_depths=include_depths,
                include_candidate_pcs=include_pointclouds,
                include_backbone=include_backbone,
                sample_key=record.entry.key,
                legacy_oracle_key=record.entry.key,
                legacy_oracle_path=record.entry.path,
                legacy_vin_key=legacy_vin_key,
                legacy_vin_path=legacy_vin_path,
            ),
        )
    return prepared


def _convert_legacy_shard(job: dict[str, Any]) -> tuple[Any, list[VinOfflineIndexRecord]]:
    """Convert one legacy-record slice into one immutable shard.

    Args:
        job: Pickle-friendly shard job payload.

    Returns:
        The written shard spec together with its shard-local sample-index rows.
    """

    rows = prepare_legacy_records(
        records=job["records"],
        oracle_cache_dir=Path(job["oracle_cache_dir"]),
        vin_cache_dir=Path(job["vin_cache_dir"]) if job["vin_cache_dir"] is not None else None,
        dataset_payload=job["dataset_payload"],
        max_candidates=int(job["max_candidates"]),
        include_backbone=bool(job["include_backbone"]),
        include_depths=bool(job["include_depths"]),
        include_pointclouds=bool(job["include_pointclouds"]),
        semidense_max_points=job["semidense_max_points"],
        semidense_include_obs_count=bool(job["semidense_include_obs_count"]),
        pad_points=int(job["pad_points"]),
    )
    return flush_prepared_samples_to_shard(
        shard_index=int(job["shard_index"]),
        shard_dir=Path(job["shard_dir"]),
        rows=rows,
    )


def finalize_migrated_store(
    *,
    store: VinOfflineStoreConfig,
    plan: LegacyOfflinePlan,
    shard_specs: list[Any],
    index_records: list[VinOfflineIndexRecord],
    max_candidates: int,
    include_backbone: bool,
    include_depths: bool,
    include_pointclouds: bool,
    semidense_max_points: int | None,
    semidense_include_obs_count: bool,
    pad_points: int,
) -> VinOfflineManifest:
    """Write the manifest, index, and split files for a migrated store.

    Args:
        store: Destination store configuration.
        plan: Scanned legacy plan.
        shard_specs: Materialized shard specs.
        index_records: Global sample-index records.
        max_candidates: Stored candidate budget.
        include_backbone: Whether backbone outputs are materialized.
        include_depths: Whether depth payloads are materialized.
        include_pointclouds: Whether candidate point clouds are materialized.
        semidense_max_points: Optional VIN collapse cap.
        semidense_include_obs_count: Whether VIN points include observation counts.
        pad_points: Stored VIN padding budget.

    Returns:
        Written migrated manifest.
    """

    split_by_legacy_key = {legacy_record.entry.key: legacy_record.split for legacy_record in plan.records}
    row_start = 0
    for shard_spec in shard_specs:
        shard_spec.row_start = row_start
        row_start += int(shard_spec.num_rows)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for sample_index, record in enumerate(index_records):
        record.sample_index = sample_index
        if record.legacy_oracle_key is None:
            record.split = "train"
            train_indices.append(sample_index)
            continue
        split = split_by_legacy_key[record.legacy_oracle_key]
        record.split = split
        if split == "val":
            val_indices.append(sample_index)
        else:
            train_indices.append(sample_index)

    manifest = VinOfflineManifest(
        version=OFFLINE_DATASET_VERSION,
        created_at=plan.oracle_metadata.get("created_at") or "",
        source={
            "dataset_config": plan.dataset_payload,
        },
        oracle={
            "labeler_config": plan.oracle_metadata.get("labeler_config"),
            "labeler_signature": plan.oracle_metadata.get("labeler_signature"),
            "backbone_config": plan.oracle_metadata.get("backbone_config"),
            "backbone_signature": plan.oracle_metadata.get("backbone_signature"),
            "max_candidates": int(max_candidates),
        },
        vin={
            "pad_points": int(pad_points),
            "semidense_max_points": semidense_max_points,
            "include_inv_dist_std": True,
            "include_obs_count": bool(semidense_include_obs_count),
        },
        materialized_blocks=VinOfflineMaterializedBlocks(
            backbone=bool(include_backbone),
            depths=bool(include_depths),
            candidate_pcs=bool(include_pointclouds),
            counterfactuals=False,
        ),
        stats={
            "num_samples": len(index_records),
            "num_shards": len(shard_specs),
            "num_train": len(train_indices),
            "num_val": len(val_indices),
        },
        provenance={
            "migration": {
                "oracle_cache_dir": plan.oracle_cache_dir.as_posix(),
                "vin_cache_dir": plan.vin_cache_dir.as_posix() if plan.vin_cache_dir is not None else None,
            }
        },
        shards=shard_specs,
    )
    manifest.write(store.manifest_path)
    VinOfflineIndexRecord.write_many(store.sample_index_path, index_records)
    store.write_split_indices(
        {
            "all": torch.arange(len(index_records), dtype=torch.long).numpy(),
            "train": torch.tensor(train_indices, dtype=torch.long).numpy(),
            "val": torch.tensor(val_indices, dtype=torch.long).numpy(),
        },
    )
    return manifest


def migrate_legacy_offline_data(
    *,
    oracle_cache: OracleRriCacheConfig,
    store: VinOfflineStoreConfig,
    vin_cache: VinSnippetCacheConfig | None = None,
    scene_ids: list[str] | tuple[str, ...] | None = None,
    split: str = "all",
    max_records: int | None = None,
    workers: int = 0,
    samples_per_shard: int = 64,
    max_candidates: int = 60,
    include_backbone: bool = True,
    include_depths: bool = True,
    include_pointclouds: bool = True,
    semidense_max_points: int | None = None,
    semidense_include_obs_count: bool = False,
    pad_points: int = 50000,
    overwrite: bool = False,
    repair_splits: bool = False,
    train_val_split: float = 0.2,
    console: Console | None = None,
) -> VinOfflineManifest:
    """Convert legacy oracle/VIN caches into one immutable VIN offline store.

    Args:
        oracle_cache: Source legacy oracle-cache configuration.
        store: Destination immutable-store configuration.
        vin_cache: Optional source legacy VIN-cache configuration.
        scene_ids: Optional scene-id allowlist used to select a migration subset.
        split: Optional legacy split selector: ``all``, ``train``, or ``val``.
        max_records: Optional cap applied after split and scene filtering.
        workers: Number of shard worker processes. Values below ``2`` keep the
            conversion in-process.
        samples_per_shard: Maximum number of samples flushed into one shard.
        max_candidates: Candidate budget stored in fixed blocks.
        include_backbone: Whether backbone outputs are materialized.
        include_depths: Whether depth payloads are materialized.
        include_pointclouds: Whether candidate point clouds are materialized.
        semidense_max_points: Optional VIN collapse-time point cap.
        semidense_include_obs_count: Whether rebuilt VIN points include
            observation counts.
        pad_points: Stored VIN padding budget.
        overwrite: Whether an existing destination store may be replaced.
        repair_splits: Whether to repair missing or stale legacy split files
            before conversion.
        train_val_split: Validation fraction used only when split repair is
            required.
        console: Optional logger.

    Returns:
        The written immutable-store manifest.
    """

    if int(samples_per_shard) <= 0:
        raise ValueError("samples_per_shard must be >= 1.")
    if int(workers) < 0:
        raise ValueError("workers must be >= 0.")

    console = console or Console.with_prefix("LegacyOfflineMigration")
    plan = scan_legacy_offline_data(
        oracle_cache=oracle_cache,
        vin_cache=vin_cache,
        scene_ids=scene_ids,
        split=split,
        max_records=max_records,
        train_val_split=train_val_split,
        repair_splits=repair_splits,
        console=console,
    )
    _validate_legacy_vin_cache_reuse(
        plan=plan,
        semidense_max_points=semidense_max_points,
        semidense_include_obs_count=bool(semidense_include_obs_count),
        pad_points=int(pad_points),
    )

    out_store = store.store_dir.expanduser().resolve()
    temp_store = out_store.with_name(f"{out_store.name}.tmp")
    if temp_store.exists():
        shutil.rmtree(temp_store)
    if out_store.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination store already exists at {out_store}. Set overwrite=True to replace it.",
            )
    temp_store.mkdir(parents=True, exist_ok=True)
    (temp_store / store.shards_dirname).mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    shard_size = int(samples_per_shard)
    for offset in range(0, len(plan.records), shard_size):
        shard_records = plan.records[offset : offset + shard_size]
        shard_index = offset // shard_size
        jobs.append(
            {
                "shard_index": shard_index,
                "records": shard_records,
                "oracle_cache_dir": plan.oracle_cache_dir.as_posix(),
                "vin_cache_dir": plan.vin_cache_dir.as_posix() if plan.vin_cache_dir is not None else None,
                "dataset_payload": plan.dataset_payload,
                "max_candidates": int(max_candidates),
                "include_backbone": bool(include_backbone),
                "include_depths": bool(include_depths),
                "include_pointclouds": bool(include_pointclouds),
                "semidense_max_points": semidense_max_points,
                "semidense_include_obs_count": bool(semidense_include_obs_count),
                "pad_points": int(pad_points),
                "shard_dir": (temp_store / store.shards_dirname / f"shard-{shard_index:06d}").as_posix(),
            },
        )

    results: list[tuple[Any, list[VinOfflineIndexRecord]]]
    if int(workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            results = list(executor.map(_convert_legacy_shard, jobs))
    else:
        results = [_convert_legacy_shard(job) for job in jobs]

    shard_specs = [result[0] for result in results]
    index_records = [record for _, records in results for record in records]
    manifest = finalize_migrated_store(
        store=store.model_copy(update={"store_dir": temp_store}),
        plan=plan,
        shard_specs=shard_specs,
        index_records=index_records,
        max_candidates=int(max_candidates),
        include_backbone=bool(include_backbone),
        include_depths=bool(include_depths),
        include_pointclouds=bool(include_pointclouds),
        semidense_max_points=semidense_max_points,
        semidense_include_obs_count=bool(semidense_include_obs_count),
        pad_points=int(pad_points),
    )
    backup_store: Path | None = None
    if out_store.exists():
        backup_store = out_store.with_name(f"{out_store.name}.bak")
        if backup_store.exists():
            shutil.rmtree(backup_store)
        out_store.rename(backup_store)
    try:
        temp_store.rename(out_store)
    except Exception:
        if backup_store is not None and backup_store.exists() and not out_store.exists():
            backup_store.rename(out_store)
        raise
    if backup_store is not None and backup_store.exists():
        shutil.rmtree(backup_store)
    console.log(f"Wrote migrated VIN offline dataset to {out_store}")
    return manifest


def verify_migrated_offline_data(
    *,
    store: VinOfflineStoreConfig,
    plan: LegacyOfflinePlan,
) -> dict[str, Any]:
    """Verify a migrated VIN offline dataset against its legacy sources.

    Args:
        store: Migrated store configuration.
        plan: Legacy plan used for migration.

    Returns:
        Verification summary dictionary.
    """

    manifest = VinOfflineManifest.read(store.manifest_path)
    records = VinOfflineIndexRecord.read_many(store.sample_index_path)
    store_reader = VinOfflineStoreReader(store)
    legacy_pairs = {(record.entry.scene_id, record.entry.snippet_id) for record in plan.records}
    migrated_pairs = {(record.scene_id, record.snippet_id) for record in records}
    train_count = sum(1 for record in records if record.split == "train")
    val_count = sum(1 for record in records if record.split == "val")
    migrated_by_pair = {(record.scene_id, record.snippet_id): record for record in records}
    result = {
        "legacy_samples": len(plan.records),
        "migrated_samples": len(records),
        "legacy_train": plan.train_count,
        "legacy_val": plan.val_count,
        "migrated_train": train_count,
        "migrated_val": val_count,
        "pair_coverage_match": legacy_pairs == migrated_pairs,
        "manifest_num_samples": int(manifest.stats.get("num_samples", -1)),
        "checked_samples": 0,
        "content_match": False,
    }
    if len(plan.records) != len(records):
        raise ValueError("Migrated dataset sample count does not match legacy oracle cache.")
    if train_count != plan.train_count or val_count != plan.val_count:
        raise ValueError("Migrated dataset split counts do not match the legacy oracle cache.")
    if legacy_pairs != migrated_pairs:
        raise ValueError("Migrated dataset (scene_id, snippet_id) coverage does not match the legacy oracle cache.")
    if int(manifest.stats.get("num_samples", -1)) != len(records):
        raise ValueError("Migrated manifest num_samples does not match sample_index.jsonl.")

    vin_loader: EfmSnippetLoader | None = None
    for legacy_record in plan.records:
        pair = (legacy_record.entry.scene_id, legacy_record.entry.snippet_id)
        migrated_record = migrated_by_pair.get(pair)
        if migrated_record is None:
            raise ValueError(f"Migrated dataset is missing sample {pair}.")
        if migrated_record.split != legacy_record.split:
            raise ValueError(
                f"Migrated split mismatch for {pair}: expected {legacy_record.split}, got {migrated_record.split}.",
            )
        if migrated_record.sample_key != legacy_record.entry.key:
            raise ValueError(
                f"Migrated sample_key mismatch for {pair}: expected {legacy_record.entry.key}, got {migrated_record.sample_key}.",
            )
        if migrated_record.legacy_oracle_key != legacy_record.entry.key:
            raise ValueError(
                f"Migrated legacy_oracle_key mismatch for {pair}: expected {legacy_record.entry.key}, got {migrated_record.legacy_oracle_key}.",
            )
        if migrated_record.legacy_oracle_path != legacy_record.entry.path:
            raise ValueError(
                f"Migrated legacy_oracle_path mismatch for {pair}: expected {legacy_record.entry.path}, got {migrated_record.legacy_oracle_path}.",
            )
        if legacy_record.vin_entry is not None:
            if migrated_record.legacy_vin_key != legacy_record.vin_entry.key:
                raise ValueError(
                    f"Migrated legacy_vin_key mismatch for {pair}: expected {legacy_record.vin_entry.key}, got {migrated_record.legacy_vin_key}.",
                )
            if migrated_record.legacy_vin_path != legacy_record.vin_entry.path:
                raise ValueError(
                    f"Migrated legacy_vin_path mismatch for {pair}: expected {legacy_record.vin_entry.path}, got {migrated_record.legacy_vin_path}.",
                )

        oracle_payload = torch.load(
            plan.oracle_cache_dir / legacy_record.entry.path,
            map_location="cpu",
            weights_only=False,
        )
        depths = CandidateDepths.from_serializable(oracle_payload["depths"], device=torch.device("cpu"))
        rri = RriResult.from_serializable(oracle_payload["rri"], device=torch.device("cpu"))
        expected_candidate_count = int(depths.poses.tensor().shape[0])
        migrated_candidate_count = int(
            store_reader.read_numeric_block(migrated_record, "oracle.candidate_count").reshape(())
        )
        if migrated_candidate_count != expected_candidate_count:
            raise ValueError(
                f"Migrated candidate_count mismatch for {pair}: expected {expected_candidate_count}, got {migrated_candidate_count}.",
            )

        _assert_exact_block(
            store_reader.read_numeric_block(migrated_record, "oracle.reference_pose_world_rig"),
            _to_numpy(depths.reference_pose.tensor(), dtype=np.float32).reshape(-1),
            field_name="oracle.reference_pose_world_rig",
            pair=pair,
        )
        _assert_padded_prefix(
            store_reader.read_numeric_block(migrated_record, "oracle.candidate_indices"),
            _to_numpy(depths.candidate_indices.reshape(-1), dtype=np.int64),
            field_name="oracle.candidate_indices",
            pair=pair,
            pad_kind="minus_one",
        )
        _assert_padded_prefix(
            store_reader.read_numeric_block(migrated_record, "oracle.candidate_poses_world_cam"),
            _to_numpy(depths.poses.tensor(), dtype=np.float32),
            field_name="oracle.candidate_poses_world_cam",
            pair=pair,
            pad_kind="zero",
        )
        for field_name, expected in (
            ("oracle.rri", _to_numpy(rri.rri.reshape(-1), dtype=np.float32)),
            ("oracle.pm_dist_before", _to_numpy(rri.pm_dist_before.reshape(-1), dtype=np.float32)),
            ("oracle.pm_dist_after", _to_numpy(rri.pm_dist_after.reshape(-1), dtype=np.float32)),
            ("oracle.pm_acc_before", _to_numpy(rri.pm_acc_before.reshape(-1), dtype=np.float32)),
            ("oracle.pm_comp_before", _to_numpy(rri.pm_comp_before.reshape(-1), dtype=np.float32)),
            ("oracle.pm_acc_after", _to_numpy(rri.pm_acc_after.reshape(-1), dtype=np.float32)),
            ("oracle.pm_comp_after", _to_numpy(rri.pm_comp_after.reshape(-1), dtype=np.float32)),
        ):
            _assert_padded_prefix(
                store_reader.read_numeric_block(migrated_record, field_name),
                expected,
                field_name=field_name,
                pair=pair,
                pad_kind="nan",
            )

        vin_snippet, _, _, vin_loader = _load_legacy_vin_snippet(
            record=legacy_record,
            vin_cache_dir=plan.vin_cache_dir,
            dataset_payload=plan.dataset_payload,
            loader=vin_loader,
            semidense_max_points=manifest.vin.get("semidense_max_points"),
            semidense_include_obs_count=bool(manifest.vin.get("include_obs_count")),
            pad_points=int(manifest.vin.get("pad_points", 0)),
        )
        _assert_exact_block(
            store_reader.read_numeric_block(migrated_record, "vin.points_world"),
            _to_numpy(vin_snippet.points_world, dtype=np.float32),
            field_name="vin.points_world",
            pair=pair,
            equal_nan=True,
        )
        _assert_exact_block(
            store_reader.read_numeric_block(migrated_record, "vin.lengths"),
            _to_numpy(vin_snippet.lengths.reshape(-1), dtype=np.int64),
            field_name="vin.lengths",
            pair=pair,
        )
        _assert_exact_block(
            store_reader.read_numeric_block(migrated_record, "vin.t_world_rig"),
            _to_numpy(vin_snippet.t_world_rig.tensor(), dtype=np.float32),
            field_name="vin.t_world_rig",
            pair=pair,
        )
        result["checked_samples"] = int(result["checked_samples"]) + 1

    result["content_match"] = True
    return result


def _to_numpy(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    """Convert tensor-like values into NumPy arrays for verification."""

    if isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return array.astype(dtype, copy=False)


def _raise_verify_mismatch(*, pair: tuple[str, str], field_name: str, detail: str) -> None:
    """Raise a stable verification error for one mismatched field."""

    raise ValueError(f"Migrated field mismatch for {pair} in {field_name}: {detail}.")


def _assert_exact_block(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    field_name: str,
    pair: tuple[str, str],
    equal_nan: bool = False,
) -> None:
    """Assert that one migrated numeric block matches the legacy source exactly."""

    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)
    if actual_arr.shape != expected_arr.shape:
        _raise_verify_mismatch(
            pair=pair,
            field_name=field_name,
            detail=f"shape {actual_arr.shape} != {expected_arr.shape}",
        )
    if not np.allclose(actual_arr, expected_arr, equal_nan=equal_nan):
        _raise_verify_mismatch(pair=pair, field_name=field_name, detail="values differ")


def _assert_padded_prefix(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    field_name: str,
    pair: tuple[str, str],
    pad_kind: str,
) -> None:
    """Assert that one migrated padded block preserves the legacy prefix exactly."""

    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)
    if actual_arr.ndim != expected_arr.ndim:
        _raise_verify_mismatch(
            pair=pair,
            field_name=field_name,
            detail=f"rank {actual_arr.ndim} != {expected_arr.ndim}",
        )
    if actual_arr.shape[0] < expected_arr.shape[0] or actual_arr.shape[1:] != expected_arr.shape[1:]:
        _raise_verify_mismatch(
            pair=pair,
            field_name=field_name,
            detail=f"incompatible shape {actual_arr.shape} for expected prefix {expected_arr.shape}",
        )
    prefix = actual_arr[: expected_arr.shape[0]]
    if not np.allclose(prefix, expected_arr, equal_nan=True):
        _raise_verify_mismatch(pair=pair, field_name=field_name, detail="legacy prefix differs")

    tail = actual_arr[expected_arr.shape[0] :]
    if tail.size == 0:
        return
    if pad_kind == "nan":
        ok = np.isnan(tail).all()
    elif pad_kind == "zero":
        ok = np.allclose(tail, 0.0)
    elif pad_kind == "minus_one":
        ok = np.all(tail == -1)
    else:
        raise ValueError(f"Unknown pad_kind={pad_kind!r}")
    if not ok:
        _raise_verify_mismatch(pair=pair, field_name=field_name, detail=f"invalid {pad_kind} padding")


__all__ = [
    "migrate_legacy_offline_data",
    "scan_legacy_offline_data",
    "verify_migrated_offline_data",
]
