"""Legacy scan, conversion, and verification helpers for VIN offline migration.

This module provides the reusable migration logic behind the temporary tools in
``.agents/workspace/data_handling_migration``. It understands the legacy
oracle-cache and VIN-snippet-cache layouts, converts them into prepared shard
rows for the new immutable VIN offline format, and verifies the migrated
dataset against the legacy sources.
"""

from __future__ import annotations

import json
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from efm3d.aria.pose import PoseTW

from ..configs import PathConfig
from ..pose_generation.types import CandidateSamplingResult
from ..rendering.candidate_depth_renderer import CandidateDepths
from ..rendering.candidate_pointclouds import CandidatePointClouds
from ..rri_metrics.types import RriResult
from ..utils import Console
from ..vin.types import EvlBackboneOutput
from ._offline_format import (
    VinOfflineIndexRecord,
    VinOfflineManifest,
    VinOfflineMaterializedBlocks,
)
from ._offline_store import OFFLINE_DATASET_VERSION, VinOfflineStoreConfig
from ._offline_writer import (
    PreparedVinOfflineSample,
    flush_prepared_samples_to_shard,
    prepare_vin_offline_sample,
)
from ._raw import EfmSnippetLoader, VinSnippetView
from ._vin_runtime import build_vin_snippet_view
from .cache_contracts import OracleRriCacheEntry, VinSnippetCacheEntry
from .cache_index import read_index, repair_oracle_split_indices, validate_oracle_split_indices
from .offline_cache_store import _read_metadata as read_legacy_oracle_metadata
from .oracle_cache import OracleRriCacheConfig
from .vin_cache import VinSnippetCacheConfig


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

    vin_metadata: dict[str, Any] | None
    """Optional legacy VIN metadata snapshot."""


def scan_legacy_offline_data(
    *,
    oracle_cache: OracleRriCacheConfig,
    vin_cache: VinSnippetCacheConfig | None = None,
    train_val_split: float = 0.2,
    repair_splits: bool = False,
    console: Console | None = None,
) -> LegacyOfflinePlan:
    """Scan legacy oracle and VIN caches and build a migration plan.

    Args:
        oracle_cache: Legacy oracle-cache configuration.
        vin_cache: Optional legacy VIN-cache configuration.
        train_val_split: Validation fraction used only when legacy split files
            need explicit repair.
        repair_splits: Whether to repair missing or stale oracle split files.
        console: Optional logger.

    Returns:
        Normalized migration plan for the legacy caches.
    """

    console = console or Console.with_prefix("LegacyOfflineScan")
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

    oracle_meta = read_legacy_oracle_metadata(oracle_cache.metadata_path)
    dataset_payload = dict(oracle_meta.dataset_config or {}) if oracle_meta.dataset_config is not None else None

    vin_entries_by_pair: dict[tuple[str, str], VinSnippetCacheEntry] = {}
    vin_meta: dict[str, Any] | None = None
    if vin_cache is not None and vin_cache.index_path.exists():
        vin_entries = read_index(vin_cache.index_path, entry_type=VinSnippetCacheEntry, allow_missing=True)
        vin_entries_by_pair = {(entry.scene_id, entry.snippet_id): entry for entry in vin_entries}
        if vin_cache.metadata_path.exists():
            vin_meta = json.loads(vin_cache.metadata_path.read_text(encoding="utf-8"))

    records: list[LegacyOfflineRecord] = []
    missing_vin_pairs: list[tuple[str, str]] = []
    for entry in base_entries:
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
        train_count=len(train_entries),
        val_count=len(val_entries),
        missing_vin_pairs=missing_vin_pairs,
        oracle_metadata=oracle_meta.__dict__.copy(),
        vin_metadata=vin_meta,
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
        train_val_split=train_val_split,
        repair_splits=repair_splits,
        console=console,
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
        shutil.rmtree(out_store)
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
    temp_store.rename(out_store)
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
    legacy_pairs = {(record.entry.scene_id, record.entry.snippet_id) for record in plan.records}
    migrated_pairs = {(record.scene_id, record.snippet_id) for record in records}
    train_count = sum(1 for record in records if record.split == "train")
    val_count = sum(1 for record in records if record.split == "val")
    result = {
        "legacy_samples": len(plan.records),
        "migrated_samples": len(records),
        "legacy_train": plan.train_count,
        "legacy_val": plan.val_count,
        "migrated_train": train_count,
        "migrated_val": val_count,
        "pair_coverage_match": legacy_pairs == migrated_pairs,
        "manifest_num_samples": int(manifest.stats.get("num_samples", -1)),
    }
    if len(plan.records) != len(records):
        raise ValueError("Migrated dataset sample count does not match legacy oracle cache.")
    if train_count != plan.train_count or val_count != plan.val_count:
        raise ValueError("Migrated dataset split counts do not match the legacy oracle cache.")
    if legacy_pairs != migrated_pairs:
        raise ValueError("Migrated dataset (scene_id, snippet_id) coverage does not match the legacy oracle cache.")
    return result


__all__ = [
    "migrate_legacy_offline_data",
    "scan_legacy_offline_data",
    "verify_migrated_offline_data",
]
