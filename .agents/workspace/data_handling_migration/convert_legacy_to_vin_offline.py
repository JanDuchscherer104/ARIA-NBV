#!/usr/bin/env python3
"""Convert legacy oracle/VIN caches into the immutable VIN offline dataset.

This temporary operator tool follows the repo's typed CLI pattern via
``pydantic-settings`` instead of hand-written ``argparse`` plumbing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from aria_nbv.data_handling import VinOfflineStoreConfig, migrate_legacy_offline_data
from aria_nbv.data_handling._legacy_cache_api import (
    OracleRriCacheConfig,
    VinSnippetCacheConfig,
)
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_scene_ids(raw: str | None) -> list[str] | None:
    """Parse a comma-separated scene-id CLI value into a normalized list."""
    if raw is None:
        return None
    scene_ids = [token.strip() for token in raw.split(",") if token.strip()]
    return scene_ids or None


class CLIMigrateLegacyOfflineData(BaseSettings):
    """CLI settings for legacy cache conversion into the immutable offline store."""

    oracle_cache: Path = Field(...)
    """Legacy oracle-cache directory."""

    vin_cache: Path | None = None
    """Optional legacy VIN-cache directory."""

    scene_ids: str | None = None
    """Optional comma-separated scene IDs to migrate."""

    split: Literal["all", "train", "val"] = "all"
    """Optional legacy split selector used before max-records truncation."""

    max_records: int | None = Field(default=None, ge=1)
    """Optional cap applied after split and scene filtering."""

    out_store: Path = Field(...)
    """Destination immutable VIN offline store."""

    workers: int = 0
    """Number of shard worker processes."""

    samples_per_shard: int = 64
    """Samples per immutable shard."""

    pad_points: int = 50000
    """Stored VIN padding budget."""

    max_candidates: int = 60
    """Stored candidate budget."""

    semidense_max_points: int | None = None
    """Optional collapse-time cap when rebuilding missing VIN snippets."""

    include_obs_count: bool = False
    """Whether rebuilt VIN points include semidense observation counts."""

    backbone: bool = True
    """Whether backbone payloads are materialized."""

    depths: bool = True
    """Whether depth payloads are materialized."""

    pointclouds: bool = True
    """Whether candidate point clouds are materialized."""

    overwrite: bool = False
    """Whether an existing destination store may be replaced."""

    repair_splits: bool = False
    """Whether missing or stale legacy split files should be repaired first."""

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_implicit_flags=True,
        extra="forbid",
    )


def main() -> None:
    """Run the legacy-to-offline conversion CLI."""
    args = CLIMigrateLegacyOfflineData()
    oracle_cfg = OracleRriCacheConfig(cache_dir=args.oracle_cache)
    vin_cfg = (
        VinSnippetCacheConfig(cache_dir=args.vin_cache)
        if args.vin_cache is not None
        else None
    )
    migrate_legacy_offline_data(
        oracle_cache=oracle_cfg,
        store=VinOfflineStoreConfig(store_dir=args.out_store),
        vin_cache=vin_cfg,
        scene_ids=_parse_scene_ids(args.scene_ids),
        split=args.split,
        max_records=args.max_records,
        workers=int(args.workers),
        samples_per_shard=int(args.samples_per_shard),
        max_candidates=int(args.max_candidates),
        include_backbone=bool(args.backbone),
        include_depths=bool(args.depths),
        include_pointclouds=bool(args.pointclouds),
        semidense_max_points=args.semidense_max_points,
        semidense_include_obs_count=bool(args.include_obs_count),
        pad_points=int(args.pad_points),
        overwrite=bool(args.overwrite),
        repair_splits=args.repair_splits,
        train_val_split=0.2,
    )
    destination = Path(args.out_store).expanduser().resolve()
    print(f"Wrote migrated VIN offline dataset to {destination}")


if __name__ == "__main__":
    main()
