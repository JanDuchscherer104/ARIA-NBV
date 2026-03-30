#!/usr/bin/env python3
"""Verify a migrated immutable VIN offline dataset against the legacy caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aria_nbv.data_handling import (
    VinOfflineStoreConfig,
    scan_legacy_offline_data,
    verify_migrated_offline_data,
)
from aria_nbv.data_handling._legacy_cache_api import (
    OracleRriCacheConfig,
    VinSnippetCacheConfig,
)


def _parse_scene_ids(raw: str | None) -> list[str] | None:
    """Parse a comma-separated scene-id argument into a normalized list."""
    if raw is None:
        return None
    scene_ids = [token.strip() for token in raw.split(",") if token.strip()]
    return scene_ids or None


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for migrated-store verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-cache",
        type=Path,
        required=True,
        help="Legacy oracle-cache directory.",
    )
    parser.add_argument(
        "--vin-cache",
        type=Path,
        default=None,
        help="Optional legacy VIN-cache directory.",
    )
    parser.add_argument(
        "--scene-ids",
        type=str,
        default=None,
        help="Optional comma-separated scene IDs to compare against.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "val"),
        default="all",
        help="Optional legacy split selector used before max-records truncation.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap applied after split and scene filtering.",
    )
    parser.add_argument(
        "--store",
        type=Path,
        required=True,
        help="Migrated immutable VIN offline store.",
    )
    parser.add_argument(
        "--repair-splits",
        action="store_true",
        help=(
            "Repair missing or stale legacy train/val split files before verification."
        ),
    )
    return parser


def main() -> None:
    """Run the migrated-store verification CLI and print the result."""
    args = _build_parser().parse_args()
    oracle_cfg = OracleRriCacheConfig(cache_dir=args.oracle_cache)
    vin_cfg = (
        VinSnippetCacheConfig(cache_dir=args.vin_cache)
        if args.vin_cache is not None
        else None
    )
    plan = scan_legacy_offline_data(
        oracle_cache=oracle_cfg,
        vin_cache=vin_cfg,
        scene_ids=_parse_scene_ids(args.scene_ids),
        split=args.split,
        max_records=args.max_records,
        repair_splits=args.repair_splits,
    )
    result = verify_migrated_offline_data(
        store=VinOfflineStoreConfig(store_dir=args.store),
        plan=plan,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
