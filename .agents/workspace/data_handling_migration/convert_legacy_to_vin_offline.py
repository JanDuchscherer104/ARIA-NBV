#!/usr/bin/env python3
"""Convert legacy oracle/VIN caches into the immutable VIN offline dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from aria_nbv.data_handling import (
    OracleRriCacheConfig,
    VinOfflineIndexRecord,
    VinOfflineShardSpec,
    VinOfflineStoreConfig,
    VinSnippetCacheConfig,
    finalize_migrated_store,
    flush_prepared_samples_to_shard,
    prepare_legacy_records,
    scan_legacy_offline_data,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for legacy-to-offline conversion."""
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
        "--out-store",
        type=Path,
        required=True,
        help="Destination immutable VIN offline store.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of shard worker processes.",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=64,
        help="Samples per immutable shard.",
    )
    parser.add_argument(
        "--pad-points",
        type=int,
        default=50000,
        help="Stored VIN padding budget.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=60,
        help="Stored candidate budget.",
    )
    parser.add_argument(
        "--semidense-max-points",
        type=int,
        default=None,
        help="Optional collapse-time cap when rebuilding missing VIN snippets.",
    )
    parser.add_argument(
        "--include-obs-count",
        action="store_true",
        help="Include semidense observation counts in rebuilt VIN snippets.",
    )
    parser.add_argument(
        "--no-backbone",
        action="store_true",
        help="Do not materialize backbone payloads.",
    )
    parser.add_argument(
        "--no-depths",
        action="store_true",
        help="Do not materialize depth payloads.",
    )
    parser.add_argument(
        "--no-pointclouds",
        action="store_true",
        help="Do not materialize candidate point clouds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing destination store.",
    )
    parser.add_argument(
        "--repair-splits",
        action="store_true",
        help="Repair missing or stale legacy train/val split files before conversion.",
    )
    return parser


def _convert_shard(job: dict[str, Any]) -> dict[str, Any]:
    """Convert one legacy-record slice into one immutable shard payload."""
    records = prepare_legacy_records(
        records=job["records"],
        oracle_cache_dir=Path(job["oracle_cache_dir"]),
        vin_cache_dir=Path(job["vin_cache_dir"])
        if job["vin_cache_dir"] is not None
        else None,
        dataset_payload=job["dataset_payload"],
        max_candidates=int(job["max_candidates"]),
        include_backbone=bool(job["include_backbone"]),
        include_depths=bool(job["include_depths"]),
        include_pointclouds=bool(job["include_pointclouds"]),
        semidense_max_points=job["semidense_max_points"],
        semidense_include_obs_count=bool(job["semidense_include_obs_count"]),
        pad_points=int(job["pad_points"]),
    )
    shard_dir = Path(job["shard_dir"])
    shard_spec, index_records = flush_prepared_samples_to_shard(
        shard_index=int(job["shard_index"]),
        shard_dir=shard_dir,
        rows=records,
    )
    return {
        "shard_spec": shard_spec.to_dict(),
        "index_records": [record.to_json() for record in index_records],
    }


def main() -> None:
    """Run the legacy-to-offline conversion CLI."""
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
        repair_splits=args.repair_splits,
    )

    out_store = args.out_store.expanduser().resolve()
    temp_store = out_store.with_name(f"{out_store.name}.tmp")
    if temp_store.exists():
        shutil.rmtree(temp_store)
    if out_store.exists():
        if not args.overwrite:
            message = (
                "Destination store already exists at "
                f"{out_store}. Use --overwrite to replace it."
            )
            raise FileExistsError(message)
        shutil.rmtree(out_store)
    temp_store.mkdir(parents=True, exist_ok=True)
    (temp_store / "shards").mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    for shard_index in range(0, len(plan.records), int(args.samples_per_shard)):
        shard_records = plan.records[
            shard_index : shard_index + int(args.samples_per_shard)
        ]
        shard_id = shard_index // int(args.samples_per_shard)
        jobs.append(
            {
                "shard_index": shard_id,
                "records": shard_records,
                "oracle_cache_dir": plan.oracle_cache_dir.as_posix(),
                "vin_cache_dir": plan.vin_cache_dir.as_posix()
                if plan.vin_cache_dir is not None
                else None,
                "dataset_payload": plan.dataset_payload,
                "max_candidates": int(args.max_candidates),
                "include_backbone": not args.no_backbone,
                "include_depths": not args.no_depths,
                "include_pointclouds": not args.no_pointclouds,
                "semidense_max_points": args.semidense_max_points,
                "semidense_include_obs_count": args.include_obs_count,
                "pad_points": int(args.pad_points),
                "shard_dir": (
                    temp_store / "shards" / f"shard-{shard_id:06d}"
                ).as_posix(),
            },
        )

    results: list[dict[str, Any]]
    if int(args.workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            results = list(executor.map(_convert_shard, jobs))
    else:
        results = [_convert_shard(job) for job in jobs]

    shard_specs = [
        VinOfflineShardSpec.from_dict(result["shard_spec"]) for result in results
    ]
    index_records = [
        VinOfflineIndexRecord.from_dict(json.loads(serialized))
        for result in results
        for serialized in result["index_records"]
    ]
    store_cfg = VinOfflineStoreConfig(store_dir=temp_store)
    finalize_migrated_store(
        store=store_cfg,
        plan=plan,
        shard_specs=shard_specs,
        index_records=index_records,
        max_candidates=int(args.max_candidates),
        include_backbone=not args.no_backbone,
        include_depths=not args.no_depths,
        include_pointclouds=not args.no_pointclouds,
        semidense_max_points=args.semidense_max_points,
        semidense_include_obs_count=args.include_obs_count,
        pad_points=int(args.pad_points),
    )
    temp_store.rename(out_store)
    print(f"Wrote migrated VIN offline dataset to {out_store}")


if __name__ == "__main__":
    main()
