#!/usr/bin/env python3
"""Scan legacy oracle/VIN caches and print a migration plan summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aria_nbv.data_handling import (
    OracleRriCacheConfig,
    VinSnippetCacheConfig,
    scan_legacy_offline_data,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the legacy-cache scan command."""
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
        "--train-val-split",
        type=float,
        default=0.2,
        help="Validation fraction used only when split repair is requested.",
    )
    parser.add_argument(
        "--repair-splits",
        action="store_true",
        help="Repair missing or stale legacy train/val split files before planning.",
    )
    return parser


def main() -> None:
    """Run the legacy-cache scan CLI and print a JSON summary."""
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
        train_val_split=args.train_val_split,
        repair_splits=args.repair_splits,
    )
    summary = {
        "oracle_cache_dir": plan.oracle_cache_dir.as_posix(),
        "vin_cache_dir": (
            plan.vin_cache_dir.as_posix() if plan.vin_cache_dir is not None else None
        ),
        "num_samples": len(plan.records),
        "train_count": plan.train_count,
        "val_count": plan.val_count,
        "missing_vin_pairs": [list(pair) for pair in plan.missing_vin_pairs],
        "has_dataset_payload": plan.dataset_payload is not None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
