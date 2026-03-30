# Data Handling Migration Workspace

This directory contains temporary operator tooling for migrating the legacy
oracle cache and VIN snippet cache into the new immutable VIN offline dataset.

## Files

- `scan_legacy_offline_data.py`
  - Scans the legacy caches and prints a JSON migration plan summary.

- `convert_legacy_to_vin_offline.py`
  - Converts the legacy caches into the new immutable VIN offline dataset.
  - Supports shard-parallel conversion via `--workers`.

- `verify_migrated_vin_offline.py`
  - Verifies migrated counts, provenance, and core per-sample content against
    the legacy caches.

- `run_migration.sh`
  - Convenience wrapper for `scan -> convert -> verify`.

## Typical usage

```bash
cd /home/jandu/repos/NBV

python .agents/workspace/data_handling_migration/scan_legacy_offline_data.py \
  --oracle-cache /path/to/legacy/oracle_cache \
  --vin-cache /path/to/legacy/vin_cache

python .agents/workspace/data_handling_migration/convert_legacy_to_vin_offline.py \
  --oracle-cache /path/to/legacy/oracle_cache \
  --vin-cache /path/to/legacy/vin_cache \
  --out-store /path/to/new/vin_offline \
  --workers 8 \
  --samples-per-shard 64 \
  --overwrite

python .agents/workspace/data_handling_migration/verify_migrated_vin_offline.py \
  --oracle-cache /path/to/legacy/oracle_cache \
  --vin-cache /path/to/legacy/vin_cache \
  --store /path/to/new/vin_offline

# Test a small subset first (single scene, first 8 selected samples)
python .agents/workspace/data_handling_migration/scan_legacy_offline_data.py \
  --oracle-cache /path/to/legacy/oracle_cache \
  --vin-cache /path/to/legacy/vin_cache \
  --scene-ids 81283 \
  --split train \
  --max-records 8

python .agents/workspace/data_handling_migration/convert_legacy_to_vin_offline.py \
  --oracle-cache /path/to/legacy/oracle_cache \
  --vin-cache /path/to/legacy/vin_cache \
  --scene-ids 81283 \
  --split train \
  --max-records 8 \
  --out-store /path/to/new/vin_offline_subset \
  --workers 0 \
  --samples-per-shard 8 \
  --overwrite

python .agents/workspace/data_handling_migration/verify_migrated_vin_offline.py \
  --oracle-cache /path/to/legacy/oracle_cache \
  --vin-cache /path/to/legacy/vin_cache \
  --scene-ids 81283 \
  --split train \
  --max-records 8 \
  --store /path/to/new/vin_offline_subset
```

## Notes

- The converter preserves exact legacy train/val membership.
- `--scene-ids` accepts a comma-separated list, for example `81283,82832`.
- `--split` and `--max-records` let you test a small deterministic slice before
  migrating the full cache.
- VIN snippets are taken from the legacy VIN cache when available and rebuilt
  live from raw ASE/EFM only for missing pairs.
- The output format is immutable. Re-run with `--overwrite` to rebuild.
- The scripts are temporary migration tools and are not part of the stable
  runtime API.
