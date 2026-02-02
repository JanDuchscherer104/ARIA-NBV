# Task Report: Offline Stats – Cache Component Memory Footprints

## Goal
Add **bar charts** to the Streamlit offline stats page showing the **estimated in-memory footprint** of the main `VinOracleBatch` components loaded from the offline oracle cache:
- `backbone_out`
- RRI targets (`rri`, `pm_*`)
- `vin_snippet` (`VinSnippetView`)
- poses + cameras
- total (sum)

## Implementation
- Added recursive tensor-size estimation helpers to `oracle_rri/oracle_rri/app/panels/offline_cache_utils.py`:
  - `_estimate_nbytes(...)`: sums tensor storage sizes across nested containers (tensors, numpy arrays, dataclasses, dict/list, and objects exposing `tensor()`).
  - `_p3d_cameras_nbytes(...)`: counts the common `PerspectiveCameras` tensor fields (`R`, `T`, `focal_length`, `principal_point`, `image_size`).
- Extended `_collect_offline_cache_stats(...)` to compute per-sample byte totals for each component and return a `memory_summary` (mean/median/p95 in MiB).
- Extended `oracle_rri/oracle_rri/app/panels/offline_stats.py` to render:
  - a small table of `mean_mib/median_mib/p95_mib` per component,
  - bar charts for mean and p95 memory footprints.

## Notes
- The footprint estimate is a **lower bound** on peak memory during loading because it excludes transient decode buffers (e.g. the cached depth maps are decoded to access poses/cameras but are not retained in `VinOracleBatch`).
- If needed, we can add a separate “peak load” estimate by instrumenting the decode path, but that would require deeper changes in `oracle_rri/data/offline_cache.py`.

## Tests
- Added unit tests covering nested-size estimation and camera accounting:
  - `oracle_rri/tests/app/test_offline_cache_memory.py`

