# VIN diagnostics: batch size + VIN snippet cache source

## What changed
- Added batch-size control in VIN diagnostics sidebar; offline cache batches are built by collating multiple cache samples.
- Added snippet source switch for offline cache runs: EFM snippet loader vs. VinSnippetCacheDataset (cached VinSnippetView).
- VIN snippet cache is cached in session state (signature + dataset instance) to avoid reloads.
- Updated VIN debug helper to accept VinSnippetView directly.

## Rationale
- Training uses vin-snippet-cache; diagnostics can now mirror that path without loading full EFM snippets.
- Batch-size support helps stress-test collate behavior and batched VIN forward.

## Notes / caveats
- Batch size > 1 is only allowed when snippets are detached or loaded from VinSnippetCacheDataset; EfmSnippetView batching is not supported.
- VinSnippetCacheDataset is used only for snippet attachment; Oracle cache samples are still the source for candidates/rri/depths.

## Tests
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/test_app_state_signature.py`
  - Pass

## Open items
- `uv run pytest ...` uses system Python 3.12 and fails with `power_spherical` missing; stick to the repo venv for tests.

## Follow-up: VIN snippet cache missing entries
- Added a “Require VIN snippet entries” toggle; when off, missing entries fallback to EFM snippets (batch_size=1) or detach snippets for batched runs.
- VIN snippet cache path now auto-resolves `<path>/vin_snippet_cache` when a parent cache dir is supplied.

## Fix: snippet_id token mismatch
- Oracle cache payload stores full snippet_id (e.g., `AriaSyntheticEnvironment_..._000028`) while vin_snippet_cache index uses the token (`000028`).
- `VinSnippetCacheDataset.get_by_scene_snippet` now normalizes via `_extract_snippet_token` if direct lookup fails, so entries resolve without rebuilding caches.

## Fix: VinSnippetView lengths
- Cache writer now stores the **pre-padding** valid point count as `points_length` (still capped when subsampling to 50k).
- Cache reader now clamps stored lengths to the finite XYZ count, fixing older entries that incorrectly stored 50k.

## Script: fix_vin_snippet_lengths
- Added `oracle_rri/scripts/fix_vin_snippet_lengths.py` to recompute and persist `points_length` based on finite XYZ rows.
- Ran against `/mnt/e/wsl-data/ase-atek-nbv/offline_cache/vin_snippet_cache` → 841 files checked, 0 updates (already consistent).

## Fix: summarize_vin with VinSnippetView
- `VinModelV3.summarize_vin` now accepts VinSnippetView inputs without attempting to access `.efm`.
- Summary output includes vin snippet fields when raw EFM inputs are unavailable.

## Pose descriptor plots
- Added R6D orthonormality error + determinant histograms.
- Added R6D component circle plots (XY/XZ/YZ for each 3D column).
- Source: `oracle_rri/oracle_rri/app/panels/vin_diag_tabs/pose.py`.

## Fix: geometry tab with VinSnippetView
- Geometry tab now treats any snippet lacking `camera_rgb` as VIN snippet cache and avoids scene_plot_options_ui (prevents AttributeError).

## Fix: candidate frustum in VIN snippet geometry
- Added frustum controls and enabled frustum drawing in the VIN snippet geometry path via `build_semidense_projection_figure(..., show_frustum=True)`.

## Fix: summarize_vin handles VinSnippetView
- `VinModelV3.summarize_vin` now routes VinSnippetView through forward and avoids `.efm` access.
- Summary includes vin snippet fields when raw EFM inputs are unavailable.

## FF encodings: scatter point size
- Increased marker sizes in LFF PCA, pose enc PCA, and pos-grid PCA plots (3 -> 6) in `vin/experimental/plotting.py`.

## FF encodings: distance plots
- Added UI to select candidate indices and compute pairwise L2 distances for R6D components and final pose encodings.
- If 2 candidates selected, show scalar distances; otherwise show distance matrices.
- Source: `oracle_rri/oracle_rri/app/panels/vin_diag_tabs/encodings.py`.

## Fix: encoding distances batch slider
- Guarded batch index slider when batch_size==1 to avoid Streamlit min==max error.
