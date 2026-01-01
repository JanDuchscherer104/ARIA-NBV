# VIN offline stats panel + VIN v2 backbone fix

## Summary
- Added an Offline Stats tab in `oracle_rri/oracle_rri/app/panels.py` that streams offline cache batches via `VinDataModule` and computes global metrics (RRI/pm_comp_after/pm_acc_after, num_valid histograms, scatter plots) using seaborn without retaining full samples in memory.
- Added backbone feature stats aggregation (mean/std/abs_mean/nz_frac per EvlBackboneOutput tensor) to help decide which fields are good global features.
- Fixed VIN v2 backbone attribute initialization to avoid `AttributeError` when accessing `self.backbone`.
- Made offline cache map_location selection robust when VIN backbone config is missing.

## Notes / Potential Follow-ups
- `num_valid` currently uses finite RRI counts; if true validity masks become available in cached batches, update stats to use that mask.
- Consider caching offline stats results on disk (e.g., a compact parquet) for large caches.
- `ruff check` currently fails due to existing style issues (relative-import lint, TODO metadata, long lines) in `panels.py` and `model_v2.py`.

## Tests
- `pytest tests/integration/test_vin_real_data.py -q` failed (module not found, uses system python).
- `uv run pytest oracle_rri/tests/integration/test_vin_real_data.py -q` failed: `ModuleNotFoundError: efm3d` during collection.

## Update
- Moved Offline Stats into its own top-level Streamlit page via `render_offline_stats_page` and navigation entry.
- Fix (2025-12-31): `_collect_offline_cache_stats` now resolves `TensorWrapper` fields via `.tensor()` (or direct tensor attr) to avoid `'function' object has no attribute 'numel'`.
- Fix (2025-12-31): forced Matplotlib `Agg` backend in `panels.py` and replaced deprecated `use_container_width` with `width="stretch"` to silence Streamlit warnings.
