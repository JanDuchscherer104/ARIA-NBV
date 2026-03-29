Cache filename scheme + nbv-summary fixes

What changed
- Updated cache filename format to `ASE_NBV_SNIPPET_<scene_id>_<snippet_token>_<config-hash>` (snippet token is the
  numeric suffix after `AtekDataSample_` when available) and stored `config_hash` in cache metadata
  (`oracle_rri/oracle_rri/data/offline_cache.py`).
- Added `rename_cache_samples` to migrate existing cache entries and used it to rename the current cache files.
- Made PyTorch3D camera decode resilient to versions that do not accept `znear`/`zfar`.
- Allowed VIN v2 summary to work with cached batches by using cached backbone outputs when raw EFM inputs are absent.

Run notes
- `uv run nbv-summary` completes successfully after the changes (cached EFM shows a note).
- Rebuilt `.data/oracle_rri_cache/index.jsonl` from cached samples and renamed files to the new naming scheme.
- `uv run nbv-cache-samples -n 1` completes and writes a cache entry using the new filename convention.

Potential issues / risks
- Older caches without `include_*` metadata will compute a config hash with defaults; consider regenerating caches if
  strict provenance is required.
- `rename_cache_samples` only renames files referenced by `index.jsonl`; use a rebuild script if the index was lost.

Suggestions / next steps
- If you want the new filenames for all existing cached samples, run `rename_cache_samples` on any legacy cache dirs.
