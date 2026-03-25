Cache CLI run + overwrite handling

What changed
- Added auto-overwrite behavior in `oracle_rri/data/offline_cache_cli.py` when the cache index exists and the user
  did not pass an explicit overwrite flag.
- Documented `-n/--num-samples` and overwrite behavior in `oracle_rri/oracle_rri/data/README.md`.

Run notes
- `uv run nbv-cache-samples -n 1` completed successfully and wrote one cached sample to
  `.data/oracle_rri_cache`.
- The full default run (no `-n`) is long; it proceeds correctly but can exceed typical CLI timeouts.

Suggestions / next steps
- For longer cache builds, run with a config file that sets `max_samples` and/or a smaller `scene_ids` subset.
