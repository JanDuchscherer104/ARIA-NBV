Offline cache append + CLI

What changed
- Added `nbv-cache-oracle` CLI entry point for building offline caches via `oracle_rri.data.offline_cache_cli`.
- Added cache append support (`OracleRriCacheAppender`) and datamodule options to append new offline samples per epoch
  (offline-first concat: cached samples then new online samples).
- Added metadata include flags for cache compatibility checks; cache samples now get unique filenames when keys collide.

Potential issues / risks
- Cache appending requires `num_workers=0`; multi-process writes are not supported and will raise.
- Appending computes EVL backbone outputs in the data pipeline, which can duplicate work with the model backbone.
- Old caches created before include flags exist will not have explicit include metadata; appender allows this but relies
  on config signatures to prevent mismatches.
- Duplicate snippet IDs can still be appended (unique filename suffix), so caches can grow with repeats if the base
  dataset repeats samples.

Suggestions / next steps
- Consider a shared EVL backbone instance (or a hook in the LightningModule) to avoid duplicate backbone compute when
  appending online samples.
- Add append-focused integration tests (real data) that verify cache growth across epochs and that metadata stays
  consistent after appends.
- Add optional duplicate detection (skip or overwrite) when `scene_id+snippet_id` repeats, for more controlled growth.
- If cache format changes further, consider bumping `CACHE_VERSION` and adding migration notes.
