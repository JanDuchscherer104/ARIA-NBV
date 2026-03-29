Offline cache simplification (config snapshot refactor)

What changed
- Replaced bespoke snapshot helpers with a single `snapshot_config()` that uses `model_dump_jsonable`, and trimmed
  dataset snapshots by excluding `tar_urls` and `scene_to_mesh`.
- Swapped writer/appender metadata to use `snapshot_config()` for labeler/backbone config capture.
- Updated `oracle_rri/oracle_rri/data/README.md` to reflect that metadata omits large path lists without counting them.

Run notes
- `uv run ruff format oracle_rri/data/offline_cache.py` (no changes)
- `uv run ruff check oracle_rri/data/offline_cache.py` (clean)
- `uv run pytest tests/data/test_offline_cache.py -q` (skipped: missing real data)

Potential issues / risks
- Metadata now captures full config dumps; this may include extra path fields compared to the prior hand-picked snapshots.
- Any tooling expecting `tar_urls_count` / `scene_to_mesh_count` fields in metadata will no longer find them.

Suggestions / next steps
- If you still want tar/mesh counts in metadata, re-add them explicitly as lightweight fields.
- Consider collapsing labeler/backbone signature handling now that config snapshots are standardized.

Additional change (dataclass serialization)
- Replaced manual encode/decode blocks in `offline_cache.py` with generic `encode_dataclass` / `decode_dataclass`
  that recurse over dataclass fields and only special-case TensorWrapper + PerspectiveCameras.
- Added `_strip_optional` + `_decode_value` helpers so typed fields control device moves without per-field code.
