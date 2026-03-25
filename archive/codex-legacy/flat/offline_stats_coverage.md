## Offline stats: dataset coverage (2026-01-03)

### Goal
- Add a lightweight way to estimate how much of the *available* ASE EFM dataset (scenes/snippets in shard tar files) is covered by the current offline cache, without running the expensive "Compute offline stats" pass.

### What changed
- `oracle_rri/oracle_rri/app/panels/offline_stats.py`
  - Added a **Dataset coverage** section.
  - Coverage scan reads `train_index.jsonl` / `val_index.jsonl` (falls back to `index.jsonl` if split indices are missing).
  - Dataset availability is obtained by scanning the **tar headers** of the configured `AseEfmDatasetConfig.tar_urls` (no sample decoding).
  - Added histograms:
    - available snippets per scene,
    - cached snippets per scene (train ∪ val),
    - per-scene coverage ratio (train ∪ val), plus optional train vs val overlay.

- `oracle_rri/oracle_rri/data/offline_cache_coverage.py`
  - New utilities for:
    - reading cache index JSONL entries,
    - expanding tar globs,
    - scanning shard tars for WebDataset sample keys,
    - computing per-scene and aggregate coverage stats.

- `tests/data/test_offline_cache_coverage.py`
  - Added synthetic-tar tests covering: tar scanning, glob expansion, JSONL parsing, filter semantics, and coverage report correctness.

### Notes / caveats
- Scanning many shard tar files can be slow (still far cheaper than decoding samples). The UI includes a `Max tar shards to scan` knob for quick checks.
- Scene ids are inferred from the tar parent directory name when it is numeric, otherwise from `AriaSyntheticEnvironment_<scene_id>_...` key prefixes.
- `snippet_key_filter` is respected using the same matching semantics as `AseEfmDataset` (`key == token or key.endswith(token)`).

### Follow-ups (optional)
- Add `st.cache_data` memoization for tar header scans keyed by tar paths + mtime to make repeated UI refreshes instant.
- Surface the worst-covered scenes in a "Top missing" table for quicker triage.

