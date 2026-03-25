# AGENTS.md Improvements (Postmortem)

## What would have helped during init

### 1) Environment + test execution
- Explicitly state: always run repo commands via `uv run ...` or `oracle_rri/.venv/bin/python ...`.
- Call out the failure mode we hit: system `pytest` (Python 3.12) missing optional deps (e.g. `power_spherical`) while `.venv` (Python 3.11) has them.
- Add a short “known good” smoke test command for this repo (e.g. `uv run pytest tests/data/test_offline_cache_split.py`).

### 2) Lightning validation defaults
- Highlight that `TrainerFactoryConfig.enable_validation` defaults to `False` and silently disables validation by forcing `limit_val_batches=0` and `check_val_every_n_epoch=0`.
- Provide a minimal TOML snippet for enabling validation.

### 3) Pydantic config pitfalls
- Add an explicit warning: do not set `Field(default=<callable>)` unless you want to serialize/store the callable; use `default_factory`.
- This would have avoided the `val_cache` ValidationError regression.

### 4) Offline cache split semantics
- Document that offline cache splitting is file-backed using `train_index.jsonl` and `val_index.jsonl`.
- Note that split files may be created/updated as a side-effect of loading `OracleRriCacheDataset(split="train"/"val")`.
- Provide the canonical rebuild entrypoint (`rebuild_cache_index`) and what it writes.

### 5) Streamlit diagnostics/stats expectations
- Mention that geometry diagnostics can optionally load `EfmSnippetView` for cached samples (`include_efm_snippet/include_gt_mesh`) and that it’s expensive.

## Related changes we made
- Added split-backed cache loading and random split rebuild.
- Fixed the Pydantic `val_cache` default regression.
- Enabled config knobs in `.configs/offline_only.toml` to make validation + cache split explicit.
