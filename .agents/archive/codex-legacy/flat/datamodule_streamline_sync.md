# DataModule Streamlining Sync (2026-01-04)

## Status
- Working branch: local workspace only.
- Goal: simplify data flow + remove duplicate batching utilities and dataset logic.
- Priority: redesign shared dataclass interfaces first, then switch datamodule + datasets to use them.

## Proposed design snapshot
- Central `VinOracleBatch` type + collate utilities live in `oracle_rri/oracle_rri/data/vin_batch_types.py` (or similar).
- Dataset variants share a base interface + ConfigAsFactory switching (online vs offline cache) via a **single** split-aware source config.
- `lit_datamodule.py` orchestrates only; no padding/collate helpers defined there.

## Key entry points (context links)
- Batch types + batching: `oracle_rri/oracle_rri/lightning/lit_datamodule.py` (current `VinOracleBatch` + collate helpers).
- Lightning usage: `oracle_rri/oracle_rri/lightning/lit_module.py` (consumes `VinOracleBatch`).
- Offline cache dataset: `oracle_rri/oracle_rri/data/offline_cache.py` (uses `VinOracleBatch` in vin mode).
- Snippet cache: `oracle_rri/oracle_rri/data/vin_snippet_cache.py`.
- EFM dataset: `oracle_rri/oracle_rri/data/efm_dataset.py`.
- Oracle labeler: `oracle_rri/oracle_rri/pipelines/oracle_rri_labeler.py`.
- PathConfig: `oracle_rri/oracle_rri/configs/path_config.py`.
- Tests (existing): `tests/data/test_vin_snippet_cache_datamodule_equivalence.py`.

## Agent assignments (3 instances total)

### Agent A — Dataclass/API redesign (primary)
**Owner:** This instance (Codex A)
**Scope:**
- Design and implement the revised shared dataclass types.
- Decide dataclass vs pydantic usage (batch container should be dataclass).
- Move collate/padding utilities into the dataclass module.
**Deliverables:**
- New module `oracle_rri/oracle_rri/data/vin_batch_types.py` (name TBD) with:
  - `VinOracleBatch` dataclass.
  - `collate(...)` classmethod + padding helpers as private staticmethods.
  - Optional `VinSnippetBatch` helper if needed.
- Update imports in `lit_module.py`, `offline_cache.py`, tests.
**Context links:** `oracle_rri/oracle_rri/lightning/lit_datamodule.py`, `oracle_rri/oracle_rri/lightning/lit_module.py`, `oracle_rri/oracle_rri/data/offline_cache.py`
**Tests to update/add:** existing equivalence test; new unit test for `VinOracleBatch.collate`.

**Progress (2026-01-04):**
- Implemented `oracle_rri/oracle_rri/data/vin_oracle_types.py` with `VinOracleBatch` + `collate` + helpers.
- Removed collate/pad helpers + `VinOracleBatch` from `lit_datamodule.py`.
- Updated imports across lightning/app/tests/offline_cache/vin models.
- Tests run: `tests/lightning/test_vin_batch_collate.py`, `tests/lightning/test_lit_module_masking.py`.

### Agent B — Dataset interface + ConfigAsFactory switching
**Owner:** Codex B
**Scope:**
- Introduce a shared dataset interface for the 3 dataset variants.
- Implement ConfigAsFactory switching, similar to LR scheduler configs.
- Ensure each dataset config has `paths: PathConfig`.
**Deliverables:**
- New dataset interface (Protocol/ABC) and config classes.
- Replace in `VinDataModuleConfig` with a single `source` using split-aware `setup_target(split=Stage)` and union configs.
- Backward-compat glue if needed (from `train_cache/val_cache`).
**Context links:** `oracle_rri/oracle_rri/lightning/optimizers.py` (scheduler pattern), `oracle_rri/oracle_rri/lightning/lit_datamodule.py`, `oracle_rri/oracle_rri/data/offline_cache.py`
**Tests to update/add:** dataset switching tests; config validation tests.

**Progress (2026-01-04):**
- Added dataset interface `VinOracleDatasetBase` in `oracle_rri/oracle_rri/data/vin_oracle_types.py`.
- Added new module `oracle_rri/oracle_rri/data/vin_oracle_datasets.py` with:
  - `VinOracleOnlineDataset`, `VinOracleCacheDataset`,
  - config factories `VinOracleOnlineDatasetConfig`, `VinOracleCacheDatasetConfig`,
  - `VinOracleDatasetConfig` discriminated union (`kind` field).
- Removed cache-append dataset/config usage in datamodule + exports.
- Updated `OracleRriCacheVinDataset` to be map-style (`is_map_style=True`) and iterable.
- Updated `lit_datamodule.py` to consume new dataset variants via `source.setup_target(split=Stage)`.
- Updated app/UI helpers and configs to use `source` (incl. `.configs/*offline*.toml`, `offline_stats.py`).
- Added tests: `tests/lightning/test_vin_datamodule_sources.py`.
- Tests run: `oracle_rri/.venv/bin/python -m pytest tests/lightning/test_vin_datamodule_sources.py`.
- Note: `source` now handles train/val via split-aware config; `use_train_as_val` reuses the train instance.
- Added dataset summary logging in `VinDataModule.setup` via `Console.plog`.
- Offline cache VIN snippet loading now warns + falls back to an empty snippet when cache/EFM lookup fails (avoids hard crash when snippets are missing).

### Agent C — Datamodule orchestration + tests
**Owner:** Codex C
**Scope:**
- Simplify `VinDataModule` logic with a shared stage-plan builder.
- Remove duplicate dataset branching from train/val/iter paths.
- Expand tests to ensure equivalence between online/offline cache + vin-snippet cache.
**Deliverables:**
- Refactored `lit_datamodule.py` with `_build_stage_plan(stage)` returning dataset + batching settings.
- Updated `iter_oracle_batches` to reuse the stage plan.
- Test additions/updates as per new interfaces.
**Context links:** `oracle_rri/oracle_rri/lightning/lit_datamodule.py`, `tests/data/test_vin_snippet_cache_datamodule_equivalence.py`
**Tests to update/add:** more thorough batching tests + stage-plan coverage.

**Progress (2026-01-04):**
- Added stage plan helper in `oracle_rri/oracle_rri/lightning/lit_datamodule.py` to unify dataset selection.
- `train_dataloader`, `val_dataloader`, and `iter_oracle_batches` now share `_build_stage_plan(...)`.
- Targeted tests run: `oracle_rri/.venv/bin/python -m pytest tests/lightning/test_vin_datamodule_sources.py`.

## Task breakdown (shared checklist)
### Core refactor
- [ ] Move `VinOracleBatch` + collate/pad helpers out of `lit_datamodule.py` into data module.
- [ ] Introduce dataset interface + config classes for 3 dataset versions.
- [ ] Update `lit_datamodule.py` to use new dataset config factories.
- [ ] Update `lit_module.py` + tests to import `VinOracleBatch` from new location.

### Cleanup
- [ ] Remove duplicated dataset selection logic with a shared stage builder.
- [ ] Ensure all dataset configs carry `paths: PathConfig`.

### Tests
- [ ] Update existing equivalence tests to use new imports.
- [ ] Add tests for new dataset interface + collate invariants.

## Open questions
- Should `VinOracleBatch` always carry `VinSnippetView` (never full `EfmSnippetView`)?
- How strict should collate be when some samples lack snippet views (hard fail vs pad empty)?

## Agent handoff notes
- Collate utilities + batch type are in `oracle_rri/oracle_rri/data/vin_oracle_types.py`.
- Dataset variants are now only: `VinOracleOnlineDataset`, `OracleRriCacheVinDataset`.
- Related logic appears in `vin_oracle_datasets.py`, `lit_datamodule.py`, and `offline_cache.py`.

## Testing log
- `tests/lightning/test_vin_batch_collate.py` (pass)
- `tests/lightning/test_lit_module_masking.py` (pass; updated dummy pose to include ndim)

## Agent C context notes (2026-01-04)
### Datamodule branching hotspots to unify
- `VinDataModule.train_dataloader`, `val_dataloader`, and `iter_oracle_batches` re-implement the same branching for:
  - offline cache only (`OracleRriCacheVinDataset`),
  - online labeler (`VinOracleOnlineDataset`),
  - `use_train_as_val` fallback.
- `use_batching` only engages when **offline-only cache** + `batch_size` set; `collate_vin_oracle_batches` is only wired there.

### Collate + batch constraints to preserve
- `collate_vin_oracle_batches` pads candidates to max length, stacks `PoseTW`, and batches `PerspectiveCameras`.
- Collation **rejects** full `EfmSnippetView` (expects `VinSnippetView` when snippets are present).
- `VinSnippetView` padding uses NaNs for points and repeats last pose for trajectories.

### External call sites that must keep working
- `oracle_rri/lightning/aria_nbv_experiment.py` calls `datamodule.iter_oracle_batches(stage=...)` for summaries and plotting.
- `oracle_rri/app/panels/vin_diagnostics.py` calls `datamodule.iter_oracle_batches(stage=...)` when not using manual cache selection.

### Tests likely impacted by datamodule refactor
- `tests/data/test_vin_snippet_cache_datamodule_equivalence.py` now uses `VinDataModuleConfig(source=...)`.
- `tests/data/test_offline_cache_split.py::test_datamodule_applies_cache_split` now asserts split handling via `train_split`/`val_split`.
## Agent B context notes (2026-01-04)
### Current dataset variants + locations
- Online iterable: `VinOracleOnlineDataset` in `oracle_rri/oracle_rri/data/vin_oracle_datasets.py` (wraps `AseEfmDataset` + `OracleRriLabeler`).
- Offline map-style: `OracleRriCacheVinDataset` in `oracle_rri/oracle_rri/data/offline_cache.py` (wrapper around `OracleRriCacheDataset` with `return_format="vin_batch"`).

### Config-as-factory patterns to mirror
- Discriminated union example: `PoseEncoderConfig` in `oracle_rri/oracle_rri/vin/pose_encoders.py` uses `Annotated[..., Field(discriminator="kind")]`.
- Standard config factories use `target: type[...] = Field(default_factory=..., exclude=True)` and `BaseConfig.setup_target()` (see `oracle_rri/oracle_rri/utils/base_config.py`).

### VinDataModuleConfig coupling points
- `source` (`VinOracleDatasetConfig`) drives dataset selection in `VinDataModule` via `setup_target(split=Stage)`.
- `use_train_as_val` reuses the train dataset instance for validation/testing.
- `batch_size` only supported for map-style datasets (offline cache / vin-snippet cache).

### Cache dataset behavior worth preserving
- `OracleRriCacheDatasetConfig.return_format="vin_batch"` uses `VinSnippetView` (via `vin_snippet_cache` or EFM loader) and requires `load_depths=True`.
- `vin_snippet_cache_mode` supports `auto`/`required`/`disabled` (see `oracle_rri/oracle_rri/data/vin_snippet_provider.py`).

### Tests to revisit under new dataset interface
- `tests/data/test_vin_snippet_cache_datamodule_equivalence.py` builds `VinDataModuleConfig(source=...)`.
- `tests/data/test_offline_cache_split.py::test_datamodule_applies_cache_split` asserts `train_split`/`val_split` usage.

## Agent B update (2026-01-04)
- Verified offline-cache training/validation for 1 batch each using `limit_train_batches=1` + `limit_val_batches=1` (tqdm progress shows val loop).
- Verified online dataset training/validation for 1 batch each using temporary config `/tmp/nbv_online_one.toml` (scene_id=81022, snippet_key_filter=000024).
- CLI cannot override `batch_size=None` for online datasets when using offline cache config; requires separate config without `batch_size` and without offline-only fields.
- Confirmed first offline cache entry (scene 81022, snippet 000024) exists in vin_snippet_cache index.

## 2026-01-05: configs + vin_snippet_cache_mode=required
- Added `.configs/online_only.toml` (online-only config without batch_size).
- Added `.configs/offline_cache_required_one_step.toml` (offline cache 1-step with vin_snippet_cache_mode="required").
- `vin_snippet_cache_mode="required"` now raises on missing config, missing cache entry, or provider failure (no fallback to empty snippet).

## 2026-01-05: Offline stats panel updates
- `offline_stats.py` now lets you switch between Oracle RRI cache and VIN snippet cache.
- New VIN snippet cache stats collector: points count / trajectory length / inv_dist_std mean.
- Coverage scan prefers vin_snippet_cache metadata when selected.
