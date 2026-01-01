# VIN Lightning training script (pydantic-settings CLI + TOML)

## What changed

- Added `oracle_rri/oracle_rri/lightning/aria_nbv_experiment.py` with `AriaNBVExperimentConfig`, mirroring the nesting pattern from `external/doc_classifier/configs/experiment_config.py`:
  - `datamodule_config: VinDataModuleConfig` (contains `AseEfmDatasetConfig` + `OracleRriLabelerConfig`)
  - `module_config: VinLightningModuleConfig` (contains `VinModelConfig`)
  - `trainer_config: TrainerFactoryConfig` (contains callbacks + optional W&B logger)
- Refactored `oracle_rri/scripts/train_vin_lightning.py` into a thin runner around `AriaNBVExperimentConfig`:
  - CLI via **pydantic-settings** (`BaseSettings`, `SettingsConfigDict`) — no `argparse` in this script.
  - TOML config loading via `--config-path` / `--config_path` with CLI overrides (deep-merge of explicitly provided CLI fields).
  - Kebab-case CLI flags + implicit boolean flags; underscore flags are normalized to kebab-case.
- Added `fit_binner_only` mode to `AriaNBVExperimentConfig`:
  - `fit_binner_and_save()` saves `rri_binner.json` and on Ctrl-C/errors saves partial fit data as `rri_binner_fit_data.pt` in `out_dir`.

## “stage” terminology (important)

There are two unrelated concepts that were previously both called “stage”:

- **Lightning stage**: `Stage.TRAIN|VAL|TEST` in `oracle_rri.utils.schemas.Stage`, used only to pick the train/val split.
- **Capture stage id** (VIN-NBV): an integer group id used to z-normalize RRI per group before binning.

We removed the capture-stage id from `VinOracleBatch`/`VinDataModule` to avoid confusion and because we currently don’t have a meaningful stage definition for ASE snippets.

`RriOrdinalBinner` still supports stage-aware normalization, but now defaults to a single stage (`stage=None` → stage id 0).

## Data handling / “why only one scene?”

Observed behavior: binner fitting may print only one `scene=<id>` for many steps.

Reason:
- WebDataset iterates **shards sequentially** by default.
- Each `.tar` shard contains **many snippet samples**, so a small `fit_snippets` can be fully satisfied by the first scene’s first shards.
- ATEK’s `shuffle_flag` only shuffles **samples within a buffer**, and does **not** reorder shards.

Fix implemented:
- In `oracle_rri/oracle_rri/data/efm_dataset.py`, `wds_shuffle=True` now also **shuffles the shard list** (`tar_urls`) so short runs see multiple scenes early.
- In `oracle_rri/oracle_rri/data/efm_dataset.py`, we also **ignore empty `.tar` shards** (e.g. `dummy.tar`) when auto-discovering tar URLs.

## Oracle speed note

- `OracleRRI.score(...)` now actually **crops the GT mesh to the provided AABB** (`extend`) before computing point↔mesh distances. This is a big speed win for binner fitting/training and matches the “crop to relevant geometry” intent from the TODOs.

## How to use

### Recommended: run from a TOML config

```bash
cd oracle_rri
uv run python scripts/train_vin_lightning.py \
  --config-path .configs/<your_run>.toml
```

You can generate a starting TOML by instantiating `AriaNBVExperimentConfig` in Python and calling `save_config(...)`
or by running once and using the `out_dir/config.toml` snapshot.

### Quick CLI overrides

```bash
cd oracle_rri
uv run python scripts/train_vin_lightning.py \
  --config-path .configs/<your_run>.toml \
  --stage train
```

## Tests / validation performed

- `ruff format` + `ruff check` on:
  - `oracle_rri/scripts/train_vin_lightning.py`
  - `oracle_rri/oracle_rri/data/efm_dataset.py`
- `oracle_rri/oracle_rri/lightning/aria_nbv_experiment.py`
- Real-data smoke run:
  - `--fit-binner-only` with `--wds-shuffle` confirmed multi-scene sampling early.
- Pytest (real data):
  - `pytest tests/integration/test_vin_lightning_real_data.py`

## Follow-ups / suggestions

- Consider separating `wds_shuffle` into:
  - `shuffle_shards` (shard order) and
  - `shuffle_samples` (buffer shuffle),
  for clearer semantics.
- Add an explicit `wds_shuffle_seed` for reproducible shard ordering independent of global RNG state.
