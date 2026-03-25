# VIN Lightning + W&B (Dec 2025)

## Goal

Add a minimal PyTorch Lightning training stack for the existing VIN implementation (`oracle_rri/oracle_rri/vin/`) with optional Weights & Biases logging, without touching any files in `oracle_rri/oracle_rri/vin/`.

## What changed

### New training package

Added `oracle_rri/oracle_rri/lightning/`:

- `oracle_rri/oracle_rri/lightning/lit_datamodule.py`
  - `VinDataModule` runs the expensive `OracleRriLabeler` online and yields `VinOracleBatch` objects.
  - Mirrors the data-flow in `oracle_rri/scripts/train_vin.py` but suitable for Lightning loops.
- `oracle_rri/oracle_rri/lightning/lit_module.py`
  - `VinLightningModule` wraps `VinModel` and trains with CORAL loss.
  - Fits `RriOrdinalBinner` on `on_fit_start` (configurable `binner_fit_snippets`) and saves it as `rri_binner.json`.
  - Stores binner state in Lightning checkpoints via `on_save_checkpoint` / `on_load_checkpoint`.
  - Logs oracle RRI diagnostics (`pm_*_{before|after}` means) alongside loss.
- `oracle_rri/oracle_rri/lightning/lit_trainer_callbacks.py`
  - `TrainerCallbacksConfig` with checkpointing + optional LR monitor/progress bars.
  - LR monitor is only enabled when a logger exists (PL 1.9 throws otherwise).
- `oracle_rri/oracle_rri/lightning/lit_trainer_factory.py`
  - `TrainerFactoryConfig` creates a `pl.Trainer` and optionally a `WandbLogger`.
  - Uses PL 1.9-compatible precision default (`"32"`).
- `oracle_rri/oracle_rri/lightning/__init__.py` exports the public surface.

### W&B + shared enums

- `oracle_rri/oracle_rri/configs/wandb_config.py`
  - `WandbConfig` copied/adapted from `external/doc_classifier` (project default changed to `"oracle-rri"`).
- `oracle_rri/oracle_rri/utils/schemas.py`
  - `Stage` enum copied/adapted (HuggingFace `datasets` import is optional).

### PathConfig update

- `oracle_rri/oracle_rri/configs/path_config.py`
  - Added `PathConfig.wandb` (`.logs/wandb`) for W&B artifact directory.

### Script entrypoint

- `oracle_rri/scripts/train_vin_lightning.py`
  - CLI entrypoint for training/validation/testing via Lightning.
  - Writes run metadata to `<out-dir>/config.json` and binner to `<out-dir>/rri_binner.json`.

### Tests

- `oracle_rri/tests/integration/test_vin_lightning_real_data.py`
  - Real-data smoke test that runs `trainer.fit(...)` with `fast_dev_run=True`.
  - Skips if EVL assets / ASE data / PyTorch3D / PL are missing.

### Dependencies

`oracle_rri/pyproject.toml`:

- Added `pytorch-lightning==1.9.5` and `wandb>=0.17.0`.
- Kept `torchmetrics==0.10.1` pinned because `projectaria-atek==1.0.0` depends on it.
  - This is why we did **not** use Lightning 2.x (it requires newer torchmetrics).

## How to run

From `oracle_rri/`:

```bash
uv run python scripts/train_vin_lightning.py --stage train --max-steps 5 --fit-snippets 2 --max-candidates 8 --device auto
```

By default, `scripts/train_vin_lightning.py` enables `RichModelSummary(max_depth=-1)` and disables Lightning’s
default model summary (`enable_model_summary=False`) so you see the full module tree once at startup.

Enable W&B:

```bash
uv run python scripts/train_vin_lightning.py --use-wandb --wandb-project oracle-rri --wandb-name my-run
```

Validate/test from a checkpoint:

```bash
uv run python scripts/train_vin_lightning.py --stage val --ckpt-path <path/to/*.ckpt>
```

## Notes / caveats

- PyTorch3D rasterizer overflow spam (“Bin size was too small…”): fixed by defaulting `Pytorch3DDepthRendererConfig.bin_size=0` (naive rasterization). If you want binned rasterization for speed, set `bin_size` to a non-zero value and tune `max_faces_per_bin` as needed.
- Lightning “ambiguous batch_size” warning: fixed by passing `batch_size=1` to all `self.log(...)` calls in `VinLightningModule`.
- `VinLightningModule.training_step` can return `None` when **no candidate is inside the EVL voxel bounds** (same masking logic as `scripts/train_vin.py`). In practice this means:
  - candidate generation + EVL voxel extent must be aligned well for stable training throughput,
  - otherwise Lightning will skip optimizer steps for those batches.
- The current setup keeps **online** oracle label generation (very expensive). For real training, we should add a caching/precompute stage and replace the online labeler in the DataModule.

## Suggested next steps (not implemented)

- Add a cached dataset of `(efm, candidates, rri, stage)` so DataLoader can use workers and avoid repeated renders.
- Add ranking metrics (Spearman / top-k recall) into `VinLightningModule` (tracked in `docs/contents/impl/vin_nbv.qmd`).
- Consider upgrading Lightning to 2.x only if/when the `projectaria-atek` ↔ `torchmetrics` pin allows it.
