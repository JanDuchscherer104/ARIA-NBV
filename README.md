# Seminar: Next Best View Estimation (NBV)

See the [GitHub Pages](https://janduchscherer104.github.io/NBV/) for more information.

# Important CLI Commands

## Entry points (project.scripts)

```sh
# Launch the streamlit app
uv run nbv-st
# Train NBV model
uv run nbv-train --config-path .configs/offline_only.toml
# Single forward pass that summarizes VIN outputs
uv run nbv-summary --config-path .configs/offline_only.toml
# Hyperparameter sweep with Optuna
uv run nbv-optuna --config-path .configs/sweep_config.toml
# Fit and save the ordinal binner only
uv run nbv-fit-binner --config-path .configs/offline_only.toml
# Dump resolved config to stdout
uv run nbv-cli --run-mode dump-config --config-path .configs/offline_only.toml
```

## Cache tooling

```sh
uv run nbv-cache-samples --config-path .configs/offline_only.toml
uv run nbv-cache-vin-snippets --config-path .configs/offline_only.toml
```

The VIN snippet cache is derived from the offline oracle cache and includes
collapsed semi-dense point clouds and snippet metadata needed by VIN.


This writes CSVs for run metadata, summaries, configs, histories, dynamics, and
a manifest of local train/val figure images under `.logs/wandb/analysis`.

## ASE downloader

`nbv-downloader` is a thin CLI over `aria_nbv.data.downloader.ASEDownloaderConfig`.
It supports two modes: `download` and `list`.

```sh
# List available scenes (first 10)
uv run nbv-downloader -m list
```
```
[DownloaderCLI]: ============================================================
[DownloaderCLI]: ASE Dataset - Available Scenes
[DownloaderCLI]: ============================================================
[DownloaderCLI]: Total scenes with GT meshes: 100
[DownloaderCLI]:
Showing 100 scenes:
[DownloaderCLI]:   Scene 83550: 49 shards
<...>
[DownloaderCLI]:   Scene 84185: 3 shards
[DownloaderCLI]:
Total shards (all GT-mesh scenes): 1641
[DownloaderCLI]: ============================================================
```

```sh
# Download meshes + ATEK shards (10 scenes, 2 shards per scene)
uv run nbv-downloader -m download --ns 10 --max-shards 2

# Download only ATEK shards (skip meshes)
uv run nbv-downloader -m download --ns 10 --skip-meshes

# Download only meshes (skip ATEK)
uv run nbv-downloader -m download --ns 10 --skip-atek
```

Notes:
- `--ns`/`--n_scenes` controls number of scenes (0 = all).
- `--max-shards` (or `--ms`) caps shards per scene.
- `--c` selects ATEK config (`efm_eval` default; `efm`, `cubercnn`, `cubercnn_eval`).
