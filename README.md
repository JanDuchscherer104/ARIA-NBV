# ARIA-NBV

ARIA-NBV develops quality-driven next-best-view planning for egocentric indoor
reconstruction. The current thesis path focuses on ASE/EFM3D snippets, oracle
Relative Reconstruction Improvement (RRI), VIN-style candidate scoring, and
bounded target-aware rollout evaluation.

The public documentation is published at
<https://janduchscherer104.github.io/ARIA-NBV/>.

## Current Focus

- Freeze the VIN offline-store, split, pose-frame, and oracle RRI contracts.
- Validate the available VIN offline store through `offline_only.toml` and VIN
  diagnostics.
- Establish a reproducible one-step VIN baseline before target-conditioned RRI
  and multi-step rollouts.
- Keep continuous control, simulator-backed RL, VLM planning, and real-device
  deployment as stretch work until the roadmap gates justify them.

## Documentation Map

- [Roadmap](docs/contents/roadmap.qmd): dated thesis milestones through
  2026-09-30.
- [Research Questions](docs/contents/questions.qmd): target-conditioned NBV
  thesis questions and evaluation criteria.
- [Implementation Notes](docs/contents/impl/overview.qmd): data, RRI, VIN, and
  diagnostics entry points.
- [API Reference](docs/reference/index.qmd): generated `aria_nbv` reference
  pages.
- [Setup](docs/contents/setup.qmd): environment and dependency notes.

## Important CLI Commands

Run package commands from `aria_nbv/` with the uv-managed environment.

```sh
# Launch the Streamlit app
uv run nbv-st

# Train the VIN model
uv run nbv-train --config-path ../.configs/offline_only.toml

# Single forward pass that summarizes VIN outputs
uv run nbv-summary --config-path ../.configs/offline_only.toml

# Hyperparameter sweep with Optuna
uv run nbv-optuna --config-path ../.configs/sweep_config.toml

# Fit and save the ordinal binner only
uv run nbv-fit-binner --config-path ../.configs/offline_only.toml

# Dump resolved config to stdout
uv run nbv-cli --run-mode dump-config --config-path ../.configs/offline_only.toml
```

## Offline Data

Offline VIN training reads the immutable store configured by
`.configs/offline_only.toml`. By default the store resolves to the `vin_offline`
store under the configured `offline_cache_dir` and is consumed through the
`kind = "offline"` datamodule source.

## ASE Downloader

`nbv-downloader` is a thin CLI over `aria_nbv.data.downloader.ASEDownloaderConfig`.
It supports `download` and `list` modes.

```sh
# List available scenes
uv run nbv-downloader -m list

# Download meshes + ATEK shards
uv run nbv-downloader -m download --ns 10 --max-shards 2

# Download only ATEK shards
uv run nbv-downloader -m download --ns 10 --skip-meshes

# Download only meshes
uv run nbv-downloader -m download --ns 10 --skip-atek
```
