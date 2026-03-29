# RRI binning data lookup (2026-01-02)

## Finding
- The RRI binning panel uses `Path(...).expanduser()` on user input, so relative paths are resolved against the process CWD.
- Training/binner artifacts are saved via `PathConfig().resolve_artifact_path(...)`, which anchors `.logs/...` under the project root (`/home/jandu/repos/NBV`), regardless of CWD.
- When Streamlit is launched from a different directory (e.g., `oracle_rri/` or elsewhere), `.logs/vin/...` resolves to the wrong location and the panel reports missing files even though they exist under the project root.

## Evidence
- Files exist at `/home/jandu/repos/NBV/.logs/vin/rri_binner_fit_data.pt` and `/home/jandu/repos/NBV/.logs/vin/rri_binner.json`.
- `oracle_rri/oracle_rri/app/panels/rri_binning.py` uses direct `Path(...)` resolution.
- `oracle_rri/oracle_rri/rri_metrics/rri_binning.py` and training code use `PathConfig.resolve_artifact_path`, which resolves relative to project root.

## Suggestions
- Run Streamlit from the project root, or enter absolute paths in the panel.
- Code fix (if desired): resolve panel paths via `PathConfig().resolve_under_root(..., create_parent=False)` / `resolve_artifact_path(..., create_parent=False)` so the panel always aligns with the project root.

## Change applied
- Updated `oracle_rri/oracle_rri/app/panels/rri_binning.py` to resolve user-provided paths via `PathConfig().resolve_artifact_path(..., create_parent=False)` with suffix validation, so `.logs/...` always anchors to the project root regardless of CWD.

## Tests
- `uv run ruff format oracle_rri/oracle_rri/app/panels/rri_binning.py`
- `uv run ruff check oracle_rri/oracle_rri/app/panels/rri_binning.py`
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/vin/test_rri_binning.py`
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/integration/test_frames_real_data.py` (FAILED)

### Failure details
- `AseEfmDatasetConfig` rejects `verbose=False` as an extra field in `oracle_rri/tests/integration/test_frames_real_data.py`.

## Update (2026-01-02)
- Added UI toggles in `oracle_rri/oracle_rri/app/panels/rri_binning.py` to show quantile edge lines and optional bin midpoint lines in the histogram.
- Edges render as thicker gold solid lines; midpoints render as thinner cyan dotted lines.

## Tests
- `uv run ruff format oracle_rri/oracle_rri/app/panels/rri_binning.py`
- `uv run ruff check oracle_rri/oracle_rri/app/panels/rri_binning.py`
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/vin/test_rri_binning.py`

## Update (2026-01-02, quantile axis)
- Added a "Quantile-normalized x-axis" toggle in the RRI binning panel. When enabled, RRI values and the edge/midpoint markers are mapped via the empirical CDF so quantile edges are equally spaced on the x-axis.

## Tests
- `uv run ruff format oracle_rri/oracle_rri/app/panels/rri_binning.py`
- `uv run ruff check oracle_rri/oracle_rri/app/panels/rri_binning.py`
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/vin/test_rri_binning.py`
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/integration/test_frames_real_data.py` (FAILED)

### Failure details
- `AseEfmDatasetConfig` rejects `verbose=False` as an extra field in `oracle_rri/tests/integration/test_frames_real_data.py`.

## Update (2026-01-02, quantile axis tick labels)
- When using the quantile-normalized x-axis, the histogram now shows x-axis tick labels in **original RRI values** (inverse CDF mapping), while positions remain quantile spaced.

## Tests
- `uv run ruff format oracle_rri/oracle_rri/app/panels/rri_binning.py`
- `uv run ruff check oracle_rri/oracle_rri/app/panels/rri_binning.py`
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/vin/test_rri_binning.py`

## Update (2026-01-02, log y-axis)
- Replaced log1p count transformation with a true log-scaled y-axis in the RRI binning histogram and label distribution plot. Zero-count bins are masked when log scale is enabled.

## Tests
- `uv run ruff format oracle_rri/oracle_rri/app/panels/rri_binning.py`
- `uv run ruff check oracle_rri/oracle_rri/app/panels/rri_binning.py`
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/vin/test_rri_binning.py`
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/integration/test_frames_real_data.py` (FAILED)

### Failure details
- `AseEfmDatasetConfig` rejects `verbose=False` as an extra field in `oracle_rri/tests/integration/test_frames_real_data.py`.

## Update (2026-01-02, offline stats PathConfig)
- `offline_cache_utils._collect_offline_cache_stats` now resolves TOML paths via `PathConfig.resolve_config_toml_path(..., must_exist=False)` and warns + falls back to default config when missing/invalid instead of raising.

## Tests
- `uv run ruff format oracle_rri/oracle_rri/app/panels/offline_cache_utils.py`
- `uv run ruff check oracle_rri/oracle_rri/app/panels/offline_cache_utils.py`
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/integration/test_frames_real_data.py` (FAILED)

### Failure details
- `AseEfmDatasetConfig` rejects `verbose=False` as an extra field in `oracle_rri/tests/integration/test_frames_real_data.py`.

## Update (2026-01-02, offline stats log y-axis)
- Offline stats histograms now use true log-scaled y-axes instead of log1p-transformed counts. Added plotly log scaling helper and updated matplotlib hist helper to set yscale log and mask zeros.

## Tests
- `uv run ruff format oracle_rri/oracle_rri/app/panels/offline_stats.py oracle_rri/oracle_rri/app/panels/plot_utils.py`
- `uv run ruff check oracle_rri/oracle_rri/app/panels/offline_stats.py oracle_rri/oracle_rri/app/panels/plot_utils.py`
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/integration/test_frames_real_data.py` (FAILED)

### Failure details
- `AseEfmDatasetConfig` rejects `verbose=False` as an extra field in `oracle_rri/tests/integration/test_frames_real_data.py`.

## Update (2026-01-02, VIN diagnostics config selector)
- VIN diagnostics now lists TOML configs from `PathConfig.configs_dir` and checkpoints from `PathConfig.checkpoints` in selectboxes (with a `(none)` option for configs).

## Tests
- `uv run ruff format oracle_rri/oracle_rri/app/panels/vin_diagnostics.py`
- `uv run ruff check oracle_rri/oracle_rri/app/panels/vin_diagnostics.py`
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/vin/test_vin_diagnostics.py` (FAILED: no such test file)
