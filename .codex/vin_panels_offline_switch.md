# Task: VIN diagnostics panel offline/online switch

## Summary
- Added a data source selector to the VIN diagnostics sidebar to switch between online oracle labeler batches and offline cache batches.
- When offline is selected, the panel constructs a cache-backed `VinDataModuleConfig` using the provided cache dir + map_location.
- `VIN` debug execution now supports offline batches by passing cached `backbone_out` and a dummy `efm` payload.
- Geometry tab is guarded to require raw EFM snippets; offline batches show an info message instead.

## Validation
- `ruff format oracle_rri/oracle_rri/app/panels.py`
- `ruff check oracle_rri/oracle_rri/app/panels.py`
- `uv run pytest tests/integration/test_vin_real_data.py -q` (from `oracle_rri/`)

## Notes
- Offline cache batches must include `backbone_out`; otherwise VIN debug will raise an error.

## Update (offline cache fixes)
- Disabled EVL backbone initialization in VIN diagnostics when offline cache is selected by setting `vin.backbone` / `vin_v2.backbone` to `None` in the panel config.
- Ensured VIN models can operate without a backbone by allowing `backbone: Optional[EvlBackboneConfig]` and requiring cached `backbone_out` during forward.
- Added device alignment in VIN forward paths and in the panel runner to prevent CPU/GPU mismatch when using cached data.
- Guarded VIN v2 diagnostics (no frustum/token fields) to avoid attribute errors; v1-only plots show an info message instead.

## Validation (after fixes)
- `uv run pytest tests/integration/test_vin_real_data.py -q` (from `oracle_rri/`)

## Update (VIN v2 summary guard)
- Guarded summary feature norms so v2 diagnostics (no `local_feat`) do not crash.

## 2025-12-31: VIN summary tab output
- Added summarize_vin output display in VIN Summary tab (Streamlit).
- Summary cached in session state keyed by cfg + snippet + torchsummary options.
- UI uses expander with optional torchsummary depth; errors are surfaced instead of crashing.
- Integration test: `uv run pytest tests/integration/test_vin_real_data.py -q`.

## 2025-12-31: VIN summary render modes
- Summary tab now offers render modes: plain (ANSI stripped), rich HTML (ANSI -> HTML via rich), raw ANSI.
- Summary HTML cached in session state; regenerated when summary inputs change.
- Integration test: `uv run pytest tests/integration/test_vin_real_data.py -q`.

## 2025-12-31: Summary render simplified
- Removed render-mode selector; Summary tab now always strips ANSI and renders plain text.
- Dropped HTML/ANSI conversion cache state.
- Integration test: `uv run pytest tests/integration/test_vin_real_data.py -q`.

## 2025-12-31: VIN summary log-axis toggle
- Added log-scale toggle for Oracle RRI vs VIN expected scatter; filters non-positive values with info message.

## 2025-12-31: Evidence threshold control
- Added evidence threshold slider on VIN Backbone Evidence tab; passed into evidence plotting helpers.

## 2025-12-31: ANSI stripping fix
- Fixed ANSI regex for stripping summary output (ESC + '['), so tokens no longer leak into Streamlit.

## 2025-12-31: Feature modality breakdown plots
- Summary tab now shows feature dimension bars (pose/global/local) and scene field channel magnitude bars.
- Channel names use FIELD_CHANNELS_V2 when available, otherwise fallback to config or generic labels.

## 2025-12-31: Scene field channel histograms
- Replaced mean magnitude bars with per-channel |value| histograms (overlay).
- Added controls for max channels and histogram bin count.

## 2025-12-31: Scene field channel multiselect
- Replaced max-channel slider with multiselect for channel histogram selection.

## 2025-12-31: Scene field histogram log transform
- Added log1p toggle for scene field histogram values and updated x-axis label.

## 2025-12-31: Histogram log-count toggle
- Replaced log1p value transform with y-axis log-scale toggle for histogram counts.

## 2025-12-31: Summary plot info popovers
- Added concise ℹ️ popovers for summary tab plots (scatter, feature dims, field histograms, feature norms).

## 2025-12-31: Popover widget fix
- Replaced keyed popovers with label-based popovers (Streamlit popover has no key arg).
- Updated summary plot info popovers to use short labels.

## 2025-12-31: VIN v2 positional encoding plots
- Added pose-grid slice + pos_proj PCA plots and LFF response heatmaps + pose_enc PCA for VIN v2.
- Added pose_vec component histogram + input range controls for LFF response.
- Integration test: `uv run pytest tests/integration/test_vin_real_data.py -q`.

## 2025-12-31: Spectrogram-style plots
- Added spectrogram-style amplitude/phase heatmaps for LFF Fourier + MLP outputs and pos_grid positional encodings.
- Integrated into FF Encodings tab with log-amplitude toggles and line selection for pos_grid.
- Integration test: `uv run pytest tests/integration/test_vin_real_data.py -q`.

## 2025-12-31: Pos grid PCA axes overlay
- Added projected rig-frame axes overlay to pos_grid PCA plot, with toggle + scale slider.

## 2025-12-31: Empirical-only FF encodings
- Removed synthetic LFF sweeps and spectrograms; FF encodings now use actual pose_vec data.
- Added empirical LFF histograms + PCA for Fourier and MLP outputs.
- Integration test: `uv run pytest tests/integration/test_vin_real_data.py -q`.

## 2025-12-31: PoseConditionedGlobalPool docs
- Expanded docstring with conceptual explanation and explicit Q/K/V definitions.
- Integration test: `uv run pytest tests/integration/test_vin_real_data.py -q`.

## 2025-12-31: scene_field_channels for VIN v2
- Added VinModelV2Config.scene_field_channels with validation and default ['occ_pr'].
- _build_scene_field_v2 now returns only requested channels + aux map; counts_norm sampled separately for valid_frac.
- Integration test: `uv run pytest tests/integration/test_vin_real_data.py -q`.
