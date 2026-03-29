# VIN Streamlit Diagnostics Plan (context + proposal)

## Context gathered
- VIN debug data is exposed by `VinModel.forward_with_debug` as `VinForwardDiagnostics`.
- Streamlit app pages are defined in `oracle_rri/oracle_rri/app/app.py` and rendered via functions in `oracle_rri/oracle_rri/app/panels.py`.
- VIN training orchestration lives in `oracle_rri/oracle_rri/lightning/aria_nbv_experiment.py`, which can construct a `VinLightningModule` + `VinDataModule` without using the Streamlit pipeline controller.
- `VinDataModule.iter_oracle_batches` yields `VinOracleBatch` (candidate poses, p3d cameras, oracle RRI) but not the `EfmSnippetView`.
- Existing VIN plotting helpers (`oracle_rri/oracle_rri/vin/plotting.py`) can generate matplotlib figures from `VinForwardDiagnostics`.

## Proposed visualizations (VIN sanity checks)
- Pose descriptor sanity: distributions of radius, center direction (u), forward direction (f), and view alignment (dot(f, -u)).
- Voxel pose sanity: voxel origin/forward in reference frame + alignment vs candidate pose statistics.
- Scene field sanity: per-channel stats (min/mean/max), mid-slice visualizations for `field_in` and `field`.
- Frustum sampling sanity: token-valid fraction per candidate; 2D grids per depth plane for a selected candidate to verify NDC→world→voxel mapping.
- Feature norms: L2 norms of `pose_enc`, `global_feat`, `local_feat`, `feats` per candidate.
- Prediction sanity: scatter of `pred.expected_normalized` vs oracle `batch.rri` (if available) + candidate_valid mask.
- Optional: reuse `plot_vin_encodings_from_debug` for the conceptual SH/radius plots.

## Integration outline
- Add a new Streamlit page (e.g., `VIN Diagnostics`) with its own state key to avoid coupling to the existing app pipeline.
- Use `AriaNBVExperimentConfig` to instantiate `VinLightningModule` + `VinDataModule` and pull a `VinOracleBatch` via `iter_oracle_batches`.
- Run `module.vin.forward_with_debug` on the batch under `torch.no_grad()` and render the diagnostics.
- Provide minimal UI controls: stage (train/val/test), batch index (skip N), EVL cfg/ckpt path overrides, and toggles for heavy plots (field slices, torchsummary).
- Cache the experiment/module and last debug output keyed by a config signature to avoid repeated heavy computation.

## Open questions / risks
- Do we want to expose a TOML path to load a full `AriaNBVExperimentConfig` instead of mirroring its fields in the UI?
- Field slice visualization needs a decision on slicing axis and channel ordering.
- `VinOracleBatch` does not include the `EfmSnippetView`; if we want mesh/trajectory overlays, we need to extend the batch or separately fetch the snippet.
- The EVL model cfg/ckpt must exist; page should show a clear error if missing.
