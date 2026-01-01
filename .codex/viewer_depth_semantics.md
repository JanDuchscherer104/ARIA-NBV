# Depth & semantic viewer update (2025-11-23)

- Added optional `distance_m` / semantic tensors to `EfmCameraView` so ATEK/EFM WDS samples expose metric depth and future semantic maps directly on camera views.
- Plotting grid now auto-collects modalities (RGB/SLAM + depth + semantics when present), colourises depth with viridis, and maps labels to a tab20 palette; first/last grids resize dynamically.
- Streamlit Data page now uses the new grid and surfaces per-modality availability; current `ase_efm` snippets provide `rgb/distance_m` only, so depth shows for RGB and a warning notes missing semantic maps.
- Sanity: loaded a real snippet via `AseEfmDatasetConfig(is_debug=True)` and ran `collect_frame_modalities` + `plot_first_last_frames` without errors; depth panels rendered successfully.
