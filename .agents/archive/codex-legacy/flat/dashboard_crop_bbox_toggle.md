Task: Remove redundant "Show crop bbox" sidebar toggle in Streamlit dashboard.

Context:
- Data cropping is controlled by `crop mesh` (dataset_config_ui) and the "Crop bbox" layer selection in `plot_options_from_ui`.
- Extra checkbox in `dashboard/app.py` did not affect cropping; it only gated passing the crop margin to the plot, leading to confusing no-op behaviour.

Changes:
- Deleted the unused sidebar checkbox and always forward `mesh_crop_margin_m` from dataset config to `render_data_page`.
- Verified formatting with `ruff format` and lint with `ruff check`.
- Ran `pytest oracle_rri/tests/test_mesh_cropping.py -q` (passes).

Notes:
- The "Crop bbox" layer still controls visibility of the crop box overlay; mesh cropping remains driven by dataset config.
