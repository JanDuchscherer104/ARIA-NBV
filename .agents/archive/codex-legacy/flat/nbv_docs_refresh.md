NBV docs refresh (Nov 22, 2025)
--------------------------------

- Consolidated package overview: new `docs/contents/impl/aria_nbv_overview.qmd` replaces the old class_diag page, with dir tree, NBV flow mermaid, RRI equations, core module map, and architecture diagram. Legacy pages now point to it.
- Navigation: Implementations section lists `overview`, `aria_nbv_overview`, `data_pipeline_overview`, `rri_computation`; external libs live under `ext-impl/` (ATEK, EFM3D, ProjectAria, symbol index).
- Data pipeline doc rewritten to mirror the current Streamlit app:
  - Quickstart: `streamlit run -m oracle_rri.streamlit_app`.
  - Data page: AseEfmDatasetConfig defaults (mesh crop/decimate sliders, require_mesh), cache reset.
  - Candidate page: CandidateViewGeneratorConfig sliders, plotting controls, background run buttons.
  - Depth page: CandidateDepthRenderer + PyTorch3D settings, run flow.
  - Logging + inline Python console; CLI download commands kept brief; programmatic dataset example; updated mermaid.
- Index links fixed to new paths (`contents/impl/rri_computation.qmd`, `contents/ext-impl/*`).
- Rendering in this sandbox works with `--no-execute`; full `quarto check` fails only because Jupyter kernel logging/sockets are blocked by sandbox perms.

Next steps (if time permits): run full-site render in a non-restricted env; consider code-level rename from `oracle_rri` to `aria_nbv` to match docs; keep streamlit app examples in sync with candidate/depth configs when they change.
