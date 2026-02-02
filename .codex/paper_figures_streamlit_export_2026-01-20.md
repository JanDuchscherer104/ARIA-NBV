# Paper housekeeping: oracle-label focus + figure export

## What changed

- Paper text now reflects current status: we compute **oracle per-candidate RRI + ordinal (CORAL) labels**, but do **not** claim a learned NBV policy yet.
- Added a concise **myopic/greedy** thought experiment (no planning horizon) in the problem formulation.
- Switched key paper figures to **Streamlit diagnostics** screenshots and added a compact **oracle-label configuration** table.

## Reproducible figure configuration

- Runnable TOML: `.configs/paper_figures_oracle_labeler.toml`
  - Contains the candidate generator + depth renderer settings used by the Streamlit screenshots (radius/elevation caps, collision checks, renderer z-range, etc.).
  - Recommendation: pin a specific snippet via `dataset.scene_ids` and `dataset.snippet_key_filter` for deterministic screenshots.

## Figure export helper (non-Streamlit)

- Script: `oracle_rri/scripts/export_paper_figures.py`
  - Runs the oracle labeler on one snippet and exports Plotly figures to `docs/figures/app/` by default.
  - Uses a deterministic filename tag derived from `(kappa, min_radius, max_radius)` for the candidate-frusta plot.
  - Plotly PNG export needs `kaleido`; otherwise it still writes `.html`.

## Typst ↔ Python/Jupyter integration (recommended workflow)

- Typst itself does not execute Python; use one of:
  1) **External figure generation**: run Python (script/notebook) → save images into `docs/figures/...` → include in `.typ`.
  2) **Quarto** for literate Python + Typst output (`format: typst`, `engine: jupyter`) when you want a notebook-like figure pipeline. This is best for plots/tables that should be regenerated automatically.

## Local compile note

- In this sandbox, the `typst` snap cannot run (permission/capability issue). Compile locally instead.

