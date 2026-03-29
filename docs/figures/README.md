# Figures Layout

Use `docs/figures/` for web-facing and shared documentation assets.

Directory conventions:
- `branding/`: shared visual identity assets such as logos.
- `dataset/`: ASE dataset examples, histograms, and mesh snapshots.
- `candidate_generation/`: pose-sampling and candidate-view illustrations.
- `app/`: Streamlit and pipeline diagnostics screenshots.
- `app-paper/`: polished paper-facing composites derived from app outputs.
- `diagrams/`: authored architecture diagrams and their rendered exports.
- `efm3d/`, `atek/`, `scene-script/`, `coral/`, `offline_cache/`, `vin_v2/`, `wandb/`: topic-specific assets.

Keep generated publish artifacts out of this tree. Quarto HTML, `site_libs/`, `*_files/`, and other rendered site output belong under `docs/_site/` only.

`docs/typst/figures/` remains the Typst-local asset tree for paper/slides sources that need stable Typst-relative paths.
