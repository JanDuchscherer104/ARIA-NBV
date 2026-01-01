# VIN pipeline refactor (modular)

## Summary
- Added a modular VIN pipeline in `oracle_rri/oracle_rri/vin/pipeline.py` that decomposes the forward pass into pose encoding, scene-field construction, frustum sampling, and feature assembly.
- Introduced `VinPipelineConfig` as a **single shared config** passed to all components.
- Added an integration equivalence test in `tests/vin/test_vin_refactor_equivalence.py` that compares the refactor against `VinModel` on real data (seeded to align random init).

## Tests
```
ruff format oracle_rri/oracle_rri/vin/pipeline.py tests/vin/test_vin_refactor_equivalence.py
ruff check oracle_rri/oracle_rri/vin/pipeline.py tests/vin/test_vin_refactor_equivalence.py
oracle_rri/.venv/bin/python -m pytest -q tests/vin/test_vin_refactor_equivalence.py
```

## Notes
- The equivalence test uses `torch.manual_seed(0)` before each model initialization to align random weights for field projection + CORAL head.
- The test imports `VinModelConfig` directly from `oracle_rri.vin.model` to avoid `__init__` side effects during collection.
- `oracle_rri/oracle_rri/vin/model.py` was left unchanged for A/B testing.

## Additional component tests
- Added `tests/vin/test_vin_pipeline_components.py` covering:
  - Synthetic scene-field channel correctness.
  - Candidate validity thresholding.
  - Feature assembler output shapes.
  - Real-data pose-encoder invariants.
  - Real-data frustum sampling shapes + valid fraction.

Tests run:
```
ruff format tests/vin/test_vin_pipeline_components.py
ruff check tests/vin/test_vin_pipeline_components.py
oracle_rri/.venv/bin/python -m pytest -q tests/vin/test_vin_pipeline_components.py
```

Result: **5 passed** (CPU, ~53s, 2 dependency warnings).

## Frustum sampling geometry check
- Added `test_frustum_sampling_matches_backproject_synthetic_camera` in `tests/vin/test_vin_pipeline_components.py`.
  - Compares `VinFrustumSampler.build_points_world` against `_backproject_depths_p3d_batch` on a synthetic pinhole camera.

Tests run:
```
ruff format tests/vin/test_vin_pipeline_components.py
ruff check tests/vin/test_vin_pipeline_components.py
oracle_rri/.venv/bin/python -m pytest -q tests/vin/test_vin_pipeline_components.py
```

Result: **6 passed** (CPU, ~38s, 2 dependency warnings).

## Update: LFF pose encoding + diagnostics UI
- Replaced legacy SH pose encoder with `LearnableFourierFeatures` (LFF) in `VinModel` and `VinPipeline`, keeping SH as a legacy alias (`pose_encoder_sh` -> `pose_encoder_lff`).
- Updated plotting utilities to support LFF weight inspection and LFF Fourier feature heatmaps.
- Simplified VIN diagnostics “FF Encodings” tab to always show actual candidate encodings (no max-candidate/pose-dim sliders, no legacy SH toggle).

Tests run:
```
ruff format oracle_rri/oracle_rri/app/panels.py
ruff check oracle_rri/oracle_rri/app/panels.py
oracle_rri/.venv/bin/python -m pytest -q tests/vin/test_vin_model_integration.py::test_vin_forward_on_real_snippet_cpu
```

## Update: pose-conditioned global pooling + coverage-aware features
- Added pose-conditioned attention pooling (`PoseConditionedGlobalPool`) and configurable global pooling mode (`mean`, `mean_max`, `attn`) in `oracle_rri/oracle_rri/vin/model.py`.
- Added unknown-token pooling for invalid frustum samples and appended `(valid_frac, 1-valid_frac)` features to candidate embeddings.
- Exposed `valid_frac` in `VinPrediction`/`VinForwardDiagnostics` and switched Lightning loss to coverage-weighted CORAL (no hard candidate masking).
- Updated docs diagram/text in `docs/contents/impl/vin_nbv.qmd`.

Tests run:
```
ruff format oracle_rri/oracle_rri/vin/model.py oracle_rri/oracle_rri/vin/types.py oracle_rri/oracle_rri/vin/pipeline.py oracle_rri/oracle_rri/lightning/lit_module.py tests/vin/test_vin_global_pool.py tests/vin/test_vin_refactor_equivalence.py
ruff check oracle_rri/oracle_rri/vin/model.py oracle_rri/oracle_rri/vin/types.py oracle_rri/oracle_rri/vin/pipeline.py oracle_rri/oracle_rri/lightning/lit_module.py tests/vin/test_vin_global_pool.py tests/vin/test_vin_refactor_equivalence.py
oracle_rri/.venv/bin/python -m pytest -q tests/vin/test_vin_global_pool.py
oracle_rri/.venv/bin/python -m pytest -q tests/vin/test_vin_model_integration.py::test_vin_forward_on_real_snippet_cpu
```

Doc validation:
```
python - <<'PY'
from pathlib import Path
text = Path('docs/contents/impl/vin_nbv.qmd').read_text()
start = text.find('```{mermaid}')
end = text.find('```', start + len('```{mermaid}'))
Path('/tmp/diagram.mmd').write_text(text[start + len('```{mermaid}') : end].strip() + '\n')
PY
npx -y @mermaid-js/mermaid-cli -i /tmp/diagram.mmd -o /tmp/diagram.svg
quarto render docs/contents/impl/vin_nbv.qmd --to html
```

## Update: pose encoding research notes (Dec 2025)
- Added a pose-encoding research subsection to `docs/contents/impl/vin_nbv.qmd` with a CPU-only synthetic experiment comparing encodings.
- Recommended switching to a simpler continuous pose vector `[t, R6d]` and linked to Zhou et al. (2019); added bib entry in `docs/references.bib`.

Doc render:
```
quarto render docs/contents/impl/vin_nbv.qmd --to html
```

## Update: R6D extraction + learned scaling docs
- Added a concise section to `docs/contents/impl/vin_nbv.qmd` describing PoseTW → R6D extraction via `matrix_to_rotation_6d` and learned per-group scaling for LFF inputs.
- Added PyTorch3D transforms doc citation in `docs/references.bib`.

Doc render:
```
quarto render docs/contents/impl/vin_nbv.qmd --to html
```

## Update: CORAL loss + binning docs
- Expanded CORAL theory section in `docs/contents/impl/vin_nbv.qmd` with explicit loss formula and random-classifier baseline (K=15 → 9.70).
- Added binning implementation details (quantiles, fallback edges, resumable fit) and inserted Seaborn threshold plot.
- Updated `oracle_rri/scripts/plot_vin_binning.py` to use Seaborn and emit `vin_rri_thresholds.png`.
- Added seaborn to `oracle_rri/pyproject.toml` and synced venv (note: `uv sync --extra dev --extra notebook` removed xformers).

Plots generated (CPU):
```
oracle_rri/.venv/bin/python oracle_rri/scripts/plot_vin_binning.py --device cpu --num-snippets 4
```

Doc render:
```
quarto render docs/contents/impl/vin_nbv.qmd --to html
```

## Update: plot_vin_binning supports logs
- Extended `oracle_rri/scripts/plot_vin_binning.py` with `--use-logs`/`--rri-npy`/`--edges-npy`/`--binner-json` flags to plot from precomputed binning artifacts.
- Ran the script against `.logs/vin` to regenerate `vin_rri_binning.png` and `vin_rri_thresholds.png`.

Plot command:
```
oracle_rri/.venv/bin/python oracle_rri/scripts/plot_vin_binning.py --use-logs --logs-dir .logs/vin --out-dir docs/figures/impl/vin
```

## Update: Phase 2 pose encoding (t + R6d)
- Added pose encoding mode switch (`shell_lff` vs `t_r6d_lff`) with per-group learned scaling in `oracle_rri/oracle_rri/vin/model.py`.
- Default pose encoder input dim updated to 9 for the new `[t, R6d]` vector; added validators for mode/dim consistency.
- Exposed `pose_vec`/`voxel_pose_vec` in `VinForwardDiagnostics` and updated plotting to use them when available.
- Kept pipeline debug compatible by passing `None` for new fields.

Tests run (CPU):
```
ruff format oracle_rri/oracle_rri/vin/model.py oracle_rri/oracle_rri/vin/types.py oracle_rri/oracle_rri/vin/plotting.py oracle_rri/oracle_rri/vin/pipeline.py tests/vin/test_vin_refactor_equivalence.py
ruff check oracle_rri/oracle_rri/vin/model.py oracle_rri/oracle_rri/vin/types.py oracle_rri/oracle_rri/vin/plotting.py oracle_rri/oracle_rri/vin/pipeline.py tests/vin/test_vin_refactor_equivalence.py
oracle_rri/.venv/bin/python -m pytest -q tests/vin/test_vin_model_integration.py::test_vin_forward_on_real_snippet_cpu
oracle_rri/.venv/bin/python -m pytest -q tests/vin/test_vin_plotting.py::test_vin_plotting_helpers_cpu
```
