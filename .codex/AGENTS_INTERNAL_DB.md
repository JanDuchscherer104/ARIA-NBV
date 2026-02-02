# AGENTS Internal Database (Codex)

Purpose: a compact, Codex-optimized knowledge base to keep the agent aligned with the NBV project goals, workflow, and tooling. Use this as the primary internal reference before taking actions.

## 0) Mission Snapshot
- Build an active Next-Best-View (NBV) planner for egocentric indoor scenes.
- Core metric: Relative Reconstruction Improvement (RRI) from oracle labels.
- Backbone: frozen EFM3D/EVL; head: lightweight RRI predictor (VIN-style).
- Dataset: Aria Synthetic Environments (ASE) with GT meshes, semi-dense SLAM points, depth.

## 1) Non-Negotiable Workflow
- Always run: `make context` at start of a task (writes `.codex/codex_make_context.md` with an embedded source index); use `rg` to extract relevant sections. Run `make context-dir-tree` only if you need a standalone tree printout.
- Always read: `docs/index.qmd`, `docs/contents/todos.qmd`.
- Never run `git restore`. Never use `git reset --hard` unless user explicitly asks.
- Assume environment is NOT fully working unless verified.
- Never make changes outside user scope. Ask if unsure.

## 2) Environment & Tooling
- Python venv: `oracle_rri/.venv` (Python 3.11). Use `uv` or the venv python.
- If venv missing: `UV_PYTHON=/home/jandu/miniforge3/envs/aria-nbv/bin/python uv sync --extra dev --extra notebook --extra pytorch3d`.
- Tests: `uv run pytest <path>` (avoid system python).
- Lint/format: `ruff format <file>` then `ruff check <file>`.
- `make context` writes `.codex/codex_make_context.md` (includes an embedded source index) with the full (unfiltered) Mermaid UML and prints a filtered UML to stdout; filter printed UML namespaces with `CONTEXT_MERMAID_EXCLUDE` (comma-separated, default `data.downloader,vin.experimental,app`).

## 3) Codex Tooling (Preferred Usage)
- File search: `rg` and `rg --files`.
- Use `apply_patch` for single-file edits; avoid for auto-generated files.
- Use MCP code-index tools for symbol lookup and file summaries when needed.
- Use Context7 for external docs; resolve library ID first unless provided.
- Web search required for up-to-date / uncertain / external facts.
- Skill helper scripts live under `.codex/skills/aria-nbv-context/scripts` with convenience wrappers in `scripts/`; use `scripts/nbv_typst_includes.py` or `rg "#include"` on `docs/typst/**` if needed.

## 4) Code Conventions (Must Follow)
- Config-as-factory pattern; instantiate via `.setup_target()`.
- Use `pathlib.Path` for paths.
- Use `PoseTW` and `CameraTW` for poses/cameras (no raw matrices).
- Use `Console` from `oracle_rri.utils` for logging.
- Use `Field(default_factory=...)` for computed defaults; avoid `Field(default=<callable>)`.
- Type hints everywhere; use `TYPE_CHECKING` for type-only imports.
- Prefer `Enum` for categories; `match-case` for multi-branch logic.

## 5) Documentation & Typst / Quarto
- Docs: `docs/references.bib` is the bibliography source of truth.
- Typst paper/slides compile expects root `docs` (so `/typst/...`, `/figures/...`, `/references.bib` resolve). Use `cd docs && typst compile typst/paper/main.typ --root .` or `cd docs && typst compile typst/slides/slides_4.typ --root .`. If `typst` is the snap wrapper and fails under sandboxing, call `/snap/typst/current/bin/typst` directly.
- Typst frame symbols: use `#symb.frame.*` (e.g., `#symb.frame.w`, `#symb.frame.r`, `#symb.frame.cq`, `#symb.frame.v`) inside math and in `#T(symb.frame.A, symb.frame.B)`; do not use legacy `fr_*` helpers.
- Single bibliography source of truth: `docs/references.bib`. Do not mirror copies under `docs/_shared`, `docs/typst`, or `docs/typst/paper`.
- Slides should import the shared `docs/typst/shared/template.typ` (and `docs/typst/shared/notes.typ`) for consistent slide overrides; avoid re-importing the base template directly unless needed.
- Streamlit-style paper figures can be regenerated with `oracle_rri/scripts/export_paper_figures.py` using `.configs/paper_figures_oracle_labeler.toml` and output to `docs/figures/app`.
- Typst builds with `--root docs` require all `/figures/...` assets under `docs/figures/`; some legacy paper images may only exist under `docs/typst/figures/` and need mirroring.
- For diagrams: validate Mermaid; use `{mermaid}` fences and `<br/>` in labels.
- MMDC flowcharts: wrap class/function names in `\\texttt{...}`; keep text vs symbols on separate lines (single `$$...$$` block with `\\begin{array}{c}`); use `\\mathcal{}`/`\\mathbf{}` symbols consistent with `docs/typst/shared/macros.typ`; for edge labels with math use quoted labels `A -- "$$...$$" --> B`.
- For doc changes: run `quarto render <file>.qmd --to html`.
- Add bib entries to `docs/references.bib` when introducing key concepts.

## 6) Must-Use Key Files
- `docs/index.qmd`
- `docs/contents/todos.qmd`
- `docs/contents/impl/*` for implementation context.
- `docs/contents/ext-impl/*` for EFM3D/ATEK references.
- `notebooks/ase_oracle_rri_simplified.ipynb` (oracle reference pipeline).

## 7) VIN / RRI Core Facts (From Current Paper/Docs)
- Oracle RRI = relative reduction of bidirectional point↔mesh error after adding candidate.
- Candidate point clouds rendered from GT mesh; fused with semi-dense SLAM points.
- Mesh is cropped to AABB covering candidate + semi-dense points to reduce compute.
- Training: CORAL ordinal regression with quantile-based bin edges.
- VIN v2 uses EVL voxel evidence + pose/trajectory features + semidense projection cues.
- VINv3: voxel gating removed; keep FiLM-only modulation. `use_voxel_valid_frac_gate` must be False (raises if True).
- VINv3 now concatenates semidense projection stats into the head and adds a tiny CNN over projection grids (occupancy + depth mean/std). Default semidense grid size is 24; `semidense_cnn_enabled` gates the CNN (diagnostics tab renders CNN grid maps when enabled).
- EVL voxel grid is local (~4m^3); add OOB validity scalars to avoid over-trust.
- EVL voxel extent default in `EvlBackbone`: `[-2, 2, 0, 4, -2, 2]` (4 m cube).
- Candidate generation applies `rotate_yaw_cw90` (90° local +Z twist) to the reference pose before sampling; this is part of the physical sampling frame (not just display) and avoids an azimuth/elevation swap under the LUF sampling assumption.
- PyTorch3D depth rendering uses `in_ndc=false` and returns metric z-buffer depth; backprojection converts pixel centers to PyTorch3D NDC and unprojects with `from_ndc=True` to match rasterizer coordinates.
- Project Aria semi-dense fields: `inv_dist_std` is the standard deviation of inverse distance (σ_ρ, m⁻¹) and `dist_std` is the standard deviation of distance (σ_d, m); smaller values indicate higher confidence.
- CW90 correction: VIN v2/v3 apply `rotate_yaw_cw90` to poses when `apply_cw90_correction=True`, but `p3d_cameras` are not rotated in-model; semidense projections can be misaligned unless cameras are already corrected upstream.
- Offline cache sanity check (2026-01-26): `candidate_poses_world_cam` matches `p3d_cameras` exactly (max abs R diff 0, max abs T diff ~1e-6). Undoing CW90 on poses only yields large mismatch (max abs R diff ~1.25, T diff ~12). For cached data, keep `apply_cw90_correction=False` or rotate `p3d_cameras` in lockstep.
- Candidate sampling: when `align_to_gravity=True`, a gravity-aligned `sampling_pose` is used for sampling; `CandidateSamplingResult.sampling_pose` stores it for plotting symmetry (reference_pose remains the physical rig pose).
- Offline cache metadata includes `labeler_config` snapshots; the candidate panel can reconstruct `CandidateViewGeneratorConfig` from `labeler_config["generator"]` for cached samples.
- `VinOracleBatch.collate` cannot batch full `EfmSnippetView` instances; batched training expects `VinSnippetView` (the cache-ready, padded subset) when snippet views are included.
- Batching EVL OBB outputs is not supported yet in `VinOracleBatch` collation: if any of `obbs_pr_nms` / `obb_pred` / `obb_pred_probs_full` are present, collation raises. For entity-aware experiments, either (a) disable OBB outputs in the backbone/cache, or (b) run with `batch_size=None` until batching support is implemented.

## 7.1) VINv3 Notes
- `VinModelV3` supports an optional trajectory encoder (`use_traj_encoder=True`, `traj_encoder=TrajectoryEncoderConfig()`), which adds a pose-dimension `traj_ctx` (attended trajectory context) to the head input and diagnostics.
- `VinPrediction.expected` / `expected_normalized` are **expected class** values (ordinal score in `[0, K-1]` / `[0, 1]`) used as a ranking proxy. A continuous expected RRI estimate uses CORAL class probabilities with bin representatives `u_k` (see `docs/typst/paper/sections/07-training-objective.typ` and `oracle_rri/oracle_rri/rri_metrics/coral.py` / `oracle_rri/oracle_rri/lightning/lit_module.py`).
- VIN Lightning logging keys live in `oracle_rri/oracle_rri/rri_metrics/logging.py` (plus a few explicit keys in `oracle_rri/oracle_rri/lightning/lit_module.py`):
  - Main scalars: `stage/<loss>` with `stage ∈ {train,val,test}` (e.g., `train/loss`, `val/coral_loss_rel_random`).
  - Aux scalars: `stage-aux/<metric>` (e.g., `train-aux/spearman`, `train-aux/top3_accuracy`).
  - Figures: `stage-figures/confusion_matrix`, `stage-figures/label_histogram` (logged as images), plus `_step` variants for train.
  - Grad norms: `train-gradnorms/grad_norm_<module>` (from `VinLightningModule.on_after_backward`; target selection in `oracle_rri/oracle_rri/utils/grad_norms.py`).
  - Robustness flags: `stage/drop_nonfinite_logits_frac`, `stage/skip_nonfinite_logits`, `stage/skip_no_valid`.
  - Lightning emits `_step`/`_epoch` suffixes when a key is logged with both `on_step=true` and `on_epoch=true`.


## 8) Housekeeping / Safety
- The repo may be dirty; never delete or reset without explicit request.
- If unexpected edits appear, stop and ask.
- Avoid GUI actions; request approval if a command needs escalated permissions.

## 9) Quick Checklists

### Start-of-task checklist
- Run `make context` (writes `.codex/codex_make_context.md`; search with `rg`, including the source index section).
- Read `docs/index.qmd`, `docs/contents/todos.qmd`.
- Clarify task scope and acceptance criteria.

### Pre-commit checklist
- Format/lint/test files touched.
- Update docs and bibliography if needed.
- Ensure no temporary citation placeholders remain.

## 11) Context7 Library IDs
- `/facebookresearch/atek`
- `/websites/facebookresearch_github_io_projectaria_tools`
- `/facebookresearch/efm3d`
- `/mikedh/trimesh`
- `/rocm/pytorch`
- `/facebookresearch/pytorch3d`
- `/plotly/plotly.py`
- `/dfki-ric/pytransform3d`
- `/isl-org/open3d`
- `/pydantic/pydantic`
- `/websites/streamlit_io`
- `/websites/typst_app`
- `/websites/quarto`
- `/websites/astral_sh_uv`

## 12) Recent Gotchas
- Pydantic v2 raises if `@field_validator` references a missing field (e.g., stale `plot_stage` validator in `AriaNBVExperimentConfig`); remove the validator or set `check_fields=False`.
- Checkpoint resume: construct the module from the current config and let `trainer.fit(..., ckpt_path=...)` restore weights/optimizers; use checkpoint hparams only for drift logging to avoid double-loading.
- BaseConfig now expects configs to override `target` as a property (not a Field) and uses `tomli_w` for TOML output; run `uv sync` after dependency updates.
- Typst cannot enumerate directories at compile time; for image sequences, generate a `frames.json` manifest and load it via `json()` for ordered frames.
- W&B API history: `wandb.Api().run(...).history(keys=[...])` can return empty/partial results for some key combinations; prefer `run.scan_history(keys=[...])` and filter non-NaN values when computing start/end metrics.
- `uv run pytest` can pick up the system python (e.g., miniforge3) if uv is not pinned; use `oracle_rri/.venv/bin/python -m pytest` when you need `efm3d` and project deps.
- `make context` previously failed with `Error 127` when `rg` or `tree` was missing; context indexing now falls back to `find` when `rg` is unavailable, and the Makefile falls back to `find` when `tree` is unavailable.
- ASE downloader / metadata: `ASEMetadata.get_scenes_with_meshes()` without `config=` returns the per-scene max-shard variant across ATEK configs. Use `get_scenes_with_meshes(config=...)` (or `filter_scenes(..., config=...)`) to get config-specific shard counts (e.g., for GT-mesh scenes in the current manifests: `efm`/`efm_eval` total 576 shards vs `cubercnn_eval` total 1641). Each shard contains multiple snippet samples (EFM shards currently have 8 snippets/shard on disk).
- VIN snippet cache metadata/hash: `vin_snippet_cache_config_hash(...)` includes `pad_points` (typically `VIN_SNIPPET_PAD_POINTS`). If you hand-write `metadata.json` (e.g., in tests), include `pad_points` and pass it into the hash to avoid mismatched compatibility checks.
