# Future Directions Scan (2026-01-30)

## Scope

Goal: extract *all hinted future directions* (research + engineering) that appear across:

- Typst paper sections included by `docs/typst/paper/main.typ` and the final slides `docs/typst/slides/slides_4.typ`.
- Quarto docs (`docs/index.qmd`, `docs/**/*.qmd`, especially `docs/contents/todos.qmd`, `docs/contents/questions.qmd`, `docs/contents/roadmap.qmd`, `docs/contents/impl/*`).
- Code (`oracle_rri/**`) where comments / TODOs / “not supported yet” constraints imply next steps.

This is a *read-only synthesis* (no integration edits were made to paper/slides), intended to be copy-pastable into “Future work / Next steps” sections.

## How this scan was done

- Context init + source index:
  - `scripts/nbv_context_index.sh` → `.codex/context_sources_index.md`
  - `make context` → `.codex/codex_make_context.md`
  - `make context-qmd-tree`
- Typst include graph:
  - `scripts/nbv_typst_includes.py --paper docs/typst/paper/main.typ --mode graph`
- Quarto heading outline:
  - `scripts/nbv_qmd_outline.sh`
- Targeted `rg` searches for: `TODO`, `next steps`, `future`, `planned`, `underused`, plus explicit task checkboxes.

## Consolidated future directions (deduped, grouped)

### A) Thesis-scale / long-horizon roadmap

- **Self-supervised NBV pretraining**: find a self-supervised objective; clarify “what would I render if I went to a candidate view?” (`docs/contents/todos.qmd:18`).
- **View synthesis integration** (explicit roadmap milestone) (`docs/contents/roadmap.qmd:38`).
- **Human-in-the-loop AR guidance**: entity selection UI, streaming NBV compute, AR overlays, “voice2voice feedback” (`docs/contents/roadmap.qmd:49`; `docs/index.qmd:36`).
- **Deploy/evaluate on real devices** (Aria, Quest 3, iPhone LiDAR) and real-world deployment/evaluation (`docs/contents/roadmap.qmd:43`; `docs/contents/roadmap.qmd:67`).
- **LLM / VLA integration** for explanation and/or high-level planning and potentially using VLAs for entity representations (`docs/contents/questions.qmd:114`).

### B) Data + scaling + offline cache evolution

- **Scale data coverage** (more scenes/snippets; track coverage systematically) (`docs/typst/slides/slides_4.typ:1835`; `docs/typst/slides/slides_4.typ:1839`).
- **Handle dataset constraints**: single prerecorded trajectory per scene; limited GT meshes; pseudo-GT beyond mesh subset (`docs/typst/slides/slides_4.typ:1806`).
- **Link ATEK shards ↔ meshes ↔ raw ASE**: robust scene/snippet mapping, metadata download, selection based on GT meshes (`docs/contents/todos.qmd:375`).
- **Offline cache “future-proofing”**:
  - cache full EVL backbone output to avoid recomputation when changing feature selection (`docs/contents/todos.qmd:387`);
  - allow mixing multiple configs in one cache directory, but add config-hash bookkeeping + filtering (`docs/contents/todos.qmd:413`);
  - optionally generate new offline samples during training and combine online/offline datasets (`docs/contents/todos.qmd:411`).
- **Multiprocessing**: make `OracleRriLabeler` work with `data_module.num_workers > 0` (`docs/contents/todos.qmd:84`).
- **Use ASE visibility metadata** as an accelerator/analysis tool instead of heavy raycasting (`docs/index.qmd:31`; `docs/contents/impl/rri_computation.qmd:891`).
- **Improve calibration for long snippets**: incorporate time-varying calibration (`mps_online_calib_processor.py`) (`docs/contents/ext-impl/atek_implementation.qmd:165`).

### C) Candidate generation + action space

- **Move from discrete candidates → continuous action**:
  - later learn continuous poses directly and score them with the learned RRI predictor; use free-space voxels to avoid collisions (`docs/contents/todos.qmd:34`);
  - combine discrete + continuous (sample continuous distribution then select among candidates) (`docs/contents/questions.qmd:71`).
- **Candidate-generation policy choices that affect labels**:
  - filter invalid views vs keep and penalize; allow backward views; allow roll (`docs/contents/todos.qmd:37`);
  - candidate sampling changes the RRI distribution → changes CORAL bins (`docs/typst/slides/slides_4.typ:1812`; `docs/contents/todos.qmd:41`).
- **Use voxel/free-space pruning before expensive mesh tests**: `sample_depths_in_grid` as a cheap pruning primitive (`docs/contents/ext-impl/efm3d_implementation.qmd:189`).
- **Differentiable pose refinement**: `diff_grid_sample` suggests gradient-based refinement of candidate poses (`docs/contents/ext-impl/efm3d_implementation.qmd:188`).

### D) Oracle pipeline: efficiency, correctness, and throughput

- **Profile + quantify oracle runtime/memory** and propagate measured numbers into paper/slides; current notes request measured cost (`docs/typst/paper/sections/10-discussion.typ:25`; `docs/typst/paper/sections/12h-appendix-offline-cache.typ:16`).
- **Make oracle scalable**:
  - improve multiprocessing/throughput (“labeler not yet optimized for large-scale multiprocessing”) (`docs/typst/paper/sections/10-discussion.typ:31`);
  - add helpers for consistent sampling/downsampling to control density effects (downsampling delegated to callers “or future helpers”) (`oracle_rri/oracle_rri/rri_metrics/oracle_rri.py:9`; `docs/contents/impl/rri_computation.qmd:874`).
- **Rendering/geometry tech debt**: keep frame conventions consistent (CW90), depth conventions correct (`docs/typst/slides/slides_4.typ:1811`; `docs/contents/todos.qmd:36`).

### E) VIN architecture + feature upgrades (v2/v3)

**Cross-cutting “core” upgrades (appear in multiple places):**

- **Candidate ordering bias**: add optional per-sample candidate shuffling in the datamodule (`docs/contents/todos.qmd:76`; `docs/typst/slides/slides_4.typ:1817`).
- **Stage awareness**: stage-aware features / stage-aware binning / stage cue scalar (`docs/contents/todos.qmd:69`; `docs/typst/paper/sections/07a-binning.typ:11`; `docs/typst/paper/sections/12g-appendix-vin-v3-streamline.typ:216`).
- **Candidate-relative positional encoding** for attention keys (rig → candidate frame) (`docs/contents/todos.qmd:68`; `docs/contents/impl/vin_nbv.qmd:616`; `docs/contents/impl/vin_v2_feature_proposals.qmd:268`).
- **Learnable CORAL bin shifts / calibration** (`docs/contents/todos.qmd:66`).

**Candidate evidence and semidense features:**

- **Visibility-aware semidense embeddings** so “invalid/unseen” is not identical to “seen but uninformative” (`docs/contents/impl/vin_coverage_aware_training.qmd:34`; `docs/typst/paper/sections/09b-ablation.typ:39`; `docs/contents/todos.qmd:483`).
- **Observation count features (`n_obs`)** and normalization choices (ablation + sweep toggles) (`docs/typst/paper/sections/09b-ablation.typ:45`; `docs/contents/impl/optuna_vin_v2_searchspace_2026-01-07.qmd:83`).
- **Add `F_empty`-style emptiness proxy + depth variance cue** (closer to VIN-NBV projection descriptors) (`docs/typst/paper/sections/12g-appendix-vin-v3-streamline.typ:220`).
- **(Optional) PointNeXt semidense encoder** (`docs/contents/impl/vin_v2_feature_proposals.qmd:167`; `docs/typst/paper/sections/09b-ablation.typ:33`).
- **(Optional) semidense frustum MHCA** for token-level view conditioning (`docs/typst/paper/sections/08a-frustum-pooling.typ:8`; `docs/typst/paper/sections/09b-ablation.typ:36`).
- **(Optional) trajectory context** (`docs/typst/paper/sections/09b-ablation.typ:48`; `docs/contents/impl/vin_v2_feature_proposals.qmd:237`).

**Feature-space / fusion improvements:**

- **Per-channel gates/thresholds** for scene-field channels (e.g., suppress low-confidence `cent_pr`) (`docs/contents/todos.qmd:65`).
- **2D appearance priors**: reuse EVL 2D tokens / DINOv2 features / RGB view-plane features (projection aligned to candidates) (`docs/contents/todos.qmd:67`; `docs/contents/impl/vin_v2_feature_proposals.qmd:254`).
- **Out-of-voxel mitigation**: explicit distance-to-voxel scalars, multi-scale pooling, OOB indicators, blending voxel with semidense features (`docs/contents/impl/vin_v2_feature_proposals.qmd:278`).

**Training and metrics:**

- **Coverage-aware curriculum weighting** to stabilize early training and then anneal toward uniform candidate weighting (`docs/contents/impl/vin_coverage_aware_training.qmd:21`; `docs/typst/paper/sections/09b-ablation.typ:63`).
- **Add more metrics + evaluation helpers** (rank correlation, top-k recall; TorchMetrics Accuracy) (`docs/contents/todos.qmd:80`; `docs/contents/todos.qmd:619`).
- **NBV rollout evaluation beyond per-snippet ranking** (explicitly called out as missing) (`docs/typst/slides/slides_4.typ:1819`; `docs/typst/slides/slides_4.typ:1843`).

### F) Entity-aware / task-driven NBV

- **Entity-aware RRI objective**: weight per-entity improvement and combine with global RRI (`docs/typst/paper/sections/10a-entity-aware.typ:14`; `docs/contents/questions.qmd:45`; `docs/contents/impl/rri_computation.qmd:909`).
- **Integrate EVL/ATEK entity tooling**:
  - use `ObbTracker` for temporal stability (`docs/contents/ext-impl/efm3d_implementation.qmd:186`);
  - run OBB metrics (`AtekObb3Metrics`, `ObbMetrics`) for candidate evaluation (`docs/contents/ext-impl/atek_implementation.qmd:167`; `docs/contents/ext-impl/efm3d_implementation.qmd:187`).
- **Key current implementation gap**: batching EVL OBB outputs is not supported yet in `VinOracleBatch` collation (blocks batched training with OBB features; likely required for entity-aware training at scale) (`oracle_rri/oracle_rri/data/vin_oracle_types.py:639`; `docs/typst/paper/sections/10a-entity-aware.typ:39`).

### G) Experiment management (Optuna/W&B)

- **Run a clean, stationary Optuna sweep** focusing on architectural toggles (avoid schedule/width confounds), and tag new trials by “sweep phase” for clean analysis (`docs/contents/impl/optuna_vin_v2_searchspace_2026-01-07.qmd:54`).
- **Prioritize candidate-specific signals and stability** based on observed W&B failure modes (collapse/NaNs), then validate via ablations (planned) (`docs/typst/paper/sections/09c-wandb.typ:108`; `docs/typst/paper/sections/09b-ablation.typ:11`).

### H) Paper/slides documentation hygiene (secondary, but recurring)

- **Single source of truth for numbers**: import cache stats + W&B summaries into paper sections to avoid drift (repeated TODOs) (`docs/typst/paper/sections/11-conclusion.typ:20`; `docs/typst/paper/sections/12g-appendix-vin-v3-streamline.typ:6`).
- **Unify notation via `docs/typst/shared/macros.typ`** (entity-aware section explicitly requests alignment) (`docs/typst/paper/sections/10a-entity-aware.typ:9`).

## Notes for `docs/typst/paper/sections/10a-entity-aware.typ`

That section is currently *entity-aware-only* and contains a TODO about integrating all future work items from `todos.qmd` (`docs/typst/paper/sections/10a-entity-aware.typ:6`). If you want “Future Extensions” to be a complete single place, a clean structure that maps to the above scan is:

- A short “Future work categories” bullet list (A–G).
- One subsection per category (short paragraph + 3–6 bullets).
- Keep entity-aware as its own subsection; reference OBB batching constraint and required cache/backbone changes.

