# Dashboard refactor overview (Streamlit)

## Context
- App complexity and recomputation risks highlighted in `oracle_rri/dashboard/app-revision.md`.
- Current `app.py` builds a mini pipeline with nested stage runners, manual reruns, and a blocking ThreadPool executor.
- Expensive steps (candidate gen, depth renders, RRI) re-run on every widget change.

## Recommended changes by file
- `app.py`: turn into a thin router; drop nested stage helpers and executor use; instantiate a `Pipeline` helper; clear caches via new resource cache; rely on forms to trigger work instead of auto reruns.
- `pipeline.py` (new): centralize stage logic (`run_data`, `run_candidates`, `run_depth`, `run_all`), config (de)serialization, staleness checks, and CUDA depth-config normalization.
- `services.py`: remove executor; add per-session cached builders for generator/renderer/oracle via `st.session_state`; keep dataset loader.
- `state.py`: remove unused task-tracking keys/types; keep simple `get/store/safe_rerun`.
- `panels.py`: make `render_depth_page` take `sample` explicitly; gate depth-hit backprojection and RRI scoring behind forms + per-session caches; apply `max_sem_pts` to oracle scoring; use cached oracle via `services.get_oracle`.
- `app.py`/`panels.py`: prefer PyTorch3D CUDA normalization in `pipeline.normalize_depth_cfg`; avoid duplicate upgrades.
- Optional: replace radio routing with `st.navigation` pages for clearer control flow.

## Acceptance criteria
- Expensive work (candidate generation, depth renders, RRI) only runs on form submit or when cache is invalidated.
- No executor/thread usage; single source of truth for pipeline state.
- Per-session resource caching prevents cross-session contention and rebuild overhead.
- Depth-hit and RRI computations respect `max_sem_pts` and stride; cached by `(sample, depth_batch, params)`.
- Clear cache button wipes both session state and resource cache.
