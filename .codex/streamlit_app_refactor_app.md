# Streamlit refactor: `oracle_rri.app` (2025-12-15)

## Problem (condensed)

The legacy Streamlit dashboard in `oracle_rri/oracle_rri/dashboard/` mixed UI, caching, and compute, making the data-flow hard to reason about and hard to reuse for training-time *online* oracle labeling.

Goals for this refactor:

- Re-implement the app in a new package `oracle_rri/oracle_rri/app/`.
- Preserve all existing functionality (Data / Candidate Poses / Candidate Renders / RRI).
- Make caching semantics explicit and **typed** (avoid untyped `st.cache_*` objects).
- Route RRI computation through the shared, non-Streamlit pipeline (`OracleRriLabeler`).

## Implementation summary

### New package layout

`oracle_rri/oracle_rri/app/` now contains:

- `state.py`: a single strongly-typed `AppState` stored in `st.session_state` (`STATE_KEY="nbv_app_state_v2"`).
- `controller.py`: `PipelineController` orchestrates compute and cache invalidation (data → candidates → depths → pcs → rri).
- `ui.py`: sidebar config widgets for dataset / candidate generator / depth renderer / oracle settings.
- `panels.py`: plotting-only page renderers (no heavy compute).
- `app.py`: top-level `NbvStreamlitApp` that wires pages and run controls.

### Typed cache + explicit invalidation

Instead of implicit Streamlit caching:

- Each stage cache stores:
  - a **config signature** (`cfg_sig`) and an **input key** (`sample_key`, `candidates_key`, `depth_key`),
  - the typed stage output (`EfmSnippetView`, `CandidateSamplingResult`, `CandidateDepths`, `CandidatePointClouds`, `RriResult`).
- `force=True` parameters bypass cache reuse for “Run / refresh …” buttons and “Run ALL”.

### `config_signature()` robustness

`oracle_rri.app.state.config_signature()` needed to be stable but also handle non-JSON types used by configs
(e.g. `torch.device`, `torch.Tensor`, `Path`, `Enum`).

We added a small serializer `_to_jsonable(...)` and a unit test:

- `oracle_rri/tests/test_app_state_signature.py`

### Oracle RRI chunking (training/perf knob)

Implemented `OracleRRIConfig.candidate_chunk_size` and chunked scoring in
`oracle_rri/oracle_rri/rri_metrics/oracle_rri.py::OracleRRI.score()` to reduce peak memory for
large candidate batches while preserving exact results. The unit test
`oracle_rri/tests/rri_metrics/test_oracle_rri_chunking.py` now exercises the real chunking path.

### RRI page uses shared pipeline

The RRI page calls `PipelineController.run_labeler()` which runs:

`OracleRriLabelerConfig.setup_target().run(sample)`

and caches returned `(candidates, depths, candidate_pcs, rri)` as typed objects.

### Entrypoint

`oracle_rri/oracle_rri/streamlit_app.py` now launches the refactored app (used by `nbv-st`).

The legacy dashboard entrypoint remains available as `oracle_rri/oracle_rri/streamlit_app_old.py`.

## Low-hanging bug fix included

`PathConfig.resolve_processed_mesh_path` previously had a signature mismatch with its only call site
(`oracle_rri/oracle_rri/data/mesh_cache.py`), breaking mesh processing and therefore dataset iteration.

Fixed signature to:

`resolve_processed_mesh_path(scene_id, spec_hash, *, snippet_id=None)`

This restores dataset + oracle pipeline execution.

## Validation performed

- `ruff format` + `ruff check` on touched files.
- `pytest` (real data integration included):
  - `oracle_rri/tests/test_efm_dataset.py`
  - `oracle_rri/tests/rri_metrics/test_oracle_rri_chunking.py`
  - `oracle_rri/tests/integration/test_oracle_rri_labeler_real_data.py`
  - `oracle_rri/tests/test_app_state_signature.py`

## Follow-ups / suggestions

- Candidate Renders UX: we now treat “renders” as `(depths + backprojected candidate point clouds)` and use Streamlit’s `st.navigation(..., position="top")` for page switching.
- CandidateDepthRenderer now always returns at most `max_candidates_final` candidates (it may render an oversampled set and then cap by valid pixel count).

- Add an optional “stage detail” sidebar showing each stage’s `cfg_sig` + input keys for easier debugging.
- Consider a small “page registry” pattern if more pages get added (keep `NbvStreamlitApp` shallow).

## Update: controller reuse + UI cleanup (2025-12-15)

### `PipelineController` is now Streamlit-free

- Moved Streamlit-free state types + cache keys into `oracle_rri/oracle_rri/app/state_types.py`.
- `oracle_rri/oracle_rri/app/state.py` now only wraps `st.session_state` and re-exports the types/helpers.
- `oracle_rri/oracle_rri/app/controller.py` no longer imports Streamlit; it accepts an injected progress callback:
  - Streamlit app passes `lambda msg: st.status(msg, expanded=False)`
  - training/CLI can pass a no-op or a `Console`-logging context manager.
- `oracle_rri/oracle_rri/app/__init__.py` now lazily imports `NbvStreamlitApp*` so non-UI modules like
  `oracle_rri.app.controller` are importable without Streamlit installed.

This makes it straightforward to reuse the same controller/caching semantics in training or CLI tools, while
the actual *pipeline* for online labeling remains `OracleRriLabeler` in `oracle_rri/oracle_rri/pipelines/oracle_rri_labeler.py`.

### Deterministic candidate sampling

- Added `CandidateViewGeneratorConfig.seed: int | None` (default `0`) and used `torch.random.fork_rng` to keep seeding
  local to candidate sampling.
- Exposed seed toggle + value in the app sidebar (generator config).
- Added a real-data integration test ensuring determinism:
  - `oracle_rri/tests/integration/test_candidate_generation_seed_real_data.py`

### Less UI clutter

- Moved **plot options** out of the sidebar and directly above their plots in `oracle_rri/oracle_rri/app/panels.py`.
- Candidate + Depth pages now use a single `Diagnostics` expander with tabs (no nested expanders).

## Update: warning cleanup (2025-12-15)

- `uv run …` warning about `tool.uv.extra-build-dependencies` is silenced by enabling preview mode in
  `oracle_rri/pyproject.toml` (`[tool.uv] preview = true`).
- Streamlit deprecation warnings about `use_container_width` are resolved by switching to `width="stretch"` in
  all `st.plotly_chart(...)` calls.
- The noisy `FutureWarning` from WebDataset / `torch.load(weights_only=...)` is filtered inside `AseEfmDataset`
  initialisation to keep logs readable.
- `AseEfmDatasetConfig` no longer logs in its Pydantic validators (validators can run many times during Streamlit
  reruns); the dataset summary is now logged once in `setup_target()`.
