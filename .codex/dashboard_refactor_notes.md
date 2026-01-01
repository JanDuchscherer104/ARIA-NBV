# Streamlit dashboard: low-hanging fixes + refactor notes (2025-12-15)

## Problem condensed

The Streamlit dashboard in `oracle_rri/oracle_rri/dashboard/` mixes UI, pipeline orchestration, caching, and compute in a way that is hard to reason about and hard to reuse for training-time label generation.

Two concrete UX bugs were reported:

- `Run ALL (data → candidates → renders)` did **not** actually run the full pipeline.
- `Run / refresh <stage>` buttons did **not** force a recomputation; cached results were reused.

## What was fixed (low-hanging fruit)

### Stage refresh actually refreshes now

`Run / refresh ...` buttons now bypass the stage-level cache and recompute the stage even if the config is unchanged.

Implementation detail: each stage runner gained a `force: bool` flag that disables the "use cached results" early-return.

### `Run ALL` runs all stages now

Previously `Run ALL` called a helper that invoked `st.rerun()` during stage execution (after the first stage), which interrupted the remaining stages. The button therefore often ran only the first stage.

Implementation detail: stage runners gained a `rerun: bool` flag, and `Run ALL` executes all stages with `rerun=False` and triggers exactly one `safe_rerun()` at the end.

## Architectural issues (why it feels “intransparent”)

1. **Pipeline code is embedded in the UI function** (`DashboardApp._render_body` defines nested `_run_*` functions).
2. **Caching semantics are implicit**: a stage run sometimes means “use cache”, sometimes “compute”, depending on config equality and internal state.
3. **State invalidation is manual and spread around** (multiple `store(..., None)` calls).
4. **Dashboard pipeline ≠ training pipeline needs**:
   - Training wants efficient, reusable stage implementations without Streamlit globals.
   - Training needs scalable batching/chunking, especially for the mesh-distance step.

## Suggested refactor (data-flow + training alignment)

### 1) Extract a reusable “oracle label pipeline” (non-Streamlit)

Create a module like `oracle_rri/oracle_rri/pipelines/oracle_rri_labeler.py` with a config-as-factory wrapper.

Minimal API sketch:

- `OracleRriLabelerConfig.setup_target() -> OracleRriLabeler`
- `OracleRriLabeler.run(sample: EfmSnippetView) -> OracleRriLabelBatch`

Where `OracleRriLabelBatch` contains (optionally) the intermediates:

- `candidates: CandidateSamplingResult`
- `depths: CandidateDepths`
- `candidate_pcs: CandidatePointClouds`
- `rri: RriResult`

The dashboard becomes a thin client that calls the same pipeline.

### 2) Make caching explicit via stage keys + run-ids

Define a small typed cache structure (not Streamlit-specific):

- `cache_key = (stage_name, cfg_hash, input_hash, run_id)`

Dashboard can store this in `st.session_state`, but training can use an LRU cache or no cache.

### 3) Add chunking everywhere candidates can be large

Training will commonly use 64–256 candidates; several steps should support chunking to avoid OOM:

- Depth render: `poses` chunking (already naturally batched via rasterizer, but should expose chunk size).
- Depth→PC backprojection: chunk over candidates and/or pixels.
- RRI scoring: chunk over candidates (and potentially mesh faces/verts if needed).

### 4) Fix the main performance cliff: point↔mesh distances

`oracle_rri/oracle_rri/rri_metrics/metrics.py::chamfer_point_mesh_batched` currently repeats `gt_verts` per candidate (`gt_verts.repeat(bsz, 1)`), which is a large memory multiplier.

Suggested options (in priority order):

1. **Chunk candidates** in `OracleRRI.score()` so `bsz` stays small (easy + safe).
2. Add an **approximate completeness mode** (sample GT surface points once, then use `knn_points` for M→P); this avoids triangle replication.
3. Investigate a **shared-mesh packing trick** for the accuracy term (P→M) using a single triangle buffer and per-cloud first indices; completeness likely still needs chunking or sampling.

### 5) Clarify “dashboard stage outputs” vs “training outputs”

Dashboard wants:

- Rich plotting objects / diagnostics.
- Caching keyed by configs.

Training wants:

- Minimal tensors on GPU.
- Chunked computation.
- No Plotly / Streamlit dependencies.

Keep these concerns separated: `pipelines/` produces tensors + typed dataclasses; `dashboard/` does plotting.

## Acceptance criteria for the refactor (when to call it “done”)

- Dashboard:
  - Each page calls a shared pipeline runner (no nested stage functions in `app.py`).
  - Cache status is visible (per-stage config hash + run-id + invalidation).
- Training:
  - A non-Streamlit pipeline can run: `sample -> candidates -> depths -> pcs -> rri`.
  - Candidate chunk size is configurable and tested on a real sample.
  - RRI scoring supports large candidate counts without OOM on typical meshes.

