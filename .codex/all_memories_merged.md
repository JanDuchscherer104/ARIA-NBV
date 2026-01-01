# NBV Project – Consolidated Memories (Nov 22, 2025)

This file merges all active .codex notes into one place.

## nbv_agent_brief
- Env: `/home/jandu/miniforge3/envs/aria-nbv/bin/python` (Py3.11); run `make context` on arrival; read `docs/index.qmd`, `docs/contents/todos.qmd`.
- Workflow: condense problem → explore files → outline with acceptance/termination criteria → implement incrementally; keep task list; pause to #think for alignment.
- Conventions: config-as-factory (`BaseConfig.setup_target()`), full typing, Google docstrings with tensor shapes; prefer `Path`, `Enum`, `Literal`, `match-case`; use ARIA constants, `PoseTW`, `CameraTW`, EFM3D/ATEK utilities; logging via `oracle_rri.utils.Console`.
- Style/quality: minimal comments, ASCII; run `ruff format` → `ruff check` → `pytest` on touched files (real data); update docs when behavior changes.
- MCP/tools: use code-index, Context7 docs, web search; `make context-dir-tree` / `context-external` for overviews.
- Guardrails: no destructive git, don’t revert user changes, typed Streamlit state.

## nbv_docs_refresh
- Docs refreshed: `aria_nbv_overview.qmd` with dir tree, NBV flow mermaid, RRI equations, core modules, architecture diagram; navigation reorganized (impl pages, ext-impl).
- Data pipeline doc rewritten to mirror Streamlit app (Data/Candidate/Depth pages, logging, CLI download, code snippet).
- Index links fixed; sandbox render works with `--no-execute`; full `quarto check` was blocked only by sandbox limits then.
- Next: full-site render in non-restricted env; consider renaming package to `aria_nbv`; keep Streamlit examples in sync with configs.

## notes
- Mission: active NBV for indoor ASE; oracle RRI from GT meshes + semidense; RRI head atop frozen EFM3D/EVL.
- Stack: Python 3.11 `aria-nbv`, heavy deps (pytorch3d, trimesh, efm3d, atek, projectaria_tools); use PoseTW/CameraTW, config factories.
- Streamlit app: Data → Candidate Poses → Candidate Renders; forms submit; background threads; typed session (`TaskState`, `SessionVars`); console sink + scratchpad; cache aware.
- Defaults: require meshes, crop/decimate sliders; pipeline “Run previous” buttons; device validation via config.
- Style/testing: Google docstrings with shapes; enums; pathlib; Console; ruff + pytest on touched files.

## rendering_backend_switch
- CandidateDepthRenderer supports GPU (PyTorch3D) and CPU (trimesh/pyembree) backends; Streamlit sidebar lets you choose.
- CPU renderer now functional with proxy walls, chunked rays, config-as-factory, Console logging, occupancy extent union.
- Integration tests on ASE scene 81283 for both backends; unit tests for hit ratios and proxy walls.
- Tips: on CPU-only, choose CPU backend; chunk_rays moderate; PyTorch3D CPU path ~90s, CPU ray ~30s (decimated mesh, 64×64).
- Next: auto-select backend by CUDA, parity checks, cached decimated meshes/calibs.

## update-key-docs
- Ran `make context`, `make context-dir-tree`; reviewed index and todos.
- AGENTS key docs section updated to current layout (aria_nbv_package, data_pipeline_overview, ext-impl refs, paths fixed).
- Suggestions: rerun `make context-qmd-tree` after doc changes; address TODOs in index (abstract, resource links).

## wall_issue_gpt_pro_reports
- Core issue: GT meshes miss interior walls; depth renders hit zfar. Fixes: add proxy walls using mesh bounds ∪ semidense volume, two-sided rendering, raise coverage threshold; proxy walls in PyTorch3D and Plotly; avoid backface culling; consider semidense AABB when mesh truncated.
- Coordinate frames verified (Aria RDF, world Z-up, PoseTW); PyTorch3D uses inverted pose correctly; proxies in world frame.
- Recommendations: use occupancy extent union for proxy placement; higher coverage threshold; possible TSDF hull; metric: hit_ratio near periphery; add regression/QA.
- Additional notes: signed_distance quirks on non-watertight meshes; Plotly uses ambient + double faces; slight offset if z-fighting.
- Logged follow-ups: broader QA, fallback when semidense missing, expose proxy toggles/stats, regression asserting hit_ratio on known scene.

## walls_issue_summary
- Observations: scene 81283 missing side walls; fragmented mesh; semidense bounds show full room.
- Fixes implemented: proxy walls via semidense bounds, two-sided rendering, occupancy extent passed through; UI defaults keep decimation off, double-sided on, hit-ratio logging.
- Validation: Streamlit Data/Render tabs should show walls, hit_ratio > ~0.3; debug shows synthesized planes.
- Risks: fallback needed when semidense absent; need QA/threshold tuning; consider UI toggle, automated regression.
- Updates: axis ordering of occupancy extents fixed; renderer parses `[xmin, xmax, ymin, ymax, zmin, zmax]`; CPU renderer parity with proxy walls and tests added.

## Formatting & tooling (cross-cutting)
- `scripts/format_qmd_lists.py` enforces blank line before lists (outside code fences), no blanks between items; handles ```/~~~; run via `make docs-lint`.
- Makefile `docs-lint` runs formatter then `quarto check`. Quarto warns about missing R but not required unless using R blocks.

