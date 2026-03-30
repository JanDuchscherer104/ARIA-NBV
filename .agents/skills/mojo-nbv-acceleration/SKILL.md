---
name: mojo-nbv-acceleration
description: Evaluate and plan incremental Mojo adoption for Aria-NBV when users ask about Mojo, Modular, kernel ports, GPU acceleration, FFI boundaries, or whether a specific `aria_nbv` hot path should move out of Python or PyTorch. Use for repo-specific decisions about what to port, how to preserve current interfaces, and when to keep code in Python.
---

# Mojo NBV Acceleration

Use this skill when the task involves Mojo, Modular, custom numeric kernels, or deciding whether a part of `aria_nbv` should move out of the current Python and PyTorch stack.

## Use When

- The user asks where Mojo could speed up `aria_nbv`.
- The task is to scope a Mojo prototype for oracle labeling, candidate generation, projection, or other hot geometry paths.
- The task needs a repo-specific integration plan for Python interop, Python-importable Mojo modules, FFI boundaries, or GPU kernels.
- The task needs a concrete answer about whether a given hotspot is a good Mojo target or should remain in PyTorch or PyTorch3D.

## Do Not Use When

- The task is a normal Python edit inside one already-localized module with no Mojo angle.
- The task is about general PyTorch, PyTorch3D, or Lightning usage without any question about Mojo.
- The user wants a generic language comparison not tied to this repo.

## First Pass

1. Localize the touched surface with the repo hot path:
   - `docs/typst/paper/main.typ`
   - `.agents/memory/state/PROJECT_STATE.md`
   - `docs/_generated/context/source_index.md`
   - `aria_nbv/AGENTS.md` plus the relevant nested module guide
2. Open `.agents/references/context7_library_ids.md` and use the Mojo library id:
   - `/websites/modular_mojo`
3. Read [references/mojo-context7-summary.md](references/mojo-context7-summary.md) before issuing more Context7 lookups.
4. Inspect the concrete code path, then classify it as:
   - Python orchestration
   - Torch or PyTorch3D kernel wrapper
   - geometry kernel with Python or NumPy control flow
   - data movement or serialization path

## Mojo Decision Rules

- Prefer Mojo for narrow, repeated, numerically heavy kernels that still have significant Python, NumPy, or per-item control overhead.
- Prefer Mojo for places where a fused kernel could remove intermediate tensors, repeated pack and unpack, or CPU round-trips.
- Keep code in PyTorch or PyTorch3D when the expensive work already happens inside optimized library kernels and the surrounding Python is thin.
- Keep code in Python when the path is primarily filesystem I/O, WebDataset iteration, JSON or msgpack serialization, or trainer orchestration.

## Repo-Specific High-Value Targets

- `aria_nbv/aria_nbv/pose_generation/candidate_generation_rules.py`
  - `MinDistanceToMeshRule`
  - `PathCollisionRule`
  - These are first-class targets when they rely on `trimesh`, NumPy conversion, or dense point sampling along rays.
- `aria_nbv/aria_nbv/rendering/candidate_pointclouds.py`
  - `_backproject_depths_p3d_batch`
  - Good target when the goal is to fuse valid-pixel masking, unprojection, compaction, and bounds reduction.
- `aria_nbv/aria_nbv/rri_metrics/oracle_rri.py`
  - Good target when the goal is to avoid materializing large repeated candidate clouds and to fuse baseline-plus-candidate distance work.
- `aria_nbv/aria_nbv/vin/model_v3.py`
  - `_project_semidense_points`
  - `_encode_semidense_projection_features`
  - `_encode_semidense_grid_features`
  - Good target when the goal is to fuse projection, clipping, binning, and reductions across many candidate views.

## Repo-Specific Low-Value Targets

- `aria_nbv/aria_nbv/rendering/pytorch3d_depth_renderer.py`
  - The expensive work is already inside PyTorch3D rasterization.
- Most of the scorer head and CNN math in `aria_nbv/aria_nbv/vin/model_v3.py`
  - These are already standard Torch kernels.
- Lightning orchestration, dataset plumbing, and cache index maintenance
  - These are not the first places to expect huge wins from Mojo.

## Integration Routes

- Python-first proof of concept:
  - Use Mojo where Python imports the Mojo module directly.
  - Best for quick CPU-side prototypes and small surface-area replacements.
- Python-importable Mojo module:
  - Use the documented `PyInit_*` plus `PythonModuleBuilder` route when the boundary should look like a normal Python module.
  - Best when the repo should call a stable Python-facing function without changing higher-level interfaces.
- Native or FFI boundary:
  - Use this when a tighter runtime boundary is needed around a low-level kernel.
  - Treat any Torch-extension wrapping plan as an integration design decision that must preserve the current tensor and device contracts.
- GPU kernel:
  - Use Mojo GPU kernels only for operations with explicit indexing, predictable memory access, and strong value from fusion.
  - Best fit here is projection, binning, ray marching, and compaction style work.

## Preserve Current Contracts

- Keep `PoseTW` and `CameraTW` as the semantic interface at module boundaries.
- Preserve config-as-factory entrypoints via `.setup_target()`.
- Avoid changing `EfmSnippetView`, `VinSnippetView`, `CandidateDepths`, `CandidatePointClouds`, or `VinOracleBatch` contracts just to make a Mojo prototype easier.
- Prefer small replacement seams behind existing helpers rather than broad rewrites.

## Porting Checklist

1. Confirm the hotspot with code inspection or profiling before proposing a port.
2. Record tensor shapes, dtypes, devices, and ownership rules at the boundary.
3. Decide whether the port is CPU Mojo, GPU Mojo, or just a Python-importable wrapper over a narrow kernel.
4. Minimize copies across Python and Mojo boundaries.
5. Keep the old path available until correctness and performance are verified.
6. Update docs if a user-visible workflow or runtime dependency changes.

## Query Refresh

If the summary file is insufficient, issue narrow Context7 lookups against `/websites/modular_mojo` for:

- Python interoperability
- Python-importable Mojo modules
- `std.ffi`
- GPU fundamentals
- GPU basics
- block and warp or shared-memory kernel guidance
