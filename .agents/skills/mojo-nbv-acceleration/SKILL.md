---
name: mojo-nbv-acceleration
description: Evaluate, plan, and implement incremental Mojo adoption for Aria-NBV, especially when replacing custom CUDA-style geometry kernels with Python-importable Mojo CPU or Metal kernels that preserve existing contracts and unlock Apple-Silicon execution.
---

# Mojo NBV Acceleration

Use this skill when the task involves Mojo, Modular, custom numeric kernels, CUDA-to-Mojo translation, Apple-Silicon acceleration, or deciding whether part of `aria_nbv` should move out of the current Python and PyTorch stack.

## Use When

- The user asks where Mojo could speed up `aria_nbv`.
- The task is to replace a custom CUDA extension or CUDA-shaped kernel with Mojo.
- The task is to scope or implement Apple-Silicon-compatible kernels for oracle labeling, candidate generation, rendering, projection, or point-cloud work.
- The task needs a repo-specific plan for Python interop, Python-importable Mojo modules, raw-pointer boundaries, CPU SIMD, or Metal GPU kernels.
- The task needs a concrete answer about whether a hotspot should become CPU Mojo, GPU Mojo, remain in PyTorch or PyTorch3D, or stay in Python orchestration.

## Do Not Use When

- The task is a normal Python edit inside one already-localized module with no Mojo angle.
- The task is about general PyTorch, PyTorch3D, or Lightning usage without any question about Mojo.
- The user wants a generic language comparison not tied to this repo or to concrete kernels.

## First Pass

1. Localize the touched surface with the repo hot path:
   - `docs/typst/paper/main.typ`
   - `.agents/memory/state/PROJECT_STATE.md`
   - `docs/_generated/context/source_index.md`
   - `docs/contents/impl/apple_silicon_mojo_oracle_backend.qmd`
   - `aria_nbv/AGENTS.md` plus the relevant nested module guide
2. Open `.agents/references/context7_library_ids.md` and use the Mojo docs id:
   - `/websites/modular_mojo`
3. Read these skill-local references before broad lookup:
   - `references/mojo-context7-summary.md`
   - `references/public-acceleration-examples.md`
4. Inspect the concrete code path and classify it as one of:
   - Python orchestration
   - PyTorch or PyTorch3D kernel wrapper
   - CPU geometry kernel with per-item control flow
   - GPU-style map, filter, reduce, gather, or scatter kernel
   - data movement or serialization path
5. If an external CUDA kernel is involved, inspect all three layers before making a port recommendation:
   - Python wrapper or autograd surface
   - C++ or pybind binding surface
   - CUDA kernel itself

## Repo Reality Check

This repo already has a real Mojo integration pattern. Reuse it before inventing a new boundary.

- `aria_nbv/aria_nbv/pose_generation/mojo_backend.py`
- `aria_nbv/aria_nbv/rendering/mojo_backend.py`
- `aria_nbv/aria_nbv/rri_metrics/mojo_backend.py`
- `aria_nbv/aria_nbv/pose_generation/mojo/*.mojo`
- `aria_nbv/aria_nbv/rendering/mojo/*.mojo`
- `aria_nbv/aria_nbv/rri_metrics/mojo/*.mojo`

Key facts about the current repo pattern:

- Python stays the orchestrator.
- Mojo modules are imported through `mojo.importer`.
- Repo kernels are exposed with `PyInit_*` plus `PythonModuleBuilder`.
- Python stages contiguous CPU buffers, passes raw addresses into Mojo, and converts outputs back to Torch tensors.
- The existing backends are opt-in and benchmarked against the PyTorch3D or Trimesh baseline.
- The current repository already treats Apple-Silicon Mojo work as an experimental backend surface, not a broad rewrite of the training stack.

When the task is implementation, first check whether the existing backend shape can be extended rather than creating a parallel integration path.

## Mojo Language Facts To Anchor On

Use official Modular docs, not memory, for unstable details. The important facts for this repo are:

- Python interoperability:
  - Mojo can call Python through `std.python`.
  - Python can import Mojo modules through `mojo.importer`.
  - Mojo can expose Python-importable modules through `PythonModuleBuilder` and `PyInit_*`.
- Raw pointer and buffer interop:
  - `UnsafePointer(..., unsafe_from_address=...)` is the documented path for external buffers.
  - This matches the repo’s current pattern of staging NumPy or Torch CPU buffers and passing raw addresses into Mojo.
- CPU parallel and SIMD surfaces:
  - `parallelize` is the documented CPU parallel workhorse for pointer-oriented kernels.
  - `SIMD`, `simd_width_of`, `vectorize`, gather, scatter, and strided load or store are the relevant CPU vectorization tools.
- GPU programming surfaces:
  - `DeviceContext` is the documented GPU execution surface.
  - `LayoutTensor`, `block_idx`, `thread_idx`, `block_dim`, `AddressSpace.SHARED`, and block or warp reduction primitives are the key idioms.
  - `map_to_host()` and explicit buffer copies are the documented host-device transfer patterns.
- Apple-Silicon specifics:
  - Apple GPU availability is detectable through `has_apple_gpu_accelerator()`.
  - Metal is a supported API surface in official docs.
  - `print()` is not a viable debugging strategy inside Apple GPU kernels; write to buffers and inspect on host instead.
- Packaging and build:
  - Source imports can work through `mojo.importer`.
  - Explicit extension builds can use `mojo build ... --emit shared-lib`.
  - Package layout with `__init__.mojo` is the documented way to present reusable Mojo modules.
- Benchmarking:
  - Use `std.benchmark` for microbenchmarks.
  - In this repo, also preserve the existing Python benchmark entrypoints and parity tests.

## Apple-Silicon Decision Rules

- Prefer CPU Mojo first when:
  - the current boundary already stages CPU buffers,
  - the kernel has heavy per-item control flow or exact-geometry branching,
  - the main win is removing Python loops rather than exploiting thousands of GPU threads,
  - or moving data to Metal would dominate runtime.
- Prefer Metal GPU Mojo when:
  - the kernel is an explicit map, stencil, tiled matmul, gather, scatter, compaction, or reduction,
  - memory access can be made predictable and mostly contiguous,
  - the kernel can stay on device for enough work to amortize transfers,
  - and the current CUDA kernel’s speedup comes from parallel indexing rather than PyTorch-only integration.
- Keep code in PyTorch or PyTorch3D when:
  - the expensive work already happens in an optimized library kernel,
  - the surrounding Python is thin,
  - or the proposed Mojo replacement would recreate mature camera or rasterization functionality with higher risk than value.
- Keep code in Python when:
  - the path is mostly filesystem I/O, WebDataset iteration, JSON or msgpack serialization, logging, config, or trainer orchestration.

Do not assume Metal is automatically the best path just because the goal is Apple Silicon. For many repo kernels, CPU Mojo plus Python importability is the shortest route to usable Apple-Silicon support.

## CUDA-To-Mojo Translation Workflow

Do not translate CUDA line-by-line. Translate the algorithm, memory pattern, and public contract.

1. Start from the public boundary.
   - Record function name, inputs, outputs, dtypes, shapes, aliasing, mutability, and whether the op is in-place.
   - If the CUDA op sits behind `torch.autograd.Function`, preserve that Python-facing contract first.
2. Reconstruct the kernel shape.
   - What does one CUDA block own?
   - What does one thread own?
   - Is the kernel map-only, tiled, shared-memory-heavy, reduction-heavy, or compaction-heavy?
   - Which data are reused inside a block?
3. Choose the first Mojo target shape.
   - Python-importable CPU kernel with raw pointers and `parallelize`.
   - Metal GPU kernel with `DeviceContext`, `LayoutTensor`, and explicit grid or block indexing.
   - FFI wrapper over an external fast library if rewriting the core is clearly lower leverage.
4. Translate memory semantics, not syntax.
   - CUDA global memory pointer work often maps to `UnsafePointer` or `LayoutTensor`.
   - CUDA shared memory usually maps to `LayoutTensor[..., address_space=AddressSpace.SHARED].stack_allocation()`.
   - CUDA thread or block indexing maps to `thread_idx`, `block_idx`, and `block_dim`.
   - CUDA block or warp reductions map to block or warp primitives, not ad hoc manual porting when primitives already exist.
5. Keep orchestration in Python.
   - Preserve config-as-factory `.setup_target()` surfaces.
   - Preserve `PoseTW`, `CameraTW`, and existing batch containers.
   - Keep the old path available behind a backend enum or explicit opt-in switch until parity and benchmarks pass.
6. Verify parity before widening scope.
   - Exact masks or ordering when the contract demands it.
   - `atol=1e-4`, `rtol=1e-4` where the repo requirements say approximate parity is allowed.
   - Benchmark both microkernels and end-to-end pipeline impact on Apple Silicon.

## External CUDA Kernel Checklist

When an external CUDA extension is the donor reference, explicitly record:

- Python module name and import path
- how the extension is built
- whether tensors are contiguous and device-specific
- launch shape: grid, block, shared memory usage
- whether the op is in-place
- whether backward is a distinct kernel or just a sign-flipped forward
- whether the code assumes CUDA-only helpers, packed accessors, or stream semantics
- licensing on the copied files

Do not vendor external code into this repo casually. If external files carry non-Apache or non-commercial headers, prefer notes and links unless the user explicitly asks for a licensed import and the repo can carry the notices cleanly.

## Repo-Specific High-Value Targets

Already-real Mojo or Mojo-shaped surfaces:

- `aria_nbv/aria_nbv/pose_generation/mojo/mesh_collision_kernels.mojo`
- `aria_nbv/aria_nbv/rendering/mojo/oracle_render_kernels.mojo`
- `aria_nbv/aria_nbv/rri_metrics/mojo/oracle_distance_kernels.mojo`

Good next targets or extension points:

- `aria_nbv/aria_nbv/pose_generation/candidate_generation_rules.py`
  - `PathCollisionRule`
  - especially if current Trimesh fallback is the runtime bottleneck
- `aria_nbv/aria_nbv/rendering/candidate_pointclouds.py`
  - `_backproject_depths_p3d_batch`
  - especially if the current per-candidate Mojo loop should become batched CPU SIMD or Metal work
- `aria_nbv/aria_nbv/vin/model_v3.py`
  - `_project_semidense_points`
  - `_encode_semidense_projection_features`
  - `_encode_semidense_grid_features`
  - these are good candidates when the win is fused projection, binning, weighting, and reduction rather than CNN math

## Repo-Specific Low-Value Targets

- Most of the scorer head and CNN math in `aria_nbv/aria_nbv/vin/model_v3.py`
- Lightning orchestration, dataset plumbing, and cache maintenance
- thin wrappers around already-optimized PyTorch3D kernels unless profiling shows surrounding Python overhead is dominant

## Existing Verification And Benchmark Surfaces

Check these before proposing a new verification story:

- `aria_nbv/tests/pose_generation/test_mojo_collision_backend.py`
- `aria_nbv/tests/rendering/test_oracle_backend_contracts.py`
- `aria_nbv/tests/rri_metrics/test_mojo_oracle_backend.py`
- `aria_nbv/tests/integration/test_oracle_rri_backend_parity.py`
- `aria_nbv/scripts/benchmark_mojo_candidate_generation.py`
- `aria_nbv/scripts/benchmark_mojo_oracle_rri.py`
- `aria_nbv/aria_nbv/pose_generation/REQUIREMENTS.md`
- `aria_nbv/aria_nbv/rendering/REQUIREMENTS.md`
- `aria_nbv/aria_nbv/rri_metrics/REQUIREMENTS.md`

When you change behavior or add a new backend seam, align the implementation with the relevant `REQUIREMENTS.md` contract and benchmark script rather than inventing new acceptance criteria.

## Output Expectations

When answering with this skill, prefer a concrete recommendation over a generic survey. The useful response shape is:

1. Current boundary and hotspot summary
2. CPU Mojo vs Metal Mojo vs keep-in-PyTorch decision with reasoning
3. Minimal implementation seam to preserve current contracts
4. Parity tests and benchmark gate
5. Risks:
   - copies across Python and Mojo
   - precision or frame mismatches
   - Apple GPU limitations
   - licensing if the task references external CUDA code

If the user asks for implementation, do not stop at brainstorming. Extend the existing repo Mojo pattern, preserve the current Python API, and carry the work through parity validation when feasible.
