# Mojo Context7 Summary For Aria-NBV

Primary Context7 library id: `/websites/modular_mojo`

This note condenses the official Modular Mojo docs that matter most for
Aria-NBV, especially for replacing CUDA-shaped custom kernels with
Python-importable Mojo code that can run on Apple Silicon.

## What To Read First In Context7

- Python interop:
  - `manual/python/mojo-from-python`
  - `std/python/bindings/PythonModuleBuilder`
  - `manual/python/python-from-mojo`
- Pointers and external buffers:
  - `manual/pointers/unsafe-pointers`
  - `std/memory/unsafe_pointer/UnsafePointer`
- CPU SIMD and vectorization:
  - `std/sys/info/simd_width_of`
  - `std/algorithm/backend/vectorize/vectorize`
- GPU and Apple-Silicon:
  - `manual/gpu/fundamentals`
  - `manual/gpu/basics`
  - `std/gpu/host/device_context/DeviceContext`
  - `std/sys/info/has_apple_gpu_accelerator`
- Packaging and build:
  - `manual/packages`
  - `cli`

## Official-Doc Takeaways

### Python interoperability

- Mojo can call into Python through `std.python`.
- Python can import Mojo modules through `mojo.importer`.
- Mojo can expose Python-importable modules through `PythonModuleBuilder` and
  `PyInit_*`.
- Mojo modules can also be built as shared libraries with
  `mojo build ... --emit shared-lib`.

Implication:

- The safest first integration path in this repo is to keep Python as the
  orchestrator and replace one hot function at a time behind a normal Python
  call boundary.
- This matches the repo’s current `*_mojo_backend.py` modules.

### Packages and reusable layout

- `__init__.mojo` is the documented way to re-export package members.
- Mojo has a documented package model rather than only single-file scripts.

Implication:

- If a repo kernel family grows beyond one file, package it cleanly instead of
  expanding one monolithic `.mojo` module.

### Raw-pointer interop

- `UnsafePointer(..., unsafe_from_address=...)` is the documented way to access
  external buffers from raw addresses.
- Pointer origins and mutability are part of the type model.
- Pointer arithmetic, vector loads and stores, gather, scatter, and strided
  access are documented surfaces.

Implication:

- The current repo pattern of passing contiguous NumPy or Torch CPU buffer
  addresses into Mojo is aligned with the official pointer model.
- CPU-first ports do not need a custom FFI layer when a Python-importable Mojo
  module plus raw addresses is sufficient.

### CPU parallelism and SIMD

- `parallelize` is a documented CPU parallel backend.
- `SIMD`, `simd_width_of`, and `vectorize` are the key CPU vectorization
  surfaces.
- SIMD gather, scatter, and strided memory ops are part of the documented
  toolbox.

Implication:

- For many Apple-Silicon bring-up tasks, CPU Mojo is the shortest path from a
  custom CUDA kernel to a usable replacement.
- Good first candidates are geometry kernels, compaction, per-point reduction,
  and weighted binning where the current bottleneck is Python or scalar loops.

### GPU programming

- `DeviceContext` is the documented GPU execution surface.
- GPU kernels use explicit thread and block indexing primitives such as
  `thread_idx`, `block_idx`, and `block_dim`.
- Shared memory and structured local tiles are expressed through `LayoutTensor`
  and `AddressSpace.SHARED`.
- Host and device buffers use explicit enqueue and copy steps, plus
  `map_to_host()` for host inspection.

Implication:

- CUDA-like kernels should be translated to the Mojo GPU model by preserving
  ownership and indexing structure, not by performing token-by-token syntax
  replacement.
- Metal GPU work is a good fit for explicit map, tile, reduction, and
  compaction kernels that can stay on device long enough to justify transfers.

### Apple-Silicon specifics

- Apple GPU availability is detectable via `has_apple_gpu_accelerator()`.
- Metal-backed devices are part of the documented GPU surface.
- `print()` is not supported as a practical debugging strategy from Apple GPU
  kernels.

Implication:

- For Apple-Silicon GPU work, debug by writing to buffers and reading them on
  host.
- Do not assume a CUDA-only debugging workflow will carry over.

### Benchmarking

- Mojo provides `std.benchmark`.
- GPU benchmarks should treat kernel launch and copy costs explicitly.

Implication:

- In this repo, keep the current parity-and-benchmark discipline:
  first pass parity tests, then compare end-to-end benchmark surfaces rather
  than celebrating microkernel wins in isolation.

## Mapping To Current Aria-NBV Surfaces

### Current repo pattern

- `aria_nbv/aria_nbv/pose_generation/mojo_backend.py`
- `aria_nbv/aria_nbv/rendering/mojo_backend.py`
- `aria_nbv/aria_nbv/rri_metrics/mojo_backend.py`

These files already implement the recommended pattern:

1. Python owns orchestration and public contracts.
2. Python imports local Mojo modules through `mojo.importer`.
3. Python stages contiguous CPU buffers and passes raw addresses.
4. Mojo fills output buffers and returns through the same boundary.

### Best current targets

- Collision distance and clearance masks
- Closest-hit depth rendering
- Depth unprojection and compaction
- Point-to-mesh and mesh-to-point reductions
- Semidense projection, grid binning, and weighted reductions in VIN

### Usually not worth porting first

- Lightning or data-module orchestration
- dataset I/O and cache plumbing
- thin wrappers around already-optimized PyTorch3D kernels
- most VIN CNN math unless a benchmark shows the custom surrounding ops, not the
  convs, are the bottleneck

## Recommended Adoption Order

1. Keep Python as the orchestrator.
2. Preserve existing public contracts and backend enums.
3. Start with a Python-importable Mojo module.
4. Use CPU Mojo first when the win is removing Python overhead or replacing a
   CUDA-only extension.
5. Move to Metal GPU only when the kernel is clearly GPU-shaped and transfer
   costs will not dominate.
6. Validate parity against the repo baseline.
7. Benchmark on Apple Silicon before widening the Mojo surface.

## Translation Reminder

For CUDA donors, translate:

- public API
- memory layout
- thread ownership
- shared-memory reuse
- reduction semantics

Do not translate:

- CUDA syntax literally
- NVIDIA-only assumptions about debugging, intrinsics, or launch semantics
