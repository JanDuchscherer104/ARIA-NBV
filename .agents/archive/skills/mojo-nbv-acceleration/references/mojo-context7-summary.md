# Mojo Context7 Summary For Aria-NBV

Primary Context7 library id: `/websites/modular_mojo`

This note condenses the official Modular Mojo docs that are relevant to incremental acceleration work in this repo. It is intentionally narrower than a general Mojo overview.

## Official-doc takeaways

### Python interoperability

Source family:
- `https://docs.modular.com/mojo/manual/python/`
- `https://docs.modular.com/mojo/manual/python/mojo-from-python`
- `https://docs.modular.com/mojo/std/python/`
- `https://docs.modular.com/mojo/std/python/python`

Documented capabilities:
- Mojo can call into Python through the `std.python` APIs.
- Mojo can import existing Python modules through the CPython runtime.
- Python can import Mojo modules through `mojo.importer`.
- Mojo can expose a Python-importable module through `PythonModuleBuilder` and a `PyInit_*` entrypoint.

Implication for this repo:
- The safest first integration path is to keep the current Python package and replace one hot function at a time behind a normal Python call boundary.
- This is a better first move than attempting a broad rewrite of `aria_nbv`.

### FFI and native boundaries

Source family:
- `https://docs.modular.com/mojo/std/ffi`

Documented capabilities:
- Mojo documents `std.ffi` and `external_call` for calling C functions.

Project-specific inference:
- The official docs retrieved here clearly cover calling C from Mojo and building Python-importable Mojo modules.
- A tighter native bridge from Python or PyTorch into Mojo is therefore a plausible integration pattern, but any Torch-extension design is an inference from these primitives, not a documented PyTorch recipe from the retrieved Mojo sources.

### GPU programming

Source family:
- `https://docs.modular.com/mojo/manual/gpu/fundamentals`
- `https://docs.modular.com/mojo/manual/gpu/basics`
- `https://docs.modular.com/mojo/manual/gpu/block-and-warp`

Documented capabilities:
- Launch kernels with `DeviceContext.enqueue_function(...)`.
- Use explicit grid, block, and thread indices.
- Use shared memory and synchronization barriers.
- Write tiled numeric kernels with explicit memory movement and reduction structure.

Implication for this repo:
- Mojo GPU work is a good fit only when the target computation has a clear kernel shape and benefits from fusion.
- That points toward ray tests, projection and binning, backprojection and compaction, and related reduction-heavy geometry kernels.

## Mapping To Current Aria-NBV Surfaces

### Best first targets

- `aria_nbv/aria_nbv/pose_generation/candidate_generation_rules.py`
  - Current code mixes Torch with `trimesh` and NumPy round-trips for mesh distance and ray intersection work.
  - Strong candidate for a narrow CPU or GPU kernel replacement.
- `aria_nbv/aria_nbv/rendering/candidate_pointclouds.py`
  - Current path unprojects dense candidate depth samples and compacts valid points afterward.
  - Strong candidate for fused valid-mask, unprojection, compaction, and bounds accumulation.
- `aria_nbv/aria_nbv/rri_metrics/oracle_rri.py`
  - Current path expands and concatenates repeated point clouds across candidates before batched distance work.
  - Strong candidate for a fused baseline-plus-candidate distance kernel if memory traffic dominates.
- `aria_nbv/aria_nbv/vin/model_v3.py`
  - Projection and grid-building helpers create large tiled candidate-point views and repeated scatter-style reductions.
  - Strong candidate when training time is materially affected by semidense projection features.

### Usually not worth porting first

- `aria_nbv/aria_nbv/rendering/pytorch3d_depth_renderer.py`
  - Real cost is already inside PyTorch3D rasterization.
- Most of the VIN scorer head in `aria_nbv/aria_nbv/vin/model_v3.py`
  - Mostly standard Torch kernels.
- `aria_nbv/aria_nbv/data_handling/_offline_writer.py`
- `aria_nbv/aria_nbv/data_handling/cache_index.py`
- `aria_nbv/aria_nbv/lightning/`
  - These are more orchestration, serialization, or I/O oriented than kernel oriented.

## Recommended adoption order

1. Keep Python as the orchestrator.
2. Replace a single helper behind an existing function boundary.
3. Validate correctness against the current implementation.
4. Measure end-to-end improvement at the snippet or batch level, not just microbenchmarks.
5. Only then consider widening the Mojo surface.

## Query recipes

Use these when the summary is not enough:

- Library id: `/websites/modular_mojo`
- Query: `Python interoperability import Python modules from Mojo PythonModuleBuilder PyInit mojo.importer`
- Query: `std.ffi external_call pointers C interop`
- Query: `GPU fundamentals DeviceContext enqueue_function block_idx thread_idx shared memory barrier`

