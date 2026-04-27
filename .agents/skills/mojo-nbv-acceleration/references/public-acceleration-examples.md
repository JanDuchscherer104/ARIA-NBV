# Public Mojo Acceleration Examples

Use this note when the task needs concrete donor examples instead of only API
docs.

## Official Modular Examples

### `modular/mojo-gpu-puzzles`

Repository:
- <https://github.com/modular/mojo-gpu-puzzles>

Useful surfaces:

- `solutions/p07/p07.mojo`
  - basic 2D grid and thread indexing with raw buffers
- `solutions/p07/p07_layout_tensor.mojo`
  - the same pattern expressed with `LayoutTensor`
- `book/src/puzzle_16/shared_memory.md`
  - shared-memory tiling mental model
- `book/src/puzzle_21/simple_embedding_kernel.md`
  - coalesced vs non-coalesced memory access and custom-op registration
- `problems/p19/op/attention.mojo`
  - tiled matmul, softmax, transposition, and composition of multiple kernels

Why it matters here:

- It shows the official Mojo GPU style for indexing, tiling, barriers,
  `LayoutTensor`, and launch configuration.
- It is the cleanest public reference for translating CUDA-style kernels into
  Mojo GPU kernels without guessing.
- The puzzle material includes Apple-oriented run targets for some exercises,
  which is useful when thinking about Metal portability instead of CUDA-only
  launch assumptions.

## ViSTA-SLAM CUDA RoPE Reference

Repository:
- <https://github.com/zhangganlin/vista-slam>

Relevant files:

- `vista_slam/sta_model/pos_embed/curope/curope2d.py`
- `vista_slam/sta_model/pos_embed/curope/curope.cpp`
- `vista_slam/sta_model/pos_embed/curope/kernels.cu`

Install hint from the upstream README:

- the repo treats the CUDA RoPE extension as optional acceleration and builds it
  with `python setup.py build_ext --inplace` under the `curope/` directory

Kernel structure to remember:

- Python autograd wrapper at the top level
- C++ extension dispatch in the middle
- CUDA kernel at the bottom
- in-place update of token buffers
- backward implemented by running the same rotation with the sign flipped

Lessons for Aria-NBV:

- This is a good example of a narrow custom kernel behind a stable Python API.
- The right first Mojo port is to preserve the Python-facing contract and
  reimplement the numeric core, not to redesign the full model stack.
- If the algorithm is mostly explicit indexing and per-element rotation, it can
  become either:
  - a Python-importable CPU Mojo kernel using raw pointers, or
  - a Metal GPU kernel when the data can stay on device long enough

Licensing note:

- The specific `curope` source files carry CC BY-NC-SA 4.0 headers upstream.
- Treat them as donor references, not copy-paste source for this repo, unless
  the task explicitly includes license handling and notices.

## Community Packaging And Acceleration Example

### `furnace-dev/sonic-mojo`

Repository:
- <https://github.com/furnace-dev/sonic-mojo>

Why it matters:

- It shows a real Mojo package around a high-performance external core.
- It is useful when the right answer is not “rewrite the whole accelerator in
  Mojo,” but “use Mojo as the repo-facing systems layer around an existing fast
  implementation.”
- It includes packaging, `mojo package`, examples, tests, and benchmark data.

Lessons for Aria-NBV:

- Keep packaging and distribution concerns separate from kernel logic.
- Consider FFI only when rewriting the numeric core is clearly lower leverage
  than wrapping an existing fast library.
- For repo-owned oracle kernels, prefer pure Mojo first because this repo
  already has a working Python-importable Mojo boundary.

## Repo-Local Donor Examples

Do not overlook the existing repo kernels:

- `aria_nbv/aria_nbv/pose_generation/mojo/mesh_collision_kernels.mojo`
- `aria_nbv/aria_nbv/rendering/mojo/oracle_render_kernels.mojo`
- `aria_nbv/aria_nbv/rri_metrics/mojo/oracle_distance_kernels.mojo`

These are the best first examples for:

- `PythonModuleBuilder`
- `PyInit_*`
- `mojo.importer` usage from Python
- raw-address buffer passing
- CPU parallel kernels with `parallelize`

When implementing in this repo, prefer extending these patterns over importing a
new packaging or runtime style from an unrelated public example.
