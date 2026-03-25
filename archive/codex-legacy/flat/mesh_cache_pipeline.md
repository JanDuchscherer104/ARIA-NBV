## Mesh cache pipeline (Nov 2025)

- Added processed-mesh directory to PathConfig (`processed_meshes`) with resolver for per-scene/snippet artifacts.
- Shared cache in `oracle_rri/data/mesh_cache.py` now crops/simplifies once, saves .ply, and returns cached torch verts/faces plus spec hash for reuse.
- Dataset attaches cached verts/faces and `mesh_cache_key` to `EfmSnippetView`; `.to()` keeps them.
- CandidateViewGenerator uses the shared P3D mesh cache via `get_pytorch3d_mesh` instead of its own per-instance cache.
- CandidateDepthRenderer hands cached verts/faces + cache key to the PyTorch3D renderer; CPU renderer still uses trimesh.
- Smoke run (scene 81283, CPU render 0.25x res) succeeded: cached key `32bec14d5fb9`, 6/6 candidates, depth batch `[3,64,64]`, render ~20s; processed meshes reside in `.data/ase_meshes_processed`.
