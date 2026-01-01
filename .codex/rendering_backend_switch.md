# Rendering Backend Switch (Nov 22, 2025)

- **What changed**: `CandidateDepthRenderer` now accepts both backends:
  - GPU raster: `Pytorch3DDepthRendererConfig`
  - CPU rays: `Efm3dDepthRendererConfig`
  The Streamlit app exposes a sidebar selector to toggle the backend and shows backend-specific knobs (faces_per_pixel vs. chunk_rays/proxy walls).

- **CPU renderer**: implemented full trimesh/pyembree ray tracer with proxy walls (mesh∪semidense bounds), chunked casting, and batch fallback. Exported via `rendering/__init__.py`.

- **Integration coverage**:
  - Added real ASE snippet test (scene 81283) for both backends (`test_candidate_renderer_integration.py`), decimating mesh + downscaling calib for runtime. Both produce non-empty depths.
  - All rendering tests (unit + integration) pass (`pytest oracle_rri/tests/rendering`).

- **Runtime tips**:
  - PyTorch3D path ~90 s on CPU-only runs for the integration test; CPU ray path ~30 s with decimated mesh (5k faces) and 64×64 images.
  - In Streamlit, pick CPU backend when no GPU is available; keep `chunk_rays` moderate (e.g., 200k) to avoid RAM spikes.

- **Open follow-ups**:
  - Optionally auto-select backend based on CUDA availability.
  - Add a parity check (depth histogram similarity) between backends on a fixed snippet.
  - Consider caching decimated meshes / downscaled calibs for UI speed.
