# Global GPU Mode Notes (Nov 23, 2025)

- Added `oracle_rri.utils.performance` with global `NBV_PERFORMANCE_MODE` (`auto|gpu|cpu`) and helper APIs (`select_device`, `pick_fast_depth_renderer`, `set_performance_mode`).
- Config defaults now resolve devices via `select_device`:
  - `CandidateViewGeneratorConfig`, `CandidatePointCloudGeneratorConfig`, `DepthDebuggerConfig` pick CUDA when GPU mode is active (CPU when mode=cpu or debug).
  - Rendering backends use the resolver; PyTorch3D renderer respects debug/availability, CPU renderer still returns tensors on the resolved device.
- `CandidateDepthRendererConfig` auto-upgrades to PyTorch3D in `gpu` mode (preserves zfar, sets device to CUDA); no upgrades in `auto`/`cpu` to avoid surprising CPU-only runs.
- New tests: `oracle_rri/tests/test_performance_mode.py` (simulated CUDA via monkeypatch) covering device resolution, renderer upgrade, and generator/device propagation.
- Docs: `oracle_rri/README.md` documents the new flag + programmatic toggle.

Open follow-ups / cautions
- Consider exposing mode in Streamlit sidebars (default to env value) to make UI consistent.
- Monitor any heavy CUDA allocations in `CandidateViewGeneratorConfig` default (num_samples=512) on smaller GPUs; may need guardrails or warnings.
