## Performance Refactor Notes (2025-11-24)

- Re-read PyTorch3D docs via MCP: collision/distance uses `point_mesh_face_distance`; rasterisation tuning via `RasterizationSettings` (`faces_per_pixel`, `bin_size`, `max_faces_per_bin`) from tutorials/renderer notes.
- GPU candidate path now mirrors CPU semantics; parity tests exist but need CUDA runner to exercise.
- Memory model for rasteriser (renderer.md) scales with `N*H*W*K`; reducing `faces_per_pixel` or enabling coarse-to-fine binning will save VRAM.

### TODO (short)
- [ ] Benchmark cached mesh sampling vs. per-call sampling; record speedup.
- [ ] Torch-compile `_point_mesh_distance_efm3d`; compare latency + correctness (blocked here: torch.compile worker perms in sandbox).
- [x] Expose fp16 / bin_size / max_faces_per_bin in config and measure impact on hit ratios.
- [x] Add `scripts/profile_nbv.py` to time sampling, collision, depth (CPU vs GPU).
- [ ] Run GPU parity tests on a CUDA host; store results.
