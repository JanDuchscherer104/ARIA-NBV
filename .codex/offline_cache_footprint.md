# Offline cache footprint snapshot

- samples/: 263 GB
- vin_snippet_cache/: 0.809 GB
- Coverage fraction: 0.8 (80 / 100 scenes)
- Full coverage estimate: ~329 GB (samples + vin_snippet_cache)

Representative per-snippet tensor footprint (CPU):
- backbone_out ~305 MB (dominant; feat2d_upsampled ~212 MB)
- candidate_pcs ~11.2 MB
- depths ~4.6 MB
- candidates ~0.03 MB
- rri metrics ~0.002 MB
- vin_snippet ~1.0 MB
- total (with backbone) ~320.8 MB
- minimal training payload (no backbone/depths/pcs) ~1.1 MB
