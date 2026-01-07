## Task
Suggest additional places to use the new Streamlit loading bars.

## Suggestions
- Offline cache scan loops (train/val label hist, cache writer runs).
- Binner fitting iterations (train dataloader loop).
- Candidate generation / depth/pointcloud rendering pipelines (per-snippet loops).
- Dataset downloads or cache rebuilds (long-running file I/O).
- Heavy plotting/analysis panels that aggregate large caches.

## Notes
- Use progress bars for per-split scans and long-running per-snippet loops; add a cancel/skip button when possible.
