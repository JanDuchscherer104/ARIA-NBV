Short task summary (2025-11-24):
- Fixed candidate frustum plotting to use all available per-frame camera calibrations instead of only the first frame. `plot_candidate_frusta` now supplies the full calibration sequence to the builder, and `add_candidate_frusta` accepts a sequence of `CameraTW` so frusta can align with each pose.
- Added a safety check to avoid empty camera inputs when plotting.

Notes / follow-ups:
- No tests were run for speed; consider a quick smoke plot to confirm visuals when time permits.
