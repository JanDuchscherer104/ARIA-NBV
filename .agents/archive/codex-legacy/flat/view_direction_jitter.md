## View direction jitter fixes (Dec 5, 2025)

- Orientation jitter now honours per-axis caps (`view_max_azimuth_deg`, `view_max_elevation_deg`) and falls back to deterministic views when all caps are zero, even if a `view_sampling_strategy` is set. This fixes the prior behaviour where enabling `view_sampling_strategy` ignored zero jitter and produced arbitrary view directions.
- Added rejection+clamp logic so sampled camera forward vectors stay within the configured azimuth/elevation bounds; jitter sampling still uses the existing uniform/power-spherical distributions.
- Backward compatibility preserved via `view_max_angle_deg` defaulting both new caps; negative jitter values are rejected during config validation.
- Regression tests: `tests/pose_generation/test_orientations.py` ensures zero-jitter keeps the base orientation and that sampled forwards respect azimuth/elevation limits. Run with the project venv (`oracle_rri/.venv/bin/python -m pytest tests/pose_generation/test_orientations.py`).

Open follow-ups:
- Consider allowing jitter when `view_sampling_strategy` is `None` by defaulting to a bounded uniform sampler; currently jitter is disabled unless a strategy is set.
- If we want tighter control over roll jitter, add a similar per-axis clamp or expose seedable sampling for deterministic test fixtures.
