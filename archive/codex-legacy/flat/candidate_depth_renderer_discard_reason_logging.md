# CandidateDepthRenderer discard-reason logging

## Goal

Make `_filter_valid_candidates` log *why* candidates are discarded: invalid renders (zero valid depth pixels) vs.
discarded due to the `max_candidates_final` cap.

## Changes

- `oracle_rri/oracle_rri/rendering/candidate_depth_renderer.py`
  - `_filter_valid_candidates` now logs:
    - `invalid_zero_hit=<n>`: number of discarded candidates with **zero** valid depth pixels.
    - `capped=<n>`: number of discarded candidates purely due to `max_candidates_final=<k>`.
  - Keeps the existing deterministic ranking logic (valid pixel count desc, then candidate index asc).
  - `render()` calls `_filter_valid_candidates` again, so the log line is emitted during normal execution.
  - `CandidateDepthRendererConfig` accepts `max_candidates` as an alias for `max_candidates_final`.

- `oracle_rri/tests/rendering/test_candidate_depth_renderer_filter_logging.py`
  - Adds regression tests using `Console.set_sink(...)` to assert the new log messages.

## Notes / follow-ups

- If needed, add a separate log in `_select_candidate_views` to make it explicit how many *valid* candidates were
  not rendered at all due to the oversampling slice (`num_render`).
