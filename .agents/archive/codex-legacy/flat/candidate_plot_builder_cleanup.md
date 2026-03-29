Candidate plot builder cleanup (2025-11-30)
------------------------------------------------

What changed
- Removed free functions `plot_candidates` and `plot_candidate_frusta`; dashboards now use `CandidatePlotBuilder` directly.
- Added `CandidatePlotBuilder.from_candidates`, internal caching for centers, and optional reference-marker support in `add_candidate_points`.
- All `add_frame_axes` calls in candidate plots now take `candidates.reference_pose` to show the true reference frame.
- `render_candidates_page` builds both the positions and frusta figures via the builder; rejected-only view uses a minimal snippet builder.
- New fluent helpers: `add_candidate_cloud`, `add_rejected_cloud`, and `add_reference_axes`; panels now consume these instead of manual colour/axis wiring. Reference/center caching reduces repeated CPU transfers.

Open follow-ups / observations
- Consider letting `add_candidate_points` accept discrete colour strings without forcing a colourscale (minor Plotly warning risk).
- If we want per-rule colouring for rejected-only view, extend the builder to accept a boolean mask to filter points instead of recomputing poses.
- The doc note in `depth-render-revision.md` still mentions the removed free functions; update documentation next time that file is touched.
