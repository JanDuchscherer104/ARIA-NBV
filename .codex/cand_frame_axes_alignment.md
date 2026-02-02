# Candidate Frame Axes Alignment (2026-01-26)

## Findings
- Candidate centers are sampled in a gravity-aligned sampling pose when `align_to_gravity=True`, but plots were using the physical reference pose axes, so the reference frame looked asymmetric relative to the candidate cloud.

## Decisions
- Store the gravity-aligned `sampling_pose` on `CandidateSamplingResult` and use it for plotting reference axes by default to keep axes symmetric with the candidate cloud.
- Clarify UI copy to indicate that axes reflect the sampling frame (gravity-aligned when enabled).

## Changes
- Added `sampling_pose` to `CandidateContext` and `CandidateSamplingResult`.
- Updated `CandidatePlotBuilder.add_reference_axes` to prefer the sampling pose and auto-title it as “Sampling frame” when it differs from the physical reference.
- Updated candidate panel popover text to reflect sampling-frame axes.

## Tests
- `oracle_rri/.venv/bin/python -m pytest tests/pose_generation/test_orientations.py`
- `oracle_rri/.venv/bin/python -m pytest tests/data/test_efm_dataset_snippet.py`

## Follow-ups
- Consider a UI toggle to display both sampling and physical reference axes for debugging.
- If `shell_offsets_ref` gets used downstream, consider renaming or adding a reference-frame variant to avoid confusion.
