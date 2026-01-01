# Candidate depth renderer oversampling

- Implemented pix_to_face based validity filtering in `CandidateDepthRenderer.render`; candidates with any miss are dropped, and outputs are capped at `max_candidates_final` after oversampling.
- Oversampling count now uses `ceil(oversample_factor * max_candidates_final)` to avoid under-drawing when factors are fractional; errors are raised when no renders remain.
- Added warnings/logs when fewer valid renders are available than requested to aid debugging mesh coverage issues.
- Follow-up: update/extend tests around `CandidateDepths` (existing `tests/rendering/test_unproject.py` still references old fields) to cover the new filtering behavior and keep the test suite aligned.
