Panels cleanup (2025-11-29)
===========================

What changed
------------
- Simplified Streamlit dashboard candidate panel to rely directly on `CandidateSamplingResult` typed fields (`poses`, `shell_poses`, `masks`, `mask_valid`) and removed legacy `_cand_*` accessors.
- Rejection plotting now uses `~mask_valid` over `shell_poses` without intermediate dict stacking.
- Rule-mask plotting uses the typed mask dict and shell pose tensor directly; candidate frusta rendering feeds `candidate_cams=poses`.

Notes / follow-ups
------------------
- Assumes `mask_valid` length matches `shell_poses`; if future generators change this contract, add an assertion near result creation instead of UI-side guards.
- Consider adding a lightweight Streamlit smoke test to exercise `render_candidates_page` with a small fixture to catch plotting regressions.
- CandidatePlotBuilder now delegates frustum plotting to the base PlotBuilder and consumes attached `CandidateSamplingResult`; call-sites can simply `attach_candidate_results(...).add_candidate_frusta()` without passing cams/poses.
- `add_candidate_frusta` was removed from `data.plotting.SnippetPlotBuilder`; the pose-generation plotting module now owns frustum plotting and routes through `_add_frusta_for_poses`, keeping dataset plotting generic.
- Resolved TODOs: `CandidateSamplingResult.views` is now the single source for frustum plotting (no extra args); `CandidatePlotBuilder` initialises `candidate_results=None` and requires `attach_candidate_results` before plotting.
- add_candidate_frusta now passes the batched `views` directly so `_add_frusta_for_poses` iterates per-camera; avoids scalar conversion errors in `get_frustum_segments` and matches the per-candidate frustum expectation.
