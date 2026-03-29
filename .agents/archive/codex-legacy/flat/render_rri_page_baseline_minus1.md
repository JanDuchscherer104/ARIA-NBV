# Task: Plot `pm_*_before` only once (index -1)

## Problem

In the Streamlit RRI page (`oracle_rri/oracle_rri/app/panels.py::render_rri_page`) we plotted the “before” point→mesh metrics (`pm_*_before`) for every candidate, even though these values are broadcast reference-only distances (semi-dense `P_t` only) and therefore identical across candidates.

This made the bar charts noisy (many identical gray bars).

## Change

- Plot `pm_dist_before`, `pm_acc_before`, and `pm_comp_before` only once at the synthetic x-category `"-1"` (representing the semi-dense-only baseline).
- Keep the per-candidate “after” bars unchanged.
- Force x-axis category order to `["-1", *candidate_ids]` so the baseline stays leftmost.

## Validation

- `ruff format oracle_rri/oracle_rri/app/panels.py`
- `ruff check oracle_rri/oracle_rri/app/panels.py`
- `pytest oracle_rri/tests/integration/test_oracle_rri_labeler_real_data.py -q`

## Notes / Follow-ups

- Unrelated failures observed when running plotting unit tests:
  - `oracle_rri/tests/test_plotting_semidense.py` currently fails because `EfmPointsView` has no `collapse_points_np` attribute.
  - This was not changed as part of this task.

