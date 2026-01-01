# Fix: Depth grid vs histogram candidate indexing

## Symptom

In the Streamlit “Candidate Renders” page:

- the **depth grid** titles showed non-contiguous candidate ids (e.g. `Candidate 0, 1, 2, 3, 8, 11, …`),
- the **depth histograms** were titled `cand 0..15` (continuous),

which looked like a depth↔histogram mismatch, especially when `max_candidates_final=16`.

## Root cause

The page displays a *filtered* batch of candidates after rendering:

- `CandidateDepths.candidate_indices` are **global indices** into the full candidate list (pre-render filtering).
- The histogram plot used **local indices** (`0..C-1`) because `depth_histogram(...)` hardcoded subplot titles as `cand {i}`.

The data order was correct, but the labels were inconsistent.

## Fix

- `oracle_rri/oracle_rri/rendering/plotting.py`
  - `depth_histogram` now accepts an optional `titles` iterable and uses it for subplot titles.
  - Reintroduced `hit_ratio_bar` (used by tests and useful for quick diagnostics).
  - `depth_grid` hit ratio now ignores non-positive depths.
- `oracle_rri/oracle_rri/app/panels.py`, `oracle_rri/oracle_rri/dashboard/panels.py`
  - Titles now show both local and global indices: `cand {local} (id {global})`.
  - Added a caption explaining what `id` means.

## Tests

- `oracle_rri/tests/rendering/test_rendering_plotting_helpers.py` exercises `depth_histogram` and `hit_ratio_bar`.
