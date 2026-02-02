# Task: Candidate source UI move

## Summary
- Moved the online/offline candidate source selection and offline cache controls into `app.py`, placing the selector above the expander and swapping expander content based on source.
- Simplified `render_candidates_page` to render only, with optional captions for offline samples.
- Removed offline UI logic from `app/panels/candidates.py` and added offline cache loading helpers in `app.py`.

## Files touched
- `oracle_rri/oracle_rri/app/app.py`
- `oracle_rri/oracle_rri/app/panels/candidates.py`

## Tests
- `ruff format oracle_rri/oracle_rri/app/app.py oracle_rri/oracle_rri/app/panels/candidates.py`
- `ruff check oracle_rri/oracle_rri/app/app.py oracle_rri/oracle_rri/app/panels/candidates.py`

## Follow-ups
- None.
