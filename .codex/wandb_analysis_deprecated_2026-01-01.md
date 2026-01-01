# W&B analysis page refresh (deprecated dashboard)

## Summary
- Added a new `render_wandb_analysis_page` in `.deprecated/dashboard/panels.py` with deeper run diagnostics (trend, gap, impact ranking) and a doc-classifier attribution explorer.
- Integrated checkpoint selection from `PathConfig.checkpoints` and a local uploader-based attribution workflow using the Captum-based engine from `external/doc_classifier/interpretability/attribution.py`.
- Added supporting helpers for W&B run resolution, history loading, and correlation/impact summaries.

## Key findings
- Attribution explorer assumes Lightning checkpoints include `hyper_parameters` (same expectation as `DocClassifierRuntime`).
- If the `traenslenzor` package is not installed, the code falls back to importing `external/doc_classifier` by injecting the repo `external/` path into `sys.path`.
- The attribution section depends on `albumentations`, `PIL`, and Captum; missing dependencies will prevent attribution rendering.

## Open suggestions
- Consider refactoring `render_wandb_analysis_page` into smaller helpers to reduce complexity warnings and improve testability.
- If the deprecated dashboard is still in use, add `__init__.py` under `.deprecated/` or relax lint rules for that subtree.
- Optionally wire the new panel into `.deprecated/dashboard/app.py` navigation if it should be user-facing.

## Tests and lint
- `ruff format .deprecated/dashboard/panels.py` ran.
- `ruff check .deprecated/dashboard/panels.py` still reports numerous pre-existing issues in the deprecated dashboard (relative imports, missing docstrings, line length, etc.).
- No integration tests were run; this UI-only change does not map cleanly to existing pytest coverage.
