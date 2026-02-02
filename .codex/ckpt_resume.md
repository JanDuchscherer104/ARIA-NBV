## Checkpoint resume with config drift (2026-01-27)

### Findings
- `nbv-cli` import failed due to a stale `@field_validator("plot_stage")` in `AriaNBVExperimentConfig` (no `plot_stage` field exists), triggering a PydanticUserError at class construction.
- Resume flow previously relied on `trainer.fit(..., ckpt_path=...)` with a freshly constructed module; checkpoint hparams could still differ from current config without explicit logging.

### Changes
- Removed the stale `plot_stage` validator.
- Streamlined resume to avoid double-loading:
  - Module is always constructed from the current config.
  - Trainer resumes weights/optimizer state via `ckpt_path`.
  - Checkpoint hyperparameters are only read to log config drift.
- Added a lightweight pytest that writes a minimal checkpoint and verifies resume setup uses current config (head dropout override) without calling `load_from_checkpoint`.

### Tests
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/lightning/test_resume_checkpoint.py`
  - Passes locally.
  - `uv run pytest ...` failed under system Python due to missing `power_spherical` (expected interpreter mismatch).

### Follow-ups / Suggestions
- The `torch.load(..., weights_only=False)` warning surfaces in resume logging; if safe, consider switching to `weights_only=True` for metadata reads.
