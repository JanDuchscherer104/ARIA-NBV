# Ctrl+C Checkpoint Handling

## Summary
- Added interrupt checkpoint saving in `AriaNBVExperimentConfig` and CLI Ctrl+C handling.
- Standardized default trainer checkpoint directory to `PathConfig.checkpoints`.
- Added tests for interrupt checkpoint saving path.

## Key Changes
- `oracle_rri/oracle_rri/lightning/aria_nbv_experiment.py`
  - Saves an interrupt checkpoint into `PathConfig.checkpoints` when training is interrupted.
  - Important: Lightning 1.9.5 *swallows* `KeyboardInterrupt` inside `trainer.fit`.
    - We therefore detect `trainer.interrupted` after `fit()` returns, save the checkpoint, and then re-raise `KeyboardInterrupt` for consistent CLI behavior.
  - Defaults `TrainerCallbacksConfig.checkpoint_dir` to `PathConfig.checkpoints`.
- `oracle_rri/oracle_rri/lightning/cli.py`: graceful Ctrl+C exit with a warning and exit code 130.
- `tests/lightning/test_interrupt_checkpoint.py`: regression test for interrupt checkpoint saving behavior.
- `docs/contents/todos.qmd`: marked the Ctrl+C checkpoint task as done.

## Findings
- If you rely only on `ModelCheckpoint`, an interrupt mid-epoch can yield *no checkpoint* (depending on save frequency / epoch completion).
- The log snippet that shows `dir=/home/jandu/repos/NBV/.logs/checkpoints` indicates the callback directory, but the directory can still be empty if no save trigger happened before Ctrl+C.

## Suggestions
- When running tests, use the uv-managed venv (`oracle_rri/.venv/bin/python -m pytest`) to ensure required deps like `power_spherical` are available.
- Consider adding a `save_last` option (or explicit `every_n_train_steps`) if you want periodic checkpoints even without epoch boundaries.
