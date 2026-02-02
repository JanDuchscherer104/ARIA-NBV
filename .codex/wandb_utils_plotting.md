# W&B utils: typing + plotting helpers

Summary
- Added typed protocols (`WandbApi`, `WandbRun`) to replace `Any` in W&B helpers.
- Made `_resolve_x_key` chain-friendly by returning `(x_key, history_copy)` without in-place mutation.
- Added seaborn plotting helpers: `plot_metric_curves`, `plot_dynamics_scatter`, `plot_dynamics_bar`, plus `prepare_history_long_dataframe`.
- Ensured pandas helpers return new dataframes suitable for chaining.

Files touched
- `oracle_rri/oracle_rri/utils/wandb_utils.py`
- `oracle_rri/oracle_rri/app/panels/wandb.py`
