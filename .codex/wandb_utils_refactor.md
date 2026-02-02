# W&B utils refactor (wandb_utils.py)

Summary
- Moved W&B analysis helpers out of `configs/wandb_config.py` into new `utils/wandb_utils.py`.
- Simplified `WandbConfig` to only contain logger setup.
- Streamlit panel now uses `wandb_utils` for run loading, history loading, dynamics summaries, and local figure discovery.
- Added local figure viewer for train/val figures in the panel (focus run).

Redundancies removed
- Duplicate metric summarization, run metadata, and history loading logic previously spread across `wandb_config.py` and `app/panels/wandb.py`.
- Multiple APIs for W&B access (panel vs config) consolidated into `wandb_utils._ensure_wandb_api` and `wandb_utils._load_runs_filtered`.

New capabilities
- `build_run_dataframes` for meta/summary/config dataframes.
- `load_run_histories` for full metric histories per run.
- `build_dynamics_dataframe` for per-run dynamics summaries.
- `collect_run_media_images` for local train/val figure lookup (latest-run or run-id directory).

Files touched
- `oracle_rri/oracle_rri/utils/wandb_utils.py`
- `oracle_rri/oracle_rri/configs/wandb_config.py`
- `oracle_rri/oracle_rri/app/panels/wandb.py`
