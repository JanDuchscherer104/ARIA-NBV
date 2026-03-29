# W&B CLI export integration

Summary
- Added a new CLI entry `nbv-wandb` (via `oracle_rri.lightning.cli:wandb_main`).
- Introduced `oracle_rri/utils/wandb_utils.py` to consolidate W&B analysis helpers and local figure discovery.
- `wandb_main` can export meta/summary/config/dynamics CSVs, per-run histories, and a train/val figure manifest under `.logs/wandb/analysis`.
- Streamlit W&B panel now reuses the new utils and shows local train/val figure galleries for the focus run.

Key usage
- `uv run nbv-wandb --entity <entity> --project <project> --max-runs 50`
- `uv run nbv-wandb --run-ids <id1,id2> --output-dir .logs/wandb/analysis`

Files touched
- `oracle_rri/oracle_rri/utils/wandb_utils.py`
- `oracle_rri/oracle_rri/lightning/cli.py`
- `oracle_rri/oracle_rri/app/panels/wandb.py`
- `oracle_rri/oracle_rri/configs/wandb_config.py`
- `oracle_rri/pyproject.toml`
- `README.md`
