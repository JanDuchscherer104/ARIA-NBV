# RichModelSummary all-modules note (2025-12-24)

- Context: In PL 1.9.5, `ModelSummary`/`RichModelSummary` use `max_depth` to control nesting; `-1` shows *all* submodules.
- Current code: `oracle_rri/oracle_rri/lightning/lit_trainer_callbacks.py` wires `RichModelSummary(max_depth=rich_summary_max_depth)` with default `rich_summary_max_depth=1`, so only the top-level module prints.
- Recommendation: Set `rich_summary_max_depth=-1` in the callbacks config (e.g., in TOML/CLI) or change the default if you want this behavior globally.
- Optional: If you also want input/output sizes in the summary table, define `example_input_array` on the LightningModule.
