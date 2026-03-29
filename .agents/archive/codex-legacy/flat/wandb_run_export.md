# W&B run export (73j3pj7v)

## What was exported
- Pickle path: `/home/jandu/repos/NBV/wandb_run_73j3pj7v.pkl`.
- Contents: run metadata, config, summary, full metric history (via `scan_history`), system history (via `history(stream="system")`), file list, logged artifacts, used artifacts.

## Counts
- history rows: 8001
- system history rows: 585
- files: 64
- logged artifacts: 2
- used artifacts: 0

## Notes / caveats
- W&B API returned `created_at`/`heartbeat_at` as strings for this run; exported as-is.
- System history uses `run.history(..., stream="system")`; if you need more system keys or additional sampling, adjust the call.
