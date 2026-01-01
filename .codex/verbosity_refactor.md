Task: Decouple console verbosity from debug mode and replace boolean `verbose` flags with a verbosity level across oracle_rri.

Changes:
- Introduced `Verbosity` IntEnum (quiet/normal/verbose) and reworked `Console` to use a global verbosity level; `set_debug(True)` now forces max verbosity while remaining independent from debug toggles.
- Updated configs and modules in data, pose_generation, rendering, analysis, and dashboard UI to accept `verbosity` (with `verbose` aliases for backward compatibility) and to wire consoles via `set_verbosity`.
- Dashboard sidebar now exposes explicit verbosity controls for dataset, candidate generator, and renderer forms; console initialization respects debug flag but no longer requires it for high verbosity.
- Adjusted tests to the new API (console semantics, downloader/assertions, dataset fixtures) and validated with `ruff format`, `ruff check`, and `pytest oracle_rri/tests/test_console.py -q`.

Notes:
- Verbosity remains a class-wide setting on `Console`; modules in debug mode still run at max verbosity but users can raise verbosity without enabling debug.
- Alias support (`validation_alias=AliasChoices("verbosity", "verbose")`) keeps old config inputs working while transitioning code/tests to the new field name.
