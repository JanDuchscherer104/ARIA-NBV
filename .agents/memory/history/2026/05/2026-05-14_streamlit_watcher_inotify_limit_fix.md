---
id: 2026-05-14_streamlit_watcher_inotify_limit_fix
date: 2026-05-14
title: "Streamlit watcher inotify limit fix"
status: done
topics: [streamlit, package, diagnostics]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/streamlit_app.py
  - aria_nbv/tests/test_streamlit_entry.py
---

## Task
Diagnose and fix `uv run nbv-st` failing with Streamlit watchdog `OSError: [Errno 28] inotify watch limit reached`.

## Method
Used `diagnose-aria` plus package guidance, inspected the `nbv-st` console entry and Streamlit watcher config, ran a bounded headless `uv run nbv-st` smoke loop, and added focused pytest coverage around the wrapper's generated Streamlit argv.

## Findings
The local host did not reproduce ENOSPC because `/proc/sys/fs/inotify/max_user_watches` was `65536`, but the user traceback localizes to Streamlit's `auto` watcher selecting watchdog and recursively scheduling inotify watches for imported ARIA-NBV package directories.

`aria_nbv/aria_nbv/streamlit_app.py` now injects `--server.fileWatcherType none` before the app path when the caller has not set `STREAMLIT_SERVER_FILE_WATCHER_TYPE` or a `--server.fileWatcherType` flag. The wrapper also preserves Streamlit flags before `--` and app script args after `--`.

`aria_nbv/tests/test_streamlit_entry.py` covers the default watcher suppression, CLI override, environment override, and `streamlit_entry()` sys.argv rewrite.

## Verification
Passed:

- `cd aria_nbv && uv run pytest tests/test_streamlit_entry.py`
- `cd aria_nbv && uv run ruff format aria_nbv/streamlit_app.py tests/test_streamlit_entry.py`
- `cd aria_nbv && uv run ruff check aria_nbv/streamlit_app.py tests/test_streamlit_entry.py`
- `cd aria_nbv && timeout 20s env STREAMLIT_SERVER_HEADLESS=true uv run nbv-st --server.port 8502`

The smoke command exited with `124` because `timeout` intentionally stopped the running server after it reached `Uvicorn server started on 0.0.0.0:8502`; no inotify watcher error appeared.

## Canonical State Impact
None. This is a wrapper robustness fix, not a new durable thesis or package contract.
