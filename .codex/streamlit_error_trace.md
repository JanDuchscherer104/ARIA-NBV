# Streamlit verbose error handling (2025-11-25)

- Added fail-soft wrapper around `DashboardApp.run` that catches any uncaught exceptions, logs the full traceback, renders `st.exception`, and shows a pre-expanded traceback block. Uses `Console` so CLI logs mirror UI output.
- Extracted prior run logic into `_render_body` to keep functionality unchanged while enabling the wrapper.
- Added `.streamlit/config.toml` with `showErrorDetails = "full"` to always surface full traces in the UI.

Notes / follow-ups:
- Consider a feature flag to hide traces in production demos while keeping logging enabled.
- If callbacks start using `on_click`/`on_change`, wrap them with similar try/except helpers to keep consistency.

## Update (2025-12-31)

- Added `_report_exception(...)` in `oracle_rri/oracle_rri/app/panels.py` to print full tracebacks to stdout and surface them in the UI (error + exception + expandable traceback).
- VIN diagnostics and offline stats now capture `traceback.format_exc()` for UI failures and show the full trace in-stream instead of only the error message.
