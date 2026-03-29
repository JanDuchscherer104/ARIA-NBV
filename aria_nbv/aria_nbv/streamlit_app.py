"""Streamlit entrypoint (refactored app).

This is the default entrypoint used by the `nbv-st` console script.
The legacy dashboard remains available via `oracle_rri.streamlit_app_old`.
"""

from __future__ import annotations

from oracle_rri.app import NbvStreamlitApp, NbvStreamlitAppConfig

__all__ = ["NbvStreamlitApp", "NbvStreamlitAppConfig", "main", "streamlit_entry"]


def main() -> None:  # pragma: no cover - Streamlit runner
    NbvStreamlitAppConfig().setup_target().run()


def streamlit_entry() -> None:  # pragma: no cover - console script
    """Launch via `nbv-st` console entry."""

    import sys
    from pathlib import Path

    from streamlit.web.cli import main as st_main

    # streamlit CLI does not accept "-m"; pass absolute path to this file
    app_path = Path(__file__).resolve()
    sys.argv = ["streamlit", "run", str(app_path)]
    st_main()


if __name__ == "__main__":  # pragma: no cover
    main()
