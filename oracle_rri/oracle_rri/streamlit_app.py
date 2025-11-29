"""Streamlit entrypoint (thin wrapper around modular dashboard app)."""

from __future__ import annotations

from oracle_rri.dashboard.app import DashboardApp
from oracle_rri.dashboard.config import DashboardConfig

__all__ = ["DashboardApp", "DashboardConfig", "main", "streamlit_entry"]


def main() -> None:  # pragma: no cover - Streamlit runner
    DashboardConfig().setup_target().run()


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
