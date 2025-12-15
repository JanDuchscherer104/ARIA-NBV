"""Streamlit application (refactored).

This package contains the revised Streamlit UI that replaces the older
`oracle_rri.dashboard` implementation. The app is intentionally split into:

- **state**: strongly typed session state + cache entries
- **controller**: pipeline orchestration (data → candidates → renders → RRI)
- **pages**: UI per tab/page (plots only; no heavy compute)
"""

from __future__ import annotations

from typing import Any

__all__ = ["NbvStreamlitApp", "NbvStreamlitAppConfig"]


def __getattr__(name: str) -> Any:
    """Lazily import Streamlit-heavy modules.

    This keeps `oracle_rri.app.controller` and other non-UI helpers importable in
    environments where Streamlit isn't installed.
    """

    if name == "NbvStreamlitApp":
        from .app import NbvStreamlitApp

        return NbvStreamlitApp
    if name == "NbvStreamlitAppConfig":
        from .config import NbvStreamlitAppConfig

        return NbvStreamlitAppConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
