"""Shared UI helpers for Streamlit panels."""

from __future__ import annotations

import re
import traceback

import numpy as np
import streamlit as st

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _pretty_label(text: str) -> str:
    """Format labels by replacing underscores and title-casing words."""
    if not text:
        return text
    return text.replace("_", " ").title()


def _info_popover(label: str, text: str) -> None:
    with st.popover(f"Info: {label.title()}", icon="ℹ️"):
        st.markdown(text, unsafe_allow_html=True)


def _report_exception(exc: Exception, *, context: str) -> None:
    """Render a full traceback in the UI and emit it to stdout."""
    trace = traceback.format_exc()
    print(trace, flush=True)
    st.error(f"{context}: {type(exc).__name__}: {exc}")
    st.exception(exc)
    with st.expander("Full traceback", expanded=False):
        st.code(trace, language="text")


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.allclose(x, x[0]):
        return float("nan")
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return float("nan")


def _segment_indices(num: int, frac: float) -> tuple[slice, slice, slice]:
    size = max(2, int(num * frac))
    early = slice(0, size)
    late = slice(max(num - size, 0), num)
    mid_start = size
    mid_end = max(num - size, mid_start)
    mid = slice(mid_start, mid_end)
    return early, mid, late


__all__ = [
    "_info_popover",
    "_linear_slope",
    "_pretty_label",
    "_report_exception",
    "_segment_indices",
    "_strip_ansi",
]
