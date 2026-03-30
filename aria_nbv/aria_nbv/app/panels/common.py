"""Shared UI helpers for Streamlit panels."""

from __future__ import annotations

import re
import traceback

import streamlit as st

from ...utils.plotting import pretty_label
from ...utils.stats import linear_slope, segment_indices

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from captured console text."""
    return _ANSI_RE.sub("", text)


def info_popover(label: str, text: str) -> None:
    """Render a small informational popover with Markdown content."""
    with st.popover(f"Info: {label.title()}", icon="ℹ️"):
        st.markdown(text, unsafe_allow_html=True)


def report_exception(exc: Exception, *, context: str) -> None:
    """Render a full traceback in the UI and emit it to stdout."""
    trace = traceback.format_exc()
    print(trace, flush=True)
    st.error(f"{context}: {type(exc).__name__}: {exc}")
    st.exception(exc)
    with st.expander("Full traceback", expanded=False):
        st.code(trace, language="text")


_info_popover = info_popover
_linear_slope = linear_slope
_pretty_label = pretty_label
_report_exception = report_exception
_segment_indices = segment_indices
_strip_ansi = strip_ansi

__all__ = [
    "info_popover",
    "linear_slope",
    "pretty_label",
    "report_exception",
    "segment_indices",
    "strip_ansi",
]
