"""Session-state keys and helpers for the Streamlit dashboard."""

from __future__ import annotations

from typing import Any, Literal, TypedDict, cast

import streamlit as st

STATE_KEYS = {
    "sample": "nbv_sample",
    "sample_cfg": "nbv_sample_cfg",
    "candidates": "nbv_candidates",
    "cand_cfg": "nbv_cand_cfg",
    "depth": "nbv_depth_batch",
    "depth_cfg": "nbv_depth_cfg",
    "sample_idx": "nbv_sample_idx",
    "dataset_iter": "nbv_dataset_iter",
}

TASK_KEYS = {
    "data": "nbv_task_data",
    "candidates": "nbv_task_candidates",
    "depth": "nbv_task_depth",
}


class TaskState(TypedDict):
    status: Literal["idle", "running", "done", "error"]
    error: str | None


class SessionVars(TypedDict, total=False):
    nbv_task_data: TaskState
    nbv_task_candidates: TaskState
    nbv_task_depth: TaskState


def state() -> SessionVars:
    return cast(SessionVars, st.session_state)


def safe_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()  # type: ignore[attr-defined]
    else:  # pragma: no cover
        raise RuntimeError("Streamlit rerun API not available.")


def init_task_state() -> None:
    for key in TASK_KEYS.values():
        state().setdefault(key, TaskState(status="idle", error=None))


def store(key: str, value: Any) -> None:
    st.session_state[key] = value


def get(key: str) -> Any:
    return st.session_state.get(key)
