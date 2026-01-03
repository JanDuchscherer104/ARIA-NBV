"""Weights & Biases comparison panel."""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:  # Optional dependency for W&B diagnostics.
    import wandb  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency guard
    wandb = None

from ...configs.wandb_config import (
    WANDB_STEP_KEYS,
    _extract_run_steps,
    _filter_runs,
    _flatten_mapping,
    _list_entities,
    _list_projects,
    _list_runs,
    _load_wandb_history,
    _metric_pairs,
    _safe_mapping,
    _select_metric_key,
)
from .common import _info_popover, _linear_slope, _pretty_label, _report_exception, _segment_indices

_ENTITY_CACHE_TTL_S = 300
_DEFAULT_METRIC_FILTER = (
    "train/coral_loss_rel_random_step|train/pred_rri_mean_epoch|val-aux/spearman|val/coral_loss_rel_random"
)


@st.cache_data(ttl=_ENTITY_CACHE_TTL_S)
def _cached_entities(api_key: str) -> list[str]:
    """Cache available W&B entities for the current user."""
    if wandb is None:
        return []
    try:
        api = wandb.Api(api_key=api_key) if api_key else wandb.Api()
    except Exception:  # pragma: no cover - API guard
        return []
    return _list_entities(api)


@st.cache_data(ttl=_ENTITY_CACHE_TTL_S)
def _cached_projects(api_key: str, entity: str) -> list[str]:
    """Cache available W&B projects for an entity."""
    if wandb is None or not entity:
        return []
    try:
        api = wandb.Api(api_key=api_key) if api_key else wandb.Api()
    except Exception:  # pragma: no cover - API guard
        return []
    return _list_projects(api, entity=entity)


def _format_timestamp(value: Any) -> str:
    """Render timestamps consistently for run tables."""
    if value is None:
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _select_with_custom(
    column: Any,
    *,
    label: str,
    options: list[str],
    default: str,
    custom_label: str,
) -> str:
    """Render a selectbox with a custom override option."""
    cleaned = [opt for opt in options if opt]
    if default and default not in cleaned:
        cleaned.insert(0, default)
    cleaned.append("Custom...")
    index = cleaned.index(default) if default in cleaned else 0
    choice = column.selectbox(label, options=cleaned, index=index)
    if choice == "Custom...":
        return column.text_input(custom_label, value=default)
    return choice


def _render_run_filters(cache: dict[str, Any]) -> dict[str, Any]:
    """Render run filter controls and return the selected values."""
    with st.expander("Run filters", expanded=False):
        name_regex_raw = st.text_input("Name regex", value=str(cache.get("name_regex", "")))
        tags_raw = st.text_input(
            "Tags (comma-separated, any-match)",
            value=str(cache.get("tags_filter", "")),
        )
        group_filter = st.text_input("Group", value=str(cache.get("group_filter", "")))
        job_type_filter = st.text_input("Job type", value=str(cache.get("job_type_filter", "")))
        state_options = ["running", "finished", "failed", "crashed", "killed"]
        states = st.multiselect(
            "States",
            options=state_options,
            default=cache.get("states", ["finished"]),
        )
        col_s1, col_s2 = st.columns(2)
        min_steps = col_s1.number_input(
            "Min steps (0 = off)",
            min_value=0,
            value=int(cache.get("min_steps", 0)),
            step=100,
        )
        max_steps = col_s2.number_input(
            "Max steps (0 = off)",
            min_value=0,
            value=int(cache.get("max_steps", 0)),
            step=100,
        )
    return {
        "name_regex_raw": name_regex_raw,
        "tags_raw": tags_raw,
        "group_filter": group_filter,
        "job_type_filter": job_type_filter,
        "states": states,
        "min_steps": min_steps,
        "max_steps": max_steps,
    }


def _normalize_step_bounds(min_steps: int, max_steps: int) -> tuple[float | None, float | None, str | None]:
    """Normalize step bounds, returning (min,max,error_message)."""
    min_steps_value = float(min_steps) if min_steps and min_steps > 0 else None
    max_steps_value = float(max_steps) if max_steps and max_steps > 0 else None
    if min_steps_value is not None and max_steps_value is not None and min_steps_value > max_steps_value:
        return None, None, "Min steps exceeds max steps; swap or reset the range."
    return min_steps_value, max_steps_value, None


def _run_metadata(run: Any) -> dict[str, Any]:
    """Extract a lightweight metadata row for a run."""
    tags = list(getattr(run, "tags", []) or [])
    steps = _extract_run_steps(run)
    return {
        "id": str(getattr(run, "id", "")),
        "name": str(getattr(run, "name", "")),
        "state": str(getattr(run, "state", "")),
        "group": str(getattr(run, "group", "")),
        "job_type": str(getattr(run, "job_type", "")),
        "tags": ", ".join(tags),
        "created_at": _format_timestamp(getattr(run, "created_at", None)),
        "steps": steps if steps is not None else np.nan,
    }


def _resolve_x_key(history: pd.DataFrame, preferred_keys: list[str]) -> str:
    """Pick the best x-axis key and add a row index when needed."""
    for key in preferred_keys:
        if key in history.columns:
            return key
    if "row" not in history.columns:
        history["row"] = np.arange(len(history))
    return "row"


def _finite_values(values: np.ndarray) -> np.ndarray:
    """Return only finite values to avoid empty-slice warnings."""
    return values[np.isfinite(values)]


def _finite_mean(values: np.ndarray) -> float:
    """Return the mean of finite values or NaN when none are finite."""
    finite = _finite_values(values)
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def _summarize_metric(
    history: pd.DataFrame,
    *,
    metric_key: str,
    x_key: str,
    segment_frac: float,
) -> dict[str, float]:
    """Compute basic dynamics statistics for a metric."""
    if metric_key not in history.columns or x_key not in history.columns:
        return {}
    df_metric = history[[x_key, metric_key]].dropna().sort_values(x_key)
    if df_metric.empty:
        return {}
    values = df_metric[metric_key].to_numpy(dtype=float)
    steps = df_metric[x_key].to_numpy(dtype=float)
    early, mid, late = _segment_indices(len(df_metric), segment_frac)
    finite = _finite_values(values)
    mean_mid = float("nan") if mid.start >= mid.stop else _finite_mean(values[mid])
    return {
        "last": float(values[-1]),
        "min": float(np.nan) if finite.size == 0 else float(finite.min()),
        "max": float(np.nan) if finite.size == 0 else float(finite.max()),
        "mean_early": _finite_mean(values[early]),
        "mean_mid": mean_mid,
        "mean_late": _finite_mean(values[late]),
        "slope_early": _linear_slope(steps[early], values[early]),
        "slope_late": _linear_slope(steps[late], values[late]),
    }


def _summarize_gap(
    history: pd.DataFrame,
    *,
    x_key: str,
    train_key: str,
    val_key: str,
) -> dict[str, float]:
    """Summarize the train/val gap over the run."""
    if train_key not in history.columns or val_key not in history.columns:
        return {}
    df_gap = history[[x_key, train_key, val_key]].dropna().sort_values(x_key)
    if df_gap.empty:
        return {}
    gap = df_gap[train_key] - df_gap[val_key]
    return {
        "gap_last": float(gap.iloc[-1]),
        "gap_mean": float(gap.mean()),
        "gap_slope": _linear_slope(df_gap[x_key].to_numpy(dtype=float), gap.to_numpy(dtype=float)),
    }


def render_wandb_analysis_page() -> None:
    """Render cross-run analytics from W&B run history."""
    st.header("W&B Run Comparison")
    _info_popover(
        "wandb comparison",
        "Compare multiple runs within the same project to connect architectural "
        "choices and hyperparameters to training dynamics and final performance.",
    )

    if wandb is None:
        st.error("wandb is not available. Install it to use this panel.")
        return

    cache_key = "wandb_compare_cache"
    cache = st.session_state.get(cache_key, {})

    default_entity = os.environ.get("WANDB_ENTITY", "")
    default_project = os.environ.get("WANDB_PROJECT", "aria-nbv")
    api_key = os.environ.get("WANDB_API_KEY", "")

    st.subheader("Run selection")
    col_a, col_b, col_c = st.columns([2, 2, 1])

    if col_c.button("Refresh entities/projects"):
        _cached_entities.clear()
        _cached_projects.clear()

    cached_entity = str(cache.get("entity", default_entity))
    entity = _select_with_custom(
        col_a,
        label="Entity",
        options=[default_entity, *_cached_entities(api_key)],
        default=cached_entity,
        custom_label="Custom entity",
    )

    cached_project = str(cache.get("project", default_project))
    project = _select_with_custom(
        col_b,
        label="Project",
        options=[default_project, *_cached_projects(api_key, entity)],
        default=cached_project,
        custom_label="Custom project",
    )

    max_runs = col_c.number_input(
        "Max runs",
        min_value=5,
        max_value=500,
        value=int(cache.get("max_runs", 60)),
        step=5,
    )

    filters = _render_run_filters(cache)

    load_runs = st.button("Load runs")
    if load_runs:
        if not entity.strip() or not project.strip():
            st.warning("Enter an entity and project to load runs.")
        else:
            name_regex = None
            if filters["name_regex_raw"].strip():
                try:
                    name_regex = re.compile(filters["name_regex_raw"].strip())
                except re.error as exc:
                    st.warning(f"Invalid regex: {exc}")
            tags_filter = {tag.strip() for tag in filters["tags_raw"].split(",") if tag.strip()}
            min_steps_value, max_steps_value, step_error = _normalize_step_bounds(
                int(filters["min_steps"]),
                int(filters["max_steps"]),
            )
            if step_error:
                st.warning(step_error)
            try:
                api = wandb.Api()
                with st.spinner("Loading runs from W&B..."):
                    runs = _list_runs(
                        api,
                        entity=entity.strip(),
                        project=project.strip(),
                        max_runs=int(max_runs),
                    )
                runs = _filter_runs(
                    runs,
                    name_regex=name_regex,
                    states=list(filters["states"]),
                    tags=tags_filter,
                    group=filters["group_filter"].strip(),
                    job_type=filters["job_type_filter"].strip(),
                    min_steps=min_steps_value,
                    max_steps=max_steps_value,
                )
                cache = {
                    "entity": entity,
                    "project": project,
                    "max_runs": int(max_runs),
                    "name_regex": filters["name_regex_raw"],
                    "tags_filter": filters["tags_raw"],
                    "group_filter": filters["group_filter"],
                    "job_type_filter": filters["job_type_filter"],
                    "states": list(filters["states"]),
                    "min_steps": int(filters["min_steps"]),
                    "max_steps": int(filters["max_steps"]),
                    "runs": runs,
                }
                st.session_state[cache_key] = cache
            except Exception as exc:  # pragma: no cover - API guard
                _report_exception(exc, context="W&B run load failed")
                return

    runs = cache.get("runs") or []
    if not runs:
        st.info("Load runs to begin comparing.")
        return

    run_by_id = {str(getattr(run, "id", "")): run for run in runs}
    meta_rows = [_run_metadata(run) for run in runs]
    meta_df = pd.DataFrame(meta_rows).set_index("id")

    summary_rows = []
    config_by_id: dict[str, dict[str, Any]] = {}
    for run in runs:
        run_id = str(getattr(run, "id", ""))
        config = _flatten_mapping(_safe_mapping(getattr(run, "config", None)))
        summary = _flatten_mapping(_safe_mapping(getattr(run, "summary", None)))
        config_by_id[run_id] = config
        summary_rows.append({"id": run_id, **summary})
    summary_df = pd.DataFrame(summary_rows).set_index("id")

    summary_keys = sorted(summary_df.columns)
    summary_filter_raw = st.text_input(
        "Summary metric filter (regex)",
        value=str(cache.get("summary_filter", "loss|rri|spearman")),
    )
    summary_filter = None
    if summary_filter_raw.strip():
        try:
            summary_filter = re.compile(summary_filter_raw.strip(), re.IGNORECASE)
        except re.error as exc:
            st.warning(f"Invalid summary filter regex: {exc}")
    filtered_summary_keys = [key for key in summary_keys if summary_filter is None or summary_filter.search(key)]
    default_summary = [
        key for key in filtered_summary_keys if "val/loss" in key or "train/loss" in key or "rri_mean" in key
    ][:4]
    selected_summary_keys = st.multiselect(
        "Summary metrics to display",
        options=filtered_summary_keys,
        default=default_summary,
    )

    summary_subset = pd.DataFrame(index=summary_df.index)
    if selected_summary_keys:
        summary_subset = summary_df[selected_summary_keys].copy()
        overlap = meta_df.columns.intersection(summary_subset.columns)
        if not overlap.empty:
            summary_subset = summary_subset.drop(columns=list(overlap))
    run_table = (
        meta_df.copy()
        .join(summary_subset, how="left")
        .reset_index()
    )
    default_selected = cache.get("selected_ids") or run_table["id"].head(3).tolist()
    run_table = run_table.assign(include=run_table["id"].isin(default_selected))

    edited = st.data_editor(
        run_table,
        hide_index=True,
        width="stretch",
        column_config={
            "include": st.column_config.CheckboxColumn(required=True),
        },
        disabled=[col for col in run_table.columns if col not in {"include"}],
    )
    selected_ids = edited.loc[edited["include"], "id"].tolist()
    cache["selected_ids"] = selected_ids
    st.session_state[cache_key] = cache

    if not selected_ids:
        st.info("Select at least one run to compare.")
        return

    selected_runs = [run_by_id[run_id] for run_id in selected_ids if run_id in run_by_id]
    st.caption(f"Selected runs: {len(selected_runs)} / {len(run_table)}")

    st.subheader("History loading")
    col_h1, col_h2 = st.columns([1, 2])
    history_rows = col_h1.number_input(
        "History rows per run",
        min_value=100,
        max_value=50000,
        value=int(cache.get("history_rows", 2000)),
        step=100,
    )
    fetch_histories = col_h2.button("Fetch history for selected runs")

    histories = cache.get("histories") or {}
    if cache.get("history_rows") != int(history_rows):
        histories = {}

    if fetch_histories:
        try:
            with st.spinner("Fetching run histories..."):
                for run in selected_runs:
                    run_id = str(getattr(run, "id", ""))
                    history = _load_wandb_history(run, keys=None, max_rows=int(history_rows))
                    history = history.replace([np.inf, -np.inf], np.nan)
                    histories[run_id] = history
            cache["histories"] = histories
            cache["history_rows"] = int(history_rows)
            st.session_state[cache_key] = cache
        except Exception as exc:  # pragma: no cover - API guard
            _report_exception(exc, context="W&B history load failed")
            return

    selected_histories = {
        run_id: histories.get(run_id)
        for run_id in selected_ids
        if run_id in histories and histories[run_id] is not None
    }
    if not selected_histories:
        st.info("Fetch histories to compare training dynamics.")
        return

    st.subheader("Training dynamics summary")
    col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
    x_pref = col_d1.selectbox(
        "X-axis preference",
        options=["auto", *WANDB_STEP_KEYS],
        index=0,
    )
    segment_frac = col_d2.slider(
        "Segment fraction",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
    )
    prefer_keys = list(WANDB_STEP_KEYS) if x_pref == "auto" else [x_pref]

    base_counts: dict[str, int] = {}
    for history in selected_histories.values():
        pairs = _metric_pairs(list(history.columns))
        for base in pairs:
            base_counts[base] = base_counts.get(base, 0) + 1
    base_choices = sorted(base_counts)
    default_bases = [base for base in base_choices if "loss" in base][:1] or base_choices[:1]
    selected_bases = col_d3.multiselect(
        "Base metrics (train/val pairs)",
        options=base_choices,
        default=default_bases,
    )

    dynamics_rows: list[dict[str, Any]] = []
    for run in selected_runs:
        run_id = str(getattr(run, "id", ""))
        history = selected_histories.get(run_id)
        if history is None or history.empty:
            continue
        history = history.copy()
        x_key = _resolve_x_key(history, prefer_keys)
        pairs = _metric_pairs(list(history.columns))
        prefer_suffix = "epoch" if "epoch" in x_key else "step"
        for base in selected_bases:
            train_key = _select_metric_key(pairs.get(base, {}).get("train", {}), prefer_suffix)
            val_key = _select_metric_key(pairs.get(base, {}).get("val", {}), prefer_suffix)
            if train_key is None and val_key is None:
                continue
            row: dict[str, Any] = {
                "run_id": run_id,
                "run_name": str(getattr(run, "name", run_id)),
                "base": base,
                "x_key": x_key,
                "train_key": train_key or "",
                "val_key": val_key or "",
            }
            if train_key:
                train_summary = _summarize_metric(
                    history,
                    metric_key=train_key,
                    x_key=x_key,
                    segment_frac=segment_frac,
                )
                row.update({f"train_{key}": value for key, value in train_summary.items()})
            if val_key:
                val_summary = _summarize_metric(
                    history,
                    metric_key=val_key,
                    x_key=x_key,
                    segment_frac=segment_frac,
                )
                row.update({f"val_{key}": value for key, value in val_summary.items()})
            if train_key and val_key:
                row.update(
                    _summarize_gap(history, x_key=x_key, train_key=train_key, val_key=val_key),
                )
            dynamics_rows.append(row)

    dynamics_df = pd.DataFrame(dynamics_rows)
    if dynamics_df.empty:
        st.info("No paired train/val metrics found for the selected bases.")
    else:
        st.dataframe(dynamics_df, width="stretch", height=320)

        if "train_last" in dynamics_df.columns and "val_last" in dynamics_df.columns:
            base_options = sorted(dynamics_df["base"].unique())
            base_scatter = st.selectbox(
                "Scatter base",
                options=base_options,
                index=0,
            )
            scatter_df = dynamics_df[dynamics_df["base"] == base_scatter].dropna(
                subset=["train_last", "val_last"],
            )
            if not scatter_df.empty:
                fig_scatter = px.scatter(
                    scatter_df,
                    x="train_last",
                    y="val_last",
                    color="run_name",
                    title=_pretty_label("Train vs val (last value)"),
                )
                fig_scatter.update_layout(
                    xaxis_title=_pretty_label("train last"),
                    yaxis_title=_pretty_label("val last"),
                )
                st.plotly_chart(fig_scatter, width="stretch")

    st.subheader("Metric curves")
    metric_filter_raw = st.text_input(
        "Metric filter (regex)",
        value=str(cache.get("metric_filter", _DEFAULT_METRIC_FILTER)),
    )
    metric_filter = None
    if metric_filter_raw.strip():
        try:
            metric_filter = re.compile(metric_filter_raw.strip(), re.IGNORECASE)
        except re.error as exc:
            st.warning(f"Invalid metric filter regex: {exc}")
    metric_candidates: set[str] = set()
    for history in selected_histories.values():
        numeric_cols = history.select_dtypes(include=[np.number]).columns
        metric_candidates.update(numeric_cols)
    metric_choices = sorted(
        metric for metric in metric_candidates if metric_filter is None or metric_filter.search(metric)
    )
    if not metric_choices:
        st.info("No metrics match the filter across selected runs.")
    else:
        col_m1, col_m2, col_m3 = st.columns([2, 1, 1])
        metric_choice = col_m1.selectbox("Metric", options=metric_choices, index=0)
        ema_alpha = col_m2.slider(
            "EMA alpha",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
        )
        max_runs_plot = col_m3.slider(
            "Max runs to plot",
            min_value=1,
            max_value=max(1, len(selected_runs)),
            value=min(6, len(selected_runs)),
        )
        show_raw = st.checkbox("Show raw curves", value=True)

        fig = go.Figure()
        x_axis_label = x_pref if x_pref != "auto" else "step/epoch"
        for run in selected_runs[:max_runs_plot]:
            run_id = str(getattr(run, "id", ""))
            history = selected_histories.get(run_id)
            if history is None or metric_choice not in history.columns:
                continue
            history = history.copy()
            x_key = _resolve_x_key(history, prefer_keys)
            df_metric = history[[x_key, metric_choice]].dropna().sort_values(x_key)
            if df_metric.empty:
                continue
            label = str(getattr(run, "name", run_id))
            if show_raw:
                fig.add_trace(
                    go.Scatter(
                        x=df_metric[x_key],
                        y=df_metric[metric_choice],
                        mode="lines",
                        name=f"{label} (raw)",
                        line={"width": 1, "dash": "dot"},
                        opacity=0.6,
                    ),
                )
            if ema_alpha > 0.0:
                smooth = df_metric[metric_choice].ewm(alpha=ema_alpha, adjust=False).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df_metric[x_key],
                        y=smooth,
                        mode="lines",
                        name=f"{label} (ema)",
                    ),
                )
        fig.update_layout(
            title=_pretty_label(f"{metric_choice} across runs"),
            xaxis_title=_pretty_label(x_axis_label),
            yaxis_title=_pretty_label(metric_choice),
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Architecture & hparam influence")
    if dynamics_df.empty:
        st.info("Load dynamics to analyze configuration influence.")
    else:
        base_options = sorted(dynamics_df["base"].unique())
        base_choice = st.selectbox(
            "Base metric for influence",
            options=base_options,
            index=0,
        )
        base_df = dynamics_df[dynamics_df["base"] == base_choice].copy()
        config_rows = []
        for run_id in base_df["run_id"]:
            config = config_by_id.get(run_id, {})
            config_rows.append({"run_id": run_id, **config})
        config_df = pd.DataFrame(config_rows)
        analysis_df = base_df.merge(config_df, on="run_id", how="left")

        target_candidates = [
            col
            for col in analysis_df.columns
            if col.startswith("train_") or col.startswith("val_") or col.startswith("gap_")
        ]
        if not target_candidates:
            st.info("No summary metrics available for influence analysis.")
            target_metric = ""
        else:
            default_target = "val_last" if "val_last" in target_candidates else target_candidates[0]
            target_metric = st.selectbox(
                "Target metric",
                options=target_candidates,
                index=target_candidates.index(default_target),
            )

        config_keys = [col for col in config_df.columns if col != "run_id" and config_df[col].nunique(dropna=True) > 1]
        config_filter_raw = st.text_input(
            "Config key filter (regex)",
            value=str(cache.get("config_filter", "pose|field|voxel|lr|weight|bin|head|global")),
        )
        config_filter = None
        if config_filter_raw.strip():
            try:
                config_filter = re.compile(config_filter_raw.strip(), re.IGNORECASE)
            except re.error as exc:
                st.warning(f"Invalid config filter regex: {exc}")
        filtered_config_keys = [key for key in config_keys if config_filter is None or config_filter.search(key)]
        if not target_metric:
            st.info("Select a target metric to explore configuration influence.")
            filtered_config_keys = []
        if not filtered_config_keys:
            st.info("No config keys match the filter.")
        else:
            config_key = st.selectbox(
                "Config key",
                options=filtered_config_keys,
                index=0,
            )
            series = analysis_df[config_key]
            numeric_series = pd.to_numeric(series, errors="coerce")
            is_numeric = numeric_series.notna().sum() == series.notna().sum()
            is_bool = pd.api.types.is_bool_dtype(series)

            if is_numeric and not is_bool:
                plot_df = analysis_df.copy()
                plot_df[config_key] = numeric_series
                plot_df = plot_df.dropna(subset=[config_key, target_metric])
                fig_hp = px.scatter(
                    plot_df,
                    x=config_key,
                    y=target_metric,
                    color="run_name",
                    title=_pretty_label(f"{target_metric} vs {config_key}"),
                )
                if len(plot_df) >= 2:
                    slope = _linear_slope(
                        plot_df[config_key].to_numpy(dtype=float),
                        plot_df[target_metric].to_numpy(dtype=float),
                    )
                    intercept = float(plot_df[target_metric].mean()) - slope * float(
                        plot_df[config_key].mean(),
                    )
                    x_vals = np.linspace(plot_df[config_key].min(), plot_df[config_key].max(), 50)
                    y_vals = slope * x_vals + intercept
                    fig_hp.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode="lines",
                            name="trend",
                        ),
                    )
                st.plotly_chart(fig_hp, width="stretch")
            else:
                plot_df = analysis_df.copy()
                plot_df[config_key] = plot_df[config_key].astype(str)
                plot_df = plot_df.dropna(subset=[config_key, target_metric])
                fig_box = px.box(
                    plot_df,
                    x=config_key,
                    y=target_metric,
                    points="all",
                    title=_pretty_label(f"{target_metric} by {config_key}"),
                )
                st.plotly_chart(fig_box, width="stretch")

        numeric_config_keys = []
        for key in config_keys:
            series = pd.to_numeric(analysis_df[key], errors="coerce")
            if series.notna().sum() == analysis_df[key].notna().sum():
                numeric_config_keys.append(key)
        if numeric_config_keys:
            min_samples = st.slider(
                "Min samples for correlation",
                min_value=3,
                max_value=max(3, len(analysis_df)),
                value=min(10, len(analysis_df)),
                step=1,
            )
            corr_rows: list[dict[str, Any]] = []
            for key in numeric_config_keys:
                df_pair = analysis_df[[target_metric, key]].dropna()
                if len(df_pair) < min_samples:
                    continue
                pearson = df_pair[target_metric].corr(df_pair[key], method="pearson")
                spearman = df_pair[target_metric].corr(df_pair[key], method="spearman")
                corr_rows.append(
                    {
                        "config_key": key,
                        "n": len(df_pair),
                        "pearson": float(pearson) if pd.notna(pearson) else float("nan"),
                        "spearman": float(spearman) if pd.notna(spearman) else float("nan"),
                        "impact": float(abs(pearson)) + float(abs(spearman)),
                    },
                )
            corr_df = pd.DataFrame(corr_rows).sort_values("impact", ascending=False)
            if not corr_df.empty:
                st.markdown("**Top numeric config correlations**")
                st.dataframe(corr_df.head(12), width="stretch", height=260)

    st.subheader("Focus run details")
    focus_id = st.selectbox(
        "Focus run",
        options=selected_ids,
        index=0,
        format_func=lambda rid: str(getattr(run_by_id.get(rid), "name", rid)),
    )
    focus_run = run_by_id.get(focus_id)
    if focus_run is not None:
        st.caption(f"Run path: {getattr(focus_run, 'path', 'n/a')}")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("**Config**")
            st.json(_safe_mapping(getattr(focus_run, "config", None)))
        with col_f2:
            st.markdown("**Summary**")
            st.json(_safe_mapping(getattr(focus_run, "summary", None)))
