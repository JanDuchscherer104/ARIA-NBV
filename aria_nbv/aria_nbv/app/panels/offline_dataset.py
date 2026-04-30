"""Standalone Streamlit diagnostics for immutable VIN offline datasets."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import plotly.express as px
import streamlit as st

from ...configs import PathConfig
from ...data_handling import (
    VinOfflineDatasetStats,
    VinOfflineSourceConfig,
    VinOfflineStoreConfig,
    collect_vin_offline_dataset_stats,
)
from ...lightning.aria_nbv_experiment import AriaNBVExperimentConfig

_STATS_CACHE_KEY = "vin_offline_dataset_page_stats"


def _load_offline_store_from_toml(toml_path: Path) -> VinOfflineStoreConfig:
    """Return the immutable VIN store configured by an experiment TOML."""

    cfg = AriaNBVExperimentConfig.from_toml(toml_path)
    source = cfg.datamodule_config.source
    if not isinstance(source, VinOfflineSourceConfig):
        raise TypeError(
            f"Experiment config uses {type(source).__name__}; expected VinOfflineSourceConfig.",
        )
    return source.offline.store


def _summary_rows(stats: VinOfflineDatasetStats) -> list[dict[str, object]]:
    """Return aggregate numeric summaries as table rows."""

    return [
        {"metric": "candidate_count", **asdict(stats.candidate_count)},
        {"metric": "rri", **asdict(stats.rri)},
        {"metric": "vin_points", **asdict(stats.vin_points)},
    ]


def _block_rows(stats: VinOfflineDatasetStats) -> list[dict[str, object]]:
    """Return manifest block diagnostics as table rows."""

    rows: list[dict[str, object]] = []
    for block in stats.block_diagnostics:
        rows.append(
            {
                "shard_id": block.shard_id,
                "name": block.name,
                "kind": block.kind,
                "dtype": block.dtype,
                "shape": block.shape,
                "optional": block.optional,
                "estimated_mib": (None if block.estimated_bytes is None else float(block.estimated_bytes) / (1024**2)),
            },
        )
    return rows


def _sample_rows(stats: VinOfflineDatasetStats) -> list[dict[str, object]]:
    """Return sampled per-row sanity summaries as table rows."""

    rows: list[dict[str, object]] = []
    for sample in stats.sample_summaries:
        rows.append(
            {
                "sample_index": sample.sample_index,
                "sample_key": sample.sample_key,
                "scene_id": sample.scene_id,
                "snippet_id": sample.snippet_id,
                "split": sample.split,
                "shard_id": sample.shard_id,
                "row": sample.row,
                "candidate_count": sample.candidate_count,
                "finite_rri_count": sample.rri.count,
                "rri_min": sample.rri.minimum,
                "rri_mean": sample.rri.mean,
                "rri_max": sample.rri.maximum,
                "vin_points_count": sample.vin_points.count,
                "vin_points_mean": sample.vin_points.mean,
            },
        )
    return rows


def _render_histogram(values: list[float], *, title: str, x_label: str, nbins: int) -> None:
    """Render one Plotly histogram when values are available."""

    if not values:
        st.info(f"No values available for {title}.")
        return
    fig = px.histogram(
        x=values,
        nbins=int(nbins),
        title=title,
        labels={"x": x_label, "y": "count"},
    )
    st.plotly_chart(fig, width="stretch")


def _render_stats(stats: VinOfflineDatasetStats, *, hist_bins: int) -> None:
    """Render one collected immutable offline-store diagnostics object."""

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Samples", stats.num_samples)
    col_b.metric("Sampled rows", stats.sampled_samples)
    col_c.metric("Scenes", stats.num_scenes)
    col_d.metric("Numeric blocks", f"{stats.numeric_bytes / (1024**2):.1f} MiB")

    tab_overview, tab_blocks, tab_samples, tab_distributions, tab_manifest = st.tabs(
        ["Overview", "Blocks", "Sample Sanity", "Distributions", "Manifest"],
    )

    with tab_overview:
        st.dataframe(_summary_rows(stats), width="stretch", hide_index=True)
        col_l, col_r = st.columns(2)
        col_l.json(stats.split_counts, expanded=True)
        col_r.json(stats.materialized_blocks, expanded=True)

    with tab_blocks:
        st.dataframe(_block_rows(stats), width="stretch", hide_index=True)

    with tab_samples:
        st.dataframe(_sample_rows(stats), width="stretch", hide_index=True)

    with tab_distributions:
        _render_histogram(
            stats.candidate_count_values,
            title="Valid Candidate Counts",
            x_label="candidate_count",
            nbins=hist_bins,
        )
        _render_histogram(
            stats.rri_values,
            title="Oracle RRI",
            x_label="rri",
            nbins=hist_bins,
        )
        _render_histogram(
            stats.vin_point_values,
            title="VIN Point Lengths",
            x_label="vin_points",
            nbins=hist_bins,
        )

    with tab_manifest:
        st.json(
            {
                "store_dir": stats.store_dir,
                "version": stats.version,
                "num_samples": stats.num_samples,
                "sampled_samples": stats.sampled_samples,
                "split_counts": stats.split_counts,
                "materialized_blocks": stats.materialized_blocks,
                "block_shapes": stats.block_shapes,
            },
            expanded=False,
        )


def render_offline_dataset_page() -> None:
    """Render immutable VIN offline dataset diagnostics as a standalone page."""

    st.header("VIN Offline Dataset")
    st.caption(
        "Inspect immutable VIN offline stores directly from their manifest and indexed shards.",
    )

    paths = PathConfig()
    config_paths = sorted(
        paths.configs_dir.glob("*.toml"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )

    with st.sidebar.form("vin_offline_dataset_form"):
        st.subheader("VIN Offline Dataset")
        source_mode = st.radio(
            "Source",
            options=["Store directory", "Experiment config TOML"],
            index=0,
            key="vin_offline_dataset_source_mode",
        )
        store_dir_text = st.text_input(
            "Store directory",
            value=str(VinOfflineStoreConfig().store_dir),
            disabled=source_mode != "Store directory",
            key="vin_offline_dataset_store_dir",
        )
        toml_options = ["(none)"] + [path.name for path in config_paths]
        toml_choice = st.selectbox(
            "Experiment config TOML",
            options=toml_options,
            index=0,
            disabled=source_mode != "Experiment config TOML",
            key="vin_offline_dataset_toml",
        )
        max_samples = int(
            st.number_input(
                "Diagnostic sample limit",
                min_value=1,
                max_value=100000,
                value=512,
                step=128,
                key="vin_offline_dataset_sample_limit",
            ),
        )
        hist_bins = int(
            st.number_input(
                "Histogram bins",
                min_value=5,
                max_value=200,
                value=40,
                step=5,
                key="vin_offline_dataset_hist_bins",
            ),
        )
        inspect = st.form_submit_button("Inspect offline store")

    if inspect:
        try:
            if source_mode == "Experiment config TOML":
                if toml_choice == "(none)":
                    raise ValueError("Select an experiment config TOML.")
                store = _load_offline_store_from_toml(paths.configs_dir / toml_choice)
            else:
                store = VinOfflineStoreConfig(store_dir=Path(store_dir_text).expanduser())

            if not store.manifest_path.is_file():
                raise FileNotFoundError(f"Missing VIN offline manifest: {store.manifest_path}")
            stats = collect_vin_offline_dataset_stats(store, max_samples=max_samples)
            st.session_state[_STATS_CACHE_KEY] = {
                "key": f"{stats.store_dir}|{max_samples}",
                "stats": stats,
            }
        except Exception as exc:  # pragma: no cover - UI guard
            st.error(f"Could not inspect VIN offline store: {type(exc).__name__}: {exc}")
            st.session_state.pop(_STATS_CACHE_KEY, None)

    cached = st.session_state.get(_STATS_CACHE_KEY)
    if not cached:
        st.info("Choose a store and click Inspect offline store.")
        return

    _render_stats(cached["stats"], hist_bins=hist_bins)


__all__ = ["render_offline_dataset_page"]
