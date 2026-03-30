"""Offline cache statistics panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]
import streamlit as st
import torch

from ...configs import PathConfig
from ...data.offline_cache_coverage import (
    compute_cache_coverage,
    expand_tar_urls,
    read_cache_index_entries,
    scan_dataset_snippets,
    snippets_by_scene,
)
from ...data.offline_cache_store import _extract_snippet_token
from ...data.offline_cache_store import _read_metadata as _read_cache_metadata
from ...data_handling import (
    AseEfmDatasetConfig,
    OracleRriCacheDatasetConfig,
    VinOracleCacheDatasetConfig,
    VinOracleOnlineDatasetConfig,
    read_vin_snippet_cache_metadata,
    rebuild_cache_index,
)
from ...lightning.aria_nbv_experiment import AriaNBVExperimentConfig
from ...pose_generation.plotting import plot_position_polar
from ...rri_metrics.plotting import _histogram_overlay, _plot_hist_counts_mpl
from ...rri_metrics.rri_binning import RriOrdinalBinner
from ...utils import Stage
from ...vin.experimental.plotting import DEFAULT_PLOT_CFG
from .common import _info_popover, _pretty_label, _report_exception
from .offline_cache_utils import (
    _collect_offline_cache_stats,
    _collect_vin_batch_shape_preview,
    _collect_vin_snippet_cache_stats,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_offline_stats_page() -> None:
    """Render offline cache statistics as a standalone page."""
    st.header("Offline Cache Statistics")
    st.caption(
        "Aggregate statistics over cached oracle batches without retaining full samples in memory.",
    )
    _info_popover(
        "offline stats",
        "Summaries are computed over cached oracle batches. This view helps "
        "validate label distributions, candidate counts, and backbone feature "
        "scales without loading full samples into memory.",
    )

    stats_key = "vin_offline_stats"
    stats_cache = st.session_state.get(stats_key, {})
    coverage_key = "vin_offline_coverage"
    coverage_cache = st.session_state.get(coverage_key, {})

    def _as_path_str(value: str | Path | None) -> str:
        if value is None:
            return ""
        if isinstance(value, Path):
            return str(value)
        return str(value)

    with st.sidebar.form("vin_offline_stats_form"):
        st.subheader("Offline stats")
        paths = PathConfig()
        config_dir = paths.configs_dir
        config_paths = sorted(
            config_dir.glob("*.toml"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        toml_options = ["(none)"] + [path.name for path in config_paths]
        toml_choice = st.selectbox(
            "Experiment config TOML",
            options=toml_options,
            index=0,
        )
        toml_path = "" if toml_choice == "(none)" else str(config_dir / toml_choice)
        cache_kind = st.selectbox(
            "Cache type",
            options=["oracle_rri_cache", "vin_snippet_cache"],
            format_func=lambda k: "Oracle RRI cache" if k == "oracle_rri_cache" else "VIN snippet cache",
            key="vin_offline_cache_kind",
        )
        stage = st.selectbox(
            "Stage",
            options=[Stage.TRAIN, Stage.VAL, Stage.TEST],
            format_func=lambda s: s.value,
            key="vin_offline_stage",
        )
        cache_dir = (
            PathConfig().offline_cache_dir
            if cache_kind == "oracle_rri_cache"
            else PathConfig().offline_cache_dir / "vin_snippet_cache"
        )
        map_location = "cpu"
        if cache_kind == "vin_snippet_cache":
            map_location = st.selectbox(
                "Cache map_location",
                options=["cpu", "cuda"],
                index=0,
                key="vin_offline_map_location",
            )
        else:
            st.caption("Oracle cache loads always use CPU map_location.")
        max_samples = st.number_input(
            "Max samples (0 = all)",
            min_value=0,
            value=0,
            step=1,
            key="vin_offline_max_samples",
        )
        num_workers = st.number_input(
            "DataLoader workers (0 = use config)",
            min_value=0,
            value=0,
            step=1,
            key="vin_offline_num_workers",
        )
        train_val_split = st.number_input(
            "Train/val split",
            min_value=0.0,
            max_value=0.95,
            value=float(OracleRriCacheDatasetConfig().train_val_split),
            step=0.05,
            key="vin_offline_train_val_split",
        )
        if cache_kind == "vin_snippet_cache":
            st.caption("Train/val split is ignored for VIN snippet cache stats.")
        run_stats = st.form_submit_button("Compute offline stats")

        st.divider()
        st.subheader("Dataset coverage")
        st.caption(
            "Scans the EFM shard tar headers to estimate how many available "
            "(scene, snippet) samples are present in the offline cache indices.",
        )
        max_tars = st.number_input(
            "Max tar shards to scan (0 = all)",
            min_value=0,
            value=0,
            step=1,
            key="vin_offline_coverage_max_tars",
        )
        scan_coverage = st.form_submit_button("Scan dataset coverage")

    index_entries: list[object] = []
    train_entries: list[object] = []
    val_entries: list[object] = []
    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path is not None:
        index_path = cache_path / "index.jsonl"
        train_index_path = cache_path / "train_index.jsonl"
        val_index_path = cache_path / "val_index.jsonl"
        samples_dir = cache_path / "samples"
        index_entries = read_cache_index_entries(index_path)
        if cache_kind == "oracle_rri_cache":
            train_entries = read_cache_index_entries(train_index_path)
            val_entries = read_cache_index_entries(val_index_path)
        else:
            train_entries = index_entries
            val_entries = []
        index_count = len(index_entries)
        train_count = len(train_entries)
        val_count = len(val_entries)
        sample_count = sum(1 for _ in samples_dir.glob("*.pt")) if samples_dir.exists() else 0
        if index_count or sample_count or train_count or val_count:
            counts_label = f"Index entries: {index_count} · Sample files: {sample_count}"
            if train_count or val_count:
                counts_label += f" · Train: {train_count} · Val: {val_count}"
            st.caption(counts_label)
        if sample_count > index_count and cache_kind == "oracle_rri_cache":
            st.warning(
                "Cache index has fewer entries than sample files. "
                "Offline stats only read index.jsonl; rebuild the index to include all samples.",
            )
            split_seed = st.number_input(
                "Split RNG seed (-1 = random)",
                min_value=-1,
                value=-1,
                step=1,
                key="vin_offline_split_seed",
            )
            if st.button("Rebuild index from samples", key="vin_offline_rebuild_index"):
                with st.spinner("Rebuilding cache index..."):
                    rebuilt = rebuild_cache_index(
                        cache_dir=cache_path,
                        train_val_split=float(train_val_split),
                        rng_seed=None if split_seed < 0 else int(split_seed),
                    )
                st.success(f"Rebuilt index with {rebuilt} entries.")
                st.session_state.pop(stats_key, None)
                st.rerun()

    def _resolve_experiment_config() -> tuple[AriaNBVExperimentConfig, PathConfig]:
        resolved_toml: Path | None = None
        if toml_path.strip():
            try:
                resolved_toml = PathConfig().resolve_config_toml_path(
                    toml_path.strip(),
                    must_exist=False,
                )
            except ValueError as exc:
                st.warning(f"Invalid config path ({toml_path}): {exc}")
                resolved_toml = None
            else:
                if not resolved_toml.exists():
                    st.warning(f"Config file not found: {resolved_toml}")
                    resolved_toml = None
        cfg = (
            AriaNBVExperimentConfig.from_toml(resolved_toml) if resolved_toml is not None else AriaNBVExperimentConfig()
        )
        paths = cfg.paths if isinstance(cfg.paths, PathConfig) else PathConfig()
        return cfg, paths

    cache_dir_str = _as_path_str(cache_dir).strip()
    toml_path_str = _as_path_str(toml_path).strip()

    cfg_key = "|".join(
        [
            cache_kind,
            toml_path_str,
            stage.value,
            cache_dir_str,
            map_location,
            str(int(max_samples)),
            str(int(num_workers)),
            f"{float(train_val_split):.3f}",
        ],
    )

    coverage_cfg_key = "|".join(
        [
            cache_kind,
            toml_path_str,
            stage.value,
            cache_dir_str,
            f"{float(train_val_split):.3f}",
            str(int(max_tars)),
        ],
    )

    log_y = st.checkbox(
        "Log-scale y-axis",
        value=False,
        key="vin_offline_log_y",
    )

    def _build_hist_ax(
        values: list[float] | np.ndarray,
        *,
        bins: int,
        title: str,
        xlabel: str,
        log_y: bool,
        color: str | None = None,
        figsize: tuple[float, float] = (7, 3),
    ) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=figsize)
        _plot_hist_counts_mpl(
            values,
            bins=bins,
            log_y=log_y,
            ax=ax,
            color=color,
        )
        ax.set_title(_pretty_label(title))
        ax.set_xlabel(_pretty_label(xlabel))
        ax.set_ylabel(_pretty_label("count (log)" if log_y else "count"))
        return fig, ax

    def _render_hist(
        values: list[float] | np.ndarray,
        *,
        bins: int,
        title: str,
        xlabel: str,
        log_y: bool,
        color: str | None = None,
        figsize: tuple[float, float] = (7, 3),
    ) -> None:
        fig, _ = _build_hist_ax(
            values,
            bins=bins,
            title=title,
            xlabel=xlabel,
            log_y=log_y,
            color=color,
            figsize=figsize,
        )
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    if scan_coverage:
        if cache_path is None:
            st.warning("Select an offline cache directory first.")
        else:
            try:
                with st.spinner("Scanning dataset shards + cache indices..."):
                    cfg, paths = _resolve_experiment_config()
                    dm_cfg = cfg.datamodule_config
                    source_cfg = dm_cfg.source
                    split_stage = stage
                    if stage is not Stage.TRAIN and dm_cfg.use_train_as_val:
                        split_stage = Stage.TRAIN

                    dataset_cfg = None
                    if cache_kind == "vin_snippet_cache":
                        meta_path = cache_path / "metadata.json"
                        if meta_path.exists():
                            meta = read_vin_snippet_cache_metadata(meta_path)
                            if meta.dataset_config is not None:
                                dataset_cfg = AseEfmDatasetConfig(**meta.dataset_config)
                                dataset_cfg.paths = paths

                    if dataset_cfg is None and isinstance(source_cfg, VinOracleOnlineDatasetConfig):
                        dataset_cfg = source_cfg.dataset.model_copy(deep=True)
                        dataset_cfg.paths = paths
                        overrides = (
                            source_cfg.train_overrides if split_stage is Stage.TRAIN else source_cfg.val_overrides
                        )
                        if overrides:
                            dataset_cfg = dataset_cfg.model_copy(deep=True, update=overrides)
                    elif dataset_cfg is None and isinstance(source_cfg, VinOracleCacheDatasetConfig):
                        cache_cfg = source_cfg.cache
                        cache_root = cache_path or cache_cfg.cache.cache_dir
                        meta_path = cache_root / "metadata.json"
                        if not meta_path.exists():
                            raise FileNotFoundError(f"Missing cache metadata: {meta_path}")
                        meta = _read_cache_metadata(meta_path)
                        if meta.dataset_config is None:
                            raise ValueError("Cache metadata does not include dataset_config.")
                        dataset_cfg = AseEfmDatasetConfig(**meta.dataset_config)
                        dataset_cfg.paths = paths
                    elif dataset_cfg is None:
                        raise TypeError("Unsupported dataset source type in experiment config.")

                    dataset_cfg = dataset_cfg.model_copy(
                        deep=True,
                        update={
                            "wds_shuffle": False,
                            "wds_repeat": False,
                            "batch_size": 1,
                            "load_meshes": False,
                            "device": "cpu",
                        },
                    )

                    tar_paths = expand_tar_urls(dataset_cfg.tar_urls)
                    if max_tars and int(max_tars) > 0:
                        tar_paths = tar_paths[: int(max_tars)]

                    progress = st.progress(0.0)

                    def _cb(done: int, total: int) -> None:
                        if total <= 0:
                            progress.progress(1.0)
                            return
                        progress.progress(min(1.0, float(done) / float(total)))

                    dataset_snippets = scan_dataset_snippets(
                        tar_paths,
                        snippet_key_filter=dataset_cfg.snippet_key_filter,
                        progress_cb=_cb,
                    )
                    progress.empty()

                    if not train_entries and not val_entries:
                        train_entries = index_entries
                        val_entries = []
                        st.info("train/val indices missing; using index.jsonl for coverage (all samples).")

                    report = compute_cache_coverage(
                        dataset_snippets=dataset_snippets,
                        cache_train_snippets=snippets_by_scene(train_entries),
                        cache_val_snippets=snippets_by_scene(val_entries),
                    )

                    coverage_cache = {
                        "key": coverage_cfg_key,
                        "tar_shards_scanned": len(tar_paths),
                        "report": report,
                    }
                    st.session_state[coverage_key] = coverage_cache
            except Exception as exc:  # pragma: no cover - UI guard
                _report_exception(exc, context="Dataset coverage scan failed")
                return

    if run_stats:
        try:
            with st.spinner("Collecting offline cache statistics..."):
                if cache_kind == "oracle_rri_cache":
                    stats_cache = _collect_offline_cache_stats(
                        toml_path=toml_path.strip() or None,
                        stage=stage,
                        cache_dir=cache_dir.as_posix().strip() or None,
                        max_samples=int(max_samples),
                        num_workers=int(num_workers) if num_workers > 0 else None,
                        train_val_split=float(train_val_split),
                    )
                else:
                    stats_cache = _collect_vin_snippet_cache_stats(
                        cache_dir=cache_dir.as_posix().strip() or None,
                        map_location=map_location,
                        max_samples=int(max_samples),
                        num_workers=int(num_workers) if num_workers > 0 else None,
                    )
                    vin_cache_dir = cache_dir.as_posix().strip() or None
                    oracle_cache_dir = None
                    if vin_cache_dir:
                        parent = Path(vin_cache_dir).parent
                        oracle_cache_dir = str(parent) if (parent / "index.jsonl").exists() else None
                    stats_cache["vin_batch_shapes"] = _collect_vin_batch_shape_preview(
                        toml_path=toml_path.strip() or None,
                        stage=stage,
                        oracle_cache_dir=oracle_cache_dir,
                        vin_cache_dir=vin_cache_dir,
                        train_val_split=float(train_val_split),
                        num_workers=int(num_workers) if num_workers > 0 else None,
                    )
                stats_cache["cache_kind"] = cache_kind
            stats_cache["key"] = cfg_key
            st.session_state[stats_key] = stats_cache
        except Exception as exc:  # pragma: no cover - UI guard
            _report_exception(exc, context="Offline stats failed")
            return

    st.subheader("Dataset coverage")
    _info_popover(
        "coverage",
        "Compares cached (scene, snippet) pairs against the list of available samples "
        "in the configured AseEfmDataset shards. This avoids loading EFM samples and "
        "only inspects the tar headers plus the cache index JSONL files.",
    )
    if not coverage_cache or coverage_cache.get("key") != coverage_cfg_key:
        st.info("Scan dataset coverage to load coverage histograms.")
    else:
        report = coverage_cache["report"]
        tar_scanned = int(coverage_cache.get("tar_shards_scanned", 0))

        col1, col2, col3 = st.columns(3)
        col1.metric("Dataset scenes", report.dataset_scenes)
        col2.metric("Cache scenes (train)", report.cache_train_scenes)
        col3.metric("Cache scenes (val)", report.cache_val_scenes)

        col4, col5, col6 = st.columns(3)
        col4.metric("Dataset snippets", report.dataset_snippets)
        col5.metric("Cache snippets (train)", report.cache_train_snippets)
        col6.metric("Cache snippets (val)", report.cache_val_snippets)

        col7, col8, col9 = st.columns(3)
        col7.metric("Cache snippets (all)", report.cache_all_snippets)
        coverage_pct = (
            0.0 if report.dataset_snippets == 0 else 100.0 * (report.cache_all_snippets / report.dataset_snippets)
        )
        col8.metric("Snippet coverage (all)", f"{coverage_pct:.2f}%")
        col9.metric("Cache outside dataset", report.cache_outside_dataset)

        st.caption(f"Tar shards scanned: {tar_scanned}")

        cov_all = [row.coverage_all for row in report.per_scene if row.coverage_all is not None]
        cov_train = [row.coverage_train for row in report.per_scene if row.coverage_train is not None]
        cov_val = [row.coverage_val for row in report.per_scene if row.coverage_val is not None]
        dataset_counts = [row.dataset_snippets for row in report.per_scene if row.dataset_snippets > 0]
        cache_counts = [row.cache_all_snippets for row in report.per_scene if row.dataset_snippets > 0]

        cov_bins = int(
            st.slider(
                "Coverage histogram bins",
                min_value=10,
                max_value=80,
                value=25,
                step=1,
                key="vin_offline_coverage_bins",
            ),
        )

        with DEFAULT_PLOT_CFG.apply():
            col_a, col_b = st.columns(2)
            with col_a:
                _render_hist(
                    dataset_counts,
                    bins=60,
                    title="Available snippets per scene (dataset)",
                    xlabel="# snippets",
                    log_y=log_y,
                )
            with col_b:
                _render_hist(
                    cache_counts,
                    bins=60,
                    title="Cached snippets per scene (train ∪ val)",
                    xlabel="# cached snippets",
                    log_y=log_y,
                    color="#1f77b4",
                )

            _render_hist(
                [float(value) for value in cov_all],
                bins=cov_bins,
                title="Per-scene coverage ratio (train ∪ val)",
                xlabel="coverage",
                log_y=log_y,
                color="#2ca02c",
            )

            if cov_train and cov_val:
                fig, ax = _build_hist_ax(
                    [float(value) for value in cov_train],
                    bins=cov_bins,
                    title="Per-scene coverage ratio (train vs val)",
                    xlabel="coverage",
                    log_y=log_y,
                    color="#ff7f0e",
                )
                _plot_hist_counts_mpl(
                    [float(value) for value in cov_val],
                    bins=cov_bins,
                    log_y=log_y,
                    ax=ax,
                    color="#9467bd",
                )
                from matplotlib.patches import Patch

                ax.legend(
                    handles=[
                        Patch(color="#ff7f0e", label=_pretty_label("train")),
                        Patch(color="#9467bd", label=_pretty_label("val")),
                    ],
                    loc="best",
                )
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

        show_table = st.checkbox(
            "Show per-scene coverage table",
            value=False,
            key="vin_offline_coverage_show_table",
        )
        if show_table:
            df = pd.DataFrame(report.as_rows()).sort_values(
                "coverage_all",
                ascending=True,
                na_position="last",
            )
            st.dataframe(df, width="stretch", height=320)

    if not stats_cache or stats_cache.get("key") != cfg_key:
        st.info("Run the offline stats to load summaries.")
        return

    def _pairs_from_entries(entries: list[object]) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for entry in entries:
            scene_id = str(getattr(entry, "scene_id", ""))
            snippet_id = str(getattr(entry, "snippet_id", ""))
            if not scene_id:
                continue
            pairs.add((scene_id, _extract_snippet_token(snippet_id)))
        return pairs

    def _render_cache_discrepancies(
        *,
        oracle_dir: Path | None,
        vin_dir: Path | None,
    ) -> None:
        if oracle_dir is None or vin_dir is None:
            st.info("Cache discrepancy check skipped: missing oracle or VIN cache dir.")
            return
        oracle_index = oracle_dir / "index.jsonl"
        vin_index = vin_dir / "index.jsonl"
        if not oracle_index.exists() or not vin_index.exists():
            st.info("Cache discrepancy check skipped: missing index.jsonl.")
            return

        oracle_entries = read_cache_index_entries(oracle_index)
        vin_entries = read_cache_index_entries(vin_index)
        oracle_pairs = _pairs_from_entries(oracle_entries)
        vin_pairs = _pairs_from_entries(vin_entries)

        missing_in_vin = sorted(oracle_pairs - vin_pairs)
        missing_in_oracle = sorted(vin_pairs - oracle_pairs)

        col1, col2, col3 = st.columns(3)
        col1.metric("Oracle cache pairs", len(oracle_pairs))
        col2.metric("VIN cache pairs", len(vin_pairs))
        col3.metric("Missing in VIN", len(missing_in_vin))
        if missing_in_oracle:
            st.caption(f"Missing in Oracle: {len(missing_in_oracle)}")

        if missing_in_vin:
            st.subheader("Oracle entries missing in VIN cache")
            df = pd.DataFrame(missing_in_vin[:200], columns=["scene_id", "snippet_token"])
            st.dataframe(df, width="stretch", height=220)
        if missing_in_oracle:
            st.subheader("VIN entries missing in Oracle cache")
            df = pd.DataFrame(missing_in_oracle[:200], columns=["scene_id", "snippet_token"])
            st.dataframe(df, width="stretch", height=220)

    st.subheader("Cache discrepancies")
    if cache_kind == "oracle_rri_cache":
        oracle_dir = Path(cache_dir) if cache_dir else None
        vin_dir = oracle_dir / "vin_snippet_cache" if oracle_dir is not None else None
    else:
        vin_dir = Path(cache_dir) if cache_dir else None
        oracle_dir = vin_dir.parent if vin_dir is not None else None
    _render_cache_discrepancies(oracle_dir=oracle_dir, vin_dir=vin_dir)

    if stats_cache.get("cache_kind") == "vin_snippet_cache":
        summary = stats_cache["summary"]
        sample_df = stats_cache["sample_df"]
        points_counts = stats_cache["points_counts"]
        traj_lengths = stats_cache["traj_lengths"]
        inv_std_values = stats_cache.get("inv_dist_std", [])
        obs_count_values = stats_cache.get("obs_count", [])
        has_obs_count = bool(stats_cache.get("has_obs_count"))

        def _fmt(value: float) -> str:
            return f"{value:.4f}" if np.isfinite(value) else "n/a"

        col1, col2, col3 = st.columns(3)
        col1.metric("Samples", summary["samples"])
        col2.metric("Points mean", _fmt(summary["points_mean"]))
        col3.metric("Traj len mean", _fmt(summary["traj_len_mean"]))
        col4, col5, col6 = st.columns(3)
        col4.metric("Points median", _fmt(summary["points_median"]))
        col5.metric("Traj len median", _fmt(summary["traj_len_median"]))
        col6.metric("Points max", _fmt(summary.get("points_max", float("nan"))))
        col7, col8, col9 = st.columns(3)
        col7.metric("inv_dist_std mean (points)", _fmt(summary.get("inv_dist_std_mean", float("nan"))))
        col8.metric("inv_dist_std std (points)", _fmt(summary.get("inv_dist_std_std", float("nan"))))
        col9.metric("inv_dist_std p95 (points)", _fmt(summary.get("inv_dist_std_p95", float("nan"))))
        if has_obs_count:
            col10, col11, col12 = st.columns(3)
            col10.metric("obs_count mean (points)", _fmt(summary.get("obs_count_mean", float("nan"))))
            col11.metric("obs_count std (points)", _fmt(summary.get("obs_count_std", float("nan"))))
            col12.metric("obs_count p95 (points)", _fmt(summary.get("obs_count_p95", float("nan"))))
            col13, col14, col15, col16 = st.columns(4)
            col13.metric("inv_dist_std min", _fmt(summary.get("inv_dist_std_min", float("nan"))))
            col14.metric("inv_dist_std max", _fmt(summary.get("inv_dist_std_max", float("nan"))))
            col15.metric("obs_count min", _fmt(summary.get("obs_count_min", float("nan"))))
            col16.metric("obs_count max", _fmt(summary.get("obs_count_max", float("nan"))))
        else:
            col10, col11 = st.columns(2)
            col10.metric("inv_dist_std min", _fmt(summary.get("inv_dist_std_min", float("nan"))))
            col11.metric("inv_dist_std max", _fmt(summary.get("inv_dist_std_max", float("nan"))))

        if not sample_df.empty:
            st.subheader("Per-snippet summary")
            st.dataframe(sample_df, width="stretch", height=240)

        with DEFAULT_PLOT_CFG.apply():
            st.subheader("VIN snippet distributions")
            if points_counts:
                _render_hist(
                    points_counts,
                    bins=60,
                    title="Collapsed semidense point count",
                    xlabel="# points",
                    log_y=log_y,
                )
            if traj_lengths:
                _render_hist(
                    traj_lengths,
                    bins=60,
                    title="Trajectory length",
                    xlabel="# poses",
                    log_y=log_y,
                )
            if inv_std_values:
                _render_hist(
                    inv_std_values,
                    bins=60,
                    title="inv_dist_std distribution (all points)",
                    xlabel="inv_dist_std",
                    log_y=log_y,
                )
            if obs_count_values:
                _render_hist(
                    obs_count_values,
                    bins=60,
                    title="obs_count distribution (all points)",
                    xlabel="obs_count",
                    log_y=log_y,
                )
        st.subheader("VIN batch tensor shapes")
        snippet_shapes = stats_cache.get("snippet_shapes")
        if snippet_shapes:
            st.caption("Raw VinSnippetView shapes")
            st.json(snippet_shapes)
        vin_batch_shapes = stats_cache.get("vin_batch_shapes")
        if vin_batch_shapes:
            st.caption("VinOracleBatch shapes (raw vs. padded)")
            st.json(vin_batch_shapes)
        return

    summary = stats_cache["summary"]
    sample_df = stats_cache["sample_df"]
    backbone_df = stats_cache["backbone_df"]
    rri_values = stats_cache["rri_values"]
    pm_comp_after_values = stats_cache["pm_comp_after_values"]
    pm_acc_after_values = stats_cache["pm_acc_after_values"]
    num_valid_values = stats_cache["num_valid_values"]
    batch_shapes = stats_cache.get("batch_shapes")

    def _fmt(value: float) -> str:
        return f"{value:.4f}" if np.isfinite(value) else "n/a"

    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", summary["samples"])
    col2.metric("Total candidates", summary["total_candidates"])
    col3.metric("RRI mean", _fmt(summary["rri_mean"]))
    col4, col5, col6 = st.columns(3)
    col4.metric("RRI median", _fmt(summary["rri_median"]))
    col5.metric("pm_comp_after mean", _fmt(summary["pm_comp_after_mean"]))
    col6.metric("pm_acc_after mean", _fmt(summary["pm_acc_after_mean"]))

    if batch_shapes:
        st.subheader("VIN batch tensor shapes")
        st.json(batch_shapes)

    memory_summary = stats_cache.get("memory_summary")
    if isinstance(memory_summary, dict) and memory_summary:
        st.subheader("Memory footprint")
        _info_popover(
            "memory footprint",
            "Estimates the in-memory footprint (CPU) of the returned VinOracleBatch "
            "components by summing tensor storage sizes. This is a lower bound on "
            "peak memory during loading (temporary decode buffers are not included).",
        )
        order = ["backbone", "vin_snippet", "rri", "pose_camera", "total"]
        rows = []
        for name in order:
            stats = memory_summary.get(name)
            if not isinstance(stats, dict):
                continue
            rows.append(
                {
                    "component": name,
                    "mean_mib": float(stats.get("mean_mib", float("nan"))),
                    "median_mib": float(stats.get("median_mib", float("nan"))),
                    "p95_mib": float(stats.get("p95_mib", float("nan"))),
                },
            )
        mem_df = pd.DataFrame(rows)
        if not mem_df.empty:
            st.dataframe(mem_df, width="stretch", height=220)
            with DEFAULT_PLOT_CFG.apply():
                plot_df = mem_df.copy()
                if log_y:
                    for col in ("mean_mib", "p95_mib"):
                        vals = plot_df[col].to_numpy(dtype=float, copy=True)
                        vals[vals <= 0] = np.nan
                        plot_df[col] = vals

                fig, ax = plt.subplots(figsize=(7, 3))
                sns.barplot(data=plot_df, x="component", y="mean_mib", ax=ax, color="#4c78a8")
                if log_y:
                    ax.set_yscale("log")
                ax.set_title(_pretty_label("Mean memory footprint by component"))
                ax.set_xlabel(_pretty_label("component"))
                ax.set_ylabel(_pretty_label("MiB (log)" if log_y else "MiB"))
                ax.tick_params(axis="x", rotation=25)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(7, 3))
                sns.barplot(data=plot_df, x="component", y="p95_mib", ax=ax, color="#f58518")
                if log_y:
                    ax.set_yscale("log")
                ax.set_title(_pretty_label("P95 memory footprint by component"))
                ax.set_xlabel(_pretty_label("component"))
                ax.set_ylabel(_pretty_label("MiB (log)" if log_y else "MiB"))
                ax.tick_params(axis="x", rotation=25)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

    def _apply_log_y(fig: go.Figure) -> None:
        if not log_y:
            return
        for trace in fig.data:
            if getattr(trace, "y", None) is None:
                continue
            y_vals = np.asarray(trace.y, dtype=float)
            y_vals[y_vals <= 0] = np.nan
            trace.y = y_vals
        fig.update_yaxes(type="log", title_text=_pretty_label("count (log)"))

    if not sample_df.empty:
        st.subheader("Per-sample summary")
        st.dataframe(sample_df, width="stretch", height=240)

    with DEFAULT_PLOT_CFG.apply():
        st.subheader("Global RRI metrics")
        _info_popover(
            "rri distributions",
            "Histograms show the global distribution of RRI and its components across "
            "all cached candidates. Skewed or heavy-tailed distributions can indicate "
            "sampling bias or failure cases in the oracle pipeline.",
        )
        if rri_values:
            _render_hist(
                rri_values,
                bins=60,
                title="Oracle RRI distribution (all candidates)",
                xlabel="RRI",
                log_y=log_y,
            )
        if pm_comp_after_values:
            _render_hist(
                pm_comp_after_values,
                bins=60,
                title="pm_comp_after distribution (all candidates)",
                xlabel="Mesh→point distance",
                log_y=log_y,
            )
        if pm_acc_after_values:
            _render_hist(
                pm_acc_after_values,
                bins=60,
                title="pm_acc_after distribution (all candidates)",
                xlabel="Point→mesh distance",
                log_y=log_y,
            )

        st.subheader("RRI binning")
        _info_popover(
            "rri binning",
            "Fits quantile-based ordinal bins for CORAL training using the cached "
            "RRI distribution. Edges are recomputed from the selected offline "
            "samples; labels show how many candidates fall into each ordinal class.",
        )
        if rri_values:
            num_classes = int(
                st.slider(
                    "Num classes (K)",
                    min_value=2,
                    max_value=50,
                    value=15,
                    step=1,
                    key="vin_offline_binner_classes",
                ),
            )
            fit_binner = st.button(
                "Fit binner from offline RRI",
                key="vin_offline_binner_fit",
            )
            binner_key = f"{cfg_key}|{num_classes}"
            if st.session_state.get("vin_offline_binner_key") != binner_key or fit_binner:
                rri_tensor = torch.tensor(rri_values, dtype=torch.float32)
                binner = RriOrdinalBinner.fit_from_iterable(
                    [rri_tensor],
                    num_classes=num_classes,
                )
                labels = binner.transform(rri_tensor)
                st.session_state["vin_offline_binner_key"] = binner_key
                st.session_state["vin_offline_binner"] = binner
                st.session_state["vin_offline_binner_labels"] = labels

            binner = st.session_state.get("vin_offline_binner")
            labels = st.session_state.get("vin_offline_binner_labels")
            if binner is not None and labels is not None:
                fig, ax = _build_hist_ax(
                    rri_values,
                    bins=60,
                    title="Raw oracle RRI + quantile edges",
                    xlabel="rri",
                    log_y=log_y,
                    figsize=(8, 3.5),
                )
                for edge in binner.edges.detach().cpu().numpy().tolist():
                    ax.axvline(float(edge), color="black", linewidth=1.0, alpha=0.25)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

                counts = (
                    torch.bincount(
                        labels.to(torch.int64),
                        minlength=int(binner.num_classes),
                    )
                    .cpu()
                    .numpy()
                )
                y_vals = counts.astype(float)
                if log_y:
                    y_vals[y_vals <= 0] = np.nan
                fig, ax = plt.subplots(figsize=(7, 3))
                sns.barplot(
                    x=np.arange(int(binner.num_classes)),
                    y=y_vals,
                    color="#285f82",
                    ax=ax,
                )
                if log_y:
                    ax.set_yscale("log")
                ax.set_title(_pretty_label("Ordinal labels (K classes)"))
                ax.set_xlabel(_pretty_label("label"))
                ax.set_ylabel(
                    _pretty_label("count (log)" if log_y else "count"),
                )
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
        else:
            st.info("No RRI values available to fit the binner.")

    if num_valid_values:
        st.subheader("Valid candidate counts")
        _info_popover(
            "valid counts",
            "Histogram of the number of valid candidates per snippet. Low counts "
            "often indicate aggressive rule filtering or challenging geometry.",
        )
        _render_hist(
            num_valid_values,
            bins=30,
            title="num_valid per snippet",
            xlabel="num_valid",
            log_y=log_y,
        )

    candidate_offsets = stats_cache.get("candidate_offsets")
    candidate_yaw = stats_cache.get("candidate_yaw")
    candidate_pitch = stats_cache.get("candidate_pitch")
    candidate_roll = stats_cache.get("candidate_roll")
    candidate_rot_deg = stats_cache.get("candidate_rot_deg")
    if isinstance(candidate_offsets, np.ndarray) and candidate_offsets.size and candidate_offsets.shape[-1] == 3:
        st.subheader("Candidate pose distributions")
        _info_popover(
            "candidate distributions",
            "These distributions are computed over **all candidates** from the "
            "offline cache, expressed in the reference rig frame. Offsets are "
            "the candidate translations `t_rc`; yaw/pitch/roll are derived from "
            "the relative rotation `R_rc`; the rotation delta is the SO(3) angle "
            "between candidate and reference orientation (a measure of jitter).",
        )
        cand_bins = int(
            st.slider(
                "Candidate histogram bins",
                min_value=20,
                max_value=180,
                value=60,
                key="vin_offline_candidate_bins",
            ),
        )
        show_polar = st.checkbox(
            "Show azimuth/elevation heatmap",
            value=True,
            key="vin_offline_candidate_polar",
        )
        if show_polar:
            fig_polar = plot_position_polar(
                candidate_offsets,
                title=_pretty_label("Offset Azimuth/Elevation (Rig Frame)"),
                bins=cand_bins,
                fixed_ranges=True,
            )
            fig_polar.update_layout(
                title=_pretty_label("Offset Azimuth/Elevation (Rig Frame)"),
                xaxis_title=_pretty_label("azimuth (deg)"),
                yaxis_title=_pretty_label("elevation (deg)"),
            )
            st.plotly_chart(fig_polar, width="stretch")

        az = np.degrees(
            np.arctan2(candidate_offsets[:, 0], candidate_offsets[:, 2]),
        )
        el = np.degrees(
            np.arctan2(
                candidate_offsets[:, 1],
                np.linalg.norm(candidate_offsets[:, [0, 2]], axis=1) + 1e-8,
            ),
        )
        radius = np.linalg.norm(candidate_offsets, axis=1)
        col_a, col_b = st.columns(2)
        with col_a:
            fig_az = _histogram_overlay(
                [("azimuth", az)],
                bins=cand_bins,
                title="Offset azimuth distribution",
                xaxis_title="azimuth (deg)",
                log1p_counts=False,
            )
            _apply_log_y(fig_az)
            st.plotly_chart(fig_az, width="stretch")
        with col_b:
            fig_el = _histogram_overlay(
                [("elevation", el)],
                bins=cand_bins,
                title="Offset elevation distribution",
                xaxis_title="elevation (deg)",
                log1p_counts=False,
            )
            _apply_log_y(fig_el)
            st.plotly_chart(fig_el, width="stretch")

        fig_r = _histogram_overlay(
            [("radius", radius)],
            bins=cand_bins,
            title="Offset radius distribution",
            xaxis_title="radius (m)",
            log1p_counts=False,
        )
        _apply_log_y(fig_r)
        st.plotly_chart(fig_r, width="stretch")

        if (
            isinstance(candidate_yaw, np.ndarray)
            and isinstance(candidate_pitch, np.ndarray)
            and isinstance(candidate_roll, np.ndarray)
        ):
            fig_angles = _histogram_overlay(
                [
                    ("yaw", candidate_yaw),
                    ("pitch", candidate_pitch),
                    ("roll", candidate_roll),
                ],
                bins=cand_bins,
                title="Candidate orientation distribution (yaw/pitch/roll)",
                xaxis_title="angle (deg)",
                log1p_counts=False,
            )
            _apply_log_y(fig_angles)
            st.plotly_chart(fig_angles, width="stretch")

        if isinstance(candidate_rot_deg, np.ndarray):
            fig_rot = _histogram_overlay(
                [("rotation_delta", candidate_rot_deg)],
                bins=cand_bins,
                title="Rotation delta distribution",
                xaxis_title="rotation delta (deg)",
                log1p_counts=False,
            )
            _apply_log_y(fig_rot)
            st.plotly_chart(fig_rot, width="stretch")

    if not sample_df.empty:
        st.subheader("Scatter diagnostics")
        _info_popover(
            "scatter diagnostics",
            "Cross-plots relate mean RRI to accuracy/completeness and to the "
            "number of valid candidates. These plots help spot correlations and "
            "outliers in the oracle labels.",
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=sample_df,
            x="rri_mean",
            y="pm_comp_after_mean",
            ax=ax,
        )
        ax.set_title(_pretty_label("RRI mean vs pm_comp_after mean"))
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=sample_df,
            x="rri_mean",
            y="pm_acc_after_mean",
            ax=ax,
        )
        ax.set_title(_pretty_label("RRI mean vs pm_acc_after mean"))
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=sample_df, x="num_valid", y="rri_mean", ax=ax)
        ax.set_title(_pretty_label("num_valid vs RRI mean"))
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        if not backbone_df.empty:
            st.subheader("Backbone feature statistics")
            _info_popover(
                "backbone stats",
                "Per-field statistics of EVL backbone outputs. High variance or "
                "extreme sparsity can indicate scale mismatches or missing modalities.",
            )
            field_options = sorted(backbone_df["field"].dropna().unique().tolist())
            selected_fields = st.multiselect(
                "Backbone fields",
                options=field_options,
                default=field_options,
                key="vin_offline_backbone_fields",
            )
            if selected_fields:
                backbone_df = backbone_df.query("field in @selected_fields")
            if backbone_df.empty:
                st.info("No backbone fields selected.")
                return
            metric_cols = [col for col in ["mean", "std", "abs_mean", "nz_frac", "numel"] if col in backbone_df.columns]
            selected_cols = st.multiselect(
                "Backbone stats columns",
                options=metric_cols,
                default=metric_cols,
                key="vin_offline_backbone_cols",
            )
            if not selected_cols:
                st.info("Select at least one metric column to summarize.")
                return
            sort_metric = "std" if "std" in selected_cols else selected_cols[0]
            sort_choice = st.selectbox(
                "Sort metric",
                options=selected_cols,
                index=selected_cols.index(sort_metric),
                key="vin_offline_backbone_sort",
            )
            summary_df = (
                backbone_df.groupby("field", as_index=False)[selected_cols]
                .mean()
                .sort_values(sort_choice, ascending=False)
            )
            st.dataframe(summary_df, width="stretch", height=280)

            plot_metric = st.selectbox(
                "Plot metric",
                options=selected_cols,
                index=selected_cols.index(sort_metric),
                key="vin_offline_backbone_plot_metric",
            )
            top_df = summary_df.head(10)
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=top_df, x=plot_metric, y="field", ax=ax)
            ax.set_title(
                _pretty_label(f"Mean feature {plot_metric} (top 10)"),
            )
            ax.set_xlabel(_pretty_label(plot_metric))
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
