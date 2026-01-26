"""Summary tab for VIN diagnostics."""

from __future__ import annotations

import numpy as np
import plotly.express as px
import streamlit as st
import torch

from ....data.efm_views import VinSnippetView
from ....vin.experimental.model_v2 import FIELD_CHANNELS_V2
from ....vin.plotting import _histogram_overlay, _parameter_distribution, _to_numpy
from ..common import _info_popover, _pretty_label, _strip_ansi
from .context import VinDiagContext


def render_summary_tab(ctx: VinDiagContext) -> None:
    """Render the Summary tab for VIN diagnostics.

    Args:
        ctx: Shared VIN diagnostics context.
    """
    state = ctx.state
    debug = ctx.debug
    pred = ctx.pred
    batch = ctx.batch
    cfg = ctx.cfg

    cache_label = "online (oracle labeler)"
    if ctx.use_offline_cache:
        cache_label = "offline: Oracle RRI cache"
        if isinstance(batch.efm_snippet_view, VinSnippetView):
            cache_label = "offline: VIN snippet cache"
        else:
            offline_cache = getattr(state, "offline_cache", None)
            if offline_cache is not None:
                cache_cfg = getattr(offline_cache, "config", None)
                if cache_cfg is not None:
                    has_vin_cache = cache_cfg.vin_snippet_cache is not None
                    mode = getattr(cache_cfg, "vin_snippet_cache_mode", "auto")
                    if has_vin_cache and mode != "disabled":
                        cache_label = "offline: VIN snippet cache"
        if batch.efm_snippet_view is None:
            cache_label = f"{cache_label} (snippet detached)"
    st.caption(f"Data source: {cache_label}")

    if batch.rri is not None:
        _info_popover(
            "scatter",
            "Each point is a candidate view. **X** is the oracle RRI computed from "
            "mesh distances (before vs after adding the candidate point cloud). "
            "**Y** is the VIN expected score from the CORAL ordinal head "
            "(mean of `P(y>k)` across bins, normalized to `[0,1]`). "
            "VIN v2 uses pose features from `[t, r6d]` in the reference rig "
            "frame plus global voxel context; VIN v1 may also use local "
            "frustum tokens.",
        )
        rri = _to_numpy(batch.rri.reshape(-1))
        expected = _to_numpy(pred.expected_normalized.reshape(-1))
        use_log_axes = st.checkbox(
            "Log scale axes",
            value=False,
            key="vin_summary_log_axes",
        )
        if use_log_axes:
            pos_mask = (rri > 0) & (expected > 0)
            if not np.all(pos_mask):
                st.info(
                    "Log axes require positive values; non-positive points are omitted.",
                )
                rri = rri[pos_mask]
                expected = expected[pos_mask]
        fig = px.scatter(
            x=rri,
            y=expected,
            labels={
                "x": _pretty_label("Oracle RRI"),
                "y": _pretty_label("VIN expected (normalized)"),
            },
            title=_pretty_label("Predicted score vs oracle RRI"),
            log_x=use_log_axes,
            log_y=use_log_axes,
        )
        st.plotly_chart(fig, width="stretch")

    feature_dims: list[tuple[str, int]] = []
    if hasattr(debug, "pose_enc"):
        feature_dims.append(("pose_enc", int(debug.pose_enc.shape[-1])))
    if getattr(debug, "global_feat", None) is not None:
        feature_dims.append(("global_feat", int(debug.global_feat.shape[-1])))
    if hasattr(debug, "local_feat"):
        feature_dims.append(("local_feat", int(debug.local_feat.shape[-1])))
    if feature_dims:
        _info_popover(
            "feature dims",
            "Feature blocks concatenated before the scorer MLP. "
            "VIN v2 uses `pose_enc` (LFF over translation + rotation-6D "
            "with learned scales) and `global_feat` (pose-conditioned "
            "attention pooling over the voxel field). "
            "VIN v1 can add `local_feat` from frustum sampling.",
        )
        dims_df = {
            "modality": [name for name, _ in feature_dims],
            "num_features": [count for _, count in feature_dims],
        }
        fig_dims = px.bar(
            dims_df,
            x="modality",
            y="num_features",
            title=_pretty_label("Feature dimensions by modality"),
        )
        st.plotly_chart(fig_dims, width="stretch")

    vin_model = state.module.vin if state.module is not None else None
    if vin_model is not None:
        params_df = _parameter_distribution(vin_model, trainable_only=True)
        if not params_df.empty:
            _info_popover(
                "param counts",
                "Trainable parameter counts grouped by top-level VIN submodule "
                "(frozen backbone params are excluded). This highlights where "
                "capacity is concentrated (pose encoder vs global pool vs head).",
            )
            fig_params = px.bar(
                params_df,
                x="module",
                y="num_params",
                title=_pretty_label("Trainable parameter counts by VIN module"),
                labels={"num_params": _pretty_label("parameters")},
            )
            st.plotly_chart(fig_params, width="stretch")
            total_params = int(params_df["num_params"].sum())
            st.caption(f"Total trainable parameters: {total_params:,}")

    field_in = getattr(debug, "field_in", None)
    if isinstance(field_in, torch.Tensor) and field_in.ndim == 5:
        _info_popover(
            "field hists",
            "Per-channel scene-field distributions **before** projection. "
            "VIN v2 builds channels from EVL heads: `occ_pr` (occupancy prob), "
            "`cent_pr` (centerness), `occ_input` (occupied evidence), "
            "`counts_norm` (log1p-normalized coverage), `observed`=counts_norm, "
            "`unknown`=1-counts_norm, `free_input` (EVL free-space or derived), "
            "`new_surface_prior`=unknown*occ_pr. Values are mostly in `[0,1]` "
            "and plotted as |value|.",
        )
        channel_count = int(field_in.shape[1])
        if channel_count == len(FIELD_CHANNELS_V2):
            channel_names = list(FIELD_CHANNELS_V2)
        elif channel_count == len(cfg.module_config.vin.scene_field_channels):
            channel_names = list(cfg.module_config.vin.scene_field_channels)
        else:
            channel_names = [f"ch_{idx}" for idx in range(channel_count)]
        default_channels = channel_names[: min(len(channel_names), 6)]
        selected_channels = st.multiselect(
            "Scene field channels",
            options=channel_names,
            default=default_channels,
            key="vin_summary_field_hist_channels",
        )
        log1p_counts = st.checkbox(
            "Log1p histogram counts",
            value=False,
            key="vin_summary_field_hist_log1p",
        )
        hist_bins = int(
            st.slider(
                "Histogram bins",
                min_value=10,
                max_value=200,
                value=60,
                key="vin_summary_field_hist_bins",
            ),
        )
        channel_vals = field_in.abs().detach().cpu()
        series: list[tuple[str, np.ndarray]] = []
        for idx, name in enumerate(channel_names):
            if name not in selected_channels:
                continue
            vals = channel_vals[:, idx, ...].reshape(-1).numpy()
            series.append((name, vals))
        fig_hist = _histogram_overlay(
            series,
            bins=hist_bins,
            title=_pretty_label("Scene field channel |value| distributions"),
            xaxis_title=_pretty_label("|value|"),
            log1p_counts=log1p_counts,
        )
        st.plotly_chart(fig_hist, width="stretch")

    _info_popover(
        "feature norms",
        "Per-candidate L2 norms of feature blocks (pose/global/local). "
        "Very low norms suggest weak signal; very high norms can dominate the "
        "MLP. Compare modalities to spot imbalance or saturation.",
    )
    feat_norms: dict[str, torch.Tensor] = {
        "pose_enc": torch.linalg.vector_norm(debug.pose_enc, dim=-1).reshape(-1),
        "feats": torch.linalg.vector_norm(debug.feats, dim=-1).reshape(-1),
    }
    if hasattr(debug, "local_feat"):
        feat_norms["local_feat"] = torch.linalg.vector_norm(
            debug.local_feat,
            dim=-1,
        ).reshape(-1)
    if getattr(debug, "global_feat", None) is not None:
        feat_norms["global_feat"] = torch.linalg.vector_norm(
            debug.global_feat,
            dim=-1,
        ).reshape(-1)

    log1p_norm_counts = st.checkbox(
        "Log1p feature histogram counts",
        value=False,
        key="vin_summary_feat_norm_log1p",
    )
    norm_series = [(name, _to_numpy(vals)) for name, vals in feat_norms.items()]
    fig = _histogram_overlay(
        norm_series,
        bins=60,
        title=_pretty_label("Feature norms (per-candidate)"),
        xaxis_title=_pretty_label("norm"),
        log1p_counts=log1p_norm_counts,
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("VIN summarize_vin output"):
        include_torchsummary = st.checkbox(
            "Include torchsummary modules",
            value=False,
            key="vin_summary_include_ts",
        )
        torchsummary_depth = int(
            st.slider(
                "Torchsummary depth",
                min_value=1,
                max_value=6,
                value=3,
                key="vin_summary_depth",
            ),
        )
        summary_key = f"{state.cfg_sig}|{batch.scene_id}|{batch.snippet_id}|{include_torchsummary}|{torchsummary_depth}"
        if state.summary_key != summary_key:
            state.summary_key = summary_key
            state.summary_text = None
            state.summary_error = None

        auto_run = state.summary_text is None and state.summary_error is None
        if st.button("Generate summary", key="vin_summary_generate") or auto_run:
            try:
                with st.spinner("Generating VIN summary..."):
                    state.summary_text = state.module.summarize_vin(
                        batch,
                        include_torchsummary=include_torchsummary,
                        torchsummary_depth=torchsummary_depth,
                    )
                state.summary_error = None
            except Exception as exc:  # pragma: no cover - UI guard
                state.summary_error = f"{type(exc).__name__}: {exc}"
                state.summary_text = None

        if state.summary_error:
            st.error(state.summary_error)
        elif state.summary_text:
            st.code(_strip_ansi(state.summary_text))
        else:
            st.info("Click 'Generate summary' to render summarize_vin output.")


__all__ = ["render_summary_tab"]
