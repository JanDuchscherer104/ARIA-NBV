"""Pose encoding tab for VIN diagnostics."""

from __future__ import annotations

import streamlit as st
import torch

from ....vin.plotting import (
    build_candidate_encoding_figures,
    build_lff_empirical_figures,
    build_pose_enc_pca_figure,
    build_pose_grid_pca_figure,
    build_pose_grid_slices_figure,
    build_pose_vec_histogram,
    build_vin_encoding_figures,
)
from ..common import _info_popover
from .context import VinDiagContext


def render_encodings_tab(ctx: VinDiagContext) -> None:
    """Render the FF Encodings tab.

    Args:
        ctx: Shared VIN diagnostics context.
    """
    debug = ctx.debug
    batch = ctx.batch
    cfg = ctx.cfg
    state = ctx.state

    _info_popover(
        "ff encodings",
        "This tab inspects the pose-encoding pathway used by VIN. "
        "VIN v1 relies on spherical features plus a learnable Fourier "
        "feature (LFF) block; VIN v2 uses LFF over the pose vector "
        "`[t_x, t_y, t_z, r6d_0..5]` and a positional grid for attention keys. "
        "The plots below show distributions and low-dimensional projections "
        "to diagnose scale, anisotropy, and feature collapse.",
    )
    pose_encoder_lff = state.module.vin.pose_encoder_lff if state.module is not None else None
    pose_enc = debug.pose_enc.reshape(-1, debug.pose_enc.shape[-1])

    if ctx.has_tokens:
        st.caption(
            "Plot Learnable Fourier Features for the actual encoded candidates.",
        )
        _info_popover(
            "lff diagnostics",
            "These plots visualize the LFF block used inside the pose encoder. "
            "Weight-space figures (Wr and its norms) show learned frequency "
            "directions, while candidate-space figures show the actual "
            "Fourier activations for the current batch. Look for mode collapse "
            "(features with near-zero variance) or extreme saturation.",
        )
        lmax = int(cfg.plot_lmax)
        sh_norm = str(cfg.plot_sh_normalization)
        freq_list = list(cfg.plot_radius_freqs)
        max_candidates = int(pose_enc.shape[0])
        max_pose_dims = int(pose_enc.shape[-1])
        max_sh_components = 64
        log1p_counts = st.checkbox(
            "Log1p histogram counts",
            value=False,
            key="vin_plot_log1p",
        )
        plot_btn = st.button("Generate encoding plots", key="vin_plot_btn")

        if plot_btn:
            figs = build_vin_encoding_figures(
                debug,
                lmax=int(lmax),
                sh_normalization=str(sh_norm),
                radius_freqs=freq_list,
                pose_encoder_lff=pose_encoder_lff,
                include_legacy_sh=False,
                log1p_counts=log1p_counts,
            )
            actual_figs = build_candidate_encoding_figures(
                debug,
                lmax=int(lmax),
                sh_normalization=str(sh_norm),
                radius_freqs=freq_list,
                pose_encoder_lff=pose_encoder_lff,
                include_legacy_sh=False,
                max_candidates=int(max_candidates),
                max_sh_components=int(max_sh_components),
                max_pose_dims=int(max_pose_dims),
            )

            for label, fig in figs.items():
                st.plotly_chart(fig, width="stretch", key=f"vin_plot_{label}")
            for label, fig in actual_figs.items():
                st.plotly_chart(
                    fig,
                    width="stretch",
                    key=f"vin_plot_actual_{label}",
                )
    else:
        st.caption("VIN v2 positional encodings (pose grid + LFF pose encoder).")
        vin_model = state.module.vin if state.module is not None else None
        if vin_model is None:
            st.info("VIN model not available.")
        else:
            pose_vec = debug.pose_vec
            if pose_vec is not None:
                _info_popover(
                    "pose vector",
                    "Histogram of a single pose-vector component. "
                    "Translation entries reflect candidate displacement "
                    "in the reference rig frame; rotation entries are "
                    "the 6D rotation representation. Extreme ranges or "
                    "heavy skew indicate scaling issues before LFF.",
                )
                dim_labels = [
                    "t_x",
                    "t_y",
                    "t_z",
                    "r6d_0",
                    "r6d_1",
                    "r6d_2",
                    "r6d_3",
                    "r6d_4",
                    "r6d_5",
                ]
                dim_index = int(
                    st.selectbox(
                        "Pose input component",
                        options=list(range(len(dim_labels))),
                        format_func=lambda idx: dim_labels[idx],
                        key="vin_pose_dim",
                    ),
                )
                log1p_pose_counts = st.checkbox(
                    "Log1p pose histogram counts",
                    value=False,
                    key="vin_pose_vec_log1p",
                )
                st.plotly_chart(
                    build_pose_vec_histogram(
                        pose_vec,
                        dim_index=dim_index,
                        num_bins=60,
                        log1p_counts=log1p_pose_counts,
                    ),
                    width="stretch",
                )

                max_features = int(
                    st.slider(
                        "Max features",
                        min_value=16,
                        max_value=256,
                        value=96,
                    ),
                )
                hist_bins = int(
                    st.slider(
                        "LFF hist bins",
                        min_value=20,
                        max_value=200,
                        value=60,
                    ),
                )
                max_points = int(
                    st.slider(
                        "LFF max points",
                        min_value=1000,
                        max_value=20000,
                        value=8000,
                    ),
                )
                log1p_lff_counts = st.checkbox(
                    "Log1p LFF histogram counts",
                    value=False,
                    key="vin_lff_hist_log1p",
                )

                if pose_encoder_lff is not None:
                    _info_popover(
                        "lff empirical",
                        "Empirical histograms and PCA for the LFF block. "
                        "Fourier features are the raw sin/cos projection "
                        "of the pose vector; the MLP output is the learned "
                        "mixture. PCA helps spot anisotropy or dead features.",
                    )
                    lff_figs = build_lff_empirical_figures(
                        pose_vec,
                        pose_encoder_lff,
                        max_features=max_features,
                        hist_bins=hist_bins,
                        max_points=max_points,
                        log1p_counts=log1p_lff_counts,
                    )
                    st.plotly_chart(
                        lff_figs["lff_empirical_fourier_hist"],
                        width="stretch",
                    )
                    st.plotly_chart(
                        lff_figs["lff_empirical_mlp_hist"],
                        width="stretch",
                    )
                    st.plotly_chart(
                        lff_figs["lff_empirical_fourier_pca"],
                        width="stretch",
                    )
                    st.plotly_chart(
                        lff_figs["lff_empirical_mlp_pca"],
                        width="stretch",
                    )

                color_mode = st.selectbox(
                    "Pose encoding color",
                    options=["translation_norm", "candidate_index"],
                    index=0,
                    key="vin_pose_enc_color",
                )
                if color_mode == "translation_norm":
                    color_values = torch.linalg.vector_norm(
                        pose_vec[..., :3],
                        dim=-1,
                    )
                    color_label = "|t|"
                else:
                    color_values = torch.arange(
                        pose_vec.shape[1],
                        device=pose_vec.device,
                    ).view(1, -1)
                    color_label = "candidate idx"
                _info_popover(
                    "pose enc pca",
                    "PCA of the final LFF pose encoding. Color can reflect "
                    "translation magnitude or candidate index. A smooth "
                    "gradient suggests that pose magnitude is well represented; "
                    "tight clumps can indicate collapsed embeddings.",
                )
                st.plotly_chart(
                    build_pose_enc_pca_figure(
                        debug.pose_enc,
                        color_values=color_values,
                        color_label=color_label,
                    ),
                    width="stretch",
                )

            try:
                field_in = debug.field_in
                grid_shape = (
                    int(field_in.shape[-3]),
                    int(field_in.shape[-2]),
                    int(field_in.shape[-1]),
                )
                pos_grid = vin_model._pos_grid_from_pts_world(
                    debug.backbone_out.pts_world,
                    t_world_voxel=debug.backbone_out.t_world_voxel,
                    pose_world_rig_ref=batch.reference_pose_world_rig,
                    voxel_extent=debug.backbone_out.voxel_extent,
                    grid_shape=grid_shape,
                )
                axis = st.selectbox(
                    "Grid axis",
                    options=["D", "H", "W"],
                    index=0,
                    key="vin_grid_axis",
                )
                max_index = {
                    "D": grid_shape[0],
                    "H": grid_shape[1],
                    "W": grid_shape[2],
                }[axis] - 1
                slice_idx = int(
                    st.slider(
                        "Grid slice index",
                        min_value=0,
                        max_value=max_index,
                        value=max_index // 2,
                    ),
                )
                _info_popover(
                    "pos grid slices",
                    "Position grid slices show the normalized voxel centers "
                    "in the reference rig frame (pos_x/pos_y/pos_z). "
                    "Expect near-linear gradients across each axis; "
                    "distortions indicate mismatched voxel extents or frames.",
                )
                st.plotly_chart(
                    build_pose_grid_slices_figure(
                        pos_grid,
                        axis=axis,
                        index=slice_idx,
                    ),
                    width="stretch",
                )
                color_by = st.selectbox(
                    "Pos grid PCA color",
                    options=["radius", "x", "y", "z"],
                    index=0,
                    key="vin_grid_color",
                )
                show_axes = st.checkbox(
                    "Show rig axes",
                    value=True,
                    key="vin_grid_axes",
                )
                axis_scale = float(
                    st.slider(
                        "Axis scale",
                        min_value=0.1,
                        max_value=2.0,
                        value=0.5,
                        step=0.1,
                    ),
                )
                _info_popover(
                    "pos grid pca",
                    "PCA of the positional embeddings used as attention keys "
                    "for the global pool. The axis overlays show how the "
                    "learned projection aligns with x/y/z directions.",
                )
                st.plotly_chart(
                    build_pose_grid_pca_figure(
                        pos_grid,
                        pos_proj=vin_model.global_pooler.pos_proj,
                        max_points=8000,
                        color_by=color_by,
                        show_axes=show_axes,
                        axis_scale=axis_scale,
                    ),
                    width="stretch",
                )
            except Exception as exc:  # pragma: no cover - optional diagnostics
                st.info(
                    f"Positional grid plots unavailable: {type(exc).__name__}: {exc}",
                )


__all__ = ["render_encodings_tab"]
