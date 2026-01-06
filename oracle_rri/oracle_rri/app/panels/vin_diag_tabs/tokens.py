"""Frustum token tab for VIN diagnostics."""

from __future__ import annotations

import streamlit as st
import torch

from ....vin.plotting import (
    build_frustum_samples_figure,
    build_semidense_projection_figure,
)
from ..common import _info_popover, _pretty_label
from ..plot_utils import _plot_slice_grid, _to_numpy
from .context import VinDiagContext


def render_tokens_tab(ctx: VinDiagContext) -> None:
    """Render the Frustum Tokens tab.

    Args:
        ctx: Shared VIN diagnostics context.
    """
    debug = ctx.debug
    batch = ctx.batch
    cfg = ctx.cfg

    if not ctx.has_tokens and not ctx.has_semidense_frustum:
        st.info("Frustum token diagnostics are only available for VIN v1/v2 frustum features.")
        return

    cand_idx = st.slider(
        "Candidate index",
        0,
        max(0, ctx.num_candidates - 1),
        0,
        key="vin_token_cand_idx",
    )

    if ctx.has_tokens:
        _info_popover(
            "frustum tokens",
            "VIN v1 samples the scene field along a frustum grid of size "
            "grid_size x grid_size x num_depths. Token norms show feature "
            "strength per sample; token_valid marks whether a sample lies "
            "inside the voxel field.",
        )
        grid_size = int(cfg.module_config.vin.frustum_grid_size)
        depth_values = list(cfg.module_config.vin.frustum_depths_m)
        num_depths = len(depth_values)

        max_depths = st.slider(
            "Max depth planes",
            1,
            max(1, num_depths),
            min(4, num_depths),
            key="vin_token_depths",
        )

        tokens = debug.tokens[0, cand_idx]
        token_valid = debug.token_valid[0, cand_idx]
        token_norm = torch.linalg.vector_norm(tokens, dim=-1)

        expected_k = grid_size * grid_size * num_depths
        if int(token_norm.numel()) != expected_k:
            st.warning(
                "Token count does not match grid_size² x num_depths; skipping grid view.",
            )
        else:
            token_norm = token_norm.view(num_depths, grid_size, grid_size)
            token_valid = token_valid.view(num_depths, grid_size, grid_size).float()

            depth_labels = [f"d={d:g}m" for d in depth_values[:max_depths]]
            norm_slices = [_to_numpy(token_norm[i]) for i in range(max_depths)]
            valid_slices = [_to_numpy(token_valid[i]) for i in range(max_depths)]

            fig_norm = _plot_slice_grid(
                norm_slices,
                titles=depth_labels,
                title=_pretty_label("Token feature norm per depth plane"),
                percentile=99.0,
                symmetric=False,
                cmap_name="viridis",
            )
            st.plotly_chart(fig_norm, width="stretch")

            fig_valid = _plot_slice_grid(
                valid_slices,
                titles=depth_labels,
                title=_pretty_label("Token validity per depth plane"),
                percentile=100.0,
                symmetric=False,
                cmap_name="gray",
            )
            st.plotly_chart(fig_valid, width="stretch")

        st.subheader("Frustum samples in world")
        st.plotly_chart(
            build_frustum_samples_figure(
                debug,
                p3d_cameras=batch.p3d_cameras,
                candidate_index=int(cand_idx),
                grid_size=int(grid_size),
                depths_m=depth_values,
            ),
            width="stretch",
        )

    if ctx.has_semidense_frustum and batch.efm_snippet_view is not None:
        st.subheader("Semidense projection (candidate visibility)")
        try:
            semidense = batch.efm_snippet_view.semidense
        except Exception:  # pragma: no cover - UI guard
            semidense = None
        if semidense is None:
            st.info("Semidense points unavailable for this snippet.")
        else:
            points_world = semidense.collapse_points(
                max_points=int(cfg.module_config.vin.semidense_proj_max_points),
                include_inv_dist_std=True,
            )
            if points_world.numel() == 0:
                st.info("Semidense points are empty for this snippet.")
            else:
                st.plotly_chart(
                    build_semidense_projection_figure(
                        points_world,
                        p3d_cameras=batch.p3d_cameras,
                        candidate_index=int(cand_idx),
                        show_frustum=True,
                    ),
                    width="stretch",
                )
    elif ctx.has_semidense_frustum:
        st.info("Attach the EFM snippet to visualize semidense frustum projections.")


__all__ = ["render_tokens_tab"]
