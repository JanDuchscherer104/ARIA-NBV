"""Pose descriptor tab for VIN diagnostics."""

from __future__ import annotations

import streamlit as st

from ....pose_generation.plotting import (
    plot_direction_polar,
    plot_direction_sphere,
    plot_radius_hist,
)
from ..common import _info_popover, _pretty_label
from ..plot_utils import _histogram_overlay, _to_numpy
from .context import VinDiagContext


def render_pose_tab(ctx: VinDiagContext) -> None:
    """Render the Pose Descriptor tab.

    Args:
        ctx: Shared VIN diagnostics context.
    """
    debug = ctx.debug
    has_tokens = ctx.has_tokens

    _info_popover(
        "pose descriptor",
        "Candidate centers are translations of `T_rig_ref_cam` "
        "(reference rig frame). Radii are `||t||` in meters. "
        "For VIN v1, direction plots show unit vectors for center "
        "directions and forward axes; view alignment is `dot(f, -u)`, "
        "measuring how much the camera looks back toward the rig. "
        "VIN v2 does not compute frustum tokens.",
    )
    centers = _to_numpy(debug.candidate_center_rig_m.reshape(-1, 3))
    st.plotly_chart(
        plot_radius_hist(
            centers,
            title=_pretty_label("Candidate radii (reference rig)"),
        ),
        width="stretch",
    )

    if has_tokens:
        center_dirs = _to_numpy(debug.candidate_center_dir_rig.reshape(-1, 3))
        forward_dirs = _to_numpy(debug.candidate_forward_dir_rig.reshape(-1, 3))
        view_align = _to_numpy(debug.view_alignment.reshape(-1))

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                plot_direction_polar(
                    center_dirs,
                    title=_pretty_label("Candidate center directions (rig frame)"),
                ),
                width="stretch",
            )
        with col2:
            st.plotly_chart(
                plot_direction_sphere(
                    center_dirs,
                    title=_pretty_label("Center directions on unit sphere"),
                ),
                width="stretch",
            )

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(
                plot_direction_polar(
                    forward_dirs,
                    title=_pretty_label("Candidate forward directions (rig frame)"),
                ),
                width="stretch",
            )
        with col4:
            st.plotly_chart(
                plot_direction_sphere(
                    forward_dirs,
                    title=_pretty_label("Forward directions on unit sphere"),
                ),
                width="stretch",
            )

        log1p_pose_counts = st.checkbox(
            "Log1p alignment histogram counts",
            value=False,
            key="vin_pose_align_log1p",
        )
        fig_align = _histogram_overlay(
            [("alignment", view_align)],
            bins=60,
            title=_pretty_label("View alignment dot(f, -u)"),
            xaxis_title=_pretty_label("dot(f, -u)"),
            log1p_counts=log1p_pose_counts,
        )
        st.plotly_chart(fig_align, width="stretch")
    else:
        st.info("Pose-direction plots are only available for VIN v1 diagnostics.")


__all__ = ["render_pose_tab"]
