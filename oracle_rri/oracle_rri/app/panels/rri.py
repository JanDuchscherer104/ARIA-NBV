"""RRI inspection panel."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ...data import EfmSnippetView
from ...rendering.candidate_depth_renderer import CandidateDepths
from ...rendering.candidate_pointclouds import CandidatePointClouds
from ...rendering.plotting import RenderingPlotBuilder
from ...rri_metrics.types import RriResult
from .common import _info_popover, _pretty_label


def render_rri_page(
    sample: EfmSnippetView,
    depth_batch: CandidateDepths,
    pcs: CandidatePointClouds,
    rri: RriResult,
) -> None:
    st.header("RRI Preview: Point Clouds vs Mesh")

    candidate_ids = depth_batch.candidate_indices.cpu().tolist()
    if len(candidate_ids) == 0:
        st.warning("No candidate renders available for RRI scoring.")
        return

    labels = [str(int(cid)) for cid in candidate_ids]
    baseline_label = "-1"

    qualitative = px.colors.qualitative.Plotly
    bar_color_map = {label: qualitative[i % len(qualitative)] for i, label in enumerate(labels)}

    _info_popover(
        "rri",
        "RRI measures relative improvement in mesh distance after adding the "
        "candidate point cloud: (d_before - d_after) / d_before. Higher is better. "
        "Scores are computed against the GT mesh with semidense points as baseline.",
    )
    st.plotly_chart(
        go.Figure(
            data=go.Bar(
                x=labels,
                y=rri.rri,
                marker_color=[bar_color_map[label] for label in labels],
            ),
            layout_title_text=_pretty_label("Oracle RRI per candidate"),
        ),
        width="stretch",
    )

    _info_popover(
        "pm dist",
        "Bidirectional point-mesh distance (Chamfer-style). "
        "Before uses semidense points only; after includes the candidate "
        "point cloud. Lower is better.",
    )
    baseline_pm_dist = float(rri.pm_dist_before[0].item())
    fig_pm_dist = go.Figure(
        data=[
            go.Bar(
                x=[baseline_label],
                y=[baseline_pm_dist],
                name="before (semi-dense, -1)",
                marker_color="lightgray",
            ),
            go.Bar(
                x=labels,
                y=rri.pm_dist_after,
                name="after",
                marker_color=[bar_color_map[label] for label in labels],
            ),
        ],
    )
    fig_pm_dist.update_layout(
        title_text=_pretty_label("Chamfer-like (bidirectional)"),
        xaxis={"categoryorder": "array", "categoryarray": [baseline_label, *labels]},
    )
    st.plotly_chart(fig_pm_dist, width="stretch")

    _info_popover(
        "pm acc",
        "Point-to-mesh accuracy: distance from reconstruction points to GT mesh. "
        "Captures how well points lie on the surface. Lower is better.",
    )
    baseline_pm_acc = float(rri.pm_acc_before[0].item())
    fig_pm_acc = go.Figure(
        data=[
            go.Bar(
                x=[baseline_label],
                y=[baseline_pm_acc],
                name="point→mesh (before, -1)",
                marker_color="lightgray",
            ),
            go.Bar(
                x=labels,
                y=rri.pm_acc_after,
                name="point→mesh (after)",
                marker_color=[bar_color_map[label] for label in labels],
            ),
        ],
    )
    fig_pm_acc.update_layout(
        title_text=_pretty_label("Point→Mesh (accuracy)"),
        xaxis={"categoryorder": "array", "categoryarray": [baseline_label, *labels]},
    )
    st.plotly_chart(fig_pm_acc, width="stretch")

    _info_popover(
        "pm comp",
        "Mesh-to-point completeness: distance from GT mesh to reconstruction points. "
        "Captures coverage of the surface. Lower is better.",
    )
    baseline_pm_comp = float(rri.pm_comp_before[0].item())
    fig_pm_comp = go.Figure(
        data=[
            go.Bar(
                x=[baseline_label],
                y=[baseline_pm_comp],
                name="mesh→point (before, -1)",
                marker_color="lightgray",
            ),
            go.Bar(
                x=labels,
                y=rri.pm_comp_after,
                name="mesh→point (after)",
                marker_color=[bar_color_map[label] for label in labels],
            ),
        ],
    )
    fig_pm_comp.update_layout(
        title_text=_pretty_label("Mesh→Point (completeness)"),
        xaxis={"categoryorder": "array", "categoryarray": [baseline_label, *labels]},
    )
    st.plotly_chart(fig_pm_comp, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        default_selection = candidate_ids[: min(6, len(candidate_ids))]
        selected_ids = st.multiselect(
            "Candidates to display",
            options=candidate_ids,
            default=default_selection,
            key="rri_cands",
        )
        cid_to_local = {int(cid): idx for idx, cid in enumerate(candidate_ids)}
        selected_local = [cid_to_local[cid] for cid in selected_ids if cid in cid_to_local]
    with col2:
        show_frusta = st.checkbox("Show frusta", value=True, key="rri_show_frusta")

    max_sem_pts = st.number_input(
        "Max semi-dense points",
        min_value=1000,
        max_value=200000,
        value=50000,
        step=1000,
        key="rri_max_sem_pts",
    )

    builder = (
        RenderingPlotBuilder.from_snippet(
            sample,
            title=_pretty_label("Mesh + Semi-dense + Candidate PCs"),
        )
        .add_mesh()
        .add_semidense(last_frame_only=False, max_points=max_sem_pts)
    )
    _info_popover(
        "rri scene",
        "3D overlay of GT mesh, semidense points, and selected candidate "
        "point clouds. This helps validate that high-RRI candidates add "
        "new surface coverage rather than noisy points.",
    )
    if show_frusta:
        builder.add_frusta_selection(
            poses=depth_batch.poses,
            camera=depth_batch.camera,
            max_frustums=min(16, len(selected_local)),
            candidate_indices=selected_local,
        )

    for idx_i, cid_int in enumerate(candidate_ids):
        if cid_int not in selected_ids:
            continue
        pts = pcs.points[idx_i, : int(pcs.lengths[idx_i].item())]
        builder.add_points(
            pts,
            name=f"Candidate {cid_int}",
            color=bar_color_map.get(
                str(cid_int),
                px.colors.qualitative.Plotly[idx_i % len(px.colors.qualitative.Plotly)],
            ),
            size=3,
            opacity=0.7,
        )

    st.plotly_chart(builder.finalize(), width="stretch")


__all__ = ["render_rri_page"]
