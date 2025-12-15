"""Page renderers for the refactored Streamlit app.

These functions are mostly UI/plotting code and intentionally contain no heavy
compute. Any expensive operations should be performed via
:class:`oracle_rri.app.controller.PipelineController`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from efm3d.aria.pose import PoseTW

from ..data import EfmSnippetView
from ..data.efm_views import EfmCameraView
from ..data.plotting import (
    SnippetPlotBuilder,
    collect_frame_modalities,
    plot_first_last_frames,
    project_pointcloud_on_frame,
)
from ..pose_generation import CandidateViewGeneratorConfig
from ..pose_generation.plotting import (
    CandidatePlotBuilder,
    _euler_histogram,
    plot_direction_marginals,
    plot_direction_polar,
    plot_direction_sphere,
    plot_euler_reference,
    plot_euler_world,
    plot_min_distance_to_mesh,
    plot_path_collision_segments,
    plot_position_polar,
    plot_position_sphere,
    plot_radius_hist,
    plot_rule_masks,
    plot_rule_rejection_bar,
)
from ..pose_generation.types import CandidateSamplingResult
from ..pose_generation.utils import stats_to_markdown_table, summarise_dirs_ref, summarise_offsets_ref
from ..rendering.candidate_depth_renderer import CandidateDepths
from ..rendering.candidate_pointclouds import CandidatePointClouds
from ..rendering.plotting import RenderingPlotBuilder, depth_grid, depth_histogram
from ..rri_metrics.types import RriResult


@dataclass(slots=True)
class Scene3DPlotOptions:
    """Plot options for the data page 3D scene view."""

    show_scene_bounds: bool
    show_crop_bounds: bool
    show_frustum: bool
    frustum_frame_indices: list[int]
    frustum_scale: float
    mark_first_last: bool
    show_gt_obbs: bool
    gt_timestamp: int | None
    semidense_mode: str
    max_sem_points: int


def pose_world_cam(sample: EfmSnippetView, cam_view: EfmCameraView, frame_idx: int) -> tuple[PoseTW, object]:
    cam_ts = cam_view.time_ns.cpu().numpy()
    traj_ts = sample.trajectory.time_ns.cpu().numpy()
    traj_idx = int(np.argmin(np.abs(traj_ts - cam_ts[frame_idx])))

    t_world_rig = sample.trajectory.t_world_rig[traj_idx]
    t_cam_rig = cam_view.calib.T_camera_rig[frame_idx]
    return t_world_rig @ t_cam_rig.inverse(), cam_view.calib[frame_idx]


def semidense_points_for_frame(sample: EfmSnippetView, frame_idx: int | None, *, all_frames: bool) -> torch.Tensor:
    sem = sample.semidense
    if sem is None or sem.points_world.numel() == 0:
        return torch.zeros((0, 3), dtype=torch.float32)

    pts = sem.points_world
    lengths = sem.lengths
    if all_frames:
        if lengths is not None:
            max_len = pts.shape[1]
            mask_valid = torch.arange(max_len, device=pts.device).unsqueeze(0) < lengths.clamp_max(max_len).unsqueeze(
                -1
            )
            pts = torch.where(mask_valid.unsqueeze(-1), pts, torch.nan)
        pts = pts.reshape(-1, 3)
    else:
        if frame_idx is None:
            frame_idx = int(torch.argmax(lengths).item()) if lengths.numel() else 0
        frame_idx = max(0, min(int(frame_idx), pts.shape[0] - 1))
        n_valid = int(lengths[frame_idx].item()) if lengths is not None else pts.shape[1]
        pts = pts[frame_idx, :n_valid]

    finite = torch.isfinite(pts).all(dim=-1)
    return pts[finite]


def scene_plot_options_ui(sample: EfmSnippetView, *, key_prefix: str = "data_scene") -> tuple[str, Scene3DPlotOptions]:
    """Render UI controls for the data-page 3D plot.

    Args:
        sample: Current snippet sample used to derive available frame indices and GT timestamps.
        key_prefix: Widget key prefix to avoid collisions across pages.

    Returns:
        Tuple of ``(frustum_camera, options)``.
    """

    st.subheader("3D scene view")

    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        frustum_camera = st.radio(
            "Frustum camera",
            options=["rgb", "slam-l", "slam-r"],
            horizontal=True,
            index=0,
            key=f"{key_prefix}_frustum_cam",
        )
        num_frames = int(sample.camera_rgb.images.shape[0])
        frustum_frame_indices = st.multiselect(
            "Frustum frame indices",
            options=list(range(num_frames)),
            default=[0],
            key=f"{key_prefix}_frustum_idx",
        )
        frustum_scale = st.slider(
            "Frustum scale",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.05,
            key=f"{key_prefix}_frustum_scale",
        )

    with opt_col2:
        semidense_mode = st.radio(
            "Semi-dense points",
            options=["off", "all frames", "last frame only"],
            horizontal=True,
            index=1,
            key=f"{key_prefix}_sem_mode",
        )
        max_sem_points = st.slider(
            "Max semi-dense points",
            min_value=1000,
            max_value=200000,
            value=20000,
            step=1000,
            key=f"{key_prefix}_max_sem_points",
        )
        show_scene_bounds = st.checkbox("Show scene bounds", value=True, key=f"{key_prefix}_scene_bounds")
        show_crop_bounds = st.checkbox("Show crop bbox", value=True, key=f"{key_prefix}_crop_bounds")
        show_frustum = st.checkbox("Show frustum", value=True, key=f"{key_prefix}_frustum_enable")
        mark_first_last = st.checkbox("Mark start/finish", value=True, key=f"{key_prefix}_mark_first_last")
        show_gt_obbs = st.checkbox("Show GT OBBs", value=False, key=f"{key_prefix}_gt_obbs")

    gt_ts = None
    if show_gt_obbs and sample.gt.timestamps:
        gt_ts = st.selectbox("GT OBB timestamp", options=sample.gt.timestamps, index=0, key=f"{key_prefix}_gt_ts")

    return (
        frustum_camera,
        Scene3DPlotOptions(
            show_scene_bounds=show_scene_bounds,
            show_crop_bounds=show_crop_bounds,
            show_frustum=show_frustum,
            frustum_frame_indices=[int(i) for i in frustum_frame_indices] if frustum_frame_indices else [0],
            frustum_scale=float(frustum_scale),
            mark_first_last=mark_first_last,
            show_gt_obbs=show_gt_obbs,
            gt_timestamp=gt_ts,
            semidense_mode=str(semidense_mode),
            max_sem_points=int(max_sem_points),
        ),
    )


def rejected_pose_tensor(candidates: CandidateSamplingResult) -> torch.Tensor | None:
    mask_valid = candidates.mask_valid
    shell_poses = candidates.shell_poses
    if mask_valid is None or shell_poses is None or mask_valid.numel() == 0:
        return None
    shell_tensor = shell_poses.tensor() if hasattr(shell_poses, "tensor") else shell_poses
    if mask_valid.shape[0] != shell_tensor.shape[0]:
        return None
    rejected_mask = ~mask_valid
    if not rejected_mask.any():
        return None
    return shell_tensor[rejected_mask]


def render_data_page(sample: EfmSnippetView, *, crop_margin: float | None = None) -> None:
    st.header("Data")
    st.write(f"Scene: **{sample.scene_id}**, snippet: **{sample.snippet_id}**")

    first_pose = sample.trajectory.t_world_rig[0]
    last_pose = sample.trajectory.t_world_rig[-1]

    def _pose_summary(pose: torch.Tensor | PoseTW):
        pt = (
            pose
            if isinstance(pose, PoseTW)
            else PoseTW.from_matrix3x4(pose.view(3, 4) if pose.shape[-1] == 12 else pose)
        )
        r, p, y = pt.to_ypr(rad=True)
        rpy = torch.rad2deg(torch.stack([r, p, y])).tolist()
        euler = torch.rad2deg(pt.to_euler(rad=True)).tolist()
        t = pt.t.tolist()
        return t, rpy, euler

    t_first, rpy_first, eul_first = _pose_summary(first_pose)
    t_last, rpy_last, eul_last = _pose_summary(last_pose)
    st.markdown(
        "- **First frame**: "
        f"t=({t_first[0]:.3f},{t_first[1]:.3f},{t_first[2]:.3f}) m, "
        f"RPY=({rpy_first[0]:.2f},{rpy_first[1]:.2f},{rpy_first[2]:.2f})°, "
        f"Euler ZYX=({eul_first[0]:.2f},{eul_first[1]:.2f},{eul_first[2]:.2f})°"
    )
    st.markdown(
        "- **Last frame**: "
        f"t=({t_last[0]:.3f},{t_last[1]:.3f},{t_last[2]:.3f}) m, "
        f"RPY=({rpy_last[0]:.2f},{rpy_last[1]:.2f},{rpy_last[2]:.2f})°, "
        f"Euler ZYX=({eul_last[0]:.2f},{eul_last[1]:.2f},{eul_last[2]:.2f})°"
    )

    modalities, missing = collect_frame_modalities(sample, include_depth=True)
    st.plotly_chart(plot_first_last_frames(sample), width="stretch")

    missing_depth = [m for m in missing if m.startswith("Depth")]
    if modalities and not missing_depth:
        st.caption("Depth maps available for rendering and overlays.")
    elif missing_depth:
        st.info(f"Depth maps missing for: {', '.join(missing_depth)}")

    st.subheader("Point cloud overlay")
    overlay_cam = st.selectbox("Overlay camera", ["rgb", "slam-l", "slam-r"], index=0, key="overlay_cam")
    overlay_frame = st.slider("Frame index for overlay", 0, int(sample.camera_rgb.images.shape[0] - 1), 0)
    overlay_source = st.selectbox(
        "Point cloud source",
        ["Semidense (all frames)", "Semidense (selected frame)", "Semidense (last with points)"],
        index=0,
    )

    if overlay_source == "Semidense (all frames)":
        points_world = semidense_points_for_frame(sample, None, all_frames=True)
    elif overlay_source == "Semidense (selected frame)":
        points_world = semidense_points_for_frame(sample, overlay_frame, all_frames=False)
    else:
        lengths = sample.semidense.lengths if sample.semidense is not None else None
        last_idx = (
            int(torch.nonzero(lengths > 0, as_tuple=False).max().item())
            if lengths is not None and torch.any(lengths > 0)
            else 0
        )
        points_world = semidense_points_for_frame(sample, last_idx, all_frames=False)

    if points_world is None or points_world.numel() == 0:
        st.info("No points available for overlay with current selection.")
    else:
        cam_view = sample.get_camera(overlay_cam.replace("-", ""))
        pose_wc, cam_tw = pose_world_cam(sample, cam_view, overlay_frame)
        fig_overlay = project_pointcloud_on_frame(
            img=cam_view.images[overlay_frame],
            cam=cam_tw,
            pose_world_cam=pose_wc,
            points_world=points_world,
            title=f"Overlay on {overlay_cam.upper()} frame {overlay_frame} ({points_world.shape[0]} pts)",
            max_points=20000,
        )
        st.plotly_chart(fig_overlay, width="stretch")

    cam_choice, plot_opts = scene_plot_options_ui(sample, key_prefix="data_scene")

    builder = (
        SnippetPlotBuilder.from_snippet(sample, title="Mesh + semidense + trajectory + camera frustum")
        .add_mesh()
        .add_trajectory(mark_first_last=plot_opts.mark_first_last, show=True)
    )

    if plot_opts.semidense_mode != "off":
        builder.add_semidense(
            max_points=plot_opts.max_sem_points,
            last_frame_only=(plot_opts.semidense_mode == "last frame only"),
        )

    if plot_opts.show_frustum:
        builder.add_frusta(
            camera=cam_choice,
            frame_indices=plot_opts.frustum_frame_indices,
            scale=plot_opts.frustum_scale,
            include_axes=True,
            include_center=True,
        )

    if plot_opts.show_scene_bounds:
        builder.add_bounds_box(name="Scene bounds", color="gray", dash="dash", width=2)

    if plot_opts.show_gt_obbs:
        builder.add_gt_obbs(camera=cam_choice, timestamp=plot_opts.gt_timestamp)

    if plot_opts.show_crop_bounds:
        crop_aabb = tuple(b.detach().cpu().numpy() for b in sample.crop_bounds)
        builder.add_bounds_box(name="Crop bounds", color="orange", dash="solid", width=3, aabb=crop_aabb)

    st.plotly_chart(builder.finalize(), width="stretch")


def render_candidates_page(
    sample: EfmSnippetView,
    candidates: CandidateSamplingResult,
    cand_cfg: CandidateViewGeneratorConfig,
) -> None:
    shell_poses = candidates.shell_poses
    mask_valid = candidates.mask_valid

    st.header("Candidate Poses")

    tab_pos, tab_frusta = st.tabs(["Positions (3D)", "Frusta (3D)"])

    with tab_pos:
        cand_fig = (
            CandidatePlotBuilder.from_candidates(
                sample, candidates, title=f"Candidate positions ({cand_cfg.camera_label})"
            )
            .add_mesh()
            .add_candidate_cloud(use_valid=True, color="royalblue", size=4, opacity=0.7)
            .add_reference_axes()
        ).finalize()
        st.plotly_chart(cand_fig, width="stretch")

    offsets_ref, dirs_ref = candidates.get_offsets_and_dirs_ref(display_rotate=False)

    with tab_frusta:
        if mask_valid is None or mask_valid.sum() == 0:
            st.warning("All candidates were rejected; frustum plot omitted.")
        else:
            opt_col1, opt_col2 = st.columns(2)
            with opt_col1:
                frustum_scale = st.slider(
                    "Frustum scale",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key="cand_frustum_scale",
                )
            with opt_col2:
                max_frustums = st.slider(
                    "Max frustums",
                    min_value=1,
                    max_value=24,
                    value=6,
                    step=1,
                    key="cand_max_frustums",
                )

            frust_fig = (
                CandidatePlotBuilder.from_candidates(
                    sample, candidates, title=f"Candidate frusta ({cand_cfg.camera_label})"
                )
                .add_mesh()
                .add_candidate_cloud(use_valid=True, color="royalblue", size=3, opacity=0.35)
                .add_candidate_frusta(
                    scale=float(frustum_scale),
                    color="crimson",
                    name="Frustum",
                    max_frustums=int(max_frustums),
                    include_axes=False,
                    include_center=False,
                )
                .add_reference_axes()
            ).finalize()
            st.plotly_chart(frust_fig, width="stretch")

    with st.expander("Distributions & Diagnostics", expanded=False):
        fixed_ranges = st.checkbox("Clamp axes to standard ranges", value=True, key="cand_angles_fixed_ranges")
        diag_offsets, diag_dirs, diag_rules, diag_rejected = st.tabs(["Offsets", "Directions", "Rules", "Rejected"])

        with diag_offsets:
            st.markdown(stats_to_markdown_table(summarise_offsets_ref(offsets_ref), header=None))
            offsets_np = offsets_ref.cpu().numpy()
            colp1, colp2 = st.columns(2)
            with colp1:
                st.plotly_chart(
                    plot_position_polar(
                        offsets_np, title="Offsets from reference pose (az/elev)", fixed_ranges=fixed_ranges
                    ),
                    width="stretch",
                )
            with colp2:
                st.plotly_chart(plot_position_sphere(offsets_np, show_axes=True), width="stretch")
            st.plotly_chart(plot_radius_hist(offsets_np), width="stretch")

        with diag_dirs:
            st.markdown(stats_to_markdown_table(summarise_dirs_ref(dirs_ref), header=None))
            dirs_np = dirs_ref.cpu().numpy()

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    plot_direction_polar(dirs_np, title="View directions (reference frame)", fixed_ranges=fixed_ranges),
                    width="stretch",
                )
            with col2:
                st.plotly_chart(
                    plot_direction_sphere(dirs_np, title="View dirs on unit sphere", show_axes=True),
                    width="stretch",
                )
            st.plotly_chart(
                plot_direction_marginals(torch.as_tensor(dirs_np), fixed_ranges=fixed_ranges), width="stretch"
            )

            yaw, pitch, roll = candidates.reference_pose.to_euler(rad=False).cpu().numpy().tolist()
            st.markdown(f"Reference Pose: Y={yaw:.2f}, P={pitch:.2f}, R={roll:.2f}  (ZYX Euler, cw90-adjusted)")
            st.plotly_chart(plot_euler_world(candidates, fixed_ranges=fixed_ranges), width="stretch")
            st.plotly_chart(plot_euler_reference(candidates, fixed_ranges=fixed_ranges), width="stretch")
            delta = candidates.extras.get("view_dirs_delta") if hasattr(candidates, "extras") else None
            if delta is not None:
                yaw_d, pitch_d, roll_d = [rad.rad2deg().cpu() for rad in delta.to_ypr(rad=True)]
                st.markdown(
                    stats_to_markdown_table(
                        {
                            "yaw_delta_deg": {
                                "min": float(yaw_d.min()),
                                "max": float(yaw_d.max()),
                                "mean": float(yaw_d.mean()),
                                "std": float(yaw_d.std(unbiased=False)),
                            },
                            "pitch_delta_deg": {
                                "min": float(pitch_d.min()),
                                "max": float(pitch_d.max()),
                                "mean": float(pitch_d.mean()),
                                "std": float(pitch_d.std(unbiased=False)),
                            },
                            "roll_delta_deg": {
                                "min": float(roll_d.min()),
                                "max": float(roll_d.max()),
                                "mean": float(roll_d.mean()),
                                "std": float(roll_d.std(unbiased=False)),
                            },
                        },
                        header="Orientation jitter stats (delta)",
                    )
                )
                st.plotly_chart(
                    _euler_histogram(
                        yaw_d,
                        pitch_d,
                        roll_d,
                        bins=90,
                        title="Orientation jitter (delta, deg)",
                        fixed_ranges=fixed_ranges,
                    ),
                    width="stretch",
                )

        with diag_rules:
            masks = candidates.masks
            if isinstance(masks, dict) and len(masks) > 0 and shell_poses is not None:
                masks_tensor = torch.stack(list(masks.values()))
                mask_fig = plot_rule_masks(
                    snippet=sample,
                    shell_poses=shell_poses.tensor() if hasattr(shell_poses, "tensor") else shell_poses,
                    masks=masks_tensor,
                    rule_names=list(masks.keys()),
                )
                st.plotly_chart(mask_fig, width="stretch")

            extras = candidates.extras if hasattr(candidates, "extras") else {}
            dist_min = extras.get("min_distance_to_mesh")
            path_collide = extras.get("path_collision_mask")

            if dist_min is not None:
                st.plotly_chart(
                    plot_min_distance_to_mesh(snippet=sample, candidates=candidates, distances=dist_min),
                    width="stretch",
                )
            if path_collide is not None:
                st.plotly_chart(
                    plot_path_collision_segments(snippet=sample, candidates=candidates, collision_mask=path_collide),
                    width="stretch",
                )

            st.plotly_chart(plot_rule_rejection_bar(candidates), width="stretch")

        with diag_rejected:
            plot_rejected_only = st.checkbox(
                "Plot rejected poses only (if any)", value=False, key="cand_plot_rejected_only"
            )
            if plot_rejected_only:
                rejected_poses = rejected_pose_tensor(candidates)
                if rejected_poses is None:
                    st.info("No rejected poses to plot; all sampled candidates survived rule filtering.")
                else:
                    rej_fig = (
                        CandidatePlotBuilder.from_candidates(
                            sample, candidates, title=f"Rejected candidate positions ({rejected_poses.shape[0]})"
                        )
                        .add_mesh()
                        .add_rejected_cloud()
                        .add_reference_axes()
                    ).finalize()
                    st.plotly_chart(rej_fig, width="stretch")


def render_depth_page(
    sample: EfmSnippetView | None,
    depth_batch: CandidateDepths,
    *,
    pcs: CandidatePointClouds | None,
) -> None:
    st.header("Candidate Renders")

    depths = depth_batch.depths
    indices = depth_batch.candidate_indices.tolist()
    titles = [f"cand {i} (id {cid})" for i, cid in enumerate(indices)]
    st.caption(
        "Local indices (cand 0..N-1) refer to the rendered batch order; "
        "`id` is the original candidate index (pre-render filtering)."
    )

    cam = depth_batch.camera
    if hasattr(cam, "valid_radius") and cam.valid_radius.numel() > 0:
        zfar_stat = float(cam.valid_radius.max().item())
    else:
        zfar_stat = float(depths.max().item()) * 1.05

    st.subheader("Depth grid")
    fig = depth_grid(depths, titles=titles, zmax=float(depths.max().item()))
    st.plotly_chart(fig, width="stretch")

    with st.expander("Diagnostics", expanded=False):
        tab_hist, tab_hits = st.tabs(["Histograms", "Depth-hit point cloud (3D)"])

        with tab_hist:
            bins = st.slider("Histogram bins", 10, 200, 50, step=10, key="depth_hist_bins")
            fig_hist = depth_histogram(depths, titles=titles, bins=int(bins), zfar=zfar_stat)
            st.plotly_chart(fig_hist, width="stretch")

        with tab_hits:
            if sample is None:
                st.info("Load data first to back-project depth hits.")
                return
            if pcs is None:
                st.info("Run / refresh renders to compute backprojected CandidatePointClouds.")
                return

            max_points = st.number_input(
                "Max points to display",
                min_value=1,
                max_value=200000,
                value=20000,
                step=1000,
                key="depth_hit_max_points",
            )

            cand_options = depth_batch.candidate_indices.tolist()
            selected_global = st.multiselect(
                "Select candidates to back-project",
                options=cand_options,
                default=cand_options,
                key="depth_hit_cands",
            )
            cand_to_local = {int(g): idx for idx, g in enumerate(depth_batch.candidate_indices.tolist())}
            selected = [cand_to_local[g] for g in selected_global if g in cand_to_local]
            num_frustums = int(depth_batch.poses.tensor().shape[0])

            points_selected = []
            for idx in selected:
                n_valid = int(pcs.lengths[idx].item())
                if n_valid == 0:
                    continue
                pts = pcs.points[idx, :n_valid]
                points_selected.append(pts)

            if points_selected:
                pts_cat = torch.cat(points_selected, dim=0)
                if pts_cat.shape[0] > max_points:
                    rand_idx = torch.randperm(pts_cat.shape[0], device=pts_cat.device)[: int(max_points)]
                    pts_cat = pts_cat[rand_idx]

                builder = (
                    RenderingPlotBuilder.from_snippet(sample, title="Depth hit back-projection")
                    .add_mesh()
                    .add_points(pts_cat, name="Depth hits", color="teal", size=3, opacity=0.8)
                    .add_frusta_selection(
                        poses=depth_batch.poses,
                        camera=depth_batch.camera,
                        max_frustums=min(16, num_frustums),
                        candidate_indices=selected,
                    )
                )
                st.plotly_chart(builder.finalize(), width="stretch")
            else:
                st.info("No valid depth hits to display for the selected candidates.")


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

    qualitative = px.colors.qualitative.Plotly
    bar_color_map = {label: qualitative[i % len(qualitative)] for i, label in enumerate(labels)}

    st.plotly_chart(
        go.Figure(
            data=go.Bar(x=labels, y=rri.rri, marker_color=[bar_color_map[label] for label in labels]),
            layout_title_text="Oracle RRI per candidate",
        ),
        width="stretch",
    )

    st.plotly_chart(
        go.Figure(
            data=[
                go.Bar(x=labels, y=rri.pm_dist_before, name="before", marker_color="lightgray"),
                go.Bar(
                    x=labels,
                    y=rri.pm_dist_after,
                    name="after",
                    marker_color=[bar_color_map[label] for label in labels],
                ),
            ],
            layout_title_text="Chamfer-like (bidirectional)",
        ),
        width="stretch",
    )

    st.plotly_chart(
        go.Figure(
            data=[
                go.Bar(x=labels, y=rri.pm_acc_before, name="point→mesh (before)", marker_color="lightgray"),
                go.Bar(
                    x=labels,
                    y=rri.pm_acc_after,
                    name="point→mesh (after)",
                    marker_color=[bar_color_map[label] for label in labels],
                ),
            ],
            layout_title_text="Point→Mesh (accuracy)",
        ),
        width="stretch",
    )

    st.plotly_chart(
        go.Figure(
            data=[
                go.Bar(x=labels, y=rri.pm_comp_before, name="mesh→point (before)", marker_color="lightgray"),
                go.Bar(
                    x=labels,
                    y=rri.pm_comp_after,
                    name="mesh→point (after)",
                    marker_color=[bar_color_map[label] for label in labels],
                ),
            ],
            layout_title_text="Mesh→Point (completeness)",
        ),
        width="stretch",
    )

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
        RenderingPlotBuilder.from_snippet(sample, title="Mesh + Semi-dense + Candidate PCs")
        .add_mesh()
        .add_semidense(last_frame_only=False, max_points=max_sem_pts)
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
                str(cid_int), px.colors.qualitative.Plotly[idx_i % len(px.colors.qualitative.Plotly)]
            ),
            size=3,
            opacity=0.7,
        )

    st.plotly_chart(builder.finalize(), width="stretch")


__all__ = [
    "render_candidates_page",
    "render_data_page",
    "render_depth_page",
    "render_rri_page",
]
