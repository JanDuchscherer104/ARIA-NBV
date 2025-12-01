"""Page renderers and plotting helpers for the Streamlit dashboard."""

from __future__ import annotations

import numpy as np
import streamlit as st
import torch
from efm3d.aria.pose import PoseTW
from plotly import graph_objects as go

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
    candidate_offsets_and_dirs_ref,
    plot_direction_marginals,
    plot_direction_polar,
    plot_direction_sphere,
    plot_min_distance_to_mesh,
    plot_path_collision_segments,
    plot_position_polar,
    plot_position_sphere,
    plot_radius_hist,
    plot_rule_masks,
    plot_rule_rejection_bar,
)
from ..pose_generation.types import CandidateSamplingResult
from ..rendering.candidate_depth_renderer import CandidateDepths
from ..rendering.plotting import RenderingPlotBuilder, depth_grid, depth_histogram, hit_ratio_bar
from .state import STATE_KEYS, get


def pose_world_cam(sample: EfmSnippetView, cam_view: EfmCameraView, frame_idx: int):
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


def plot_options_from_ui(sample: EfmSnippetView) -> dict:
    st.sidebar.subheader("Plot options")
    layer_choices = st.sidebar.multiselect(
        "Layers",
        options=[
            "Semidense",
            "Semidense (last frame only)",
            "Scene bounds",
            "Crop bbox",
            "Frustum",
            "Mark start/finish",
            "GT OBBs",
        ],
        default=["Semidense", "Scene bounds", "Frustum", "Mark start/finish"],
        key="plot_layers",
    )
    max_sem = st.sidebar.slider("Max semidense points", 1000, 50000, 20000, step=1000, key="max_sem_points")
    num_frames = int(sample.camera_rgb.images.shape[0])
    frustum_idx = st.sidebar.multiselect(
        "Frustum frame indices",
        options=list(range(num_frames)),
        default=[0],
        key="frustum_idx",
    )
    gt_ts = None
    if "GT OBBs" in layer_choices and sample.gt.timestamps:
        gt_ts = st.sidebar.selectbox("GT OBB timestamp", options=sample.gt.timestamps, index=0)

    return {
        "show_semidense": "Semidense" in layer_choices,
        "max_sem_points": max_sem,
        "pc_from_last_only": "Semidense (last frame only)" in layer_choices,
        "show_scene_bounds": "Scene bounds" in layer_choices,
        "show_crop_bounds": "Crop bbox" in layer_choices,
        "show_frustum": "Frustum" in layer_choices,
        "frustum_scale": 1.0,
        "frustum_frame_indices": frustum_idx if len(frustum_idx) > 0 else [0],
        "mark_first_last": "Mark start/finish" in layer_choices,
        "show_gt_obbs": "GT OBBs" in layer_choices,
        "gt_timestamp": gt_ts,
    }


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

    cam_choice = st.sidebar.selectbox("Camera for frustum", ["rgb", "slam-l", "slam-r"], index=0, key="frustum_cam")
    plot_opts = plot_options_from_ui(sample)
    crop_bounds = None

    builder = (
        SnippetPlotBuilder.from_snippet(sample, title="Mesh + semidense + trajectory + camera frustum")
        .add_mesh()
        .add_trajectory(mark_first_last=plot_opts["mark_first_last"], show=True)
    )

    if plot_opts["show_semidense"]:
        builder.add_semidense(max_points=plot_opts["max_sem_points"], last_frame_only=plot_opts["pc_from_last_only"])

    if plot_opts["show_frustum"]:
        builder.add_frusta(
            camera=cam_choice,
            frame_indices=plot_opts["frustum_frame_indices"],
            scale=plot_opts["frustum_scale"],
            include_axes=True,
            include_center=True,
        )

    if plot_opts["show_scene_bounds"]:
        builder.add_bounds_box(name="Scene bounds", color="gray", dash="dash", width=2)

    if plot_opts["show_gt_obbs"]:
        builder.add_gt_obbs(camera=cam_choice, timestamp=plot_opts["gt_timestamp"])

    if crop_bounds is not None:
        builder.add_bounds_box(name="Crop bounds", color="orange", dash="solid", width=3, aabb=crop_bounds)

    st.plotly_chart(builder.finalize(), width="stretch")


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


def render_candidates_page(
    sample: EfmSnippetView,
    candidates: CandidateSamplingResult,
    cand_cfg: CandidateViewGeneratorConfig,
    frustum_scale: float,
    max_frustums: int,
    plot_rejected_only: bool,
) -> None:
    shell_poses = candidates.shell_poses
    mask_valid = candidates.mask_valid

    st.header("Candidate Poses")
    with st.status("Building candidate plots...", expanded=False):
        cand_fig = (
            CandidatePlotBuilder.from_candidates(
                sample, candidates, title=f"Candidate positions ({cand_cfg.camera_label})"
            )
            .add_mesh()
            .add_candidate_cloud(use_valid=True, color="royalblue", size=4, opacity=0.7)
            .add_reference_axes()
        ).finalize()
    st.plotly_chart(cand_fig, width="stretch")

    offsets_ref, dirs_ref = candidate_offsets_and_dirs_ref(candidates)

    with st.expander("Sampling distributions", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                plot_direction_polar(dirs_ref, title="View directions (reference frame)"),
                width="stretch",
            )
        with col2:
            st.plotly_chart(
                plot_direction_sphere(dirs_ref, title="View dirs on unit sphere", show_axes=True),
                width="stretch",
            )
        st.plotly_chart(plot_direction_marginals(dirs_ref), width="stretch")

    with st.expander("Position distributions (reference frame)", expanded=False):
        colp1, colp2 = st.columns(2)
        with colp1:
            st.plotly_chart(
                plot_position_polar(offsets_ref, title="Offsets from reference pose (az/elev)"),
                width="stretch",
            )
        with colp2:
            st.plotly_chart(
                plot_position_sphere(offsets_ref, title="Offsets on unit sphere", show_axes=True),
                width="stretch",
            )
        st.plotly_chart(plot_radius_hist(offsets_ref), width="stretch")

    # Rule masks overlay
    masks = candidates.masks
    if isinstance(masks, dict) and len(masks) > 0:
        with st.expander("Rule-wise pruning", expanded=False):
            masks_tensor = torch.stack(list(masks.values()))
            mask_fig = plot_rule_masks(
                snippet=sample,
                shell_poses=shell_poses.tensor() if hasattr(shell_poses, "tensor") else shell_poses,
                masks=masks_tensor,
                rule_names=list(masks.keys()),
            )
            st.plotly_chart(mask_fig, width="stretch")
    # Rule statistics
    extras = candidates.extras if hasattr(candidates, "extras") else {}
    dist_min = extras.get("min_distance_to_mesh")
    path_collide = extras.get("path_collision_mask")

    if dist_min is not None:
        with st.expander("MinDistanceToMesh stats", expanded=False):
            st.plotly_chart(
                plot_min_distance_to_mesh(snippet=sample, candidates=candidates, distances=dist_min),
                width="stretch",
            )
    if path_collide is not None:
        with st.expander("PathCollisionRule segments", expanded=False):
            st.plotly_chart(
                plot_path_collision_segments(snippet=sample, candidates=candidates, collision_mask=path_collide),
                width="stretch",
            )
    with st.expander("Rejections per rule", expanded=False):
        st.plotly_chart(plot_rule_rejection_bar(candidates), width="stretch")

    if mask_valid is None or mask_valid.sum() == 0:
        st.warning("All candidates were rejected; frustum plot omitted.")
    else:
        with st.status("Candidate Frusta", expanded=False):
            frust_fig = (
                CandidatePlotBuilder.from_candidates(
                    sample, candidates, title=f"Candidate frusta ({cand_cfg.camera_label})"
                )
                .add_mesh()
                .add_candidate_cloud(use_valid=True, color="royalblue", size=3, opacity=0.35)
                .add_candidate_frusta(
                    scale=frustum_scale,
                    color="crimson",
                    name="Frustum",
                    max_frustums=max_frustums,
                    include_axes=False,
                    include_center=False,
                )
                .add_reference_axes()
            ).finalize()
            st.plotly_chart(frust_fig, width="stretch")

    rejected_poses = rejected_pose_tensor(candidates)
    if plot_rejected_only:
        if rejected_poses is None:
            st.info("No rejected poses to plot; all sampled candidates survived rule filtering.")
        else:
            with st.status("Rendering rejected candidates plot...", expanded=False):
                rej_fig = (
                    CandidatePlotBuilder.from_candidates(
                        sample, candidates, title=f"Rejected candidate positions ({rejected_poses.shape[0]})"
                    )
                    .add_mesh()
                    .add_rejected_cloud()
                    .add_reference_axes()
                ).finalize()
            st.plotly_chart(rej_fig, width="stretch")


def render_depth_page(depth_batch: CandidateDepths) -> None:
    st.header("Candidate Renders")

    depths = depth_batch.depths
    indices = depth_batch.candidate_indices.tolist()
    titles = [f"Candidate {i}" for i in indices]
    # Best-effort zfar for stats; fall back to depth max if missing.
    cam = depth_batch.camera
    if hasattr(cam, "valid_radius") and cam.valid_radius.numel() > 0:
        zfar_stat = float(cam.valid_radius.max().item())
    else:
        zfar_stat = float(depths.max().item()) * 1.05

    with st.expander("Depth grid", expanded=True):
        with st.status("Building depth grid...", expanded=False):
            fig = depth_grid(depths, titles=titles, zmax=float(depths.max().item()))
        st.plotly_chart(fig, width="stretch")

    with st.expander("Depth histograms", expanded=False):
        fig_hist = depth_histogram(depths, bins=50, zfar=zfar_stat)
        st.plotly_chart(fig_hist, width="stretch")

    with st.expander("Hit-ratio bars", expanded=False):
        fig_hits = hit_ratio_bar(depths, zfar=zfar_stat)
        st.plotly_chart(fig_hits, width="stretch")

    with st.expander("Valid-pixel counts (ranked)", expanded=False):
        counts = depth_batch.valid_counts.detach().cpu()
        fig_counts = go.Figure(
            go.Bar(
                x=list(range(counts.numel())),
                y=counts.tolist(),
                text=[str(int(c)) for c in counts],
                textposition="auto",
            )
        )
        fig_counts.update_layout(
            title="Valid depth pixels per candidate (after ranking)",
            yaxis_title="valid pixels",
            xaxis_title="candidate rank",
            height=280,
        )
        st.plotly_chart(fig_counts, width="stretch")

    sample = get(STATE_KEYS["sample"])
    with st.expander("Frusta + image planes (3D)", expanded=False):
        if sample is None:
            st.info("Load data first to plot frusta.")
        else:
            cand_options_frusta = depth_batch.candidate_indices.tolist()
            selected_frusta_global = st.multiselect(
                "Select candidates to display (frusta)",
                options=cand_options_frusta,
                default=cand_options_frusta,
                key="frusta_cands",
            )
            # map global candidate ids to local batch indices
            cand_to_local = {int(g): idx for idx, g in enumerate(depth_batch.candidate_indices.tolist())}
            selected_frusta = [cand_to_local[g] for g in selected_frusta_global if g in cand_to_local]
            plane_dist = st.slider("Image plane distance (m)", 0.2, 3.0, 1.0, step=0.1, key="plane_dist_slider")
            builder = RenderingPlotBuilder.from_snippet(sample, title="Rendered frusta with image planes").add_mesh()
            num_frustums = int(depth_batch.poses.tensor().shape[0])
            builder.add_frusta_with_image_plane(
                poses=depth_batch.poses,
                camera=depth_batch.camera,
                plane_dist=float(plane_dist),
                max_frustums=min(16, num_frustums),
                candidate_indices=selected_frusta,
            )
            st.plotly_chart(builder.finalize(), width="stretch")

    with st.expander("Depth hit point cloud (3D)", expanded=False):
        if sample is None:
            st.info("Load data first to back-project depth hits.")
        else:
            stride = st.slider("Depth hit stride", 1, 32, 8, step=1, key="depth_hit_stride")
            builder_hits = RenderingPlotBuilder.from_snippet(sample, title="Depth hit back-projection").add_mesh()
            cand_options = depth_batch.candidate_indices.tolist()
            selected_global = st.multiselect(
                "Select candidates to back-project", options=cand_options, default=cand_options, key="depth_hit_cands"
            )
            cand_to_local = {int(g): idx for idx, g in enumerate(depth_batch.candidate_indices.tolist())}
            selected = [cand_to_local[g] for g in selected_global if g in cand_to_local]
            builder_hits.add_depth_hits(
                depths=depth_batch.depths,
                poses=depth_batch.poses,
                camera=depth_batch.p3d_cameras,
                stride=int(stride),
                zfar=float(depth_batch.depths.max().item()),
                max_points=20000,
                candidate_indices=selected,
            )
            st.plotly_chart(builder_hits.finalize(), width="stretch")
