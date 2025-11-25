"""Page renderers and plotting helpers for the Streamlit dashboard."""

from __future__ import annotations

import numpy as np
import streamlit as st
import torch
from efm3d.aria.pose import PoseTW

from ...data import EfmSnippetView
from ...data.efm_views import EfmCameraView
from ...data.plotting import (
    SnippetPlotBuilder,
    collect_frame_modalities,
    crop_aabb_from_semidense,
    plot_first_last_frames,
    project_pointcloud_on_frame,
)
from ...pose_generation import CandidateViewGeneratorConfig
from ...pose_generation.plotting import plot_candidate_frusta, plot_candidates, plot_sampling_shell
from ...pose_generation.types import CandidateSamplingResult
from ...rendering.candidate_depth_renderer import CandidateDepthBatch
from ...rendering.plotting import depth_grid


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
            mask_valid = torch.arange(max_len, device=pts.device).unsqueeze(0) < lengths.clamp_max(max_len).unsqueeze(-1)
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
        pt = pose if isinstance(pose, PoseTW) else PoseTW.from_matrix3x4(pose.view(3, 4) if pose.shape[-1] == 12 else pose)
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
        last_idx = int(torch.nonzero(lengths > 0, as_tuple=False).max().item()) if lengths is not None and torch.any(lengths > 0) else 0
        points_world = semidense_points_for_frame(sample, last_idx, all_frames=False)

    if points_world is None or points_world.numel() == 0:
        st.info("No points available for overlay with current selection.")
    else:
        cam_map = {"rgb": sample.camera_rgb, "slam-l": sample.camera_slam_left, "slam-r": sample.camera_slam_right}
        cam_view = cam_map[overlay_cam]
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
    if plot_opts["show_crop_bounds"] and crop_margin is not None:
        crop_bounds = crop_aabb_from_semidense(sample, margin=float(crop_margin))
        if crop_bounds is None and sample.mesh is not None:
            mesh_min, mesh_max = sample.mesh.bounds
            crop_bounds = (mesh_min, mesh_max)

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
    masks = candidates.get("masks")
    shell_poses = candidates.get("shell_poses")
    if masks is None or shell_poses is None or masks.numel() == 0 or shell_poses.numel() == 0:
        return None
    mask_valid_shell = masks.all(dim=0)
    if mask_valid_shell.shape[0] != shell_poses.shape[0]:
        return None
    rejected_mask = ~mask_valid_shell
    if not rejected_mask.any():
        return None
    return shell_poses[rejected_mask]


def render_candidates_page(
    sample: EfmSnippetView,
    candidates: CandidateSamplingResult,
    cand_cfg: CandidateViewGeneratorConfig,
    frustum_scale: float,
    max_frustums: int,
    plot_rejected_only: bool,
) -> None:
    last_pose_rig = sample.trajectory.final_pose
    last_center = last_pose_rig.t.detach().cpu().numpy()

    st.header("Candidate Poses")
    with st.status("Building candidate plots...", expanded=False):
        cand_fig = plot_candidates(snippet=sample, poses=candidates["poses"], title="Candidate positions", center=last_center)
        shell_fig = plot_sampling_shell(
            snippet=sample,
            shell_poses=candidates["shell_poses"],
            last_pose=last_pose_rig,
            min_radius=float(cand_cfg.min_radius),
            max_radius=float(cand_cfg.max_radius),
            min_elev_deg=float(cand_cfg.min_elev_deg),
            max_elev_deg=float(cand_cfg.max_elev_deg),
            azimuth_full_circle=bool(cand_cfg.azimuth_full_circle),
        )
    st.plotly_chart(cand_fig, width="stretch")
    st.plotly_chart(shell_fig, width="stretch")
    if candidates["mask_valid"].sum() == 0:
        st.warning("All candidates were rejected; frustum plot omitted.")
    else:
        with st.status("Rendering frustums...", expanded=False):
            frust_fig = plot_candidate_frusta(
                snippet=sample,
                poses=candidates["poses"],
                camera_view=sample.camera_rgb,
                frustum_scale=frustum_scale,
                max_frustums=max_frustums,
            )
        st.plotly_chart(frust_fig, width="stretch")

    rejected_poses = rejected_pose_tensor(candidates)
    if plot_rejected_only:
        if rejected_poses is None:
            st.info("No rejected poses to plot; all sampled candidates survived rule filtering.")
        else:
            with st.status("Rendering rejected candidates plot...", expanded=False):
                rej_fig = plot_candidates(
                    snippet=sample,
                    poses=rejected_poses,
                    title=f"Rejected candidate positions ({rejected_poses.shape[0]})",
                    center=last_center,
                )
            st.plotly_chart(rej_fig, width="stretch")


def render_depth_page(depth_batch: CandidateDepthBatch) -> None:
    st.header("Candidate Renders")
    with st.status("Building depth grid...", expanded=False):
        depths = depth_batch["depths"]
        indices = depth_batch["candidate_indices"].tolist()
        titles = [f"Candidate {i}" for i in indices]
        fig = depth_grid(depths, titles=titles, zmax=float(depths.max().item()))
    st.plotly_chart(fig, width="stretch")
