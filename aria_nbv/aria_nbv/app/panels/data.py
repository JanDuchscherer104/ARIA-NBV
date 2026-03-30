"""Data panel rendering and helpers."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st
import torch
from efm3d.aria.pose import PoseTW

from ...data_handling import EfmSnippetView
from ...utils.data_plotting import (
    SnippetPlotBuilder,
    collect_frame_modalities,
    plot_first_last_frames,
    pose_world_cam,
    project_pointcloud_on_frame,
    semidense_points_for_frame,
)
from .common import _info_popover, _pretty_label


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


def scene_plot_options_ui(
    sample: EfmSnippetView,
    *,
    key_prefix: str = "data_scene",
) -> tuple[str, Scene3DPlotOptions]:
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
        show_scene_bounds = st.checkbox(
            "Show scene bounds",
            value=True,
            key=f"{key_prefix}_scene_bounds",
        )
        show_crop_bounds = st.checkbox(
            "Show crop bbox",
            value=True,
            key=f"{key_prefix}_crop_bounds",
        )
        show_frustum = st.checkbox(
            "Show frustum",
            value=True,
            key=f"{key_prefix}_frustum_enable",
        )
        mark_first_last = st.checkbox(
            "Mark start/finish",
            value=True,
            key=f"{key_prefix}_mark_first_last",
        )
        show_gt_obbs = st.checkbox(
            "Show GT OBBs",
            value=False,
            key=f"{key_prefix}_gt_obbs",
        )

    gt_ts = None
    if show_gt_obbs and sample.gt.timestamps:
        gt_ts = st.selectbox(
            "GT OBB timestamp",
            options=sample.gt.timestamps,
            index=0,
            key=f"{key_prefix}_gt_ts",
        )

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


def render_data_page(
    sample: EfmSnippetView,
    *,
    crop_margin: float | None = None,
) -> None:
    st.header("Data")
    st.write(f"Scene: **{sample.scene_id}**, snippet: **{sample.snippet_id}**")

    first_pose = sample.trajectory.t_world_rig[0]
    last_pose = sample.trajectory.t_world_rig[-1]

    def _pose_summary(pose: torch.Tensor | PoseTW):
        pt = (
            pose
            if isinstance(pose, PoseTW)
            else PoseTW.from_matrix3x4(
                pose.view(3, 4) if pose.shape[-1] == 12 else pose,
            )
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
        f"Euler ZYX=({eul_first[0]:.2f},{eul_first[1]:.2f},{eul_first[2]:.2f})°",
    )
    st.markdown(
        "- **Last frame**: "
        f"t=({t_last[0]:.3f},{t_last[1]:.3f},{t_last[2]:.3f}) m, "
        f"RPY=({rpy_last[0]:.2f},{rpy_last[1]:.2f},{rpy_last[2]:.2f})°, "
        f"Euler ZYX=({eul_last[0]:.2f},{eul_last[1]:.2f},{eul_last[2]:.2f})°",
    )

    modalities, missing = collect_frame_modalities(sample, include_depth=True)
    st.plotly_chart(plot_first_last_frames(sample), width="stretch")

    missing_depth = [m for m in missing if m.startswith("Depth")]
    if modalities and not missing_depth:
        st.caption("Depth maps available for rendering and overlays.")
    elif missing_depth:
        st.info(f"Depth maps missing for: {', '.join(missing_depth)}")

    st.subheader("Point cloud overlay")
    _info_popover(
        "overlay",
        "Projects world-space points into the selected camera image using the "
        "camera intrinsics and the per-frame pose. Use this to verify that "
        "the semidense SLAM points align with RGB/SLAM frames and to spot "
        "pose or calibration mismatches.",
    )
    overlay_cam = st.selectbox(
        "Overlay camera",
        ["rgb", "slam-l", "slam-r"],
        index=0,
        key="overlay_cam",
    )
    overlay_frame = st.slider(
        "Frame index for overlay",
        0,
        int(sample.camera_rgb.images.shape[0] - 1),
        0,
    )
    overlay_source = st.selectbox(
        "Point cloud source",
        [
            "Semidense (all frames)",
            "Semidense (selected frame)",
            "Semidense (last with points)",
        ],
        index=0,
    )

    if overlay_source == "Semidense (all frames)":
        points_world = semidense_points_for_frame(sample, None, all_frames=True)
    elif overlay_source == "Semidense (selected frame)":
        points_world = semidense_points_for_frame(
            sample,
            overlay_frame,
            all_frames=False,
        )
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
            title=_pretty_label(
                f"Overlay on {overlay_cam.upper()} frame {overlay_frame} ({points_world.shape[0]} pts)",
            ),
            max_points=20000,
        )
        st.plotly_chart(fig_overlay, width="stretch")

    cam_choice, plot_opts = scene_plot_options_ui(sample, key_prefix="data_scene")
    _info_popover(
        "scene overview",
        "3D scene view combines GT mesh, semidense points, and the trajectory. "
        "Frusta are drawn from camera intrinsics/extrinsics. Bounds boxes show "
        "scene extents and any crop applied during RRI computation.",
    )

    builder = (
        SnippetPlotBuilder.from_snippet(
            sample,
            title=_pretty_label("Mesh + semidense + trajectory + camera frustum"),
        )
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
        builder.add_bounds_box(
            name="Crop bounds",
            color="orange",
            dash="solid",
            width=3,
            aabb=crop_aabb,
        )

    st.plotly_chart(builder.finalize(), width="stretch")


__all__ = ["render_data_page"]
