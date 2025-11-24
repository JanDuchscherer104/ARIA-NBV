"""Streamlit UI for inspecting ASE snippets, candidate poses, and depth renders."""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict, cast

import numpy as np
import streamlit as st
import torch
from efm3d.aria.pose import PoseTW

from oracle_rri.data import AseEfmDatasetConfig, EfmSnippetView
from oracle_rri.data.efm_views import EfmCameraView
from oracle_rri.data.plotting import (
    SnippetPlotBuilder,
    collect_frame_modalities,
    crop_aabb_from_semidense,
    plot_first_last_frames,
    project_pointcloud_on_frame,
)
from oracle_rri.pose_generation import CandidateViewGeneratorConfig
from oracle_rri.pose_generation.plotting import plot_candidate_frustums_simple, plot_candidates, plot_sampling_shell
from oracle_rri.pose_generation.types import CandidateSamplingResult
from oracle_rri.rendering import (
    CandidateDepthRendererConfig,
    Efm3dDepthRendererConfig,
    Pytorch3DDepthRendererConfig,
)
from oracle_rri.rendering.candidate_depth_renderer import CandidateDepthBatch
from oracle_rri.rendering.plotting import depth_grid
from oracle_rri.utils import Console

STATE_KEYS = {
    "sample": "nbv_sample",
    "sample_cfg": "nbv_sample_cfg",
    "candidates": "nbv_candidates",
    "cand_cfg": "nbv_cand_cfg",
    "depth": "nbv_depth_batch",
    "depth_cfg": "nbv_depth_cfg",
    "sample_idx": "nbv_sample_idx",
}
TASK_KEYS = {
    "data": "nbv_task_data",
    "candidates": "nbv_task_candidates",
    "depth": "nbv_task_depth",
}


class TaskState(TypedDict):
    """Background task state tracked in session_state."""

    status: Literal["idle", "running", "done", "error"]
    error: str | None


class SessionVars(TypedDict, total=False):
    """Typed view over Streamlit session_state keys used in this app."""

    nbv_task_data: TaskState
    nbv_task_candidates: TaskState
    nbv_task_depth: TaskState


def _state() -> SessionVars:
    """Typed accessor for session_state."""

    return cast(SessionVars, st.session_state)


def _safe_rerun() -> None:
    """Call Streamlit rerun with backward compatibility."""

    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()  # type: ignore[attr-defined]
    else:  # pragma: no cover - unexpected env
        raise RuntimeError("Streamlit rerun API not available.")


@st.cache_resource(
    show_spinner=False,
    hash_funcs={AseEfmDatasetConfig: lambda c: json.dumps(c.model_dump(mode="json", round_trip=True), sort_keys=True)},
)
def _load_dataset(cfg: AseEfmDatasetConfig):
    """Cache dataset construction to avoid reloading shards on reruns."""

    return cfg.setup_target()


def _load_sample(cfg: AseEfmDatasetConfig, sample_idx: int = 0) -> EfmSnippetView:
    """Load a single snippet from the configured dataset."""

    dataset = _load_dataset(cfg)
    it = iter(dataset)
    for _ in range(sample_idx + 1):
        sample = next(it)
    return sample


def _candidate_config_from_ui(
    default: CandidateViewGeneratorConfig,
    ui: st.delta_generator.DeltaGenerator,
    *,
    super_fast: bool = False,
    is_debug: bool = False,
) -> CandidateViewGeneratorConfig:
    """Sidebar controls for candidate generation."""

    ui.subheader("Candidate Generator")
    num_samples_default = 2 if super_fast else default.num_samples
    num_samples = ui.slider("num_samples", 2, 512, num_samples_default, step=2)
    min_radius = ui.slider("min_radius (m)", 0.1, 2.0, float(default.min_radius), step=0.05)
    max_radius = ui.slider("max_radius (m)", 0.2, 3.0, float(default.max_radius), step=0.05)
    min_elev = ui.slider("min_elev_deg", -45.0, 0.0, float(default.min_elev_deg), step=1.0)
    max_elev = ui.slider("max_elev_deg", 0.0, 75.0, float(default.max_elev_deg), step=1.0)
    ensure_collision_free = ui.checkbox("ensure_collision_free", value=default.ensure_collision_free)
    ensure_free_space = ui.checkbox("ensure_free_space", value=default.ensure_free_space)
    min_distance = ui.slider(
        "min_distance_to_mesh (m)",
        0.0,
        0.5,
        float(default.min_distance_to_mesh),
        step=0.01,
    )
    device = ui.selectbox("Generator device", ["cpu", "cuda"], index=0 if str(default.device) == "cpu" else 1)
    return default.model_copy(
        update={
            "num_samples": int(num_samples),
            "min_radius": float(min_radius),
            "max_radius": float(max_radius),
            "min_elev_deg": float(min_elev),
            "max_elev_deg": float(max_elev),
            "min_distance_to_mesh": float(min_distance),
            "ensure_collision_free": ensure_collision_free,
            "ensure_free_space": ensure_free_space,
            "device": device,
            "is_debug": is_debug,
        }
    )


def _renderer_config_from_ui(
    default: CandidateDepthRendererConfig,
    ui: st.delta_generator.DeltaGenerator,
    *,
    super_fast: bool = False,
    is_debug: bool = False,
) -> CandidateDepthRendererConfig:
    """Sidebar controls for depth rendering."""

    ui.subheader("Depth Renderer")
    backend_options = ["pytorch3d (raster)", "cpu (trimesh rays)"]
    default_backend_idx = 0 if isinstance(default.renderer, Pytorch3DDepthRendererConfig) else 1
    backend_choice = ui.selectbox("backend", backend_options, index=default_backend_idx)
    max_candidates_default = 2 if super_fast else (default.max_candidates if default.max_candidates is not None else 4)
    max_candidates = ui.slider("max_candidates", 1, 16, max_candidates_default)
    low_res = ui.checkbox("Low-res render (downscale to 320x240)", value=super_fast)
    res_scale = ui.slider("Render resolution scale", 0.1, 1.0, 0.25 if super_fast else 1.0, step=0.05)
    renderer_device_options = ["cpu", "cuda"]
    default_device_idx = renderer_device_options.index(str(getattr(default.renderer, "device", "cpu")))
    renderer_device = ui.selectbox("renderer device", renderer_device_options, index=default_device_idx)
    zfar = ui.slider("zfar (m)", 5.0, 50.0, default.renderer.zfar, step=1.0)
    if backend_choice.startswith("pytorch3d"):
        faces_default = (
            default.renderer.faces_per_pixel if isinstance(default.renderer, Pytorch3DDepthRendererConfig) else 1
        )
        faces_pp = ui.slider("faces_per_pixel", 1, 4, faces_default)
        renderer_cfg: Pytorch3DDepthRendererConfig | Efm3dDepthRendererConfig = Pytorch3DDepthRendererConfig(
            device=renderer_device,
            zfar=float(zfar),
            faces_per_pixel=int(faces_pp),
            is_debug=is_debug,
        )
    else:
        chunk_rays = ui.slider("chunk_rays", 50_000, 500_000, 200_000, step=50_000)
        renderer_cfg = Efm3dDepthRendererConfig(
            device="cpu",
            zfar=float(zfar),
            chunk_rays=int(chunk_rays),
            is_debug=is_debug,
        )
    return default.model_copy(
        update={
            "max_candidates": int(max_candidates),
            "renderer": renderer_cfg,
            "is_debug": is_debug,
            "low_res": low_res,
            "resolution_scale": float(res_scale),
        }
    )


def _dataset_config_from_ui(
    ui: st.delta_generator.DeltaGenerator, *, super_fast: bool, is_debug: bool
) -> AseEfmDatasetConfig:
    """Sidebar controls for dataset (data page only)."""

    ui.subheader("Dataset")
    # Fixed defaults (no user control).
    mesh_ratio = ui.slider("mesh decimation ratio", 0.0, 1.0, 0.02 if super_fast else 0.02, step=0.02)
    mesh_max_faces = ui.number_input(
        "max mesh faces (cap after decimation)",
        min_value=1_000,
        max_value=2_000_000,
        value=100_000 if super_fast else 300_000,
        step=10_000,
    )
    mesh_crop_margin = ui.slider(
        "crop margin (m)",
        0.0,
        2.0,
        0.2 if super_fast else 0.5,
        step=0.05,
    )
    mesh_keep_ratio = ui.slider(
        "min keep ratio (after crop)",
        0.0,
        1.0,
        0.9 if super_fast else 0.7,
        step=0.05,
    )
    batch_size = None
    verbose = True
    require_mesh = ui.checkbox("require mesh", value=True)

    return AseEfmDatasetConfig(
        atek_variant="efm",
        mesh_simplify_ratio=mesh_ratio if mesh_ratio > 0 else None,
        mesh_crop_margin_m=mesh_crop_margin,
        mesh_crop_min_keep_ratio=mesh_keep_ratio,
        mesh_max_faces=int(mesh_max_faces),
        require_mesh=require_mesh,
        batch_size=batch_size,
        verbose=verbose,
        is_debug=is_debug,
    )


def _plot_options_from_ui(sample: EfmSnippetView) -> dict[str, Any]:
    """Sidebar controls for plot layers (separate from dataset options)."""

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
    show_sem = "Semidense" in layer_choices
    sem_last_only = "Semidense (last frame only)" in layer_choices
    show_scene_bounds = "Scene bounds" in layer_choices
    show_crop_bounds = "Crop bbox" in layer_choices
    show_frustum = "Frustum" in layer_choices
    mark_first_last = "Mark start/finish" in layer_choices
    show_gt_obbs = "GT OBBs" in layer_choices

    max_sem = st.sidebar.slider("Max semidense points", 1000, 50000, 20000, step=1000, key="max_sem_points")
    num_frames = int(sample.camera_rgb.images.shape[0])
    frustum_idx = st.sidebar.multiselect(
        "Frustum frame indices",
        options=list(range(num_frames)),
        default=[0],
        key="frustum_idx",
    )

    mark_first_last = st.sidebar.checkbox("Mark start / finish", value=True, key="mark_first_last")
    gt_ts = None
    if show_gt_obbs and sample.gt.timestamps:
        gt_ts = st.sidebar.selectbox("GT OBB timestamp", options=sample.gt.timestamps, index=0)

    return {
        "show_semidense": show_sem,
        "max_sem_points": max_sem,
        "pc_from_last_only": sem_last_only,
        "show_scene_bounds": show_scene_bounds,
        "show_crop_bounds": show_crop_bounds,
        "show_frustum": show_frustum,
        "frustum_scale": 1.0,
        "frustum_frame_indices": frustum_idx if len(frustum_idx) > 0 else [0],
        "mark_first_last": mark_first_last,
        "show_gt_obbs": show_gt_obbs,
        "gt_timestamp": gt_ts,
    }


def _pose_world_cam(sample: EfmSnippetView, cam_view: EfmCameraView, frame_idx: int):
    """Compute T_world_cam for a specific frame without display reorientation."""

    cam_ts = cam_view.time_ns.cpu().numpy()
    traj_ts = sample.trajectory.time_ns.cpu().numpy()
    traj_idx = int(np.argmin(np.abs(traj_ts - cam_ts[frame_idx])))

    t_world_rig = sample.trajectory.t_world_rig[traj_idx]
    t_cam_rig = cam_view.calib.T_camera_rig[frame_idx]
    return t_world_rig @ t_cam_rig.inverse(), cam_view.calib[frame_idx]


def _semidense_points_for_frame(sample: EfmSnippetView, frame_idx: int | None, *, all_frames: bool) -> torch.Tensor:
    """Return semidense points in world coords (Torch, CPU)."""

    sem = sample.semidense
    if sem is None or sem.points_world.numel() == 0:
        return torch.zeros((0, 3), dtype=torch.float32)

    pts = sem.points_world
    lengths = sem.lengths
    if all_frames:
        if lengths is not None:
            max_len = pts.shape[1]
            mask_valid = torch.arange(max_len).unsqueeze(0) < lengths.clamp_max(max_len).unsqueeze(-1)
            pts = torch.where(mask_valid.unsqueeze(-1), pts, torch.nan)
        pts = pts.reshape(-1, 3)
    else:
        if frame_idx is None:
            frame_idx = int(torch.argmax(lengths).item()) if lengths.numel() else 0
        frame_idx = max(0, min(int(frame_idx), pts.shape[0] - 1))
        n_valid = int(lengths[frame_idx].item()) if lengths is not None else pts.shape[1]
        pts = pts[frame_idx, :n_valid]

    finite = torch.isfinite(pts).all(dim=-1)
    pts = pts[finite]
    return pts.cpu()


def _render_data_page(sample: EfmSnippetView, *, crop_margin: float | None = None) -> None:
    """Render the Data page plots."""

    st.header("Data")
    st.write(f"Scene: **{sample.scene_id}**, snippet: **{sample.snippet_id}**")

    # Final pose pose_world_rig with orientation summaries (world frame, LUF cameras, gravity in -Z).

    first_pose = sample.trajectory.t_world_rig[0]
    last_pose = sample.trajectory.t_world_rig[-1]

    def _pose_summary(pose: torch.Tensor | PoseTW):
        if isinstance(pose, PoseTW):
            pt = pose
        else:
            tensor = pose
            if tensor.shape[-1] == 12:
                tensor = tensor.view(3, 4)
            pt = PoseTW.from_matrix3x4(tensor)
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

    available_depth = [name for name, _, _ in modalities if name.startswith("Depth")]

    if not available_depth:
        st.info("No metric depth maps present in this snippet.")
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
        points_world = _semidense_points_for_frame(sample, None, all_frames=True)
    elif overlay_source == "Semidense (selected frame)":
        points_world = _semidense_points_for_frame(sample, overlay_frame, all_frames=False)
    elif overlay_source == "Semidense (last with points)":
        lengths = sample.semidense.lengths if sample.semidense is not None else None
        if lengths is not None and torch.any(lengths > 0):
            last_idx = int(torch.nonzero(lengths > 0, as_tuple=False).max().item())
        else:
            last_idx = 0
        points_world = _semidense_points_for_frame(sample, last_idx, all_frames=False)
    else:
        points_world = torch.zeros((0, 3), dtype=torch.float32)

    if points_world is None or points_world.numel() == 0:
        st.info("No points available for overlay with current selection.")
    else:
        cam_map = {
            "rgb": sample.camera_rgb,
            "slam-l": sample.camera_slam_left,
            "slam-r": sample.camera_slam_right,
        }
        cam_view = cam_map[overlay_cam]
        pose_wc, cam_tw = _pose_world_cam(sample, cam_view, overlay_frame)
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
    plot_opts = _plot_options_from_ui(sample)
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


def _rejected_pose_tensor(candidates: CandidateSamplingResult) -> torch.Tensor | None:
    """Return tensor of rejected shell poses if any were filtered out."""

    masks = candidates.get("masks")
    shell_poses = candidates.get("shell_poses")
    if masks is None or shell_poses is None:
        return None
    if masks.numel() == 0 or shell_poses.numel() == 0:
        return None
    mask_valid_shell = masks.all(dim=0)
    if mask_valid_shell.shape[0] != shell_poses.shape[0]:
        return None
    rejected_mask = ~mask_valid_shell
    if not rejected_mask.any():
        return None
    return shell_poses[rejected_mask]


def _render_candidates_page(
    sample: EfmSnippetView,
    candidates: CandidateSamplingResult,
    cand_cfg: CandidateViewGeneratorConfig,
    frustum_scale: float,
    max_frustums: int,
    plot_rejected_only: bool,
) -> None:
    """Render the Candidate Poses page."""

    cam_map = {
        "rgb": sample.camera_rgb,
        "rgb_depth": sample.camera_rgb,
        "slam_l": sample.camera_slam_left,
        "slam_r": sample.camera_slam_right,
    }
    cam_view = cam_map.get(cand_cfg.camera_index, sample.camera_rgb)
    t_cam_rig = cam_view.calib.T_camera_rig[0]
    last_pose_cam = sample.trajectory.final_pose @ t_cam_rig.inverse()
    last_center = last_pose_cam.t.detach().cpu().numpy()

    st.header("Candidate Poses")
    with st.status("Building candidate plots...", expanded=False):
        cand_fig = plot_candidates(candidates["poses"], sample.mesh, title="Candidate positions", center=last_center)
        shell_fig = plot_sampling_shell(
            shell_poses=candidates["shell_poses"],
            last_pose=last_pose_cam,
            min_radius=float(cand_cfg.min_radius),
            max_radius=float(cand_cfg.max_radius),
            min_elev_deg=float(cand_cfg.min_elev_deg),
            max_elev_deg=float(cand_cfg.max_elev_deg),
            azimuth_full_circle=bool(cand_cfg.azimuth_full_circle),
            mesh=sample.mesh,
        )
    st.plotly_chart(cand_fig, width="stretch")
    st.plotly_chart(shell_fig, width="stretch")
    if candidates["mask_valid"].sum() == 0:
        st.warning("All candidates were rejected; frustum plot omitted.")
    else:
        with st.status("Rendering frustums...", expanded=False):
            frust_fig = plot_candidate_frustums_simple(
                poses=candidates["poses"],
                camera_view=sample.camera_rgb,
                mesh=sample.mesh,
                frustum_scale=frustum_scale,
                max_frustums=max_frustums,
            )
        st.plotly_chart(frust_fig, width="stretch")

    rejected_poses = _rejected_pose_tensor(candidates)
    if plot_rejected_only:
        if rejected_poses is None:
            st.info("No rejected poses to plot; all sampled candidates survived rule filtering.")
        else:
            with st.status("Rendering rejected candidates plot...", expanded=False):
                rej_fig = plot_candidates(
                    rejected_poses,
                    sample.mesh,
                    title=f"Rejected candidate positions ({rejected_poses.shape[0]})",
                    center=last_center,
                )
            st.plotly_chart(rej_fig, width="stretch")


def _render_depth_page(depth_batch: CandidateDepthBatch) -> None:
    """Render the Candidate Renders page."""

    st.header("Candidate Renders")
    with st.status("Building depth grid...", expanded=False):
        depths = depth_batch["depths"]
        indices = depth_batch["candidate_indices"].tolist()
        titles = [f"Candidate {i}" for i in indices]
        fig = depth_grid(depths, titles=titles, zmax=float(depths.max().item()))
    st.plotly_chart(fig, width="stretch")


def _init_task_state() -> None:
    """Initialise task state containers in session_state."""

    for key in TASK_KEYS.values():
        _state().setdefault(
            key,
            TaskState(
                status="idle",
                error=None,
            ),
        )


def main() -> None:
    """Entry point for `streamlit run -m oracle_rri.streamlit_app`."""

    st.set_page_config(page_title="NBV Explorer", layout="wide")
    _init_task_state()
    st.sidebar.markdown("---")
    super_fast = st.sidebar.checkbox(
        "Super-fast debug mode", value=False, help="Use tiny meshes, 2 candidates, low-res renders."
    )
    is_debug_global = st.sidebar.checkbox("Debug logging (all modules)", value=True)
    console = Console.with_prefix("streamlit_app").set_verbose(True).set_debug(is_debug_global)

    st.sidebar.write("Pages")
    page = st.sidebar.radio("Select view", ("Data", "Candidate Poses", "Candidate Renders"))

    # Helpers for dirty/stale detection
    def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
        return cfg.model_dump() if hasattr(cfg, "model_dump") else dict(cfg)

    def _cfg_from_state(key: str, cfg_cls: type[Any]) -> Any:
        stored = _get(key)
        if isinstance(stored, cfg_cls):
            return stored
        if isinstance(stored, dict):
            return cfg_cls.model_validate(stored)
        return cfg_cls()

    def _cfg_from_state_optional(key: str, cfg_cls: type[Any]) -> Any | None:
        stored = _get(key)
        if stored is None:
            return None
        if isinstance(stored, cfg_cls):
            return stored
        if isinstance(stored, dict):
            return cfg_cls.model_validate(stored)
        return None

    def _store(key: str, value: Any) -> None:
        st.session_state[key] = value

    def _get(key: str) -> Any:
        return st.session_state.get(key)

    sample = cast(EfmSnippetView | None, _get(STATE_KEYS["sample"]))
    candidates = cast(CandidateSamplingResult | None, _get(STATE_KEYS["candidates"]))
    depth_batch = cast(CandidateDepthBatch | None, _get(STATE_KEYS["depth"]))

    # Status flags
    cfg_changed = {"sample": False, "cand": False, "depth": False}
    pipeline_order: tuple[str, ...] = ("data", "candidates", "depth")

    def _refresh_stage_vars() -> None:
        nonlocal sample, candidates, depth_batch
        sample = cast(EfmSnippetView | None, _get(STATE_KEYS["sample"]))
        candidates = cast(CandidateSamplingResult | None, _get(STATE_KEYS["candidates"]))
        depth_batch = cast(CandidateDepthBatch | None, _get(STATE_KEYS["depth"]))

    def _run_data_stage(
        cfg: AseEfmDatasetConfig | None, sample_idx: int, *, allow_ui: bool = True
    ) -> EfmSnippetView | None:
        if cfg is None:
            if allow_ui:
                st.warning("No cached dataset config. Configure the dataset on the Data page first.")
            return None
        try:
            sample_local = _load_sample(cfg, sample_idx=sample_idx)
            _store(STATE_KEYS["sample"], sample_local)
            _store(STATE_KEYS["sample_cfg"], _cfg_to_dict(cfg))
            _store(STATE_KEYS["sample_idx"], sample_idx)
            _store(STATE_KEYS["candidates"], None)
            _store(STATE_KEYS["cand_cfg"], None)
            _store(STATE_KEYS["depth"], None)
            _store(STATE_KEYS["depth_cfg"], None)
            cfg_changed["sample"] = False
            if allow_ui:
                _safe_rerun()
            return sample_local
        except Exception as exc:  # pragma: no cover - UI feedback
            if allow_ui:
                st.error(f"Failed to load dataset sample: {exc}")
            console.error(str(exc))
            return None

    def _run_candidates_stage(
        cfg: CandidateViewGeneratorConfig | None, *, allow_ui: bool = True
    ) -> CandidateSamplingResult | None:
        if cfg is None:
            if allow_ui:
                st.warning("No cached candidate config. Run candidate generation once to cache settings.")
            return None
        local_sample = cast(EfmSnippetView | None, _get(STATE_KEYS["sample"]))
        if local_sample is None:
            if allow_ui:
                st.warning("Load data first on the Data page, then run candidates.")
            return None
        try:
            generator = cfg.setup_target()
            with st.status("Generating candidates...", expanded=False):
                candidates_local = generator.generate_from_typed_sample(local_sample)
            _store(STATE_KEYS["candidates"], candidates_local)
            _store(STATE_KEYS["cand_cfg"], _cfg_to_dict(cfg))
            _store(STATE_KEYS["depth"], None)
            _store(STATE_KEYS["depth_cfg"], None)
            cfg_changed["cand"] = False
            if allow_ui:
                _safe_rerun()
            return candidates_local
        except Exception as exc:  # pragma: no cover
            if allow_ui:
                st.error(f"Candidate generation failed: {exc}")
            console.error(str(exc))
            return None

    def _run_depth_stage(
        cfg: CandidateDepthRendererConfig | None,
        render_depths: bool = True,
        *,
        allow_ui: bool = True,
    ) -> CandidateDepthBatch | None:
        if not render_depths:
            return None
        if cfg is None:
            if allow_ui:
                st.warning("No cached renderer config. Configure renderer settings first.")
            return None
        local_sample = cast(EfmSnippetView | None, _get(STATE_KEYS["sample"]))
        local_candidates = cast(CandidateSamplingResult | None, _get(STATE_KEYS["candidates"]))
        if local_sample is None or local_candidates is None:
            if allow_ui:
                st.warning("Need data and candidates. Run previous stages before rendering.")
            return None
        renderer = cfg.setup_target()
        try:
            with st.status("Rendering depth maps...", expanded=False):
                depth_local = renderer.render(local_sample, local_candidates)
            depth_local = renderer.render(local_sample, local_candidates)
            _store(STATE_KEYS["depth"], depth_local)
            _store(STATE_KEYS["depth_cfg"], _cfg_to_dict(cfg))
            cfg_changed["depth"] = False
            if allow_ui:
                _safe_rerun()
            return depth_local
        except Exception as exc:  # pragma: no cover - rendering failures
            if allow_ui:
                st.warning(f"Depth rendering failed: {exc}")
            console.warn(str(exc))
            return None

    def _run_previous_stages(current_stage: str) -> None:
        """Run all stages that precede the current one using cached configs."""

        try:
            target_idx = pipeline_order.index(current_stage)
        except ValueError:
            st.warning(f"Unknown stage: {current_stage}")
            return

        for stage in pipeline_order[:target_idx]:
            if stage == "data":
                data_cfg_state = _cfg_from_state_optional(STATE_KEYS["sample_cfg"], AseEfmDatasetConfig)
                sample_idx_state = int(_get(STATE_KEYS["sample_idx"]) or 0)
                _run_data_stage(data_cfg_state, sample_idx_state)
            elif stage == "candidates":
                cand_cfg_state = _cfg_from_state_optional(STATE_KEYS["cand_cfg"], CandidateViewGeneratorConfig)
                if cand_cfg_state is None:
                    cand_cfg_state = CandidateViewGeneratorConfig()
                _run_candidates_stage(cand_cfg_state)
            elif stage == "depth":
                renderer_cfg_state = _cfg_from_state_optional(STATE_KEYS["depth_cfg"], CandidateDepthRendererConfig)
                _run_depth_stage(renderer_cfg_state, True, allow_ui=False)
        _refresh_stage_vars()

    if page == "Data":
        with st.sidebar.form("data_form"):
            dataset_cfg = _dataset_config_from_ui(st.sidebar, super_fast=super_fast, is_debug=is_debug_global)
            sample_idx = int(_get(STATE_KEYS["sample_idx"]) or 0)
            next_sample = st.form_submit_button("Next sample")
            run_data = st.form_submit_button("Run / refresh data")
        if next_sample:
            sample_idx += 1
            run_data = True
        clear = st.sidebar.button("Clear cache")
        if clear:
            for key in STATE_KEYS.values():
                st.session_state.pop(key, None)
            st.rerun()

        cfg_changed["sample"] = _get(STATE_KEYS["sample_cfg"]) != _cfg_to_dict(dataset_cfg)

        if run_data:
            sample = _run_data_stage(dataset_cfg, sample_idx)
            _refresh_stage_vars()

        if sample is None:
            st.info("No sample loaded. Configure dataset and click 'Run / refresh data' (or 'Next sample').")
        else:
            cfg_state = _cfg_from_state(STATE_KEYS["sample_cfg"], AseEfmDatasetConfig)
            show_crop_box = st.sidebar.checkbox("Show crop bbox", value=True, key="show_crop_box")
            _render_data_page(sample, crop_margin=cfg_state.mesh_crop_margin_m if show_crop_box else None)
        return

    # Candidate page controls
    if page == "Candidate Poses":
        with st.sidebar.form("cand_form"):
            candidate_cfg = _candidate_config_from_ui(
                CandidateViewGeneratorConfig(), st.sidebar, super_fast=super_fast, is_debug=is_debug_global
            )
            st.sidebar.subheader("Candidate plot options")
            frustum_scale = st.sidebar.slider("Frustum scale", 0.1, 1.0, 0.5, step=0.05)
            max_frustums = st.sidebar.slider("Max frustums", 1, 24, 6)
            plot_rejected_only = st.sidebar.checkbox("Plot rejected poses only (if any)", value=False)
            run_prev = st.form_submit_button("Run previous")
            run_cand = st.form_submit_button("Run / refresh candidates")
        cfg_changed["cand"] = _get(STATE_KEYS["cand_cfg"]) != _cfg_to_dict(candidate_cfg)

        if run_prev:
            _run_previous_stages("candidates")

        if run_cand:
            _run_candidates_stage(candidate_cfg, allow_ui=False)
        _refresh_stage_vars()

        if candidates is None:
            st.info("No candidates yet. Configure generator and click 'Run / refresh candidates'.")
        else:
            _render_candidates_page(
                sample,
                candidates,
                candidate_cfg,
                frustum_scale,
                max_frustums,
                plot_rejected_only,
            )
        if cfg_changed["cand"]:
            st.info("Candidate settings changed; rerun to refresh results.")
        return

    # Render page controls
    with st.sidebar.form("depth_form"):
        renderer_cfg = _renderer_config_from_ui(
            CandidateDepthRendererConfig(renderer=Pytorch3DDepthRendererConfig(device="cpu")),
            st.sidebar,
            super_fast=super_fast,
            is_debug=is_debug_global,
        )
        render_depths = st.checkbox("Compute depth renders", value=True, key="compute_depths")
        run_prev = st.form_submit_button("Run previous")
        run_depth = st.form_submit_button("Run / refresh renders")
    cfg_changed["depth"] = _get(STATE_KEYS["depth_cfg"]) != _cfg_to_dict(renderer_cfg)

    if run_prev:
        _run_previous_stages("depth")

    if run_depth:
        if render_depths:
            _run_depth_stage(renderer_cfg, render_depths, allow_ui=True)
        elif depth_batch is None:
            st.warning("No cached depths available; enable 'Compute depth renders' or run renders first.")
        else:
            st.info("Using cached depths to rebuild plots.")
        _refresh_stage_vars()

    if depth_batch is None:
        if candidates is None:
            st.info("No renders yet. Run candidates first, then click 'Run / refresh renders'.")
        elif not render_depths:
            st.info("Enable 'Compute depth renders' to produce depth maps.")
        else:
            st.info("Click 'Run / refresh renders' to compute depth maps.")
    else:
        _render_depth_page(depth_batch)

    stale_parts = [name for name, changed in cfg_changed.items() if changed]
    if stale_parts:
        st.info(f"Cached results are stale for: {', '.join(stale_parts)}. Click the page's run button to update.")
    return


if __name__ == "__main__":
    main()
