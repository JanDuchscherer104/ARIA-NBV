"""Streamlit UI for inspecting ASE snippets, candidate poses, and depth renders."""

from __future__ import annotations

from typing import Any, cast

import streamlit as st
import torch

from oracle_rri.data import AseEfmDatasetConfig, EfmSnippetView
from oracle_rri.data.plotting import crop_aabb_from_semidense, plot_first_last_frames, plot_trajectory
from oracle_rri.pose_generation import CandidateViewGeneratorConfig
from oracle_rri.pose_generation.plotting import (
    plot_candidate_frustums,
    plot_candidates,
    plot_sampling_shell,
)
from oracle_rri.pose_generation.types import CandidateSamplingResult
from oracle_rri.rendering import CandidateDepthRendererConfig, Pytorch3DDepthRendererConfig
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


def _load_sample(cfg: AseEfmDatasetConfig) -> EfmSnippetView:
    """Load a single snippet from the configured dataset."""

    dataset = cfg.setup_target()
    return next(iter(dataset))


def _candidate_config_from_ui(default: CandidateViewGeneratorConfig) -> CandidateViewGeneratorConfig:
    """Sidebar controls for candidate generation."""

    st.sidebar.subheader("Candidate Generator")
    num_samples = st.sidebar.slider("num_samples", 8, 512, default.num_samples, step=8)
    min_radius = st.sidebar.slider("min_radius (m)", 0.1, 2.0, float(default.min_radius), step=0.05)
    max_radius = st.sidebar.slider("max_radius (m)", 0.2, 3.0, float(default.max_radius), step=0.05)
    min_elev = st.sidebar.slider("min_elev_deg", -45.0, 0.0, float(default.min_elev_deg), step=1.0)
    max_elev = st.sidebar.slider("max_elev_deg", 0.0, 75.0, float(default.max_elev_deg), step=1.0)
    ensure_collision_free = st.sidebar.checkbox("ensure_collision_free", value=default.ensure_collision_free)
    ensure_free_space = st.sidebar.checkbox("ensure_free_space", value=default.ensure_free_space)
    min_distance = st.sidebar.slider(
        "min_distance_to_mesh (m)",
        0.0,
        0.5,
        float(default.min_distance_to_mesh),
        step=0.01,
    )
    device = st.sidebar.selectbox("Generator device", ["cpu", "cuda"], index=0 if str(default.device) == "cpu" else 1)
    is_debug = st.sidebar.checkbox("generator is_debug", value=default.is_debug)

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


def _renderer_config_from_ui(default: CandidateDepthRendererConfig) -> CandidateDepthRendererConfig:
    """Sidebar controls for depth rendering."""

    st.sidebar.subheader("Depth Renderer")
    max_candidates = st.sidebar.slider(
        "max_candidates",
        1,
        16,
        default.max_candidates if default.max_candidates is not None else 4,
    )
    renderer_device = st.sidebar.selectbox("renderer device", ["cpu", "cuda"], index=0)
    zfar = st.sidebar.slider("zfar (m)", 5.0, 50.0, default.renderer.zfar, step=1.0)
    faces_pp = st.sidebar.slider("faces_per_pixel", 1, 4, default.renderer.faces_per_pixel)
    is_debug = st.sidebar.checkbox("renderer is_debug", value=default.is_debug)

    renderer_cfg = default.renderer.model_copy(
        update={
            "device": renderer_device,
            "zfar": float(zfar),
            "faces_per_pixel": int(faces_pp),
            "is_debug": is_debug,
        }
    )
    return default.model_copy(
        update={
            "max_candidates": int(max_candidates),
            "renderer": renderer_cfg,
            "is_debug": is_debug,
        }
    )


def _dataset_config_from_ui() -> AseEfmDatasetConfig:
    """Sidebar controls for dataset (data page only)."""

    st.sidebar.subheader("Dataset")
    # Fixed defaults (no user control).
    scene_ids: list[str] = []
    atek_variant = "efm"
    load_meshes = True
    mesh_ratio = st.sidebar.slider("mesh decimation ratio", 0.0, 1.0, 0.1, step=0.05)
    mesh_crop_margin = st.sidebar.slider(
        "crop margin (m)",
        0.0,
        2.0,
        0.5,
        step=0.05,
    )
    mesh_face_cap = None
    batch_size = None
    verbose = True
    is_debug = True
    require_mesh = st.sidebar.checkbox("require mesh", value=True)

    return AseEfmDatasetConfig(
        scene_ids=scene_ids,
        atek_variant=atek_variant,
        load_meshes=load_meshes,
        mesh_simplify_ratio=mesh_ratio if mesh_ratio > 0 else None,
        mesh_crop_margin_m=mesh_crop_margin,
        mesh_max_faces=mesh_face_cap,
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
            "Show walls (double-sided mesh)",
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
    double_sided_mesh = "Show walls (double-sided mesh)" in layer_choices

    max_sem = st.sidebar.slider("Max semidense points", 1000, 50000, 20000, step=1000, key="max_sem_points")
    num_frames = int(sample.camera_rgb.images.shape[0])
    frustum_idx = st.sidebar.multiselect(
        "Frustum frame indices",
        options=list(range(num_frames)),
        default=[0],
        key="frustum_idx",
    )

    mark_first_last = st.sidebar.checkbox("Mark start / finish", value=True, key="mark_first_last")

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
        "double_sided_mesh": double_sided_mesh,
    }


def _render_data_page(sample: EfmSnippetView, *, crop_margin: float | None = None) -> None:
    """Render the Data page plots."""

    st.header("Data")
    st.write(f"Scene: **{sample.scene_id}**, snippet: **{sample.snippet_id}**")

    # Final pose pose_world_rig with orientation summaries (world frame, RDF cameras, gravity in -Z).
    final_pose = sample.trajectory.final_pose
    yaw_rad, pitch_rad, roll_rad = final_pose.to_ypr(rad=True)
    yaw_deg = torch.rad2deg(yaw_rad).item()
    pitch_deg = torch.rad2deg(pitch_rad).item()
    roll_deg = torch.rad2deg(roll_rad).item()
    euler_zyx_deg = torch.rad2deg(final_pose.to_euler(rad=True)).tolist()  # [roll, pitch, yaw] ZYX
    tx, ty, tz = final_pose.t.tolist()

    st.markdown(
        "**Pose frame:** `T_world_rig` (VIO world frame, gravity in -Z; rig/camera frame is RDF: X-right, Y-down, Z-forward)"
    )
    st.markdown(
        "- Translation (m): "
        f"x={tx:.3f}, y={ty:.3f}, z={tz:.3f}\n"
        f"- Roll/Pitch/Yaw (deg, ZYX): roll={roll_deg:.2f}, pitch={pitch_deg:.2f}, yaw={yaw_deg:.2f}\n"
        f"- Euler ZYX (deg): roll={euler_zyx_deg[0]:.2f}, pitch={euler_zyx_deg[1]:.2f}, yaw={euler_zyx_deg[2]:.2f}"
    )

    st.plotly_chart(plot_first_last_frames(sample), width="stretch")

    cam_choice = st.sidebar.selectbox("Camera for frustum", ["rgb", "slam-l", "slam-r"], index=0, key="frustum_cam")
    plot_opts = _plot_options_from_ui(sample)

    st.plotly_chart(
        plot_trajectory(
            sample,
            camera=cam_choice,
            crop_aabb=crop_aabb_from_semidense(sample, margin=float(crop_margin))
            if plot_opts["show_crop_bounds"]
            else None,
            **plot_opts,
        ),
        width="stretch",
    )


def _render_candidates_page(
    sample: EfmSnippetView, candidates: CandidateSamplingResult, cand_cfg: CandidateViewGeneratorConfig
) -> None:
    """Render the Candidate Poses page."""

    st.header("Candidate Poses")
    st.plotly_chart(
        plot_candidates(candidates["poses"], sample.mesh, title="Candidate positions"),
        width="stretch",
    )
    st.plotly_chart(
        plot_sampling_shell(
            shell_poses=candidates["shell_poses"],
            last_pose=sample.trajectory.final_pose,
            min_radius=float(cand_cfg.min_radius),
            max_radius=float(cand_cfg.max_radius),
            min_elev_deg=float(cand_cfg.min_elev_deg),
            max_elev_deg=float(cand_cfg.max_elev_deg),
            azimuth_full_circle=bool(cand_cfg.azimuth_full_circle),
            mesh=sample.mesh,
        ),
        width="stretch",
    )
    traj_positions = sample.trajectory.t_world_rig.matrix3x4[..., :3, 3].detach().cpu().numpy()
    st.plotly_chart(
        plot_candidate_frustums(
            poses=candidates["poses"],
            camera=sample.camera_rgb,
            mesh=sample.mesh,
            traj_positions=traj_positions,
        ),
        width="stretch",
    )


def _render_depth_page(depth_batch: CandidateDepthBatch) -> None:
    """Render the Candidate Renders page."""

    st.header("Candidate Renders")
    depths = depth_batch["depths"]
    indices = depth_batch["candidate_indices"].tolist()
    titles = [f"Candidate {i}" for i in indices]
    fig = depth_grid(depths, titles=titles, zmax=float(depths.max().item()))
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    """Entry point for `streamlit run -m oracle_rri.streamlit_app`."""

    st.set_page_config(page_title="NBV Explorer", layout="wide")
    console = Console.with_prefix("streamlit_app").set_verbose(True)

    st.sidebar.markdown("---")
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

    def _store(key: str, value: Any) -> None:
        st.session_state[key] = value

    def _get(key: str) -> Any:
        return st.session_state.get(key)

    sample = cast(EfmSnippetView | None, _get(STATE_KEYS["sample"]))
    candidates = cast(CandidateSamplingResult | None, _get(STATE_KEYS["candidates"]))
    depth_batch = cast(CandidateDepthBatch | None, _get(STATE_KEYS["depth"]))

    # Status flags
    cfg_changed = {"sample": False, "cand": False, "depth": False}

    if page == "Data":
        dataset_cfg = _dataset_config_from_ui()
        sample_idx = int(_get(STATE_KEYS["sample_idx"]) or 0)
        run_data = st.sidebar.button("Run / refresh data")
        next_sample = st.sidebar.button("Next sample")
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
            try:
                sample = _load_sample(dataset_cfg)
                # advance to indexed sample
                if sample_idx > 0:
                    dataset_iter = iter(dataset_cfg.setup_target())
                    for _ in range(sample_idx + 1):
                        sample = next(dataset_iter)
                _store(STATE_KEYS["sample"], sample)
                _store(STATE_KEYS["sample_cfg"], _cfg_to_dict(dataset_cfg))
                _store(STATE_KEYS["sample_idx"], sample_idx)
                # invalidate downstream caches
                _store(STATE_KEYS["candidates"], None)
                _store(STATE_KEYS["cand_cfg"], None)
                _store(STATE_KEYS["depth"], None)
                _store(STATE_KEYS["depth_cfg"], None)
                cfg_changed["sample"] = False
            except Exception as exc:  # pragma: no cover - UI feedback
                st.error(f"Failed to load dataset sample: {exc}")
                console.error(str(exc))
                return

        if sample is None:
            st.info("No sample loaded. Configure dataset and click 'Run / refresh data' (or 'Next sample').")
        else:
            cfg_state = _cfg_from_state(STATE_KEYS["sample_cfg"], AseEfmDatasetConfig)
            show_crop_box = st.sidebar.checkbox("Show crop bbox", value=True, key="show_crop_box")
            _render_data_page(sample, crop_margin=cfg_state.mesh_crop_margin_m if show_crop_box else None)
        return

    # Candidate page controls
    if page == "Candidate Poses":
        candidate_cfg = _candidate_config_from_ui(CandidateViewGeneratorConfig())
        run_cand = st.sidebar.button("Run / refresh candidates")
        cfg_changed["cand"] = _get(STATE_KEYS["cand_cfg"]) != _cfg_to_dict(candidate_cfg)

        if run_cand:
            if sample is None:
                st.warning("Load data first on the Data page, then run candidates.")
            else:
                generator = candidate_cfg.setup_target()
                candidates = generator.generate_from_typed_sample(sample)
                _store(STATE_KEYS["candidates"], candidates)
                _store(STATE_KEYS["cand_cfg"], _cfg_to_dict(candidate_cfg))
                # invalidate downstream depth
                _store(STATE_KEYS["depth"], None)
                _store(STATE_KEYS["depth_cfg"], None)
                cfg_changed["cand"] = False

        if candidates is None:
            st.info("No candidates yet. Configure generator and click 'Run / refresh candidates'.")
        else:
            _render_candidates_page(sample, candidates, candidate_cfg)
        if cfg_changed["cand"]:
            st.info("Candidate settings changed; rerun to refresh results.")
        return

    # Render page controls
    renderer_cfg = _renderer_config_from_ui(
        CandidateDepthRendererConfig(renderer=Pytorch3DDepthRendererConfig(device="cpu"))
    )
    render_depths = st.sidebar.checkbox("Compute depth renders", value=True)
    run_depth = st.sidebar.button("Run / refresh renders")
    cfg_changed["depth"] = _get(STATE_KEYS["depth_cfg"]) != _cfg_to_dict(renderer_cfg)

    if run_depth and render_depths:
        if sample is None or candidates is None:
            st.warning("Need data and candidates. Load data, then run candidates before rendering.")
        else:
            renderer = renderer_cfg.setup_target()
            try:
                depth_batch = renderer.render(sample, candidates)
                _store(STATE_KEYS["depth"], depth_batch)
                _store(STATE_KEYS["depth_cfg"], _cfg_to_dict(renderer_cfg))
                cfg_changed["depth"] = False
            except Exception as exc:  # pragma: no cover - rendering failures
                st.warning(f"Depth rendering failed: {exc}")
                console.warn(str(exc))

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

    # Page rendering -----------------------------------------------------
    if page == "Data":
        if sample is None:
            st.warning("No sample loaded yet. Click 'Run / refresh' to load one.")
        else:
            cfg_state = _cfg_from_state(STATE_KEYS["sample_cfg"], AseEfmDatasetConfig)
            _render_data_page(sample, crop_margin=cfg_state.mesh_crop_margin_m)
    elif page == "Candidate Poses":
        if candidates is None:
            st.warning("No candidates generated. Click 'Run / refresh' to generate with current settings.")
        else:
            cand_cfg_state = _cfg_from_state(STATE_KEYS["cand_cfg"], CandidateViewGeneratorConfig)
            _render_candidates_page(sample, candidates, cand_cfg_state)
    else:
        if depth_batch is None:
            if candidates is None:
                st.warning("No candidate renders yet. Click 'Run / refresh' on this page (compute depths checked).")
            else:
                st.info("Depth renders unavailable (rendering not run or disabled). Enable 'Compute depth renders'.")
        else:
            _render_depth_page(depth_batch)


if __name__ == "__main__":
    main()
