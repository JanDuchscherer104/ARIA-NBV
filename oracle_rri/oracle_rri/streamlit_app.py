"""Streamlit UI for inspecting ASE snippets, candidate poses, and depth renders."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Literal, TypedDict, cast

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
LOG_STATE_KEY = "nbv_console_logs"
MAX_LOG_LINES = 500
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

    nbv_console_logs: list[str]
    nbv_task_data: TaskState
    nbv_task_candidates: TaskState
    nbv_task_depth: TaskState
    py_console_globals: dict[str, Any]


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


@st.cache_resource(show_spinner=False)
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
    default: CandidateViewGeneratorConfig, ui: st.delta_generator.DeltaGenerator, super_fast: bool = False
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
    is_debug = ui.checkbox("generator is_debug", value=True)

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
    default: CandidateDepthRendererConfig, ui: st.delta_generator.DeltaGenerator, super_fast: bool = False
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
    is_debug = ui.checkbox("renderer is_debug", value=True)

    if backend_choice.startswith("pytorch3d"):
        faces_default = default.renderer.faces_per_pixel if isinstance(default.renderer, Pytorch3DDepthRendererConfig) else 1
        faces_pp = ui.slider("faces_per_pixel", 1, 4, faces_default)
        renderer_cfg: Pytorch3DDepthRendererConfig | Efm3dDepthRendererConfig = Pytorch3DDepthRendererConfig(
            device=renderer_device,
            zfar=float(zfar),
            faces_per_pixel=int(faces_pp),
            is_debug=is_debug,
        )
    else:
        chunk_rays = ui.slider("chunk_rays", 50_000, 500_000, 200_000, step=50_000)
        proxy_default = getattr(default.renderer, "add_proxy_walls", True)
        renderer_cfg = Efm3dDepthRendererConfig(
            device="cpu",
            zfar=float(zfar),
            add_proxy_walls=bool(proxy_default),
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


def _dataset_config_from_ui(ui: st.delta_generator.DeltaGenerator, *, super_fast: bool) -> AseEfmDatasetConfig:
    """Sidebar controls for dataset (data page only)."""

    ui.subheader("Dataset")
    # Fixed defaults (no user control).
    scene_ids: list[str] = []
    load_meshes = True
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
    is_debug = True
    require_mesh = ui.checkbox("require mesh", value=True)

    return AseEfmDatasetConfig(
        scene_ids=scene_ids,
        atek_variant="efm",
        load_meshes=load_meshes,
        mesh_simplify_ratio=mesh_ratio if mesh_ratio > 0 else None,
        mesh_crop_margin_m=mesh_crop_margin,
        mesh_crop_min_keep_ratio=mesh_keep_ratio,
        mesh_max_faces=int(mesh_max_faces),
        require_mesh=require_mesh,
        batch_size=batch_size,
        verbose=verbose,
        is_debug=is_debug,
        DEBUG_DEFAULTS=AseEfmDatasetConfig.DEBUG_DEFAULTS,
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
        default=["Semidense", "Scene bounds", "Frustum", "Mark start/finish", "Show walls (double-sided mesh)"],
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
        cand_fig = plot_candidates(
            candidates["poses"], sample.mesh, title="Candidate positions", center=last_center
        )
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
            frust_fig = plot_candidate_frustums(
                poses=candidates["poses"],
                camera=sample.camera_rgb,
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


def _append_log(line: str) -> None:
    """Append a line to the Streamlit log buffer with bounded length."""

    state = cast(dict[str, Any], st.session_state)
    buf_obj = state.get(LOG_STATE_KEY, [])
    buf: list[str] = list(buf_obj) if isinstance(buf_obj, list) else []
    buf.append(line)
    if len(buf) > MAX_LOG_LINES:
        buf = buf[-MAX_LOG_LINES:]
    state[LOG_STATE_KEY] = buf


def _tap_console(console: Console) -> Console:
    """Wrap console methods to also push lines into the UI log buffer."""

    def wrap_simple(method_name: str, level: str) -> None:
        orig = getattr(console, method_name)

        def wrapped(message: str) -> None:
            _append_log(f"{level}: {message}")
            orig(message)

        setattr(console, method_name, wrapped)

    wrap_simple("log", "INFO")
    wrap_simple("warn", "WARN")
    wrap_simple("error", "ERROR")
    wrap_simple("dbg", "DEBUG")

    orig_log_summary = console.log_summary

    def log_summary(label: str, value: Any) -> None:
        summary_line = f"{label}: {value}"
        _append_log(f"INFO: {summary_line}")
        orig_log_summary(label, value)

    console.log_summary = log_summary  # type: ignore[assignment]
    return console


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


def _render_log_panel() -> None:
    """Render console log buffer at bottom of each page."""

    st.divider()
    st.subheader("Console Logs")
    state = cast(dict[str, Any], st.session_state)
    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Clear logs", key="clear_logs"):
            state[LOG_STATE_KEY] = []
    log_lines_obj = state.get(LOG_STATE_KEY, [])
    log_lines = list(log_lines_obj) if isinstance(log_lines_obj, list) else []
    st.text_area(
        "Console output",
        value="\n".join(log_lines),
        height=250,
        label_visibility="collapsed",
        key="console_text_area",
    )

    with st.expander("Interactive Python console (runs locally)", expanded=False):
        code = st.text_area("Python code", key="py_console_code", height=140)
        if st.button("Run code", key="run_py_console"):
            buf = io.StringIO()
            err = io.StringIO()
            # reuse globals across runs
            globs = _state().setdefault("py_console_globals", {})
            globs.setdefault("st", st)
            globs.setdefault("torch", torch)
            with redirect_stdout(buf), redirect_stderr(err):
                try:
                    exec(code, globs)  # noqa: S102 - user-triggered eval within local app
                except Exception as exc:  # pragma: no cover - UI feedback
                    print(f"Exception: {exc}")
            out = err.getvalue() + buf.getvalue()
            if out.strip():
                for line in out.strip().splitlines():
                    _append_log(f"PYCON: {line}")
            st.toast("Code executed; output appended to console logs.")


def main() -> None:
    """Entry point for `streamlit run -m oracle_rri.streamlit_app`."""

    st.set_page_config(page_title="NBV Explorer", layout="wide")
    _state().setdefault(LOG_STATE_KEY, [])
    Console.set_sink(_append_log)
    _init_task_state()
    console = Console.with_prefix("streamlit_app").set_verbose(True)

    st.sidebar.markdown("---")
    super_fast = st.sidebar.checkbox("Super-fast debug mode", value=False, help="Use tiny meshes, 2 candidates, low-res renders.")
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
            dataset_cfg = _dataset_config_from_ui(st.sidebar, super_fast=super_fast)
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
        _render_log_panel()
        return

    # Candidate page controls
    if page == "Candidate Poses":
        with st.sidebar.form("cand_form"):
            candidate_cfg = _candidate_config_from_ui(CandidateViewGeneratorConfig(), st.sidebar, super_fast=super_fast)
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
        _render_log_panel()
        return

    # Render page controls
    with st.sidebar.form("depth_form"):
        renderer_cfg = _renderer_config_from_ui(
            CandidateDepthRendererConfig(renderer=Pytorch3DDepthRendererConfig(device="cpu")),
            st.sidebar,
            super_fast=super_fast,
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

    _render_log_panel()
    return


if __name__ == "__main__":
    main()
