"""Shared Streamlit controls for snippet-level 3D Plotly scene views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import streamlit as st

from ..data_handling import EfmSnippetView
from ..utils.data_plotting import SnippetPlotBuilder

PlotBuilderT = TypeVar("PlotBuilderT", bound=SnippetPlotBuilder)


@dataclass(frozen=True, slots=True)
class Scene3DPlotDefaults:
    """Default visibility and style choices for a snippet 3D scene view."""

    show_mesh: bool = True
    mesh_opacity: float = 0.35
    show_trajectory: bool = True
    show_scene_bounds: bool = True
    show_crop_bounds: bool = True
    show_frustum: bool = True
    frustum_scale: float = 1.0
    mark_first_last: bool = True
    show_gt_obbs: bool = False
    semidense_mode: str = "all frames"
    max_sem_points: int = 20000


@dataclass(slots=True)
class Scene3DPlotOptions:
    """Resolved options for a snippet 3D scene view."""

    show_mesh: bool
    mesh_opacity: float
    show_trajectory: bool
    show_scene_bounds: bool
    show_crop_bounds: bool
    show_frustum: bool
    frustum_frame_indices: list[int]
    frustum_scale: float
    mark_first_last: bool
    show_gt_obbs: bool
    gt_timestamp: str | int | None
    semidense_mode: str
    max_sem_points: int


DATA_SCENE_DEFAULTS = Scene3DPlotDefaults()
"""Default 3D scene choices for the Data page."""

ROLLOUT_SCENE_DEFAULTS = Scene3DPlotDefaults(
    show_mesh=True,
    mesh_opacity=0.18,
    show_trajectory=False,
    show_scene_bounds=False,
    show_crop_bounds=False,
    show_frustum=False,
    frustum_scale=0.7,
    mark_first_last=False,
    show_gt_obbs=False,
    semidense_mode="off",
    max_sem_points=20000,
)
"""Minimal evidence-view defaults for Counterfactual Rollouts."""


def scene_plot_options_ui(
    sample: EfmSnippetView,
    *,
    key_prefix: str,
    title: str = "3D scene view",
    defaults: Scene3DPlotDefaults = DATA_SCENE_DEFAULTS,
) -> tuple[str, Scene3DPlotOptions]:
    """Render shared 3D scene controls and return the chosen camera/options."""

    st.subheader(title)
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        show_mesh = st.checkbox(
            "Show mesh",
            value=defaults.show_mesh,
            key=f"{key_prefix}_mesh_enable",
        )
        mesh_opacity = float(
            st.slider(
                "Mesh opacity",
                min_value=0.02,
                max_value=1.0,
                value=float(defaults.mesh_opacity),
                step=0.01,
                key=f"{key_prefix}_mesh_opacity",
                disabled=not show_mesh,
            )
        )
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
        frustum_scale = float(
            st.slider(
                "Frustum scale",
                min_value=0.1,
                max_value=2.0,
                value=float(defaults.frustum_scale),
                step=0.05,
                key=f"{key_prefix}_frustum_scale",
            )
        )

    with opt_col2:
        semidense_options = ["off", "all frames", "last frame only"]
        semidense_mode = st.radio(
            "Semi-dense points",
            options=semidense_options,
            horizontal=True,
            index=semidense_options.index(defaults.semidense_mode),
            key=f"{key_prefix}_sem_mode",
        )
        max_sem_points = int(
            st.slider(
                "Max semi-dense points",
                min_value=1000,
                max_value=200000,
                value=int(defaults.max_sem_points),
                step=1000,
                key=f"{key_prefix}_max_sem_points",
                disabled=semidense_mode == "off",
            )
        )
        show_trajectory = st.checkbox(
            "Show source trajectory",
            value=defaults.show_trajectory,
            key=f"{key_prefix}_trajectory",
        )
        show_scene_bounds = st.checkbox(
            "Show scene bounds",
            value=defaults.show_scene_bounds,
            key=f"{key_prefix}_scene_bounds",
        )
        show_crop_bounds = st.checkbox(
            "Show crop bbox",
            value=defaults.show_crop_bounds,
            key=f"{key_prefix}_crop_bounds",
        )
        show_frustum = st.checkbox(
            "Show source frustum",
            value=defaults.show_frustum,
            key=f"{key_prefix}_frustum_enable",
        )
        mark_first_last = st.checkbox(
            "Mark source start/finish",
            value=defaults.mark_first_last,
            key=f"{key_prefix}_mark_first_last",
            disabled=not show_trajectory,
        )
        show_gt_obbs = st.checkbox(
            "Show all GT OBBs",
            value=defaults.show_gt_obbs,
            key=f"{key_prefix}_gt_obbs",
        )

    gt_ts = None
    sample_gt = getattr(sample, "gt", None)
    gt_timestamps = getattr(sample_gt, "timestamps", []) if sample_gt is not None else []
    if show_gt_obbs and gt_timestamps:
        gt_ts = st.selectbox(
            "GT OBB timestamp",
            options=gt_timestamps,
            index=0,
            key=f"{key_prefix}_gt_ts",
        )

    return (
        frustum_camera,
        Scene3DPlotOptions(
            show_mesh=bool(show_mesh),
            mesh_opacity=float(mesh_opacity),
            show_trajectory=bool(show_trajectory),
            show_scene_bounds=bool(show_scene_bounds),
            show_crop_bounds=bool(show_crop_bounds),
            show_frustum=bool(show_frustum),
            frustum_frame_indices=[int(i) for i in frustum_frame_indices] if frustum_frame_indices else [0],
            frustum_scale=float(frustum_scale),
            mark_first_last=bool(mark_first_last),
            show_gt_obbs=bool(show_gt_obbs),
            gt_timestamp=gt_ts,
            semidense_mode=str(semidense_mode),
            max_sem_points=int(max_sem_points),
        ),
    )


def apply_scene_plot_options(
    builder: PlotBuilderT,
    sample: EfmSnippetView,
    *,
    camera: str,
    options: Scene3DPlotOptions,
) -> PlotBuilderT:
    """Apply shared scene options to a `SnippetPlotBuilder` subclass."""

    if options.show_mesh:
        builder.add_mesh(opacity=options.mesh_opacity)

    if options.show_trajectory:
        builder.add_trajectory(mark_first_last=options.mark_first_last, show=True)

    if options.semidense_mode != "off":
        builder.add_semidense(
            max_points=options.max_sem_points,
            last_frame_only=(options.semidense_mode == "last frame only"),
        )

    if options.show_frustum:
        builder.add_frusta(
            camera=camera,
            frame_indices=options.frustum_frame_indices,
            scale=options.frustum_scale,
            include_axes=True,
            include_center=True,
        )

    if options.show_scene_bounds:
        builder.add_bounds_box(name="Scene bounds", color="gray", dash="dash", width=2)

    if options.show_gt_obbs:
        builder.add_gt_obbs(camera=camera, timestamp=options.gt_timestamp)

    if options.show_crop_bounds:
        crop_bounds = getattr(sample, "crop_bounds", None)
        if crop_bounds is not None:
            crop_aabb = tuple(bound.detach().cpu().numpy() for bound in crop_bounds)
            builder.add_bounds_box(
                name="Crop bounds",
                color="orange",
                dash="solid",
                width=3,
                aabb=crop_aabb,
            )

    return builder


__all__ = [
    "DATA_SCENE_DEFAULTS",
    "ROLLOUT_SCENE_DEFAULTS",
    "Scene3DPlotDefaults",
    "Scene3DPlotOptions",
    "apply_scene_plot_options",
    "scene_plot_options_ui",
]
