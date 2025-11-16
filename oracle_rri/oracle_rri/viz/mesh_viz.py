"""Mesh/point/pose visualization helpers for oracle RRI.

Supports Plotly rendering, minimal Streamlit embedding, and mesh-aware pose
sampling that avoids collisions with the current GT mesh. All transforms follow
the Aria convention (x-left, y-up, z-forward) and use the ``T_A_B`` notation
(``PoseTW`` that maps points from frame B into frame A).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import trimesh
from efm3d.aria import PoseTW
from pydantic import Field

from ..utils import BaseConfig, Console


@dataclass(slots=True)
class PlotlyPose:
    """Pose container for Plotly visualization.

    Attributes:
        position: XYZ in world frame, shape (3,).
        direction: Unit Z axis in world frame, shape (3,). Defaults to [0, 0, 1].
        color: RGB hex color (e.g. ``"#ff0000"``).
        name: Legend label.
    """

    position: np.ndarray
    direction: np.ndarray = np.array([0.0, 0.0, 1.0])
    color: str = "#ff0000"
    name: str = "pose"


def _segment_intersects_mesh(mesh: trimesh.Trimesh, start: np.ndarray, end: np.ndarray) -> bool:
    """Check if the segment start→end intersects the mesh."""
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return False
    direction_unit = direction / length
    try:
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=[start],
            ray_directions=[direction_unit],
            multiple_hits=False,
        )
    except BaseException:  # pragma: no cover - backend specific errors
        return False

    if len(locations) == 0:
        return False
    # Accept only hits that lie between start and end.
    distances = np.linalg.norm(locations - start, axis=1)
    return bool(np.any(distances <= length))


def _point_is_inside(mesh: trimesh.Trimesh, point: np.ndarray) -> bool:
    """Check if a point is inside the mesh volume."""
    try:
        return bool(mesh.contains([point])[0])
    except BaseException:
        signed = mesh.nearest.signed_distance([point])
        return bool((signed < 0).any())


def sample_collision_free_poses(
    mesh: trimesh.Trimesh,
    last_pose: PoseTW,
    num_samples: int = 16,
    max_tries: int = 256,
    margin: float = 0.2,
) -> list[PoseTW]:
    """Sample poses inside the room bounds without intersecting the mesh.

    Args:
        mesh: GT mesh in world frame.
        last_pose: Current device pose ``T_world_device``.
        num_samples: Number of candidate poses to return.
        max_tries: Max random draws before giving up.
        margin: Extra margin (meters) around mesh bounds for sampling.

    Returns:
        List of ``PoseTW`` candidates with the same orientation as ``last_pose``,
        filtered to collision-free straight-line motion from ``last_pose``.
    """
    console = Console.with_prefix("mesh_viz", "sample_collision_free_poses")
    last_t = last_pose.translation().cpu().numpy()
    bounds_min, bounds_max = mesh.bounds
    bounds_min -= margin
    bounds_max += margin

    candidates: list[PoseTW] = []
    tries = 0
    while len(candidates) < num_samples and tries < max_tries:
        tries += 1
        sample = np.random.uniform(bounds_min, bounds_max)
        if _point_is_inside(mesh, sample):
            continue
        if _segment_intersects_mesh(mesh, last_t, sample):
            continue
        pose = last_pose.clone()
        pose._data = pose._data.clone()
        pose._data[..., 9:12] = pose._data.new_tensor(sample)  # update translation in matrix3x4
        candidates.append(pose)

    console.dbg(f"Generated {len(candidates)} / {num_samples} candidates (tries={tries})")
    return candidates


def plot_mesh_scene(
    mesh: trimesh.Trimesh,
    points: np.ndarray | None = None,
    poses: Sequence[PlotlyPose] | None = None,
    title: str = "Mesh Scene",
) -> go.Figure:
    """Create a Plotly 3D figure with mesh, points, and poses.

    Args:
        mesh: GT mesh in world frame.
        points: Optional points in world frame ``[N,3]``.
        poses: Optional list of ``PlotlyPose`` for camera/device markers.
        title: Figure title.
    """
    # TODO: doesn't trimesh offer direct Plotly export?
    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            opacity=0.35,
            color="lightgray",
            name="GT Mesh",
            hoverinfo="skip",
        )
    )

    if points is not None:
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={"size": 2, "color": points[:, 2], "colorscale": "Viridis"},
                name="Points",
            )
        )

    if poses:
        for pose in poses:
            p = pose.position
            d = pose.direction
            fig.add_trace(
                go.Scatter3d(
                    x=[p[0], p[0] + d[0] * 0.3],
                    y=[p[1], p[1] + d[1] * 0.3],
                    z=[p[2], p[2] + d[2] * 0.3],
                    mode="lines+markers",
                    line={"color": pose.color, "width": 6},
                    marker={"size": 6, "color": pose.color},
                    name=pose.name,
                )
            )

    fig.update_layout(
        title=title,
        scene={"aspectmode": "data"},
        legend={"itemsizing": "constant"},
        height=720,
    )
    return fig


class StreamlitMeshViewerConfig(BaseConfig[None]):
    """Config for spawning a minimal Streamlit mesh viewer."""

    target: type[None] = Field(default=None, exclude=True)
    title: str = "Mesh Viewer"
    point_size: int = 2
    show_legend: bool = True


def streamlit_mesh_viewer(
    mesh: trimesh.Trimesh,
    points: np.ndarray | None = None,
    poses: Sequence[PlotlyPose] | None = None,
    config: StreamlitMeshViewerConfig | None = None,
) -> None:
    """Render a Streamlit page with mesh + points + poses using Plotly."""
    import streamlit as st
    from plotly.errors import PlotlyError

    cfg = config or StreamlitMeshViewerConfig()
    st.set_page_config(page_title=cfg.title, layout="wide")
    st.title(cfg.title)

    try:
        fig = plot_mesh_scene(mesh, points, poses, title=cfg.title)
        if not cfg.show_legend:
            fig.update_layout(showlegend=False)
        if points is not None:
            for trace in fig.data:
                if isinstance(trace, go.Scatter3d) and trace.name == "Points":
                    trace.marker.size = cfg.point_size

        st.plotly_chart(fig, use_container_width=True)
    except PlotlyError as exc:  # pragma: no cover
        st.error(f"Plotly error: {exc}")
    except Exception as exc:  # pragma: no cover
        st.error(f"Unexpected error: {exc}")
        st.exception(exc)
