"""Plotting utilities for the Streamlit data view.

Conventions (efm3d / ATEK):
- Camera frame is **LUF** (x left, y up, z forward) per Project Aria docs; world is Z-up (gravity = [0,0,-g]).
- Pose: T_world_cam = T_world_rig @ T_camera_rig.inverse() (matches efm3d.render_frustum).
- Frustum geometry is built directly in world+LUF (no display-frame tweaks); Plotly merely visualises it.
- Images are rotated -90° for display, but the underlying 3D poses stay in LUF.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import torch
import trimesh
from efm3d.aria.camera import CameraTW
from efm3d.aria.pose import PoseTW
from matplotlib import colormaps
from plotly.subplots import make_subplots

from oracle_rri.data import EfmCameraView, EfmSnippetView, EfmTrajectoryView
from oracle_rri.utils import Console
from oracle_rri.utils.frames import world_from_rig_camera_pose

console = Console.with_prefix("plotting")


def _frustum_segments(
    cam: CameraTW,
    pose_world_cam: PoseTW,
    scale: float = 1.0,
) -> list[np.ndarray]:
    """Return frustum wireframe segments in world frame using CameraTW.unproject (LUF pose).

    The four image-plane corners are first unprojected in the camera's intrinsic
    frame, then re-ordered into a canonical sequence

        0: top-left
        1: bottom-left
        2: bottom-right
        3: top-right

    so that:

        - left  edge  is (0, 1)
        - bottom edge is (1, 2)
        - right edge is (2, 3)
        - top   edge is (3, 0)

    This ordering is independent of the initial pixel-corner indexing and
    robust to the sign conventions used inside CameraTW.project/unproject.
    A small "hat" is added above the top edge by offsetting along the
    physical camera +Y axis (LUF), optionally undoing a display roll.
    """
    # Construct four pixel corners on an inscribed square
    c = cam.c.squeeze(0)  # (2,)
    rs = cam.valid_radius.squeeze(0) / np.sqrt(2.0)  # inscribed square radius per axis

    corners_px = (
        torch.stack(
            [
                c + torch.tensor([-rs[0], -rs[1]]),  # TL
                c + torch.tensor([-rs[0], rs[1]]),  # BL
                c + torch.tensor([rs[0], rs[1]]),  # BR
                c + torch.tensor([rs[0], -rs[1]]),  # TR
            ],
            dim=0,
        )
        .to(c)
        .unsqueeze(0)
    )  # 1 x 4 x 2

    # Unproject to camera-frame rays (LUF) and normalise
    rays_cam = cam.unproject(corners_px)[0].squeeze(0)  # 4 x 3
    rays_cam = torch.nn.functional.normalize(rays_cam, dim=-1, eps=1e-6)
    frustum_cam = rays_cam * scale  # 4 x 3

    # Transform to world frame and build triangular faces
    frustum_np = pose_world_cam.transform(frustum_cam).detach().cpu().numpy()  # (4, 3)
    center_np = pose_world_cam.t.detach().cpu().numpy()  # (3,)

    segments: list[np.ndarray] = []
    ring_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, j in ring_edges:
        segments.append(np.vstack([center_np, frustum_np[i], frustum_np[j]]))

    # Top-edge "hat": offset along physical camera +Y
    edge_top = (frustum_np[2], frustum_np[1])
    mid_top = 0.5 * (edge_top[0] + edge_top[1])

    # camera +Y = "up" in LUF
    up_dir = pose_world_cam.R @ torch.tensor([0.0, 1.0, 0.0], device=cam.device, dtype=cam.dtype)

    up_dir_np = up_dir.detach().cpu().numpy()
    up_dir_np = up_dir_np / np.linalg.norm(up_dir_np)

    hat_h = 0.1 * scale  # height of the cue relative to frustum scale
    top_pts = np.array(
        [
            edge_top[0] + hat_h * up_dir_np,
            mid_top + 2.5 * hat_h * up_dir_np,
            edge_top[1] + hat_h * up_dir_np,
        ]
    )
    segments.append(top_pts)

    return segments


def _bbox_edges(min_pt: np.ndarray, max_pt: np.ndarray) -> list[np.ndarray]:
    corners = np.array(
        [
            [min_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], max_pt[1], min_pt[2]],
            [min_pt[0], max_pt[1], min_pt[2]],
            [min_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], max_pt[1], max_pt[2]],
            [min_pt[0], max_pt[1], max_pt[2]],
        ]
    )
    edges_idx = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return [corners[list(pair)] for pair in edges_idx]


def _proxy_box_mesh(bounds: np.ndarray) -> "trimesh.Trimesh":
    import trimesh

    vmin, vmax = bounds
    extents = vmax - vmin
    box = trimesh.creation.box(
        extents=extents, transform=trimesh.transformations.translation_matrix(vmin + extents / 2.0)
    )
    return box


def _mesh_to_plotly(
    mesh,
    *,
    double_sided: bool = False,
    add_proxy_walls: bool = True,
    occupancy_extent: np.ndarray | None = None,
) -> list[go.Mesh3d]:
    """Convert a Trimesh to Plotly Mesh3d traces.

    double_sided=True duplicates faces with reversed winding so interior walls remain
    visible when the camera is inside the mesh (Plotly only shades front faces).
    add_proxy_walls=True adds missing wall planes from the mesh bounds when coverage is low.
    """

    vertices = mesh.vertices
    faces = mesh.faces
    traces: list[go.Mesh3d] = []

    base_trace = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color="lightgray",
        opacity=0.35,
        flatshading=True,
        lighting={"ambient": 1.0, "diffuse": 0.3, "specular": 0.0, "fresnel": 0.0, "roughness": 1.0},
        name="GT Mesh",
        hoverinfo="skip",
        showscale=False,
    )
    traces.append(base_trace)

    if double_sided:
        faces_rev = faces[:, [0, 2, 1]]
        traces.append(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces_rev[:, 0],
                j=faces_rev[:, 1],
                k=faces_rev[:, 2],
                color="#c0c0c0",
                opacity=0.35,
                flatshading=True,
                lighting={"ambient": 1.0, "diffuse": 0.0, "specular": 0.0, "fresnel": 0.0, "roughness": 1.0},
                name="GT Mesh (inner)",
                hoverinfo="skip",
                showscale=False,
            )
        )

    if add_proxy_walls:
        # Detect missing wall coverage (very low area near bounds); add thin box faces if needed.
        centers = mesh.triangles_center
        areas = mesh.area_faces
        if occupancy_extent is not None and occupancy_extent.shape == (2, 3):
            vmin, vmax = occupancy_extent
        else:
            vmin, vmax = mesh.bounds
        extents = vmax - vmin
        desired = {
            "xmin": extents[1] * extents[2],
            "xmax": extents[1] * extents[2],
            "ymin": extents[0] * extents[2],
            "ymax": extents[0] * extents[2],
            "zmin": extents[0] * extents[1],
            "zmax": extents[0] * extents[1],
        }
        eps = 0.05
        planes = {
            "xmin": centers[:, 0] <= vmin[0] + eps,
            "xmax": centers[:, 0] >= vmax[0] - eps,
            "ymin": centers[:, 1] <= vmin[1] + eps,
            "ymax": centers[:, 1] >= vmax[1] - eps,
            "zmin": centers[:, 2] <= vmin[2] + eps,
            "zmax": centers[:, 2] >= vmax[2] - eps,
        }
        coverage = {k: areas[m].sum() for k, m in planes.items()}
        missing = [k for k, cov in coverage.items() if desired[k] > 0 and cov < 0.2 * desired[k]]
        if missing:
            proxy_bounds = occupancy_extent if occupancy_extent is not None else mesh.bounds
            proxy = _proxy_box_mesh(proxy_bounds)
            traces.append(
                go.Mesh3d(
                    x=proxy.vertices[:, 0],
                    y=proxy.vertices[:, 1],
                    z=proxy.vertices[:, 2],
                    i=proxy.faces[:, 0],
                    j=proxy.faces[:, 1],
                    k=proxy.faces[:, 2],
                    color="#b0c4de",
                    opacity=0.3,
                    flatshading=True,
                    lighting={"ambient": 1.0, "diffuse": 0.0, "specular": 0.0, "fresnel": 0.0, "roughness": 1.0},
                    name="Proxy walls",
                    hoverinfo="skip",
                    showscale=False,
                )
            )

    return traces


def _aligned_pose_world_cam(cam: EfmCameraView, traj: EfmTrajectoryView, frame_idx: int) -> PoseTW:
    """Nearest rig-aligned camera pose in **world+LUF** (no display corrections)."""

    cam_ts = np.atleast_1d(cam.time_ns.cpu().numpy())
    traj_ts = np.atleast_1d(traj.time_ns.cpu().numpy())
    if cam_ts.size == 0 or traj_ts.size == 0:
        return PoseTW.from_Rt(torch.eye(3), torch.zeros(3))
    traj_idx = int(np.argmin(np.abs(traj_ts - cam_ts[frame_idx])))

    t_world_rig = traj.t_world_rig[traj_idx]
    # Keep the physical pose in LUF; Plotly should render the true sensor frame.
    return world_from_rig_camera_pose(t_world_rig, cam.calib, frame_idx)


def _apply_display_roll(pose_world_cam: PoseTW, roll_rad: float = np.pi / 2) -> PoseTW:
    """Optionally roll the pose about its +Z (forward) axis for display-only alignment."""

    if roll_rad == 0.0:
        return pose_world_cam
    c, s = np.cos(roll_rad), np.sin(roll_rad)
    r_roll = torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        device=pose_world_cam.R.device,
        dtype=pose_world_cam.R.dtype,
    )
    r_disp = pose_world_cam.R @ r_roll
    return PoseTW.from_Rt(r_disp, pose_world_cam.t)


class SnippetPlotBuilder:
    """Composable builder for mesh/points/frusta visuals with optional trajectory."""

    def __init__(self, *, scene_ranges: dict, title: str, height: int = 900):
        self.fig = go.Figure()
        self.scene_ranges = scene_ranges
        self.title = title
        self.height = height

    def add_mesh(
        self,
        mesh,
        *,
        double_sided: bool = False,
        occupancy_extent: np.ndarray | None = None,
    ) -> "SnippetPlotBuilder":
        if mesh is None:
            return self
        for trace in _mesh_to_plotly(mesh, double_sided=double_sided, occupancy_extent=occupancy_extent):
            self.fig.add_trace(trace)
        return self

    def add_semidense(self, pts_np: np.ndarray, *, name: str = "Semidense points") -> "SnippetPlotBuilder":
        if pts_np.size == 0:
            return self
        self.fig.add_trace(
            go.Scatter3d(
                x=pts_np[:, 0],
                y=pts_np[:, 1],
                z=pts_np[:, 2],
                mode="markers",
                marker={"size": 2, "color": pts_np[:, 2], "colorscale": "Viridis", "opacity": 0.5},
                name=name,
            )
        )
        return self

    def add_trajectory(
        self,
        traj_positions: np.ndarray,
        *,
        mark_first_last: bool = True,
        first_color: str = "green",
        last_color: str = "red",
        show: bool = True,
    ) -> "SnippetPlotBuilder":
        if not show:
            return self
        traj_x, traj_y, traj_z = traj_positions.T
        self.fig.add_trace(
            go.Scatter3d(
                x=traj_x,
                y=traj_y,
                z=traj_z,
                mode="lines+markers",
                marker={"size": 3, "color": "black"},
                line={"color": "black", "width": 4},
                name="Trajectory",
            )
        )
        if mark_first_last and traj_positions.shape[0] > 0:
            first = traj_positions[0]
            last = traj_positions[-1]
            self.fig.add_trace(
                go.Scatter3d(
                    x=[first[0]],
                    y=[first[1]],
                    z=[first[2]],
                    mode="markers",
                    marker={"size": 2, "color": first_color, "symbol": "diamond"},
                    name="Start",
                )
            )
            if traj_positions.shape[0] > 1:
                self.fig.add_trace(
                    go.Scatter3d(
                        x=[last[0]],
                        y=[last[1]],
                        z=[last[2]],
                        mode="markers",
                        marker={"size": 2, "color": last_color, "symbol": "x"},
                        name="Final",
                    )
                )
        return self

    def add_frustum(
        self,
        cam: CameraTW,
        pose_world_cam: PoseTW,
        *,
        scale: float = 1.0,
        name: str = "Frustum",
    ) -> "SnippetPlotBuilder":
        frustum_segments = _frustum_segments(cam, pose_world_cam, scale=scale)
        for idx, seg in enumerate(frustum_segments):
            self.fig.add_trace(
                go.Scatter3d(
                    x=seg[:, 0],
                    y=seg[:, 1],
                    z=seg[:, 2],
                    mode="lines",
                    line={"color": "red", "width": 4},
                    name=name if idx == 0 else None,
                    showlegend=idx == 0,
                )
            )

        return self

    def add_camera_axes(self, cam_center: np.ndarray, cam_axes: np.ndarray) -> "SnippetPlotBuilder":
        axis_colors = ["red", "green", "blue"]
        for axis, color in enumerate(axis_colors):
            axis_end = cam_center + cam_axes[axis] * 0.4
            self.fig.add_trace(
                go.Scatter3d(
                    x=[cam_center[0], axis_end[0]],
                    y=[cam_center[1], axis_end[1]],
                    z=[cam_center[2], axis_end[2]],
                    mode="lines",
                    line={"color": color, "width": 8},
                    name="Camera axes" if axis == 0 else None,
                    showlegend=axis == 0,
                )
            )
        return self

    def add_camera_center(
        self, cam_center: np.ndarray, *, color: str = "red", symbol: str = "diamond"
    ) -> "SnippetPlotBuilder":
        self.fig.add_trace(
            go.Scatter3d(
                x=[cam_center[0]],
                y=[cam_center[1]],
                z=[cam_center[2]],
                mode="markers",
                marker={"size": 10, "color": color, "symbol": symbol},
                name="Camera center",
            )
        )
        return self

    def add_bounds_box(
        self,
        min_pt: np.ndarray,
        max_pt: np.ndarray,
        *,
        name: str,
        color: str = "gray",
        dash: str = "dash",
        width: int = 2,
    ) -> "SnippetPlotBuilder":
        for idx, seg in enumerate(_bbox_edges(min_pt, max_pt)):
            self.fig.add_trace(
                go.Scatter3d(
                    x=seg[:, 0],
                    y=seg[:, 1],
                    z=seg[:, 2],
                    mode="lines",
                    line={"color": color, "width": width, "dash": dash},
                    name=name if idx == 0 else None,
                    showlegend=idx == 0,
                    hoverinfo="skip",
                )
            )
        return self

    def add_gt_obbs(
        self,
        ts_world_object: np.ndarray,
        object_dimensions: np.ndarray,
        *,
        color: str = "purple",
        name: str = "GT OBBs",
        opacity: float = 0.35,
    ) -> "SnippetPlotBuilder":
        """Add oriented boxes defined by world←object poses and dimensions."""

        if ts_world_object.size == 0 or object_dimensions.size == 0:
            return self

        def _obb_corners(rot: np.ndarray, t: np.ndarray, dims: np.ndarray) -> np.ndarray:
            half = dims / 2.0
            signs = np.array(
                [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]
            )
            local = signs * half
            return (rot @ local.T).T + t

        for corners in (
            _obb_corners(ts[:3, :3], ts[:3, 3], dims)
            for ts, dims in zip(ts_world_object, object_dimensions, strict=False)
        ):
            edges_idx = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ]
            for idx, (i0, i1) in enumerate(edges_idx):
                seg = np.vstack([corners[i0], corners[i1]])
                self.fig.add_trace(
                    go.Scatter3d(
                        x=seg[:, 0],
                        y=seg[:, 1],
                        z=seg[:, 2],
                        mode="lines",
                        line={"color": color, "width": 3},
                        name=name if idx == 0 else None,
                        showlegend=idx == 0,
                        opacity=opacity,
                    )
                )
        return self

    def finalize(self) -> go.Figure:
        self.fig.update_layout(title=self.title, scene=dict(aspectmode="data", **self.scene_ranges), height=self.height)
        return self.fig


def _to_uint8_image(img: torch.Tensor, *, rotate_k: int = 0) -> np.ndarray:
    """Convert tensor image (C,H,W or H,W) to uint8 HWC."""
    arr = img.detach().cpu()
    if arr.ndim == 2:
        arr = arr.unsqueeze(0)
    if arr.shape[0] == 1:
        arr = arr.repeat(3, 1, 1)
    arr = arr.permute(1, 2, 0).clamp(0, 1).mul(255).to(torch.uint8).numpy()
    if rotate_k:
        arr = np.rot90(arr, k=rotate_k)
    return arr


def _depth_to_color(depth: torch.Tensor, *, rotate_k: int = 0, percentile: float = 99.5) -> np.ndarray:
    """Colorise a single depth/distance map to uint8 RGB using a perceptual colormap."""

    depth_np = depth.detach().cpu().squeeze().numpy()
    finite_mask = np.isfinite(depth_np)
    if not finite_mask.any():
        rgb = np.zeros(depth_np.shape + (3,), dtype=np.uint8)
    else:
        finite_vals = depth_np[finite_mask]
        vmin = max(np.nanmin(finite_vals), 0.0)
        vmax = np.nanpercentile(finite_vals, percentile)
        if vmax <= vmin:
            vmax = vmin + 1e-3
        norm = np.clip((depth_np - vmin) / (vmax - vmin), 0.0, 1.0)
        cmap = colormaps.get_cmap("viridis")
        rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    if rotate_k:
        rgb = np.rot90(rgb, k=rotate_k)
    return rgb


def _semantic_to_color(mask: torch.Tensor, *, rotate_k: int = 0) -> np.ndarray:
    """Map integer semantic/instance labels to RGB colours."""

    mask_np = mask.detach().cpu().squeeze().numpy()
    if mask_np.ndim != 2:
        mask_np = np.reshape(mask_np, mask_np.shape[-2:])
    mask_np = np.nan_to_num(mask_np, nan=-1.0).astype(np.int64)
    if rotate_k:
        mask_np = np.rot90(mask_np, k=rotate_k)

    palette = (colormaps.get_cmap("tab20", 256)(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    mapped = palette[np.mod(mask_np, 256)]
    mapped[mask_np < 0] = 0
    return mapped


class FrameGridBuilder:
    """Builder for image grids (2D modalities)."""

    def __init__(self, rows: int, cols: int, *, titles: list[str], height: int, width: int, title: str):
        self.fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, specs=[[{"type": "image"}] * cols] * rows)
        self.height = height
        self.width = width
        self.title = title

    def add_image(self, img: np.ndarray, *, row: int, col: int) -> "FrameGridBuilder":
        self.fig.add_trace(go.Image(z=img), row=row, col=col)
        return self

    def finalize(self) -> go.Figure:
        self.fig.update_layout(height=self.height, width=self.width, title_text=self.title)
        return self.fig


def collect_frame_modalities(
    sample: EfmSnippetView,
    *,
    include_depth: bool = True,
    include_semantic: bool = True,
    rotate_k: int = -1,
) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], list[str]]:
    """Collect first/last-frame visualisations for available modalities."""

    missing: list[str] = []
    missing_set: set[str] = set()
    modalities: list[tuple[str, np.ndarray, np.ndarray]] = []

    cams: list[tuple[str, EfmCameraView]] = [
        ("RGB", sample.camera_rgb),
        ("SLAM-L", sample.camera_slam_left),
        ("SLAM-R", sample.camera_slam_right),
    ]

    for name, cam in cams:
        if cam.images.numel() == 0:
            if name not in missing_set:
                missing.append(name)
                missing_set.add(name)
            continue
        modalities.append(
            (
                name,
                _to_uint8_image(cam.images[0], rotate_k=rotate_k),
                _to_uint8_image(cam.images[-1], rotate_k=rotate_k),
            )
        )

    if include_depth:
        depth_streams = [
            ("Depth (RGB)", sample.camera_rgb.distance_m),
            ("Depth (SLAM-L)", sample.camera_slam_left.distance_m),
            ("Depth (SLAM-R)", sample.camera_slam_right.distance_m),
        ]
        for name, depth in depth_streams:
            if depth is None or depth.numel() == 0:
                if name not in missing_set:
                    missing.append(name)
                    missing_set.add(name)
                    console.dbg(f"{name} missing in sample {sample.scene_id}/{sample.snippet_id}")
                continue
            modalities.append(
                (name, _depth_to_color(depth[0], rotate_k=rotate_k), _depth_to_color(depth[-1], rotate_k=rotate_k))
            )

    return modalities, missing


def _rotate_uv(uv: np.ndarray, width: int, height: int, rotate_k: int) -> np.ndarray:
    """Rotate pixel coords to match an image rotated by np.rot90."""

    if rotate_k % 4 == 0:
        return uv
    k = rotate_k % 4
    u, v = uv[:, 0], uv[:, 1]
    if k == 1:  # 90° CCW
        u_new = v
        v_new = width - 1 - u
    elif k == 2:  # 180°
        u_new = width - 1 - u
        v_new = height - 1 - v
    elif k == 3:  # 90° CW (equivalent to k=-1)
        u_new = height - 1 - v
        v_new = u
    else:  # pragma: no cover
        u_new, v_new = u, v
    return np.stack([u_new, v_new], axis=-1)


def project_pointcloud_on_frame(
    *,
    img: torch.Tensor,
    cam: CameraTW,
    pose_world_cam: PoseTW,
    points_world: torch.Tensor,
    rotate_k: int = -1,
    max_points: int | None = 20000,
    title: str = "Point cloud overlay",
    point_size: int = 5,
) -> go.Figure:
    """Project 3D points into image plane and overlay on the frame."""

    if points_world.numel() == 0:
        return go.Figure()

    img_np = _to_uint8_image(img, rotate_k=rotate_k)
    h, w = img_np.shape[:2]

    pts = points_world
    if pts.ndim == 3:
        pts = pts.reshape(-1, pts.shape[-1])
    finite = torch.isfinite(pts).all(dim=-1)
    pts = pts[finite]
    if pts.numel() == 0:
        return go.Figure()
    if max_points is not None and pts.shape[0] > max_points:
        idx = torch.randperm(pts.shape[0], device=pts.device)[:max_points]
        pts = pts[idx]

    pts_cam = pose_world_cam.inverse().transform(pts)
    pts_cam_b = pts_cam.unsqueeze(0)  # 1 x N x 3 to satisfy fisheye projection utils
    p2d, valid = cam.project(pts_cam_b)
    # p2d: 1 x N x 2, valid: 1 x N (or 1 x N x 1)
    if valid.ndim == 3:
        valid = valid.squeeze(0)
    if valid.ndim > 2:
        valid = valid.squeeze()
    valid_flat = valid.squeeze()
    p2d = p2d.squeeze(0)[valid_flat]
    depth = pts_cam[valid_flat, -1]
    if p2d.numel() == 0:
        return go.Figure()

    uv = p2d.detach().cpu().numpy()
    depth_np = depth.detach().cpu().numpy()
    uv = _rotate_uv(uv, width=w, height=h, rotate_k=rotate_k)

    depth_norm = np.clip((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-6), 0, 1)
    cmap = colormaps.get_cmap("turbo")
    colors = (cmap(depth_norm)[..., :3] * 255).astype(np.uint8)
    colors_hex = ["#" + "".join(f"{c:02x}" for c in rgb) for rgb in colors]

    fig = go.Figure()
    fig.add_trace(go.Image(z=img_np))
    fig.add_trace(
        go.Scattergl(
            x=uv[:, 0],
            y=uv[:, 1],
            mode="markers",
            marker={"size": point_size, "color": colors_hex, "opacity": 0.8},
            name="Projected points",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=600, width=600, title=title)
    return fig


def plot_frames(sample: EfmSnippetView) -> go.Figure:
    """First RGB/SLAM frames side-by-side."""
    modalities, _ = collect_frame_modalities(sample, include_depth=False, include_semantic=False)
    if len(modalities) == 0:
        return go.Figure()

    cols = len(modalities)
    builder = FrameGridBuilder(
        rows=1,
        cols=cols,
        titles=[name for name, _, _ in modalities],
        height=500,
        width=max(800, 320 * cols),
        title="Camera views (frame 0)",
    )
    for col, (_, first_img, _) in enumerate(modalities, start=1):
        builder.add_image(first_img, row=1, col=col)
    return builder.finalize()


def plot_first_last_frames(sample: EfmSnippetView) -> go.Figure:
    """First and final frames for all available modalities (RGB/SLAM + optional depth/semantic)."""

    modalities, _ = collect_frame_modalities(sample, include_depth=True, include_semantic=True)
    if len(modalities) == 0:
        return go.Figure()

    cols = len(modalities)
    subplot_titles = [f"First {name}" for name, _, _ in modalities] + [f"Last {name}" for name, _, _ in modalities]
    builder = FrameGridBuilder(
        rows=2,
        cols=cols,
        titles=subplot_titles,
        height=500 if cols <= 3 else int(500 * cols / 3),
        width=max(1200, 320 * cols),
        title="First and final frames (rotated for display)",
    )
    for col, (_, first_img, last_img) in enumerate(modalities, start=1):
        builder.add_image(first_img, row=1, col=col)
        builder.add_image(last_img, row=2, col=col)

    return builder.finalize()


def plot_trajectory(
    sample: EfmSnippetView,
    camera: str = "rgb",
    show_semidense: bool = True,
    max_sem_points: int | None = 20000,
    pc_from_last_only: bool = True,
    show_scene_bounds: bool = True,
    show_crop_bounds: bool = False,
    crop_aabb: tuple[np.ndarray, np.ndarray] | None = None,
    show_frustum: bool = True,
    frustum_frame_indices: list[int] | None = None,
    frustum_scale: float = 1.0,
    display_roll_rad: float = np.pi / 2,  # viewer roll to match rotated RGB display; set 0 to disable
    mark_first_last: bool = True,
    double_sided_mesh: bool = False,
    show_gt_obbs: bool = False,
    gt_timestamp: str | int | None = None,
) -> go.Figure:
    """Mesh + semidense + trajectory + optional frusta/bounds."""

    mesh = sample.mesh
    traj = sample.trajectory
    sem = sample.semidense

    cam_map = {
        "rgb": sample.camera_rgb,
        "slam-l": sample.camera_slam_left,
        "slam-r": sample.camera_slam_right,
    }
    cam = cam_map.get(camera, sample.camera_rgb)

    if frustum_frame_indices is None or len(frustum_frame_indices) == 0:
        frustum_frame_indices = [0]

    clamped_indices = [max(0, min(idx, cam.time_ns.shape[0] - 1)) for idx in frustum_frame_indices]
    frustum_poses = [_aligned_pose_world_cam(cam, traj, idx) for idx in clamped_indices]
    if display_roll_rad is not None:
        frustum_poses = [_apply_display_roll(p, display_roll_rad) for p in frustum_poses]

    traj_positions = traj.t_world_rig.t.detach().cpu().numpy()
    pts_np = np.zeros((0, 3), dtype=np.float32)
    if show_semidense and sem is not None:
        pts_np = (
            sem.last_frame_points_np(max_sem_points) if pc_from_last_only else sem.collapse_points_np(max_sem_points)
        )

    cam_centers = np.stack([p.t.detach().cpu().numpy() for p in frustum_poses], axis=0)
    # PoseTW.R stores R_world_cam; its columns are the camera basis in world coords.
    cam_axes = [p.R.detach().cpu().numpy().T for p in frustum_poses]

    bbox = np.vstack(
        [
            pts_np,
            cam_centers.reshape(-1, 3),
            traj_positions,
            mesh.vertices if mesh is not None else np.zeros((0, 3)),
        ]
    )
    finite_mask = np.isfinite(bbox).all(axis=1)
    bbox = bbox[finite_mask]
    if bbox.shape[0] == 0:
        bbox = np.array([[0.0, 0.0, 0.0]])

    vmin, vmax = bbox.min(axis=0), bbox.max(axis=0)
    max_extent = (vmax - vmin).max()
    padding = max(0.5, 0.1 * max_extent)
    scene_ranges = {
        "xaxis": {"title": "X (m)", "range": [vmin[0] - padding, vmax[0] + padding]},
        "yaxis": {"title": "Y (m)", "range": [vmin[1] - padding, vmax[1] + padding]},
        "zaxis": {"title": "Z (m)", "range": [vmin[2] - padding, vmax[2] + padding]},
    }

    scene_min = scene_max = None
    if sem is not None and hasattr(sem, "volume_min") and hasattr(sem, "volume_max"):
        scene_min = sem.volume_min.detach().cpu().numpy()
        scene_max = sem.volume_max.detach().cpu().numpy()

    builder = SnippetPlotBuilder(title="Mesh + semidense + trajectory + camera frustum", scene_ranges=scene_ranges)
    builder.add_mesh(
        mesh,
        double_sided=double_sided_mesh,
        occupancy_extent=None if scene_min is None else np.stack([scene_min, scene_max]),
    ).add_trajectory(traj_positions, mark_first_last=mark_first_last, show=True)

    if pts_np.size > 0:
        builder.add_semidense(pts_np)

    if show_frustum:
        for pose, idx in zip(frustum_poses, clamped_indices, strict=False):
            builder.add_frustum(
                cam.calib[idx],
                pose,
                scale=frustum_scale,
            )
        builder.add_camera_axes(cam_centers[0], cam_axes[0])
        builder.add_camera_center(cam_centers[0])

    if show_scene_bounds:
        if scene_min is not None and scene_max is not None:
            builder.add_bounds_box(scene_min, scene_max, name="Scene bounds", color="gray", dash="dash", width=2)
        else:
            builder.add_bounds_box(vmin, vmax, name="Scene bounds", color="gray", dash="dash", width=2)

    if show_gt_obbs and sample.gt is not None:
        ts_key = (
            gt_timestamp if gt_timestamp is not None else (sample.gt.timestamps[0] if sample.gt.timestamps else None)
        )
        if ts_key is not None:
            cam_id_map = {"rgb": "camera-rgb", "slam-l": "camera-slam-left", "slam-r": "camera-slam-right"}
            try:
                cam_gt = sample.gt.cameras_at(ts_key)[cam_id_map.get(camera, "camera-rgb")]
                builder.add_gt_obbs(
                    ts_world_object=cam_gt.ts_world_object.detach().cpu().numpy(),
                    object_dimensions=cam_gt.object_dimensions.detach().cpu().numpy(),
                )
            except KeyError:
                console.dbg(f"No GT OBBs for ts={ts_key} and camera={camera}")

    if show_crop_bounds and crop_aabb is not None:
        builder.add_bounds_box(crop_aabb[0], crop_aabb[1], name="Crop bounds", color="orange", dash="solid", width=3)
    elif show_crop_bounds and mesh is not None:
        mesh_min, mesh_max = mesh.bounds
        builder.add_bounds_box(mesh_min, mesh_max, name="Crop bounds", color="orange", dash="solid", width=3)

    return builder.finalize()


def plot_crop_box(sample: EfmSnippetView, margin: float = 0.0, color: str = "orange") -> go.Figure:
    """Visualize the semidense AABB (with optional margin) and mesh/trajectory context."""

    # Prefer computing bounds from finite semidense points to avoid oversized volumes.
    sem = sample.semidense
    if sem is None or not hasattr(sem, "points_world"):
        return go.Figure()
    lengths = sem.lengths
    frame_idx = int(torch.argmax(lengths).item())
    n_valid = int(lengths[frame_idx].item())
    points = sem.points_world[frame_idx, :n_valid]
    finite = torch.isfinite(points).all(dim=-1)
    points = points[finite]
    if points.shape[0] == 0:
        return go.Figure()
    pts_np = points.detach().cpu().numpy()
    # Outlier-robust bounds (1st-99th percentile) to avoid a few stray points inflating the box.
    lo_np = np.nanpercentile(pts_np, 1, axis=0) - margin
    hi_np = np.nanpercentile(pts_np, 99, axis=0) + margin
    console.dbg(
        f"crop box from semidense frame {frame_idx}: n={pts_np.shape[0]}, lo={lo_np}, hi={hi_np}, margin={margin}"
    )

    def _edges(a: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
        corners = np.array(
            [
                [a[0], a[1], a[2]],
                [b[0], a[1], a[2]],
                [b[0], b[1], a[2]],
                [a[0], b[1], a[2]],
                [a[0], a[1], b[2]],
                [b[0], a[1], b[2]],
                [b[0], b[1], b[2]],
                [a[0], b[1], b[2]],
            ]
        )
        edges_idx = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        return [corners[list(pair)] for pair in edges_idx]

    edges = _edges(lo_np, hi_np)
    fig = go.Figure()

    for idx, seg in enumerate(edges):
        fig.add_trace(
            go.Scatter3d(
                x=seg[:, 0],
                y=seg[:, 1],
                z=seg[:, 2],
                mode="lines",
                line={"color": color, "width": 6 if idx < 4 else 3},
                name="Crop AABB" if idx == 0 else None,
                showlegend=idx == 0,
            )
        )

    if sample.mesh is not None:
        fig.add_trace(
            go.Mesh3d(
                x=sample.mesh.vertices[:, 0],
                y=sample.mesh.vertices[:, 1],
                z=sample.mesh.vertices[:, 2],
                i=sample.mesh.faces[:, 0],
                j=sample.mesh.faces[:, 1],
                k=sample.mesh.faces[:, 2],
                color="lightgray",
                opacity=0.2,
                name="Mesh",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    traj_positions = sample.trajectory.t_world_rig.matrix3x4[..., :3, 3].detach().cpu().numpy()
    fig.add_trace(
        go.Scatter3d(
            x=traj_positions[:, 0],
            y=traj_positions[:, 1],
            z=traj_positions[:, 2],
            mode="lines+markers",
            marker={"size": 3, "color": "black"},
            line={"color": "black", "width": 3},
            name="Trajectory",
        )
    )

    pts = sample.semidense.points_world
    n_valid = int(sample.semidense.lengths[0].item()) if hasattr(sample.semidense, "lengths") else 0
    if n_valid > 0:
        pts = pts[0, :n_valid]
        finite = torch.isfinite(pts).all(dim=-1)
        pts = pts[finite]
        if pts.shape[0] > 5000:
            pts = pts[torch.randperm(pts.shape[0])[:5000]]
        pts_np = pts.detach().cpu().numpy()
        fig.add_trace(
            go.Scatter3d(
                x=pts_np[:, 0],
                y=pts_np[:, 1],
                z=pts_np[:, 2],
                mode="markers",
                marker={"size": 2, "opacity": 0.5, "color": "purple"},
                name="Semidense (sampled)",
            )
        )

    ranges = {
        "xaxis": {"title": "X (m)", "range": [lo_np[0], hi_np[0]]},
        "yaxis": {"title": "Y (m)", "range": [lo_np[1], hi_np[1]]},
        "zaxis": {"title": "Z (m)", "range": [lo_np[2], hi_np[2]]},
    }
    fig.update_layout(
        title=f"Crop AABB (margin={margin:.2f} m)",
        scene=dict(aspectmode="data", **ranges),
        height=700,
    )
    return fig


def crop_aabb_from_semidense(sample: EfmSnippetView, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (lo, hi) crop bounds from semidense points; robust to outliers."""

    sem = sample.semidense
    if sem is None or not hasattr(sem, "points_world"):
        return None
    lengths = sem.lengths
    if lengths.numel() == 0:
        return None
    frame_idx = int(torch.argmax(lengths).item())
    n_valid = int(lengths[frame_idx].item())
    if n_valid == 0:
        return None
    points = sem.points_world[frame_idx, :n_valid]
    finite = torch.isfinite(points).all(dim=-1)
    points = points[finite]
    if points.shape[0] == 0:
        return None
    pts_np = points.detach().cpu().numpy()
    lo_np = np.nanpercentile(pts_np, 1, axis=0) - margin
    hi_np = np.nanpercentile(pts_np, 99, axis=0) + margin
    return lo_np, hi_np


if __name__ == "__main__":
    from oracle_rri.data import AseEfmDatasetConfig
    from oracle_rri.utils import Console

    debug = True
    verbose = True
    console = Console.with_prefix("tst_import").set_verbose(True).set_debug(debug)

    ds_cfg = AseEfmDatasetConfig(
        scene_ids=["81283"],
        mesh_simplify_ratio=0.01,
        load_meshes=True,
        require_mesh=True,
        verbose=verbose,
        is_debug=debug,
    )
    dataset = ds_cfg.setup_target()
    sample = next(iter(dataset))

    plot_trajectory(sample, pc_from_last_only=False).show()
    plot_crop_box(sample, margin=0.2).show()
