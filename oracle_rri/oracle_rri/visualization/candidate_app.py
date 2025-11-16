"""Streamlit/Plotly viewer for candidate view sampling pipeline.

Runs even without local ASE data by falling back to synthetic poses.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass

import torch
import trimesh
from pydantic import Field

# Prefer installed efm3d; fall back to vendored copy for local runs.
try:
    from efm3d.aria import CameraTW, PoseTW
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    vendor = Path(__file__).resolve().parents[2] / "external" / "efm3d"
    if vendor.exists():
        sys.path.append(str(vendor))
        from efm3d.aria import CameraTW, PoseTW  # type: ignore
    else:
        raise

from oracle_rri.data import ASEDataset, ASEDatasetConfig
from oracle_rri.utils import BaseConfig, Console
from oracle_rri.views.candidate_generation import CandidateViewGeneratorConfig
from oracle_rri.views.candidate_rendering import (
    CandidatePointCloudGenerator,
    CandidatePointCloudGeneratorConfig,
)


class CandidateVizConfig(BaseConfig["CandidateVizApp"]):
    """Configuration for the candidate visualization app."""

    target: type["CandidateVizApp"] = Field(default_factory=lambda: CandidateVizApp, exclude=True)

    dataset: ASEDatasetConfig | None = None
    """Dataset config used to pull an example snippet."""

    generator: CandidateViewGeneratorConfig = Field(default_factory=CandidateViewGeneratorConfig)
    """Candidate view generator configuration."""

    renderer: CandidatePointCloudGeneratorConfig = Field(default_factory=CandidatePointCloudGeneratorConfig)
    """Depth/point rendering configuration."""

    max_candidates: int = 256
    """Cap number of candidates visualized for speed."""

    verbose: bool = False
    """Enable verbose console output."""


@dataclass(slots=True)
class CandidateVizApp:
    """Minimal Streamlit app for inspecting candidate sampling."""

    config: CandidateVizConfig

    def __call__(self) -> None:
        self.run()

    def run(self) -> None:
        import plotly.graph_objects as go
        import streamlit as st

        console = Console.with_prefix("CandidateViz").set_verbose(self.config.verbose)
        st.set_page_config(page_title="NBV Candidate Visualizer", layout="wide")
        st.title("NBV Candidate Sampling Explorer")

        try:
            num_samples = st.sidebar.slider("Candidates", 64, 1024, self.config.generator.num_samples, 64)
            device = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)

            generator_cfg = self.config.generator.model_copy(update={"num_samples": num_samples, "device": device})
            generator = generator_cfg.setup_target()
            renderer_cfg = self.config.renderer.model_copy(update={"device": device})
            renderer = renderer_cfg.setup_target()

            sample = None
            if self.config.dataset is not None:
                try:
                    ds = ASEDataset(self.config.dataset)
                    sample = next(iter(ds))
                except Exception as data_exc:  # pragma: no cover - UI feedback
                    st.warning(f"Dataset load failed, using synthetic scene. Details: {data_exc}")
                    console.warn(f"Dataset load failed: {data_exc}")

            if sample is None:
                # Synthetic sample: identity pose, small cube mesh for visualization
                last_pose = PoseTW.from_matrix3x4(torch.eye(3, 4))
                mesh = _make_unit_cube_mesh()
                scene_id = "synthetic"
                snippet_id = "demo"
                camera_tw = _make_default_camera_tw(device)
            else:
                traj = sample.trajectory.ts_world_device
                last_pose = PoseTW.from_matrix3x4(traj[-1])
                mesh = sample.mesh
                scene_id = sample.scene_id
                snippet_id = sample.snippet_id
                camera_tw = _camera_tw_from_sample(sample, device=device)

            result = generator.generate(last_pose=last_pose, gt_mesh=mesh, occupancy_extent=None)
            poses = result["poses"]
            mask_valid = result.get("mask_valid")
            if mask_valid is not None and mask_valid.any():
                valid_indices = torch.nonzero(mask_valid, as_tuple=False).squeeze(-1).tolist()
                if isinstance(valid_indices, int):
                    valid_indices = [valid_indices]
            else:
                valid_indices = list(range(min(len(poses), self.config.max_candidates)))
            pos_t = poses.t[: self.config.max_candidates].detach().cpu()
        except Exception as exc:  # pragma: no cover - show error in UI
            st.error(f"App error: {exc}")
            st.code(traceback.format_exc())
            return

        tabs = st.tabs(
            [
                "Candidate Sampling",
                "Depth Rendering",
                "Point Cloud Fusion",
                "RRI Scoring",
            ]
        )

        with tabs[0]:
            st.subheader("Candidate Positions")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter3d(
                    x=pos_t[:, 0].cpu(),
                    y=pos_t[:, 1].cpu(),
                    z=pos_t[:, 2].cpu(),
                    mode="markers",
                    marker={"size": 3, "color": "blue", "opacity": 0.6},
                    name="Candidates",
                )
            )
            if mesh is not None:
                verts = mesh.vertices
                faces = mesh.faces
                fig.add_trace(
                    go.Mesh3d(
                        x=verts[:, 0],
                        y=verts[:, 1],
                        z=verts[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        opacity=0.15,
                        color="gray",
                        name="Mesh",
                    )
                )
            fig.update_layout(scene_aspectmode="data")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            st.subheader("Depth Rendering (RGB intrinsics)")
            if mesh is None:
                st.info("No GT mesh attached; depth rendering unavailable.")
            elif not valid_indices:
                st.warning("No valid candidates to render.")
            else:
                idx = st.slider("Candidate index", 0, max(0, len(valid_indices) - 1), 0)
                cand_pose = poses[valid_indices[idx]]
                depth = renderer.render_depth(
                    pose_world_cam=cand_pose,
                    mesh=mesh,
                    camera=camera_tw,
                )
                depth_np = depth.detach().cpu().numpy()
                st.write(f"Depth stats (m): min={depth_np.min():.3f}, max={depth_np.max():.3f}")
                st.image(depth_np, caption="Depth map (m)", clamp=True)

        with tabs[2]:
            st.subheader("Depth → Point Cloud (world)")
            if mesh is None:
                st.info("No GT mesh attached; point cloud unavailable.")
            elif not valid_indices:
                st.warning("No valid candidates to render.")
            else:
                idx = st.slider("Candidate index (PC)", 0, max(0, len(valid_indices) - 1), 0, key="pc_idx")
                cand_pose = poses[valid_indices[idx]]
                depth, pts_world = renderer.render_point_cloud(
                    pose_world_cam=cand_pose,
                    mesh=mesh,
                    camera=camera_tw,
                )
                st.write(f"Points: {pts_world.shape[0]:,}")
                limit = min(20_000, pts_world.shape[0])
                pts_plot = pts_world
                if pts_world.shape[0] > limit:
                    choice = torch.randperm(pts_world.shape[0], device=pts_world.device)[:limit]
                    pts_plot = pts_world[choice]
                pts_np = pts_plot.detach().cpu().numpy()
                fig_pc = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=pts_np[:, 0],
                            y=pts_np[:, 1],
                            z=pts_np[:, 2],
                            mode="markers",
                            marker=dict(size=1, color=pts_np[:, 2], colorscale="Viridis", opacity=0.7),
                        )
                    ]
                )
                fig_pc.update_layout(scene_aspectmode="data")
                st.plotly_chart(fig_pc, use_container_width=True)

        with tabs[3]:
            st.info("RRI scoring visualization pending metric module integration.")

        console.log(
            f"Rendered {min(len(pos_t), self.config.max_candidates)} candidates for scene {scene_id}/{snippet_id}"
        )


def launch_candidate_viz(config: CandidateVizConfig) -> None:
    """Convenience launcher used by `streamlit run -m oracle_rri.visualization.candidate_app`."""

    app = config.setup_target()
    app.run()


def _camera_tw_from_sample(sample, frame_idx: int = -1, device: str = "cpu") -> CameraTW:
    """Build CameraTW from a TypedSample's RGB stream.

    Uses fisheye projection params: [fx, fy, cx, cy, k0..k5, p0, p1, s0, s1, s2].
    Extrinsics are handled via candidate PoseTW (world←camera).
    """

    cam_view = sample.camera_rgb
    proj = cam_view.projection_params
    if proj.ndim > 1:
        proj = proj[frame_idx]
    fx, fy, cx, cy = proj[:4].to(device)
    h, w = cam_view.images.shape[-2], cam_view.images.shape[-1]
    valid_radius = cam_view.camera_valid_radius
    if valid_radius is not None:
        if valid_radius.ndim > 1:
            valid_radiusx = valid_radius[frame_idx, 0].to(device)
            valid_radiusy = valid_radius[frame_idx, 0].to(device)
        else:
            valid_radiusx = valid_radiusy = valid_radius.to(device)
    else:
        valid_radiusx = valid_radiusy = torch.tensor([99999.0], device=device)
    gain = cam_view.gains[frame_idx] if cam_view.gains is not None else torch.tensor([-1.0], device=device)
    exposure = (
        cam_view.exposure_durations_s[frame_idx]
        if cam_view.exposure_durations_s is not None
        else torch.tensor([0.001], device=device)
    )
    dist = torch.zeros((1, 0), device=device)  # keep length 22
    t_cam_rig = PoseTW.from_matrix3x4(torch.eye(3, 4, device=device).unsqueeze(0))
    return CameraTW.from_parameters(
        width=torch.tensor([[w]], device=device),
        height=torch.tensor([[h]], device=device),
        fx=fx.unsqueeze(0),
        fy=fy.unsqueeze(0),
        cx=cx.unsqueeze(0),
        cy=cy.unsqueeze(0),
        gain=gain.unsqueeze(0),
        exposure_s=exposure.unsqueeze(0),
        valid_radiusx=valid_radiusx.unsqueeze(0),
        valid_radiusy=valid_radiusy.unsqueeze(0),
        T_camera_rig=t_cam_rig,
        dist_params=dist,
    )


def _make_default_camera_tw(device: str = "cpu") -> CameraTW:
    """Fallback pinhole camera (fx=fy=700, 704x704) for synthetic demo."""

    fx = fy = torch.tensor([[700.0]], device=device)
    cx = cy = torch.tensor([[352.0]], device=device)
    dist = torch.zeros((1, 0), device=device)
    t_cam_rig = PoseTW.from_matrix3x4(torch.eye(3, 4, device=device).unsqueeze(0))
    return CameraTW.from_parameters(
        width=torch.tensor([[704]], device=device),
        height=torch.tensor([[704]], device=device),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        T_camera_rig=t_cam_rig,
        dist_params=dist,
    )


def _make_unit_cube_mesh() -> trimesh.Trimesh:
    """Create a small cube mesh for fallback visualization."""

    verts = torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    ).numpy()
    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=torch.int64,
    ).numpy()
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
