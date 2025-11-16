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
    from efm3d.aria import PoseTW
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    vendor = Path(__file__).resolve().parents[2] / "external" / "efm3d"
    if vendor.exists():
        sys.path.append(str(vendor))
        from efm3d.aria import PoseTW  # type: ignore
    else:
        raise

from oracle_rri.data import ASEDataset, ASEDatasetConfig
from oracle_rri.utils import BaseConfig, Console
from oracle_rri.views.candidate_generation import CandidateViewGeneratorConfig


class CandidateVizConfig(BaseConfig["CandidateVizApp"]):
    """Configuration for the candidate visualization app."""

    target: type["CandidateVizApp"] = Field(default_factory=lambda: CandidateVizApp, exclude=True)

    dataset: ASEDatasetConfig | None = None
    """Dataset config used to pull an example snippet."""

    generator: CandidateViewGeneratorConfig = Field(default_factory=CandidateViewGeneratorConfig)
    """Candidate view generator configuration."""

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

            sample = None
            if self.config.dataset is not None:
                ds = ASEDataset(self.config.dataset)
                sample = next(iter(ds))

            if sample is None:
                # Synthetic sample: identity pose, small cube mesh for visualization
                last_pose = PoseTW.from_matrix3x4(torch.eye(3, 4))
                mesh = _make_unit_cube_mesh()
                scene_id = "synthetic"
                snippet_id = "demo"
            else:
                traj = sample.trajectory.ts_world_device
                last_pose = PoseTW.from_matrix3x4(traj[-1])
                mesh = sample.mesh
                scene_id = sample.scene_id
                snippet_id = sample.snippet_id

            result = generator.generate(last_pose=last_pose, gt_mesh=mesh, occupancy_extent=None)
            poses = result["poses"]
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
            st.info("Depth rendering integration pending oracle renderer hookup.")

        with tabs[2]:
            st.info("Point cloud fusion view pending fusion module wiring.")

        with tabs[3]:
            st.info("RRI scoring visualization pending metric module integration.")

        console.log(
            f"Rendered {min(len(pos_t), self.config.max_candidates)} candidates for scene {scene_id}/{snippet_id}"
        )


def launch_candidate_viz(config: CandidateVizConfig) -> None:
    """Convenience launcher used by `streamlit run -m oracle_rri.visualization.candidate_app`."""

    app = config.setup_target()
    app.run()


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
