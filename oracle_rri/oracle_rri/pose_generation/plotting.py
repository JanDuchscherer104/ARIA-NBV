"""Plotting helpers for candidate sampling."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go  # type: ignore[import]
import torch
from efm3d.aria.pose import PoseTW

from oracle_rri.data import AseEfmDatasetConfig, EfmCameraView, EfmSnippetView
from oracle_rri.data.plotting import SnippetPlotBuilder
from oracle_rri.pose_generation import CandidateViewGenerator, CandidateViewGeneratorConfig
from oracle_rri.utils import Console, Verbosity

console = Console.with_prefix("pose_plotting")


def plot_candidate_frusta(
    *,
    snippet: EfmSnippetView,
    poses: PoseTW,
    camera_view: EfmCameraView,
    frustum_scale: float,
    max_frustums: int,
) -> go.Figure:
    """Plot mesh, candidate centres, and frustums using the shared builder."""

    calib = camera_view.calib
    num_frames = calib.shape[0] if calib.ndim > 1 else 1
    cams = [calib[idx] for idx in range(num_frames)] if num_frames > 1 else [calib]

    return (
        SnippetPlotBuilder.from_snippet(snippet, title="Candidates with frustums")
        .add_mesh()
        .add_points(poses, name="Candidates", color="royalblue", size=3, opacity=0.35)
        .add_candidate_frusta(
            cam=cams,
            poses=poses,
            scale=frustum_scale,
            color="crimson",
            name="Frustum",
            max_frustums=max_frustums,
            include_axes=False,
            include_center=False,
        )
    ).finalize()


def plot_candidates(
    *,
    snippet: EfmSnippetView,
    poses: PoseTW,
    title: str = "Candidate positions",
    center: np.ndarray | None = None,
) -> go.Figure:
    """Plot candidate camera centres with optional mesh overlay."""
    builder = (
        SnippetPlotBuilder.from_snippet(snippet, title=title)
        .add_mesh()
        .add_points(poses, name="Candidates", color="royalblue", size=4, opacity=0.7)
    )

    if center is not None:
        builder = builder.add_points(
            np.asarray(center).reshape(1, 3),
            name="sampling center",
            color="red",
            size=5,
            symbol="x",
            opacity=1.0,
        )

    return builder.finalize()


def plot_sampling_shell(
    *,
    snippet: EfmSnippetView,
    shell_poses: torch.Tensor,
    last_pose: PoseTW,
    min_radius: float,
    max_radius: float,
    min_elev_deg: float,
    max_elev_deg: float,
    azimuth_full_circle: bool,
    sample_n: int = 200,
) -> go.Figure:
    """Visualise the spherical sampling band and a subset of raw shell samples."""

    return (
        SnippetPlotBuilder.from_snippet(snippet, title="Sampling shell (r_min/r_max + elev cap)")
        .add_mesh()
        .add_sampling_shell(
            center=last_pose.t.detach().cpu().numpy(),
            min_radius=min_radius,
            max_radius=max_radius,
            min_elev_deg=min_elev_deg,
            max_elev_deg=max_elev_deg,
            azimuth_full_circle=azimuth_full_circle,
            samples=None,
            sample_n=sample_n,
        )
    ).finalize()


def main() -> None:
    # load one real snippet
    sample_cfg = AseEfmDatasetConfig(
        scene_ids=["81283"],
        verbosity=Verbosity.QUIET,
        load_meshes=True,
        mesh_simplify_ratio=None,
    )
    sample = next(iter(sample_cfg.setup_target()))

    gen_cfg = CandidateViewGeneratorConfig(
        num_samples=128,
        max_resamples=2,
        ensure_collision_free=False,
        ensure_free_space=True,
        min_distance_to_mesh=0.0,
        device=torch.device("cpu"),
    )
    gen = CandidateViewGenerator(gen_cfg)
    result = gen.generate_from_typed_sample(sample)

    plot_candidates(snippet=sample, poses=result["poses"], title="Candidate positions")
    plot_sampling_shell(
        snippet=sample,
        shell_poses=result["shell_poses"],
        last_pose=sample.trajectory.final_pose,
        min_radius=gen_cfg.min_radius,
        max_radius=gen_cfg.max_radius,
        min_elev_deg=gen_cfg.min_elev_deg,
        max_elev_deg=gen_cfg.max_elev_deg,
        azimuth_full_circle=gen_cfg.azimuth_full_circle,
        sample_n=300,
    )


if __name__ == "__main__":
    main()
