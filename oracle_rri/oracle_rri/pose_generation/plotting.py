"""Plotting helpers for candidate sampling."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import torch
import trimesh

from oracle_rri.configs import PathConfig
from oracle_rri.data import AseEfmDatasetConfig
from oracle_rri.pose_generation import CandidateViewGenerator, CandidateViewGeneratorConfig


def _ensure_plot_dir() -> Path:
    out = PathConfig().data_root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_candidates(poses: torch.Tensor, mesh: trimesh.Trimesh | None, title: str, path: Path) -> None:
    """Save a 3D scatter/mesh plot of candidate positions."""
    fig = go.Figure()
    pos = poses.t.cpu()
    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="markers",
            marker=dict(size=3, color="blue", opacity=0.6),
            name="candidates",
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
                name="mesh",
            )
        )
    fig.update_layout(scene_aspectmode="data", title=title)
    fig.write_html(str(path))


def plot_rule_masks(masks: list[torch.Tensor], path: Path) -> None:
    """Bar plot of per-rule survival counts."""
    counts = [int(m.sum().item()) for m in masks]
    labels = [f"rule_{i}" for i in range(len(masks))]
    fig = go.Figure(go.Bar(x=labels, y=counts, text=counts, textposition="outside"))
    fig.update_layout(title="Candidates surviving per rule")
    fig.write_html(str(path))


def plot_shell(shell_poses: torch.Tensor, path: Path) -> None:
    """Plot raw shell samples before pruning."""
    if hasattr(shell_poses, "t") and not callable(shell_poses.t):
        pos = shell_poses.t.detach().cpu()
    else:
        arr = shell_poses.tensor() if hasattr(shell_poses, "tensor") else shell_poses
        if arr.shape[-1] == 12:
            arr = arr.view(-1, 3, 4)[..., :3, 3]
        if arr.ndim == 2 and arr.shape[1] == 3:
            pos = arr.cpu()
        else:
            pos = arr.t().cpu()
    fig = go.Figure(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="markers",
            marker=dict(size=2, color="red", opacity=0.4),
        )
    )
    fig.update_layout(scene_aspectmode="data", title="Shell samples (raw)")
    fig.write_html(str(path))


def main() -> None:
    # load one real snippet
    sample_cfg = AseEfmDatasetConfig(scene_ids=["81283"], verbose=False, load_meshes=True, mesh_simplify_ratio=None)
    sample = next(iter(sample_cfg.setup_target()))

    gen_cfg = CandidateViewGeneratorConfig(
        num_samples=128,
        max_resamples=2,
        ensure_collision_free=False,
        ensure_free_space=True,
        min_distance_to_mesh=0.0,
        device="cpu",
    )
    gen = CandidateViewGenerator(gen_cfg)
    result = gen.generate_from_typed_sample(sample)

    out_dir = _ensure_plot_dir()
    plot_candidates(result["poses"], sample.mesh, "Candidate positions", out_dir / "candidates.html")
    plot_rule_masks(result["masks"], out_dir / "rule_masks.html")
    if len(result["shell_poses"]) > 0:
        plot_shell(result["shell_poses"][0], out_dir / "shell.html")


if __name__ == "__main__":
    main()
