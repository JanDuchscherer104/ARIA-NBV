"""Generate VIN v2 architecture diagrams (DOT + renders + optional draw.io export).

This script is intended for an **iterative** workflow:

1) run the script to generate `*.dot` + `*.svg`/`*.png`,
2) inspect the render,
3) tweak labels/layout in `oracle_rri/oracle_rri/vin/arch_viz.py`,
4) re-run.

The default mode runs VIN v2 on **synthetic inputs** to collect shapes without
requiring EVL checkpoints or ASE data.
"""

from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

# VIN v2 imports optional PointNeXt code paths which currently emit AMP deprecation warnings.
# They are unrelated to diagram generation and would otherwise spam iterative runs.
warnings.filterwarnings(
    "ignore",
    message=".*torch\\.cuda\\.amp\\.custom_fwd.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*torch\\.cuda\\.amp\\.custom_bwd.*",
    category=FutureWarning,
)

import torch
from efm3d.aria.pose import PoseTW
from pytorch3d.renderer.cameras import PerspectiveCameras  # type: ignore[import-untyped]
from torch import Tensor

from oracle_rri.data.efm_views import VinSnippetView
from oracle_rri.utils import Console
from oracle_rri.vin.arch_viz import VinV2ArchDotConfig, VinV2ArchInputShapes, build_vin_v2_arch_dot, write_dot
from oracle_rri.vin.model_v2 import VinModelV2Config
from oracle_rri.vin.types import EvlBackboneOutput


@dataclass(slots=True)
class SyntheticVinInputs:
    """Minimal synthetic inputs to run VIN v2 forward_with_debug."""

    snippet: VinSnippetView
    candidate_poses_world_cam: PoseTW
    reference_pose_world_rig: PoseTW
    p3d_cameras: PerspectiveCameras
    backbone_out: EvlBackboneOutput


def _make_grid_points_world(
    *,
    extent_xyz: Tensor,
    grid_shape: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return voxel center points in world coords as a dense grid.

    Args:
        extent_xyz: ``Tensor["6"]`` extent in voxel/world frame
            ``[x_min,x_max,y_min,y_max,z_min,z_max]``.
        grid_shape: ``(D,H,W)`` voxel grid shape.
    """
    d, h, w = grid_shape
    x_min, x_max, y_min, y_max, z_min, z_max = (float(v) for v in extent_xyz.tolist())
    xs = torch.linspace(x_min, x_max, w, device=device, dtype=dtype)
    ys = torch.linspace(y_min, y_max, h, device=device, dtype=dtype)
    zs = torch.linspace(z_min, z_max, d, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    pts = torch.stack([xx, yy, zz], dim=-1)  # D H W 3
    return pts


def _make_synthetic_inputs(
    *,
    num_candidates: int,
    grid_shape: tuple[int, int, int],
    num_points: int,
    num_traj_frames: int,
    seed: int,
) -> SyntheticVinInputs:
    """Create a CPU-only synthetic VIN v2 batch."""
    torch.manual_seed(int(seed))
    device = torch.device("cpu")
    dtype = torch.float32

    # ------------------------------------------------------------------ snippet: semidense points + trajectory
    # points_world: K x (3 + extras). Provide (x,y,z,inv_dist_std,obs_count).
    xyz = torch.randn((num_points, 3), device=device, dtype=dtype)
    xyz[:, 2] = xyz[:, 2].abs() + 1.0  # bias in front of the camera
    inv_dist_std = torch.rand((num_points, 1), device=device, dtype=dtype).clamp_min(0.0)
    obs_count = torch.randint(1, 20, (num_points, 1), device=device, dtype=torch.int64).to(dtype=dtype)
    points_world = torch.cat([xyz, inv_dist_std, obs_count], dim=-1)

    # trajectory: PoseTW[F,12]
    rot = torch.eye(3, device=device, dtype=dtype).expand(num_traj_frames, 3, 3).contiguous()
    trans = torch.zeros((num_traj_frames, 3), device=device, dtype=dtype)
    trans[:, 0] = torch.linspace(0.0, 0.2, num_traj_frames, device=device, dtype=dtype)
    t_world_rig = PoseTW.from_Rt(rot, trans)

    lengths = torch.tensor([points_world.shape[0]], device=points_world.device, dtype=torch.int64)
    snippet = VinSnippetView(points_world=points_world, lengths=lengths, t_world_rig=t_world_rig)
    reference_pose_world_rig = t_world_rig[-1]

    # ------------------------------------------------------------------ candidates: poses + cameras
    # Arrange candidates on a small circle around the origin in world frame.
    angles = torch.linspace(0.0, 2.0 * torch.pi, num_candidates + 1, device=device, dtype=dtype)[:-1]
    radius = 0.25
    cand_t = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles), torch.zeros_like(angles)], dim=-1)
    cand_r = torch.eye(3, device=device, dtype=dtype).expand(num_candidates, 3, 3).contiguous()
    candidate_poses_world_cam = PoseTW.from_Rt(cand_r, cand_t)

    poses_cw = candidate_poses_world_cam.inverse().to(device=device)
    rotations = poses_cw.R.transpose(-1, -2).contiguous()
    translations = poses_cw.t

    width, height = 160, 120
    image_size = torch.tensor([[height, width]], device=device, dtype=dtype).expand(num_candidates, -1)
    focal_length = torch.tensor([[80.0, 80.0]], device=device, dtype=dtype).expand(num_candidates, -1)
    principal_point = torch.tensor([[width * 0.5, height * 0.5]], device=device, dtype=dtype).expand(num_candidates, -1)
    p3d_cameras = PerspectiveCameras(
        device=device,
        R=rotations,
        T=translations,
        focal_length=focal_length,
        principal_point=principal_point,
        image_size=image_size,
        in_ndc=False,
    )

    # ------------------------------------------------------------------ synthetic backbone output
    d, h, w = grid_shape
    occ_pr = torch.rand((1, 1, d, h, w), device=device, dtype=dtype)
    cent_pr = torch.rand_like(occ_pr)
    occ_input = (torch.rand_like(occ_pr) > 0.5).to(dtype=dtype)
    free_input = (torch.rand_like(occ_pr) > 0.5).to(dtype=dtype)
    counts = torch.randint(0, 10, (1, d, h, w), device=device, dtype=torch.int64)
    extent = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], device=device, dtype=dtype)
    pts_grid = _make_grid_points_world(extent_xyz=extent, grid_shape=grid_shape, device=device, dtype=dtype)
    pts_world = pts_grid.reshape(1, d * h * w, 3)
    t_world_voxel = PoseTW.from_Rt(torch.eye(3, device=device, dtype=dtype).unsqueeze(0), torch.zeros((1, 3)))

    backbone_out = EvlBackboneOutput(
        t_world_voxel=t_world_voxel,
        voxel_extent=extent,
        occ_pr=occ_pr,
        occ_input=occ_input,
        free_input=free_input,
        counts=counts,
        cent_pr=cent_pr,
        pts_world=pts_world,
    )

    return SyntheticVinInputs(
        snippet=snippet,
        candidate_poses_world_cam=candidate_poses_world_cam,
        reference_pose_world_rig=reference_pose_world_rig,
        p3d_cameras=p3d_cameras,
        backbone_out=backbone_out,
    )


def _render_dot(dot_path: Path, *, svg_path: Path, png_path: Path) -> None:
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)
    subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(png_path)], check=True)


def _render_vdx(dot_path: Path, *, vdx_path: Path) -> None:
    """Render a Visio VDX file which draw.io can import as editable shapes."""
    vdx_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["dot", "-Tvdx", str(dot_path), "-o", str(vdx_path)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures/vin_v2"))
    parser.add_argument("--stem", type=str, default="vin_v2_arch_shapes")
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--grid", type=int, nargs=3, default=(12, 12, 12), metavar=("D", "H", "W"))
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--traj-frames", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-sem-frustum", action="store_true", help="Enable semidense frustum MHCA in VIN v2.")
    parser.add_argument("--disable-trajectory", action="store_true", help="Disable trajectory encoder + MHCA.")
    parser.add_argument("--drawio", action="store_true", help="Also export an editable draw.io import via VDX.")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--include-formulas", action="store_true")
    parser.add_argument("--include-node-shapes", action="store_true")
    parser.add_argument("--no-symbolic", action="store_true")
    parser.add_argument("--no-numeric", action="store_true")
    args = parser.parse_args()

    console = Console.with_prefix("generate_vin_v2_arch")
    console.set_verbose(True).set_debug(True)

    inputs = _make_synthetic_inputs(
        num_candidates=int(args.num_candidates),
        grid_shape=(int(args.grid[0]), int(args.grid[1]), int(args.grid[2])),
        num_points=int(args.num_points),
        num_traj_frames=int(args.traj_frames),
        seed=int(args.seed),
    )

    vin_cfg = VinModelV2Config(
        backbone=None,
        enable_semidense_frustum=bool(args.enable_sem_frustum),
        use_traj_encoder=not bool(args.disable_trajectory),
    )
    vin = vin_cfg.setup_target().eval()

    with torch.no_grad():
        pred, debug = vin.forward_with_debug(
            inputs.snippet,
            inputs.candidate_poses_world_cam,
            inputs.reference_pose_world_rig,
            inputs.p3d_cameras,
            backbone_out=inputs.backbone_out,
        )

    dot_cfg = VinV2ArchDotConfig(
        include_edge_shapes=True,
        include_node_shapes=bool(args.include_node_shapes),
        include_formulas=bool(args.include_formulas),
        symbolic_shapes=not bool(args.no_symbolic),
        show_numeric_shapes=not bool(args.no_numeric),
        include_semidense_frustum=bool(args.enable_sem_frustum),
        include_trajectory=not bool(args.disable_trajectory),
    )
    shape_inputs = VinV2ArchInputShapes(
        semidense_points_world=tuple(int(d) for d in inputs.snippet.points_world.shape),
        traj_world_rig=tuple(int(d) for d in inputs.snippet.t_world_rig.shape),
        cameras_r=tuple(int(d) for d in inputs.p3d_cameras.R.shape),
        cameras_t=tuple(int(d) for d in inputs.p3d_cameras.T.shape),
        cameras_image_size=tuple(int(d) for d in inputs.p3d_cameras.image_size.shape),
    )
    dot = build_vin_v2_arch_dot(pred=pred, debug=debug, inputs=shape_inputs, cfg=dot_cfg)

    out_dir: Path = args.out_dir
    dot_path = out_dir / f"{args.stem}.dot"
    svg_path = out_dir / f"{args.stem}.svg"
    png_path = out_dir / f"{args.stem}.png"
    write_dot(dot_path, dot)
    console.log(f"Wrote DOT: {dot_path}")

    if args.drawio:
        vdx_path = out_dir / f"{args.stem}.vdx"
        _render_vdx(dot_path, vdx_path=vdx_path)
        console.log(f"Wrote draw.io import (VDX): {vdx_path}")

    if not args.no_render:
        _render_dot(dot_path, svg_path=svg_path, png_path=png_path)
        console.log(f"Rendered: {svg_path}")
        console.log(f"Rendered: {png_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
