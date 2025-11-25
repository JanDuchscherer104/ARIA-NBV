"""Depth-to-mesh sanity checks for ASE snippets."""

from __future__ import annotations

import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from atek.data_loaders.atek_wds_dataloader import load_atek_wds_dataset
from atek.data_preprocess.atek_data_sample import create_atek_data_sample_from_flatten_dict
from pydantic import AliasChoices, Field, field_validator

try:  # Prefer installed efm3d, otherwise fall back to vendored source.
    from efm3d.aria import CameraTW, PoseTW
    from efm3d.utils.ray import ray_grid, transform_rays
except ModuleNotFoundError:  # pragma: no cover - exercised in CI fallback
    vendor_root = Path(__file__).resolve().parents[2] / "external" / "efm3d"
    sys.path.append(str(vendor_root))
    from efm3d.aria import CameraTW, PoseTW
    from efm3d.utils.ray import ray_grid, transform_rays

from ..utils import BaseConfig, Console, Verbosity, select_device


@dataclass(frozen=True, slots=True)
class DepthDebugResult:
    """Summary statistics for depth-to-mesh projection."""

    mean: float
    median: float
    p90: float
    max: float
    num_points: int
    variant: str


class DepthDebugger:
    """Project depth into 3D, align with poses, and measure distance to GT mesh."""

    def __init__(self, config: DepthDebuggerConfig):
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__).set_verbosity(config.verbosity)

    def _load_sample(self) -> tuple[dict[str, Any], trimesh.Trimesh]:
        tar_paths = sorted(Path().glob(self.config.tar_glob))
        if not tar_paths:
            raise FileNotFoundError(f"No shards found matching {self.config.tar_glob}")

        if not self.config.mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {self.config.mesh_path}")

        dataset = load_atek_wds_dataset(
            urls=[str(p) for p in tar_paths],
            batch_size=None,
            shuffle_flag=False,
            repeat_flag=False,
        )
        flat = next(iter(dataset))
        mesh = trimesh.load(self.config.mesh_path, process=False)
        if self.config.mesh_simplify_ratio is not None:
            try:
                mesh = mesh.simplify_quadric_decimation(self.config.mesh_simplify_ratio)
            except ModuleNotFoundError:
                self.console.warn("fast_simplification not installed; using full-resolution mesh.")

        self.console.plog(
            {
                "sequence": flat.get("sequence_name", "<unknown>"),
                "keys": sorted(flat.keys())[:10],
            }
        )
        return flat, mesh

    def _compute_distances(self, points: np.ndarray, mesh: trimesh.Trimesh, variant: str) -> DepthDebugResult:
        try:
            pq = trimesh.proximity.ProximityQuery(mesh)
            signed = pq.signed_distance(points)
        except ModuleNotFoundError:
            self.console.warn("rtree not installed; falling back to vertex-distance approximation.")
            vertices = mesh.vertices
            if self.config.mesh_vertex_cap is not None and len(vertices) > self.config.mesh_vertex_cap:
                rng = np.random.default_rng(0)
                indices = rng.choice(len(vertices), size=self.config.mesh_vertex_cap, replace=False)
                vertices = vertices[indices]
            sq_dists = np.sum((points[:, None, :] - vertices[None, :, :]) ** 2, axis=-1)
            signed = np.sqrt(np.min(sq_dists, axis=1))
        abs_dist = np.abs(signed)
        stats = DepthDebugResult(
            mean=float(abs_dist.mean()),
            median=float(np.median(abs_dist)),
            p90=float(np.percentile(abs_dist, 90)),
            max=float(abs_dist.max()),
            num_points=points.shape[0],
            variant=variant,
        )
        self.console.plog(asdict(stats))
        return stats

    def run(self) -> DepthDebugResult:
        """Execute the depth projection pipeline on one snippet."""
        flat, mesh = self._load_sample()
        sample = create_atek_data_sample_from_flatten_dict(flat)

        if sample.mps_traj_data is None or sample.mps_traj_data.Ts_World_Device is None:
            raise ValueError("Trajectory data missing in sample.")

        depth_stream = sample.camera_rgb_depth
        depth_ready = (
            depth_stream is not None
            and depth_stream.images is not None
            and depth_stream.projection_params is not None
            and depth_stream.T_Device_Camera is not None
        )
        if not depth_ready:
            self.console.warn("Depth stream incomplete; falling back to semidense points.")
            semidense = sample.mps_semidense_point_data
            if semidense is None or not semidense.points_world:
                raise ValueError("Neither depth stream nor semidense points are available in sample.")
            cleaned_frames = []
            for frame_pts in semidense.points_world:
                if frame_pts is None:
                    continue
                mask = torch.isfinite(frame_pts).all(dim=-1)
                if mask.any():
                    cleaned_frames.append(frame_pts[mask])
            if not cleaned_frames:
                raise ValueError("Semidense points exist but are all invalid (NaN/inf).")
            points = torch.cat(cleaned_frames, dim=0)
            if points.shape[0] > self.config.max_points:
                indices = torch.tensor(
                    random.sample(range(points.shape[0]), self.config.max_points), device=points.device
                )
                points = points[indices]
            return self._compute_distances(points.detach().cpu().numpy(), mesh, variant="semidense_points")

        depth = depth_stream.images
        device = self.config.device
        depth = depth.to(device)

        camera_tw = CameraTW.from_surreal(
            width=torch.full((depth.shape[0], 1), depth.shape[-1], device=device, dtype=torch.float32),
            height=torch.full((depth.shape[0], 1), depth.shape[-2], device=device, dtype=torch.float32),
            type_str=depth_stream.camera_model_name or "",
            params=depth_stream.projection_params.to(device),
            gain=depth_stream.gains.to(device) if depth_stream.gains is not None else None,
            exposure_s=depth_stream.exposure_durations_s.to(device)
            if depth_stream.exposure_durations_s is not None
            else None,
            valid_radius=depth_stream.camera_valid_radius.to(device)
            if depth_stream.camera_valid_radius is not None
            else torch.zeros((depth.shape[0], 1), device=device),
            T_camera_rig=PoseTW.from_matrix3x4(depth_stream.T_Device_Camera).inverse().to(device),
        )

        pose_world_device = PoseTW.from_matrix3x4(sample.mps_traj_data.Ts_World_Device).to(device)
        pose = pose_world_device[0]

        rays_rig, valid = ray_grid(camera_tw)
        rays_rig = rays_rig.to(device)
        valid = valid.to(device)

        depth_first = depth[0, 0]  # [H, W]
        rays_world = transform_rays(rays_rig, pose)
        origins = rays_world[..., :3]
        directions = rays_world[..., 3:]
        points = origins + depth_first.unsqueeze(-1) * directions
        points_valid = points[valid]
        mask = torch.isfinite(points_valid).all(dim=-1)
        points_valid = points_valid[mask]

        if points_valid.shape[0] > self.config.max_points:
            indices = torch.tensor(random.sample(range(points_valid.shape[0]), self.config.max_points), device=device)
            points_valid = points_valid[indices]

        points_np = points_valid.detach().cpu().numpy()
        return self._compute_distances(points_np, mesh, variant="luf")


class DepthDebuggerConfig(BaseConfig[DepthDebugger]):
    """Config-as-factory wrapper for :class:`DepthDebugger`."""

    target: type[DepthDebugger] = Field(default=DepthDebugger, exclude=True)  # type: ignore[assignment]

    scene_id: str
    tar_glob: str
    mesh_path: Path
    mesh_simplify_ratio: float | None = 0.2
    max_points: int = 1000
    mesh_vertex_cap: int | None = 5000
    device: torch.device = Field(default_factory=lambda: select_device("auto", component="DepthDebugger"))
    verbosity: Verbosity = Field(
        default=Verbosity.NORMAL,
        validation_alias=AliasChoices("verbosity", "verbose"),
        description="Verbosity level for logging (0=quiet, 1=normal, 2=verbose).",
    )

    @field_validator("device", mode="before")
    @classmethod
    def _resolve_device(cls, value: str | torch.device) -> torch.device:
        return select_device(value, component="DepthDebugger")

    @field_validator("verbosity", mode="before")
    @classmethod
    def _coerce_verbosity(cls, value: Any) -> Verbosity:
        return Verbosity.from_any(value)

    def setup_target(self) -> DepthDebugger:  # type: ignore[override]
        return DepthDebugger(self)
