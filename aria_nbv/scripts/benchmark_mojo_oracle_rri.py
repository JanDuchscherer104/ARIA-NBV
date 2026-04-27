"""Benchmark the full oracle RRI path against the current PyTorch3D baseline."""

from __future__ import annotations

import argparse
import time
from statistics import mean

import numpy as np
import torch

from aria_nbv.configs import PathConfig
from aria_nbv.data_handling import AseEfmDatasetConfig
from aria_nbv.pipelines import OracleRriLabelerConfig
from aria_nbv.pose_generation import CandidateViewGeneratorConfig
from aria_nbv.pose_generation.types import ViewDirectionMode
from aria_nbv.rendering import CandidateDepthRendererConfig, DepthRendererBackend
from aria_nbv.rendering.candidate_pointclouds import CandidatePointCloudBuilderConfig, PointCloudBackend
from aria_nbv.rendering.mojo_backend import is_mojo_available as rendering_mojo_available
from aria_nbv.rri_metrics.mojo_backend import is_mojo_available as rri_mojo_available
from aria_nbv.rri_metrics.oracle_rri import OracleDistanceBackend, OracleRRIConfig
from aria_nbv.utils import Verbosity


def _require_data() -> None:
    paths = PathConfig()
    atek_dir = paths.resolve_atek_data_dir("efm")
    if not atek_dir.exists() or not any(atek_dir.glob("**/*.tar")):
        raise RuntimeError(f"ATEK data dir or shards missing under {atek_dir}")
    mesh_dir = paths.ase_meshes
    if not any(mesh_dir.glob("scene_ply_*.ply")):
        raise RuntimeError(f"No ASE meshes found under {mesh_dir}")


def _require_mojo() -> None:
    if not rendering_mojo_available() or not rri_mojo_available():
        raise RuntimeError("Mojo oracle backends are not available locally.")


def _small_sample(scene_id: str, keep_faces: int):
    cfg = AseEfmDatasetConfig(
        scene_ids=[scene_id],
        batch_size=None,
        load_meshes=True,
        require_mesh=True,
        mesh_simplify_ratio=0.02,
        device="cpu",
        verbosity=Verbosity.QUIET,
        is_debug=False,
    )
    ds = cfg.setup_target()
    sample = next(iter(ds))
    mesh = sample.mesh
    assert mesh is not None
    keep_idx = np.linspace(0, int(mesh.faces.shape[0]) - 1, num=min(int(mesh.faces.shape[0]), keep_faces), dtype=int)
    mesh_small = mesh.submesh([keep_idx], append=True)
    return type(sample)(
        efm=sample.efm,
        scene_id=sample.scene_id,
        snippet_id=sample.snippet_id,
        mesh=mesh_small,
        crop_bounds=sample.crop_bounds,
        mesh_verts=torch.as_tensor(mesh_small.vertices, dtype=torch.float32),
        mesh_faces=torch.as_tensor(mesh_small.faces, dtype=torch.int64),
    )


def _labeler_config(
    *,
    depth_backend: DepthRendererBackend,
    pointcloud_backend: PointCloudBackend,
    distance_backend: OracleDistanceBackend,
    stride: int,
    resolution_scale: float,
) -> OracleRriLabelerConfig:
    generator_cfg = CandidateViewGeneratorConfig(
        num_samples=3,
        oversample_factor=1.0,
        max_resamples=0,
        min_radius=0.0,
        max_radius=0.0,
        view_direction_mode=ViewDirectionMode.FORWARD_RIG,
        min_distance_to_mesh=0.0,
        ensure_collision_free=False,
        ensure_free_space=False,
        device="cpu",
        verbosity=Verbosity.QUIET,
        is_debug=False,
    )
    depth_cfg = CandidateDepthRendererConfig(
        backend=depth_backend,
        max_candidates_final=3,
        oversample_factor=1.0,
        resolution_scale=resolution_scale,
        verbosity=Verbosity.QUIET,
        is_debug=False,
    )
    pointcloud_cfg = CandidatePointCloudBuilderConfig(
        backend=pointcloud_backend,
        backprojection_stride=stride,
    )
    oracle_cfg = OracleRRIConfig(backend=distance_backend)
    return OracleRriLabelerConfig(
        generator=generator_cfg,
        depth=depth_cfg,
        pointcloud=pointcloud_cfg,
        oracle=oracle_cfg,
        device="cpu",
    )


def _time_labeler(cfg: OracleRriLabelerConfig, sample, repeats: int) -> tuple[list[float], torch.Tensor]:
    labeler = cfg.setup_target()
    labeler.run(sample)
    timings: list[float] = []
    last_rri: torch.Tensor | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        batch = labeler.run(sample)
        timings.append((time.perf_counter() - start) * 1000.0)
        last_rri = batch.rri.rri.detach().cpu()
    assert last_rri is not None
    return timings, last_rri


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", default="81283", help="ASE scene id to load.")
    parser.add_argument("--keep-faces", type=int, default=2000, help="Maximum faces kept in the benchmark mesh.")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repetitions per backend.")
    parser.add_argument("--stride", type=int, default=32, help="Backprojection stride.")
    parser.add_argument("--resolution-scale", type=float, default=0.1, help="Candidate render resolution scale.")
    args = parser.parse_args()

    _require_data()
    _require_mojo()
    sample = _small_sample(args.scene_id, args.keep_faces)

    baseline_cfg = _labeler_config(
        depth_backend=DepthRendererBackend.PYTORCH3D,
        pointcloud_backend=PointCloudBackend.PYTORCH3D,
        distance_backend=OracleDistanceBackend.PYTORCH3D,
        stride=args.stride,
        resolution_scale=args.resolution_scale,
    )
    mojo_cfg = _labeler_config(
        depth_backend=DepthRendererBackend.MOJO,
        pointcloud_backend=PointCloudBackend.MOJO,
        distance_backend=OracleDistanceBackend.MOJO,
        stride=args.stride,
        resolution_scale=args.resolution_scale,
    )

    baseline_times, baseline_rri = _time_labeler(baseline_cfg, sample, args.repeats)
    mojo_times, mojo_rri = _time_labeler(mojo_cfg, sample, args.repeats)
    if not torch.allclose(baseline_rri, mojo_rri, atol=1e-4, rtol=1e-4):
        raise RuntimeError("Oracle parity check failed before reporting timings.")

    print(f"scene_id={args.scene_id}")
    print(f"keep_faces={args.keep_faces}")
    print(f"repeats={args.repeats}")
    print(f"baseline_mean_ms={mean(baseline_times):.3f}")
    print(f"mojo_mean_ms={mean(mojo_times):.3f}")
    print(f"speedup_vs_baseline={mean(baseline_times) / mean(mojo_times):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
