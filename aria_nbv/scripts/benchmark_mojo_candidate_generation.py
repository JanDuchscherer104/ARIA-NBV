"""Benchmark the experimental Mojo mesh-clearance backend against Trimesh.

This script compares full candidate-generation runs on the same synthetic mesh
and asserts equivalence before reporting timing numbers.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from statistics import mean

import torch
import trimesh  # type: ignore[import-untyped]
from efm3d.aria import CameraTW, PoseTW

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aria_nbv.pose_generation import CandidateViewGenerator, CandidateViewGeneratorConfig, CollisionBackend
from aria_nbv.pose_generation.mojo_backend import is_mojo_available


def _identity_pose(device: torch.device | str = "cpu") -> PoseTW:
    data = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=device)
    return PoseTW(data)


def _dummy_camera(device: torch.device | str = "cpu") -> CameraTW:
    return CameraTW.from_surreal(
        width=torch.tensor([64.0], device=device),
        height=torch.tensor([64.0], device=device),
        type_str="Pinhole",
        params=torch.tensor([[60.0, 60.0, 32.0, 32.0]], device=device),
        gain=torch.zeros(1, device=device),
        exposure_s=torch.zeros(1, device=device),
        valid_radius=torch.tensor([64.0], device=device),
        T_camera_rig=PoseTW.from_matrix3x4(torch.eye(3, 4, device=device).unsqueeze(0)),
    )


def _default_extent(device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor([-10.0, 10.0, -10.0, 10.0, -10.0, 10.0], dtype=torch.float32, device=device)


def _benchmark_generator(
    cfg: CandidateViewGeneratorConfig,
    mesh: trimesh.Trimesh,
    verts: torch.Tensor,
    faces: torch.Tensor,
    repeats: int,
) -> tuple[list[float], torch.Tensor, torch.Tensor]:
    generator = CandidateViewGenerator(cfg)
    reference_pose = _identity_pose(device=cfg.device)
    camera = _dummy_camera(cfg.device)
    extent = _default_extent(cfg.device)

    generator.generate(
        reference_pose=reference_pose,
        gt_mesh=mesh,
        mesh_verts=verts,
        mesh_faces=faces,
        camera_calib_template=camera,
        occupancy_extent=extent,
    )

    timings_ms: list[float] = []
    last_mask: torch.Tensor | None = None
    last_views: torch.Tensor | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = generator.generate(
            reference_pose=reference_pose,
            gt_mesh=mesh,
            mesh_verts=verts,
            mesh_faces=faces,
            camera_calib_template=camera,
            occupancy_extent=extent,
        )
        timings_ms.append((time.perf_counter() - start) * 1000.0)
        last_mask = result.mask_valid.cpu()
        last_views = result.views.tensor().cpu()

    assert last_mask is not None
    assert last_views is not None
    return timings_ms, last_mask, last_views


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=256, help="Requested valid candidates.")
    parser.add_argument("--repeats", type=int, default=5, help="Timed repetitions per backend.")
    parser.add_argument("--mesh-subdivisions", type=int, default=3, help="Icosphere subdivision level.")
    parser.add_argument("--min-distance", type=float, default=0.2, help="Mesh clearance threshold in metres.")
    parser.add_argument(
        "--ensure-collision-free",
        action="store_true",
        help="Keep the existing path-collision rule enabled during the benchmark.",
    )
    args = parser.parse_args()

    if not is_mojo_available():
        raise SystemExit(
            "Mojo backend is not available. Install Mojo into `<repo>/.mojo-venv` or set "
            "`ARIA_NBV_MOJO_SITE_PACKAGES` before running this benchmark."
        )

    mesh = trimesh.creation.icosphere(subdivisions=args.mesh_subdivisions, radius=0.6)
    mesh.apply_translation([0.2, 0.0, 0.0])
    verts = torch.from_numpy(mesh.vertices).to(dtype=torch.float32)
    faces = torch.from_numpy(mesh.faces).to(dtype=torch.int64)

    cfg_common = {
        "num_samples": args.num_samples,
        "oversample_factor": 2.0,
        "max_resamples": 1,
        "min_radius": 0.4,
        "max_radius": 1.4,
        "ensure_collision_free": args.ensure_collision_free,
        "ensure_free_space": False,
        "min_distance_to_mesh": args.min_distance,
        "device": "cpu",
        "seed": 0,
        "verbosity": 0,
        "is_debug": False,
    }

    cfg_trimesh = CandidateViewGeneratorConfig(collision_backend=CollisionBackend.TRIMESH, **cfg_common)
    cfg_mojo = CandidateViewGeneratorConfig(collision_backend=CollisionBackend.MOJO, **cfg_common)

    trimesh_times, trimesh_mask, trimesh_views = _benchmark_generator(cfg_trimesh, mesh, verts, faces, args.repeats)
    mojo_times, mojo_mask, mojo_views = _benchmark_generator(cfg_mojo, mesh, verts, faces, args.repeats)

    if not torch.equal(trimesh_mask, mojo_mask):
        raise SystemExit("Backend equivalence failed: mask_valid differs between Trimesh and Mojo.")
    if not torch.allclose(trimesh_views, mojo_views, atol=1e-5):
        raise SystemExit("Backend equivalence failed: valid candidate poses differ between Trimesh and Mojo.")

    trimesh_mean = mean(trimesh_times)
    mojo_mean = mean(mojo_times)
    speedup = trimesh_mean / mojo_mean if mojo_mean > 0 else float("inf")

    print(f"mesh_faces={faces.shape[0]}")
    print(f"num_samples={args.num_samples}")
    print(f"repeats={args.repeats}")
    print(f"ensure_collision_free={args.ensure_collision_free}")
    print(f"trimesh_mean_ms={trimesh_mean:.3f}")
    print(f"mojo_mean_ms={mojo_mean:.3f}")
    print(f"speedup_vs_trimesh={speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
