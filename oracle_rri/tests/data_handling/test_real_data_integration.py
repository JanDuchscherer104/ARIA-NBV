"""Integration tests that exercise real ASE/ATEK data end-to-end."""

from pathlib import Path

import pytest

pytest.importorskip("efm3d")

import torch
from efm3d.utils.depth import dist_im_to_point_cloud_im
from efm3d.utils.mesh_utils import compute_pts_to_mesh_dist

from oracle_rri.data import AseEfmDataset, AseEfmDatasetConfig
from oracle_rri.utils import Console


def _find_first_scene_with_data() -> tuple[str, Path, Path] | None:
    """Locate a scene that has both tar shards and a mesh locally."""
    mesh_dir = Path(".data/ase_meshes")
    tar_root = Path(".data/ase_efm")
    if not mesh_dir.exists() or not tar_root.exists():
        return None
    for mesh_path in sorted(mesh_dir.glob("scene_ply_*.ply")):
        scene_id = mesh_path.stem.replace("scene_ply_", "")
        scene_dir = tar_root / scene_id
        if scene_dir.exists() and list(scene_dir.glob("shards-*.tar")):
            return scene_id, scene_dir, mesh_path
    return None


@pytest.mark.integration
def test_depth_to_mesh_distance_real_sample():
    """Load a real snippet, unproject depth via efm3d, and compute distance to GT mesh."""

    located = _find_first_scene_with_data()
    if located is None:
        pytest.skip("Real ASE shards or meshes not available locally.")

    scene_id, scene_dir, mesh_path = located
    console = Console.with_prefix("integration", scene_id)

    config = AseEfmDatasetConfig(
        scene_ids=[scene_id],
        scene_to_mesh={scene_id: mesh_path},
        load_meshes=True,
        batch_size=None,
        verbosity=0,
    )

    ds = AseEfmDataset(config)
    sample = next(iter(ds))

    cam_rgb = sample.camera_rgb
    if cam_rgb.distance_m is None:
        pytest.skip("Depth channel not present in sample.")
    depth = cam_rgb.distance_m  # [T,1,H,W]
    depth0 = depth[0, 0]  # [H,W] for a single frame
    cam0 = cam_rgb.calib[0]

    pts_cam, valid = dist_im_to_point_cloud_im(depth0, cam0)
    pts_cam = pts_cam[valid].reshape(-1, 3)
    if pts_cam.numel() == 0:
        pytest.skip("No valid backprojected points in first depth frame.")

    # Convert camera points to world points: world<-rig @ rig<-cam @ p_cam.
    t_world_rig = sample.trajectory.t_world_rig[0]
    t_rig_cam = cam0.T_camera_rig.inverse()
    pts_w = t_world_rig.transform(t_rig_cam.transform(pts_cam))

    mesh = sample.mesh
    assert mesh is not None
    verts = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).long()
    if pts_w.shape[0] > 5_000:
        pts_w = pts_w[:5_000]
    if faces.shape[0] > 20_000:
        keep = torch.linspace(0, faces.shape[0] - 1, 20_000, dtype=torch.int64)
        faces = faces[keep]
    dists_np = compute_pts_to_mesh_dist(pts_w, faces, verts, step=max(1, min(int(faces.shape[0]), 5_000)))
    dists = torch.as_tensor(dists_np)

    console.log(f"points={pts_w.shape[0]}, dist_mean={dists.mean().item():.4f}m")
    assert pts_w.shape[0] > 0
    assert torch.isfinite(dists).all()
    assert dists.mean().item() >= 0.0
