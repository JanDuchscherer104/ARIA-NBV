"""Integration tests that exercise real ASE/ATEK data end-to-end."""

from pathlib import Path

import pytest
import torch

from efm3d.utils.depth import dist_im_to_point_cloud_im
from efm3d.utils.mesh_utils import compute_pts_to_mesh_dist
from oracle_rri.data_handling import ASEDataset, ASEDatasetConfig
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

    config = ASEDatasetConfig(
        tar_urls=[str(scene_dir / "shards-*.tar")],
        scene_to_mesh={scene_id: mesh_path},
        load_meshes=True,
        batch_size=None,
        shuffle=False,
        repeat=False,
        verbose=False,
    )

    ds = ASEDataset(config)
    sample = next(iter(ds))

    if sample.atek.camera_rgb_depth is None or sample.atek.camera_rgb_depth.images is None:
        pytest.skip("Depth stream not present in sample.")

    cam_tw = sample.atek.camera_rgb_depth.to_camera_tw()
    depth = sample.atek.camera_rgb_depth.images  # [T,1,H,W]
    # Use first frame
    depth0 = depth[:1]
    cam0 = cam_tw[0:1]

    pts_w, _ = dist_im_to_point_cloud_im(depth0, cam0)
    pts_w = pts_w.reshape(-1, 3)

    # Compute point-to-mesh distance using efm3d helper
    mesh = sample.gt_mesh
    assert mesh is not None
    verts = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).long()
    dists = compute_pts_to_mesh_dist(pts_w, faces, verts)

    console.log(f"points={pts_w.shape[0]}, dist_mean={dists.mean().item():.4f}m")
    assert pts_w.shape[0] > 0
    assert torch.isfinite(dists).all()
    assert dists.mean().item() >= 0.0
