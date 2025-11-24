import torch
import trimesh
from efm3d.aria.aria_constants import ARIA_POINTS_VOL_MAX, ARIA_POINTS_VOL_MIN, ARIA_POINTS_WORLD

from oracle_rri.data.efm_dataset import crop_mesh_with_bounds, infer_semidense_bounds


def test_infer_bounds_prefers_volume_keys():
    efm = {
        ARIA_POINTS_VOL_MIN: torch.tensor([0.0, 0.0, 0.0]),
        ARIA_POINTS_VOL_MAX: torch.tensor([1.0, 1.0, 1.0]),
        ARIA_POINTS_WORLD: torch.zeros((1, 2, 3)),
    }

    bounds = infer_semidense_bounds(efm)
    assert bounds is not None
    lo, hi = bounds
    assert torch.allclose(lo, torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(hi, torch.tensor([0.0, 0.0, 0.0]))


def test_infer_bounds_falls_back_to_points_and_lengths():
    points = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
                [float("nan"), float("nan"), float("nan")],
                [10.0, 10.0, 10.0],
            ]
        ]
    )
    lengths = torch.tensor([2])
    efm = {ARIA_POINTS_WORLD: points, "msdpd#points_world_lengths": lengths}

    bounds = infer_semidense_bounds(efm)
    assert bounds is not None
    lo, hi = bounds
    assert torch.allclose(lo, torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(hi, torch.tensor([1.0, 2.0, 3.0]))


def test_crop_mesh_with_bounds_removes_far_geometry_and_caps_faces():
    box_near = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box_far = box_near.copy()
    box_far.apply_translation([5.0, 0.0, 0.0])
    mesh = trimesh.util.concatenate([box_near, box_far])

    bounds = (torch.tensor([-1.0, -1.0, -1.0]), torch.tensor([1.0, 1.0, 1.0]))

    cropped = crop_mesh_with_bounds(mesh, bounds, margin_m=0.2)
    assert cropped.faces.shape[0] < mesh.faces.shape[0]
    assert cropped.bounds[1, 0] < 2.0  # far box removed

    capped = crop_mesh_with_bounds(mesh, bounds, margin_m=0.2, max_faces=8)
    assert capped.faces.shape[0] <= 8
