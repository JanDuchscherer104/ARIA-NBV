"""Tests for VinOracleBatch collation with variable candidate counts."""

from __future__ import annotations

import pytest
import torch
from efm3d.aria.pose import PoseTW
from oracle_rri.data.efm_views import VinSnippetView
from oracle_rri.data.vin_oracle_types import VinOracleBatch
from oracle_rri.vin.types import EvlBackboneOutput

pytorch3d_cameras = pytest.importorskip("pytorch3d.renderer.cameras")
PerspectiveCameras = pytorch3d_cameras.PerspectiveCameras


def _identity_pose(num: int) -> PoseTW:
    eye = torch.eye(3, dtype=torch.float32).reshape(1, 9).repeat(num, 1)
    t = torch.zeros((num, 3), dtype=torch.float32)
    return PoseTW(torch.cat([eye, t], dim=-1))


def _make_cameras(num: int) -> PerspectiveCameras:
    rot = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(num, 1, 1)
    trans = torch.zeros((num, 3), dtype=torch.float32)
    focal = torch.full((num, 2), 250.0, dtype=torch.float32)
    principal = torch.zeros((num, 2), dtype=torch.float32)
    image_size = torch.tensor([[640.0, 480.0]], dtype=torch.float32).expand(num, -1)
    return PerspectiveCameras(
        R=rot,
        T=trans,
        focal_length=focal,
        principal_point=principal,
        image_size=image_size,
        in_ndc=False,
    )


def _make_backbone() -> EvlBackboneOutput:
    t_world_voxel = _identity_pose(1)
    voxel_extent = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    occ = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
    counts = torch.zeros((1, 2, 2, 2), dtype=torch.int64)
    pts_world = torch.zeros((1, 8, 3), dtype=torch.float32)
    return EvlBackboneOutput(
        t_world_voxel=t_world_voxel,
        voxel_extent=voxel_extent,
        occ_pr=occ.clone(),
        occ_input=occ.clone(),
        free_input=occ.clone(),
        counts=counts,
        cent_pr=occ.clone(),
        pts_world=pts_world,
    )


def test_collate_vin_oracle_batches_pads_candidates() -> None:
    """Pad candidate sets and backbone outputs to a shared batch shape."""
    batch_size = 2
    max_candidates = 3
    total_cameras = batch_size * max_candidates
    batch_a = VinOracleBatch(
        efm_snippet_view=None,
        candidate_poses_world_cam=_identity_pose(2),
        reference_pose_world_rig=PoseTW(_identity_pose(1).tensor().squeeze(0)),
        rri=torch.tensor([0.1, 0.2], dtype=torch.float32),
        pm_dist_before=torch.tensor([1.0, 1.1], dtype=torch.float32),
        pm_dist_after=torch.tensor([0.9, 1.0], dtype=torch.float32),
        pm_acc_before=torch.tensor([0.5, 0.6], dtype=torch.float32),
        pm_comp_before=torch.tensor([0.4, 0.5], dtype=torch.float32),
        pm_acc_after=torch.tensor([0.3, 0.4], dtype=torch.float32),
        pm_comp_after=torch.tensor([0.2, 0.3], dtype=torch.float32),
        p3d_cameras=_make_cameras(2),
        scene_id="scene-a",
        snippet_id="snip-a",
        backbone_out=_make_backbone(),
    )
    batch_b = VinOracleBatch(
        efm_snippet_view=None,
        candidate_poses_world_cam=_identity_pose(3),
        reference_pose_world_rig=PoseTW(_identity_pose(1).tensor().squeeze(0)),
        rri=torch.tensor([0.3, 0.4, 0.5], dtype=torch.float32),
        pm_dist_before=torch.tensor([1.2, 1.3, 1.4], dtype=torch.float32),
        pm_dist_after=torch.tensor([1.1, 1.2, 1.3], dtype=torch.float32),
        pm_acc_before=torch.tensor([0.7, 0.8, 0.9], dtype=torch.float32),
        pm_comp_before=torch.tensor([0.6, 0.7, 0.8], dtype=torch.float32),
        pm_acc_after=torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32),
        pm_comp_after=torch.tensor([0.3, 0.4, 0.5], dtype=torch.float32),
        p3d_cameras=_make_cameras(3),
        scene_id="scene-b",
        snippet_id="snip-b",
        backbone_out=_make_backbone(),
    )

    batched = VinOracleBatch.collate([batch_a, batch_b])

    assert batched.candidate_poses_world_cam.shape == (batch_size, max_candidates, 12)  # noqa: S101
    assert batched.rri.shape == (batch_size, max_candidates)  # noqa: S101
    assert torch.isnan(batched.rri[0, 2])  # noqa: S101
    assert torch.isfinite(batched.rri[1]).all()  # noqa: S101

    cams = batched.p3d_cameras
    assert cams.R.shape[0] == total_cameras  # noqa: S101
    assert cams.R.shape[1:] == (3, 3)  # noqa: S101

    backbone = batched.backbone_out
    assert backbone is not None  # noqa: S101
    assert backbone.occ_pr is not None  # noqa: S101
    assert backbone.occ_pr.shape[0] == batch_size  # noqa: S101
    assert backbone.voxel_extent.shape == (batch_size, 6)  # noqa: S101


def test_collate_vin_snippet_view_pads_points_and_traj() -> None:
    """Batch VinSnippetView payloads with padded points and trajectory."""
    points_a = torch.randn(5, 4, dtype=torch.float32)
    points_b = torch.randn(3, 4, dtype=torch.float32)
    traj_a = PoseTW(torch.randn(4, 12, dtype=torch.float32))
    traj_b = PoseTW(torch.randn(4, 12, dtype=torch.float32))
    snippet_a = VinSnippetView(
        points_world=points_a,
        lengths=torch.tensor([5], dtype=torch.int64),
        t_world_rig=traj_a,
    )
    snippet_b = VinSnippetView(
        points_world=points_b,
        lengths=torch.tensor([3], dtype=torch.int64),
        t_world_rig=traj_b,
    )

    batch_a = VinOracleBatch(
        efm_snippet_view=snippet_a,
        candidate_poses_world_cam=_identity_pose(2),
        reference_pose_world_rig=PoseTW(_identity_pose(1).tensor().squeeze(0)),
        rri=torch.tensor([0.1, 0.2], dtype=torch.float32),
        pm_dist_before=torch.tensor([1.0, 1.1], dtype=torch.float32),
        pm_dist_after=torch.tensor([0.9, 1.0], dtype=torch.float32),
        pm_acc_before=torch.tensor([0.5, 0.6], dtype=torch.float32),
        pm_comp_before=torch.tensor([0.4, 0.5], dtype=torch.float32),
        pm_acc_after=torch.tensor([0.3, 0.4], dtype=torch.float32),
        pm_comp_after=torch.tensor([0.2, 0.3], dtype=torch.float32),
        p3d_cameras=_make_cameras(2),
        scene_id="scene-a",
        snippet_id="snip-a",
        backbone_out=_make_backbone(),
    )
    batch_b = VinOracleBatch(
        efm_snippet_view=snippet_b,
        candidate_poses_world_cam=_identity_pose(2),
        reference_pose_world_rig=PoseTW(_identity_pose(1).tensor().squeeze(0)),
        rri=torch.tensor([0.3, 0.4], dtype=torch.float32),
        pm_dist_before=torch.tensor([1.2, 1.3], dtype=torch.float32),
        pm_dist_after=torch.tensor([1.1, 1.2], dtype=torch.float32),
        pm_acc_before=torch.tensor([0.7, 0.8], dtype=torch.float32),
        pm_comp_before=torch.tensor([0.6, 0.7], dtype=torch.float32),
        pm_acc_after=torch.tensor([0.4, 0.5], dtype=torch.float32),
        pm_comp_after=torch.tensor([0.3, 0.4], dtype=torch.float32),
        p3d_cameras=_make_cameras(2),
        scene_id="scene-b",
        snippet_id="snip-b",
        backbone_out=_make_backbone(),
    )

    batched = VinOracleBatch.collate([batch_a, batch_b])
    assert isinstance(batched.efm_snippet_view, VinSnippetView)  # noqa: S101
    snippet = batched.efm_snippet_view
    assert snippet.points_world.shape == (2, 5, 4)  # noqa: S101
    assert torch.isnan(snippet.points_world[1, 4]).all()  # noqa: S101
    assert snippet.t_world_rig.shape == (2, 4, 12)  # noqa: S101
