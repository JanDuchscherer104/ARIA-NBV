"""Tests for mixed finite-candidate generation."""

# ruff: noqa: S101

from __future__ import annotations

import pytest

pytest.importorskip("efm3d")

import torch
import trimesh
from efm3d.aria import CameraTW, PoseTW

from aria_nbv.pose_generation import (
    CandidateGenerationRuntimeContext,
    CandidateMixtureComponentConfig,
    CandidateMixtureViewGenerator,
    CandidateMixtureViewGeneratorConfig,
    CandidateViewGeneratorConfig,
    ViewDirectionMode,
    candidate_strategy_id,
)


def _identity_pose(device: torch.device | str = "cpu") -> PoseTW:
    return PoseTW(
        torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            device=device,
        )
    )


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


def _mesh_triplet(device: torch.device | str = "cpu") -> tuple[trimesh.Trimesh, torch.Tensor, torch.Tensor]:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    verts = torch.from_numpy(mesh.vertices).to(dtype=torch.float32, device=device)
    faces = torch.from_numpy(mesh.faces).to(dtype=torch.int64, device=device)
    return mesh, verts, faces


def _base_cfg() -> CandidateViewGeneratorConfig:
    return CandidateViewGeneratorConfig(
        num_samples=6,
        oversample_factor=1.0,
        min_radius=0.8,
        max_radius=0.8,
        ensure_collision_free=False,
        ensure_free_space=False,
        min_distance_to_mesh=0.0,
        view_max_azimuth_deg=0.0,
        view_max_elevation_deg=0.0,
        verbosity=0,
        seed=0,
        is_debug=True,
    )


def _run_generate(cfg: CandidateMixtureViewGeneratorConfig):
    mesh, verts, faces = _mesh_triplet(cfg.device)
    return CandidateMixtureViewGenerator(cfg).generate(
        reference_pose=_identity_pose(device=cfg.device),
        gt_mesh=mesh,
        mesh_verts=verts,
        mesh_faces=faces,
        camera_calib_template=_dummy_camera(cfg.device),
        occupancy_extent=torch.tensor([-10.0, 10.0, -10.0, 10.0, -10.0, 10.0], dtype=torch.float32),
        runtime_context=CandidateGenerationRuntimeContext(target_center_world=torch.zeros(3)),
    )


def test_mixed_sampler_fixed_counts_and_full_shell_provenance() -> None:
    cfg = CandidateMixtureViewGeneratorConfig(
        base=_base_cfg(),
        components=[
            CandidateMixtureComponentConfig(name="target", count=4, strategy=ViewDirectionMode.TARGET_POINT),
            CandidateMixtureComponentConfig(name="away", count=2, strategy=ViewDirectionMode.RADIAL_AWAY),
        ],
    )

    result = _run_generate(cfg)

    assert result.mask_valid.shape[0] == 6
    assert result.strategy_id is not None
    assert result.mixture_id is not None
    assert result.sampler_probability is not None
    assert (
        result.strategy_id.tolist()
        == [candidate_strategy_id(ViewDirectionMode.TARGET_POINT)] * 4
        + [candidate_strategy_id(ViewDirectionMode.RADIAL_AWAY)] * 2
    )
    assert result.mixture_id.tolist() == [0, 0, 0, 0, 1, 1]
    assert torch.allclose(
        result.sampler_probability,
        torch.full((6,), 1.0 / 6.0, device=result.sampler_probability.device),
    )
    assert result.views.tensor().shape[0] == int(result.mask_valid.sum().item())


def test_target_point_component_requires_runtime_target_context() -> None:
    cfg = CandidateMixtureViewGeneratorConfig(
        base=_base_cfg(),
        components=[CandidateMixtureComponentConfig(name="target", count=2, strategy=ViewDirectionMode.TARGET_POINT)],
    )
    mesh, verts, faces = _mesh_triplet(cfg.device)

    with pytest.raises(ValueError, match="target_center_world"):
        CandidateMixtureViewGenerator(cfg).generate(
            reference_pose=_identity_pose(device=cfg.device),
            gt_mesh=mesh,
            mesh_verts=verts,
            mesh_faces=faces,
            camera_calib_template=_dummy_camera(cfg.device),
            occupancy_extent=torch.tensor([-10.0, 10.0, -10.0, 10.0, -10.0, 10.0], dtype=torch.float32),
        )


def test_target_point_component_orients_towards_actor_visible_target_center() -> None:
    cfg = CandidateMixtureViewGeneratorConfig(
        base=_base_cfg(),
        components=[CandidateMixtureComponentConfig(name="target", count=4, strategy=ViewDirectionMode.TARGET_POINT)],
    )

    result = _run_generate(cfg)
    centers = result.shell_poses.t.reshape(-1, 3)
    forward = result.shell_poses.R[:, :, 2]
    to_target = torch.nn.functional.normalize(-centers, dim=1)
    cosine = (forward * to_target).sum(dim=1)

    assert torch.all(cosine > 0.99)
