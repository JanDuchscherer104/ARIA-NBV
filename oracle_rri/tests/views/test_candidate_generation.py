"""Functional tests for candidate view generation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import trimesh

from oracle_rri.views.candidate_generation import (
    CandidateViewGenerator,
    CandidateViewGeneratorConfig,
)


class _DummyPose:
    def __init__(self, mat: torch.Tensor):
        self._mat = mat

    def to(self, device: torch.device):
        # Pose content is small; keep on CPU but satisfy interface.
        return self

    @property
    def matrix3x4(self) -> torch.Tensor:
        return self._mat


def _identity_pose() -> _DummyPose:
    return _DummyPose(torch.eye(4, dtype=torch.float32)[:3])


def test_generate_basic_no_filters(monkeypatch: pytest.MonkeyPatch):
    """Generates deterministic poses without collision/free-space pruning."""

    cfg = CandidateViewGeneratorConfig(
        num_samples=4,
        ensure_collision_free=False,
        ensure_free_space=False,
        device="cpu",
    )
    gen = CandidateViewGenerator(cfg)

    def _fixed_positions(ctx: dict) -> dict:
        n = cfg.num_samples
        ctx["poses"] = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]] * n,
            device="cpu",
            dtype=torch.float32,
        )
        ctx["mask"] = torch.ones(n, dtype=torch.bool)
        return ctx

    gen.rules = [_fixed_positions]

    out = gen.generate(last_pose=_identity_pose(), gt_mesh=None)

    poses: PoseTW = out["poses"]
    mask = out["mask_valid"]
    assert mask.shape[0] == 4
    assert mask.all()
    # All positions should sit at x=+1 in world frame (since r=1, dir=+x)
    xyz = poses.matrix3x4[:, :3, 3]
    assert torch.allclose(xyz[:, 0], torch.ones(4))
    assert torch.allclose(xyz[:, 1:], torch.zeros(4, 2))


def test_min_distance_to_mesh_blocks_close_samples(monkeypatch: pytest.MonkeyPatch):
    """Min-distance rule should reject viewpoints that are inside a mesh."""

    sphere = trimesh.creation.icosphere(radius=0.05)
    cfg = CandidateViewGeneratorConfig(
        num_samples=1,
        min_distance_to_mesh=0.06,
        ensure_collision_free=False,
        ensure_free_space=False,
        device="cpu",
    )
    gen = CandidateViewGenerator(cfg)

    def _fixed_inside(ctx: dict) -> dict:
        ctx["poses"] = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        ctx["mask"] = torch.ones(1, dtype=torch.bool)
        return ctx

    gen.rules = [
        _fixed_inside,
        gen._rule_min_distance_to_mesh,
    ]

    out = gen.generate(last_pose=_identity_pose(), gt_mesh=sphere)

    assert out["mask_valid"].sum() == 0, "Candidate inside sphere should be rejected by distance rule"


def test_path_collision_rejects_blocked_rays(monkeypatch: pytest.MonkeyPatch):
    """Ray intersects blocking mesh between rig and candidate."""

    wall = trimesh.creation.box(extents=(0.1, 2.0, 2.0))
    wall.apply_translation((0.5, 0.0, 0.0))
    cfg = CandidateViewGeneratorConfig(
        num_samples=1,
        min_distance_to_mesh=0.0,
        ensure_collision_free=True,
        ensure_free_space=False,
        device="cpu",
    )
    gen = CandidateViewGenerator(cfg)

    def _fixed_target(ctx: dict) -> dict:
        ctx["poses"] = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        ctx["mask"] = torch.ones(1, dtype=torch.bool)
        return ctx

    gen.rules = [
        _fixed_target,
        gen._rule_path_collision,
    ]

    out = gen.generate(last_pose=_identity_pose(), gt_mesh=wall)

    assert out["mask_valid"].sum() == 0, "Ray should collide with wall placed at x=0.5"


def test_free_space_extent_filters(monkeypatch: pytest.MonkeyPatch):
    """Free-space rule should cull poses outside occupancy extent."""

    cfg = CandidateViewGeneratorConfig(
        num_samples=2,
        ensure_collision_free=False,
        ensure_free_space=True,
        occupancy_extent=torch.tensor([0.0, 0.5, -0.5, 0.5, -0.5, 0.5]),
        device="cpu",
    )
    gen = CandidateViewGenerator(cfg)

    def _fixed_outside(ctx: dict) -> dict:
        n = cfg.num_samples
        ctx["poses"] = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]] * n,
            dtype=torch.float32,
        )
        ctx["mask"] = torch.ones(n, dtype=torch.bool)
        return ctx

    gen.rules = [
        _fixed_outside,
        gen._rule_free_space,
    ]

    out = gen.generate(last_pose=_identity_pose(), gt_mesh=None)

    assert out["mask_valid"].sum() == 0, "All candidates lie at x=1.0 outside occupancy extent"


@pytest.mark.integration
def test_generate_with_real_mesh(monkeypatch: pytest.MonkeyPatch):
    """Smoke-test against a real ASE mesh to ensure pipeline works end-to-end."""

    mesh_paths = sorted(Path(".data/ase_meshes").glob("scene_ply_*.ply"))
    if not mesh_paths:
        pytest.skip("No ASE meshes available locally")
    mesh = trimesh.load(mesh_paths[0], process=False)

    cfg = CandidateViewGeneratorConfig(
        num_samples=8,
        min_radius=1.0,
        max_radius=1.0,
        ensure_collision_free=True,
        ensure_free_space=False,
        device="cpu",
    )
    gen = CandidateViewGenerator(cfg)
    gen.rules = [
        lambda ctx: {
            **ctx,
            "poses": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]],
                dtype=torch.float32,
            ).expand(cfg.num_samples, -1),
            "mask": torch.ones(cfg.num_samples, dtype=torch.bool),
        },
        gen._rule_path_collision,
    ]

    out = gen.generate(last_pose=_identity_pose(), gt_mesh=mesh)

    assert out["poses"].matrix3x4.shape[0] <= cfg.num_samples
    assert out["mask_valid"].dtype == torch.bool
