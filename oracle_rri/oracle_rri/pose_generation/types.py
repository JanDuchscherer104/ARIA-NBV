"""Type definitions for candidate pose generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import torch
from efm3d.aria.pose import PoseTW

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from trimesh import Trimesh  # type: ignore[import-untyped]

    from .candidate_generation import CandidateViewGeneratorConfig


class SamplingStrategy(StrEnum):
    """Sampling strategy for candidate viewpoints."""

    SHELL_UNIFORM = "shell_uniform"
    FORWARD_GAUSSIAN = "forward_gaussian"


class ViewDirectionMode(StrEnum):
    """How to derive the base camera orientation for candidates."""

    FORWARD_RIG = "forward_rig"
    RADIAL_AWAY = "radial_away"
    RADIAL_TOWARDS = "radial_towards"
    TARGET_POINT = "target_point"


class CollisionBackend(StrEnum):
    """Backend for collision tests."""

    P3D = "pytorch3d"
    PYEMBREE = "pyembree"
    TRIMESH = "trimesh"


@dataclass
class CandidateContext:
    """Mutable state passed between sampling and pruning rules."""

    # TODO Context should contain Console from CandidateViewGenerator

    cfg: "CandidateViewGeneratorConfig"
    last_pose: PoseTW

    gt_mesh: "Trimesh | None" = None
    mesh_verts: torch.Tensor | None = None
    mesh_faces: torch.Tensor | None = None
    occupancy_extent: torch.Tensor | None = None
    camera_fov: torch.Tensor | None = None

    shell_poses: PoseTW | None = None
    centers_world: torch.Tensor | None = None
    shell_offsets_rig: torch.Tensor | None = None
    mask_valid: torch.Tensor | None = None

    rule_masks: dict[str, torch.Tensor] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateSamplingResult:
    """Immutable result of candidate sampling + rule-based pruning."""

    poses: PoseTW
    mask_valid: torch.Tensor
    masks: dict[str, torch.Tensor]
    shell_poses: PoseTW
    extras: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "SamplingStrategy",
    "CollisionBackend",
    "ViewDirectionMode",
    "CandidateContext",
    "CandidateSamplingResult",
]
