"""Type definitions for candidate pose generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import torch
from efm3d.aria.camera import CameraTW
from efm3d.aria.pose import PoseTW

if TYPE_CHECKING:
    from trimesh import Trimesh  # type: ignore[import-untyped]

    from .candidate_generation import CandidateViewGeneratorConfig


class SamplingStrategy(StrEnum):
    r"""Angular sampling strategy for candidate directions on S^2.

    The strategy controls how unit directions on the sphere :math:`\mathbb{S}^2` are drawn for both:

    * positional sampling of candidate camera centers (see :class:`pose_generation.samplers.PositionSampler`), and
    * optional view-direction jitter in the camera frame (see :class:`pose_generation.orientations.OrientationBuilder`).
    """

    UNIFORM_SPHERE = "uniform_sphere"
    r"""
    Draw directions uniformly on :math:`\mathbb{S}^2` using a :class:`HypersphericalUniform` distribution (constant
            density over the sphere; no directional prior).
    """
    FORWARD_POWERSPHERICAL = "forward_powerspherical"
    r"""
    Draw directions from a forward-biased Power Spherical distribution :math:`\mathcal{PS}(\mu, \kappa)` centerd on
            the device forward axis with concentration ``kappa``. Larger :math:`\kappa` yields views clustered around
            the *mean direction*; :math:`\kappa \rightarrow 0` approaches the uniform sphere.
    """


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

    cfg: "CandidateViewGeneratorConfig"
    reference_pose: PoseTW

    gt_mesh: Trimesh
    mesh_verts: torch.Tensor
    mesh_faces: torch.Tensor
    occupancy_extent: torch.Tensor
    camera_calib_template: CameraTW

    shell_poses: PoseTW
    centers_world: torch.Tensor
    shell_offsets_ref: torch.Tensor
    mask_valid: torch.Tensor

    rule_masks: dict[str, torch.Tensor] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    def record_mask(self, name: str, mask: torch.Tensor) -> None:
        """Store a copy of the cumulative validity mask for diagnostics."""

        self.rule_masks[name] = mask.clone()

    def invalidate(self, reject_mask: torch.Tensor) -> None:
        """Apply a rejection mask (True = reject) to the current validity mask."""

        self.mask_valid = self.mask_valid & (~reject_mask)

    def mark_debug(self, key: str, value: torch.Tensor) -> None:
        """Attach debug tensors in a consistent shape (clone kept to avoid side-effects)."""

        self.debug[key] = value.clone()


@dataclass
class CandidateSamplingResult:
    """Immutable result of candidate sampling + rule-based pruning."""

    views: CameraTW
    """views.T_camera_rig are candidate_camera <- reference_pose (camera pose in reference frame; intrinsics as per CandidateGenerationConfig :: camera_label)."""
    reference_pose: PoseTW
    """World <- reference_pose around which candidates are defined."""
    mask_valid: torch.Tensor
    masks: dict[str, torch.Tensor]
    shell_poses: PoseTW
    """cam2world poses for all sampled candidates (pre-pruning)."""
    shell_offsets_ref: torch.Tensor | None = None
    """Sampled offsets in reference frame for the full shell (pre-pruning)."""
    extras: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "SamplingStrategy",
    "CollisionBackend",
    "ViewDirectionMode",
    "CandidateContext",
    "CandidateSamplingResult",
]
