"""Type definitions for candidate pose generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import torch
from efm3d.aria.camera import CameraTW
from efm3d.aria.pose import PoseTW

from ..utils.typed_payloads import from_serializable, to_serializable

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
class CandidateGenerationRuntimeContext:
    """Runtime-only context for target-conditioned candidate generation."""

    target_center_world: torch.Tensor | None = None
    """Actor-visible target center in world coordinates, required by TARGET_POINT components."""

    target_id: str | None = None
    """Optional stable actor-visible target id for diagnostics."""


@dataclass
class CandidateContext:
    """Mutable state passed between sampling and pruning rules."""

    cfg: "CandidateViewGeneratorConfig"
    reference_pose: PoseTW
    sampling_pose: PoseTW

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
    """World <- physical reference pose (rig) used to express candidate extrinsics. FIXME: pose is not gravity-aligned; This pose is used downstream!"""
    mask_valid: torch.Tensor
    masks: dict[str, torch.Tensor]
    shell_poses: PoseTW
    """cam2world poses for all sampled candidates (pre-pruning)."""
    shell_offsets_ref: torch.Tensor | None = None
    """Sampled offsets in the sampling frame for the full shell (pre-pruning)."""
    sampling_pose: PoseTW | None = None
    """World <- sampling pose (gravity-aligned when enabled) used to generate candidate centers. FIXME: Previously we only provided reference_pose, which was *not* gravity-aligned."""
    strategy_id: torch.Tensor | None = None
    """Full-shell candidate strategy ids aligned with ``mask_valid``."""
    mixture_id: torch.Tensor | None = None
    """Full-shell mixture component ids aligned with ``mask_valid``."""
    sampler_probability: torch.Tensor | None = None
    """Full-shell sampler probabilities aligned with ``mask_valid``."""
    component_name: tuple[str, ...] | None = None
    """Optional per-shell component names aligned with ``mask_valid``."""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_serializable(self) -> dict[str, Any]:
        """Serialize this result into a cache-friendly CPU payload."""

        return to_serializable(self)

    @classmethod
    def from_serializable(
        cls,
        payload: dict[str, Any],
        *,
        device: torch.device | None = None,
    ) -> "CandidateSamplingResult":
        """Reconstruct one result from a serialized payload.

        Args:
            payload: Serialized payload produced by :meth:`to_serializable`.
            device: Optional destination device for tensors and wrappers.

        Returns:
            Reconstructed candidate-sampling result.
        """

        return from_serializable(cls, payload, device=device)

    def poses_world_cam(self, *, device: torch.device | None = None) -> PoseTW:
        """World <- camera poses for **valid** candidates."""
        t_cam_ref = self.views.T_camera_rig.to(device=device)  # camera <- reference
        t_world_ref = self.reference_pose.to(t_cam_ref.device)  # world <- reference

        # Compose to world <- camera.
        return t_world_ref @ t_cam_ref.inverse()

    def candidate_shell_indices(self, *, device: torch.device | None = None) -> torch.Tensor:
        """Return full-shell indices aligned with ``views`` and labels.

        ``CandidateViewGenerator`` stores ``views`` as the compact valid-candidate
        table and keeps ``mask_valid``/``shell_poses`` as full-shell diagnostics.
        Some synthetic diagnostics use full-shell ``views`` directly. This helper
        accepts those two explicit layouts and rejects ambiguous combinations so
        candidate poses, depth renders, oracle labels, and serialized diagnostics
        cannot silently drift out of order.

        Args:
            device: Optional destination device for the returned indices.

        Returns:
            ``Tensor["N"]`` integer indices into the full sampled shell.

        Raises:
            ValueError: If ``views`` cannot be mapped unambiguously to the full
                candidate shell.
        """

        view_count = int(self.views.tensor().shape[0])
        target_device = device or self.views.tensor().device
        if self.mask_valid is None:
            return torch.arange(view_count, device=target_device, dtype=torch.long)

        valid_mask = self.mask_valid.to(device=target_device, dtype=torch.bool).reshape(-1)
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).reshape(-1)
        if valid_indices.numel() == view_count:
            return valid_indices
        if valid_mask.numel() == view_count:
            return torch.arange(view_count, device=target_device, dtype=torch.long)

        raise ValueError(
            "Candidate views cannot be mapped to full-shell indices: "
            f"views={view_count}, valid_count={valid_indices.numel()}, mask_width={valid_mask.numel()}. "
            "Expected compact valid views or full-shell views.",
        )

    def get_offsets_and_dirs_ref(
        self,
        *,
        display_rotate: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Offsets and forward directions in the **physical reference frame**.

        Args:
            display_rotate: If ``True``, apply the visual 90° CW yaw rotation
                (``rotate_yaw_cw90``) to match UI plots. Defaults to ``False`` so
                downstream geometry keeps physical Aria frames.

        Returns:
            Tuple ``(offsets, dirs)`` with shapes ``(N,3)`` each.
        """

        poses_cam_ref = self.views.T_camera_rig  # camera<-reference
        if display_rotate:
            from aria_nbv.utils import rotate_yaw_cw90

            poses_cam_ref = rotate_yaw_cw90(poses_cam_ref)

        offsets = poses_cam_ref.inverse().t.view(-1, 3)  # camera->reference
        z_cam = (
            torch.tensor([0.0, 0.0, 1.0], device=offsets.device, dtype=offsets.dtype)
            .view(1, 3)
            .expand(offsets.shape[0], 3)
        )
        dirs = poses_cam_ref.inverse().rotate(z_cam).view(-1, 3)
        dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
        return offsets, dirs


__all__ = [
    "SamplingStrategy",
    "CandidateGenerationRuntimeContext",
    "CollisionBackend",
    "ViewDirectionMode",
    "CandidateContext",
    "CandidateSamplingResult",
]
