"""Type definitions for candidate pose generation.

This module centralises small enums and TypedDicts shared between the
candidate generation core and individual rule implementations.
"""
from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from efm3d.aria import PoseTW
    from torch import Tensor, device
    from trimesh import Trimesh


class SamplingStrategy(StrEnum):
    """Sampling strategy for candidate viewpoints.

    The strategy controls how directions on the spherical shell around the
    current camera pose are sampled. Both strategies operate on a *spherical
    cap* (elevation band) defined by ``min_elev_deg`` and ``max_elev_deg`` and
    use standard spherical coordinates:

    .. math::

        x = \\cos(\\text{elev}) \\cos(\\text{az}), \\\\
        y = \\sin(\\text{elev}), \\\\
        z = \\cos(\\text{elev}) \\sin(\\text{az}).

    For background on sphere sampling see, e.g.,
    `Sphere point picking <https://mathworld.wolfram.com/SpherePointPicking.html>`_
    or the discussion in
    `Ray Tracing: The Rest of Your Life <https://raytracing.github.io/books/RayTracingTheRestOfYourLife.html>`_.
    """

    SHELL_UNIFORM = "shell_uniform"
    """Sample directions area-uniformly over a spherical band.

    Elevation is drawn using the inverse-CDF of sin(elev) between
    `min_elev_deg` and `max_elev_deg`, so that equal solid angle regions
    have equal probability. Azimuth is uniform over either a full circle
    or half-sphere depending on `azimuth_full_circle`.
    """

    FORWARD_GAUSSIAN = "forward_gaussian"
    """Sample directions from a Gaussian around the forward axis.

    Elevation is drawn from a truncated normal centred between
    `min_elev_deg` and `max_elev_deg` and clamped to that band. This
    biases viewpoints towards the current forward direction while still
    respecting the configured angular limits.
    """


class CollisionBackend(StrEnum):
    """Backend for collision tests.

    The backend controls how rays are intersected with the GT mesh in
    :class:`oracle_rri.pose_generation.candidate_generation_rules.PathCollisionRule`.
    """

    PYEMBREE = "pyembree"
    """Use :class:`trimesh.ray.ray_pyembree.RayMeshIntersector` for fast
    ray-mesh tests when available.

    See
    [:class:`trimesh.ray.ray_pyembree.RayMeshIntersector`](https://trimesh.org/trimesh.ray.ray_pyembree.html#trimesh.ray.ray_pyembree.RayMeshIntersector)
    for implementation details.
    """

    TRIMESH = "trimesh"


class CandidateContext(TypedDict):
    """Internal mutable context passed between sampling rules."""

    last_pose: PoseTW
    gt_mesh: Trimesh | None
    occupancy_extent: Tensor | None
    device: device
    poses: Tensor
    mask: Tensor


class CandidateSamplingResult(TypedDict):
    """Typed mapping for candidate generation output.

    Attributes:
        poses: PoseTW containing the final, filtered candidate viewpoints in Aria world frame (T_world_camera).
        mask_valid: Boolean mask of length `poses.batch_size` indicating which sampled poses survived all rules.
        masks: Per-rule boolean masks in the order rules were applied; useful for debugging which rule killed which candidates.
        shell_poses: Raw sampled poses on the spherical shell *before* applying any rules, shape (N_shell, 12) as flattened [R|t] blocks. This is used purely for visualising the sampling
            distribution.
    """

    poses: PoseTW
    """PoseTW containing the final, filtered candidate viewpoints in Aria world frame (T_world_camera). """
    mask_valid: Tensor
    """Boolean mask of length `poses.batch_size` indicating which sampled poses survived all rules."""
    masks: list[Tensor]
    """Per-rule boolean masks in the order rules were applied; useful for debugging which rule killed which candidates."""
    shell_poses: Tensor  # (N_shell, 12) float32
    """Raw sampled poses on the spherical shell *before* applying any rules, shape (N_shell, 12) as flattened [R|t] blocks. This is used purely for visualising the sampling distribution."""
