"""Small typed containers for VIN (View Introspection Network)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from efm3d.aria.pose import PoseTW

Tensor = torch.Tensor


@dataclass(slots=True)
class EvlBackboneOutput:
    """EVL backbone features used by VIN.

    Attributes:
        occ_feat: Optional ``Tensor["B C D H W", float32]`` neck features for occupancy.
        obb_feat: Optional ``Tensor["B C D H W", float32]`` neck features for OBB detection.
        occ_pr: Optional ``Tensor["B 1 D H W", float32]`` EVL occupancy probability.
        occ_input: Optional ``Tensor["B 1 D H W", float32]`` voxelized occupied evidence from input points.
        free_input: Optional ``Tensor["B 1 D H W", float32]`` voxelized free-space evidence (ray samples).
        counts: Optional ``Tensor["B D H W", int64]`` per-voxel observation counts.
        counts_m: Optional ``Tensor["B D H W", int64]`` masked/debug variant of counts.
        t_world_voxel: ``PoseTW["B 12"]`` world←voxel pose for the voxel grid.
        voxel_extent: ``Tensor["6", float32]`` voxel grid extent in voxel frame
            ``[x_min,x_max,y_min,y_max,z_min,z_max]`` (meters).
    """

    t_world_voxel: PoseTW
    voxel_extent: Tensor
    occ_feat: Tensor | None = None
    obb_feat: Tensor | None = None
    occ_pr: Tensor | None = None
    occ_input: Tensor | None = None
    free_input: Tensor | None = None
    counts: Tensor | None = None
    counts_m: Tensor | None = None


@dataclass(slots=True)
class VinPrediction:
    """VIN predictions for a candidate set."""

    logits: Tensor
    """``Tensor["B N K-1", float32]`` CORAL logits (K ordinal classes)."""

    prob: Tensor
    """``Tensor["B N K", float32]`` Class probabilities derived from CORAL logits."""

    expected: Tensor
    """``Tensor["B N", float32]`` Expected class value in ``[0, K-1]``."""

    expected_normalized: Tensor
    """``Tensor["B N", float32]`` Expected value normalized to ``[0, 1]``."""

    candidate_valid: Tensor
    """``Tensor["B N", bool]`` Candidate-in-voxel-grid validity mask."""
