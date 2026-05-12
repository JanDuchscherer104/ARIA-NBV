r"""High-level oracle RRI computation orchestrator.

This module wires together rendering outputs (candidate depth → point clouds),
distance primitives (see ``metrics.py``), and configuration defaults to produce
per-candidate Relative Reconstruction Improvement (RRI) scores as defined in
``docs/contents/theory/rri_theory.qmd``. The implementation is intentionally
kept modular:

* **Sampling / downsampling** is delegated to callers (or future helpers) so
  that density can be harmonised between ``P_t`` and ``P_{t∪q}``.
* **Distance evaluation** is performed via PyTorch3D on GPU for efficiency.
* **Config-as-factory** pattern is used to keep runtime objects
  serialisable and consistent with the rest of the codebase.

The implemented scalar score is

$$
\mathrm{RRI}(q) =
\frac{\Delta(P_t, M) - \Delta(P_t \cup P_q, M)}
     {\max(\Delta(P_t, M), \epsilon)}
$$

where $\Delta$ is the configured point-mesh error. Target-specific callers pass
target-cropped points and meshes; invalid crops raise upstream and must not be
silently converted to scene-level labels.
"""

from __future__ import annotations

import torch

from aria_nbv.utils.base_config import TargetConfig

from .metrics import chamfer_point_mesh, chamfer_point_mesh_batched
from .types import RriResult


class OracleRRIConfig(TargetConfig["OracleRRI"]):
    """Config-as-factory wrapper for oracle RRI computation."""

    @property
    def target_type(self) -> type["OracleRRI"]:
        return OracleRRI


class OracleRRI:
    """Facade to compute oracle RRI for one or more candidates.

    Conceptual steps:
        1. Merge ``P_t`` (semi-dense SLAM) with candidate view point cloud
           ``P_q`` to obtain ``P_{t∪q}``.
        2. (Optional) Voxel-downsample both ``P_t`` and ``P_{t∪q}`` to ensure
           comparable density when evaluating point-mesh distances.
        3. Compute accuracy/completeness distances to the GT mesh using the
           PyTorch3D backend.
        4. Form RRI = (d_before - d_after) / d_before and return diagnostics.
    """

    config: OracleRRIConfig

    def __init__(self, config: OracleRRIConfig) -> None:
        self.config = config

    def score(
        self,
        *,
        points_t: torch.Tensor,
        points_q: torch.Tensor,
        lengths_q: torch.Tensor,
        gt_verts: torch.Tensor,
        gt_faces: torch.Tensor,
        extend: torch.Tensor,
    ) -> RriResult:
        """Compute RRI for one or more candidates in a single forward pass.

        Args:
            points_t: ``Tensor['N_t', 3]`` semi-dense SLAM point cloud up to time *t*.
            points_q: ``Tensor['N_q', 3]`` candidate-view point cloud rendered from GT.
            gt_verts: ``Tensor['V', 3]`` ground-truth mesh vertices.
            gt_faces: ``Tensor['F', 3]`` ground-truth mesh face indices (int64).
            extend: ``Tensor[6]`` [xmin, xmax, ymin, ymax, zmin, zmax] AABB in world frame used to crop the GT mesh.
        Returns:
            ``RriResult`` containing scalar RRI and distance breakdowns.
        """

        gt_verts_crop, gt_faces_crop = _crop_mesh_to_aabb(gt_verts, gt_faces, extend)
        lengths_q = lengths_q.to(device=points_q.device)

        dist_before = chamfer_point_mesh(points_t, gt_verts_crop, gt_faces_crop)
        num_t = points_t.shape[0]
        points_t_exp = points_t.unsqueeze(0).expand(points_q.shape[0], num_t, 3)
        points_tq = torch.cat([points_t_exp, points_q], dim=1)
        lengths_tq = lengths_q + num_t

        dist_after = chamfer_point_mesh_batched(points_tq, lengths_tq, gt_verts_crop, gt_faces_crop)

        denom = dist_before.bidirectional.clamp_min(1e-12)
        rri_all = (dist_before.bidirectional - dist_after.bidirectional) / denom

        return RriResult(
            rri=rri_all,
            pm_dist_before=dist_before.bidirectional.expand_as(rri_all),
            pm_dist_after=dist_after.bidirectional,
            pm_acc_before=dist_before.accuracy.expand_as(rri_all),
            pm_comp_before=dist_before.completeness.expand_as(rri_all),
            pm_acc_after=dist_after.accuracy,
            pm_comp_after=dist_after.completeness,
        )

    def score_batch(
        self,
        *,
        points_t: torch.Tensor,
        points_q: torch.Tensor,
        lengths_q: torch.Tensor,
        gt_verts: torch.Tensor,
        gt_faces: torch.Tensor,
        extend: torch.Tensor,
    ) -> RriResult:
        """Alias kept for callers using the old batch name."""

        return self.score(
            points_t=points_t,
            points_q=points_q,
            lengths_q=lengths_q,
            gt_verts=gt_verts,
            gt_faces=gt_faces,
            extend=extend,
        )


def _crop_mesh_to_aabb(
    verts: torch.Tensor,
    faces: torch.Tensor,
    aabb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop a mesh to an AABB with explicit empty-crop failure.

    The current crop contract keeps faces whose **any** vertex lies inside the
    world-frame box ``[xmin, xmax, ymin, ymax, zmin, zmax]``. Empty crops raise
    instead of silently falling back to full-scene scoring, because target-aware
    RRI must not turn a missing object crop into a scene-level label.
    """

    if aabb.numel() != 6:
        raise ValueError("extend must be Tensor[6] = [xmin, xmax, ymin, ymax, zmin, zmax]")

    bounds = aabb.reshape(6).to(device=verts.device, dtype=verts.dtype)
    lower = bounds[[0, 2, 4]]
    upper = bounds[[1, 3, 5]]
    if bool(torch.any(lower > upper).item()):
        raise ValueError("extend min bounds must be <= max bounds for x/y/z.")

    vmask = torch.all((verts >= lower) & (verts <= upper), dim=1)

    # Keep faces that intersect the AABB (coarse test via any-vertex-inside).
    fmask = vmask[faces].any(dim=1)
    faces_kept = faces[fmask]
    if faces_kept.numel() == 0:
        raise ValueError(
            "AABB crop produced no mesh faces; refusing to fall back to full-scene RRI.",
        )

    unique_idx, new_idx = torch.unique(faces_kept.reshape(-1), sorted=True, return_inverse=True)
    verts_crop = verts[unique_idx]
    faces_crop = new_idx.reshape(faces_kept.shape)
    return verts_crop, faces_crop


__all__ = ["OracleRRI", "OracleRRIConfig"]
