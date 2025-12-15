"""End-to-end oracle RRI label generation pipeline (non-Streamlit).

This module provides a small orchestration layer that is reusable in:

- the Streamlit dashboard (visualization),
- offline dataset preprocessing, and
- training-time *online* label generation (candidates + oracle RRI).

The goal is to make data-flow explicit and to keep performance controls
local to the compute code (chunk sizes, backprojection stride, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from pydantic import Field

from ..data.efm_views import EfmSnippetView
from ..pose_generation import CandidateViewGeneratorConfig
from ..pose_generation.types import CandidateSamplingResult
from ..rendering import CandidateDepthRendererConfig, CandidateDepths, CandidatePointClouds, build_candidate_pointclouds
from ..rri_metrics.oracle_rri import OracleRRIConfig
from ..rri_metrics.types import RriResult
from ..utils import BaseConfig, Console


@dataclass(slots=True)
class OracleRriLabelBatch:
    """Outputs of the oracle label pipeline for one snippet.

    Attributes:
        sample: The input snippet.
        candidates: Candidate poses (and per-rule diagnostics if enabled).
        depths: Rendered depth maps for a subset (or all) candidates.
        candidate_pcs: Backprojected candidate point clouds + collapsed semi-dense points.
        rri: Oracle RRI per rendered candidate.
    """

    sample: EfmSnippetView
    candidates: CandidateSamplingResult
    depths: CandidateDepths
    candidate_pcs: CandidatePointClouds
    rri: RriResult


def _target_cls():
    return OracleRriLabeler


class OracleRriLabelerConfig(BaseConfig["OracleRriLabeler"]):
    """Config-as-factory wrapper for :class:`OracleRriLabeler`.

    This config composes the existing stage configs (generation, rendering,
    scoring) and adds a small number of pipeline-level knobs.
    """

    target: type["OracleRriLabeler"] = Field(default_factory=_target_cls, exclude=True)

    generator: CandidateViewGeneratorConfig = Field(default_factory=CandidateViewGeneratorConfig)
    """Candidate generation configuration."""

    depth: CandidateDepthRendererConfig = Field(default_factory=CandidateDepthRendererConfig)
    """Depth rendering configuration."""

    oracle: OracleRRIConfig = Field(default_factory=OracleRRIConfig)
    """Oracle RRI scoring configuration."""

    backprojection_stride: int = 8
    """Pixel stride used when backprojecting depth maps to point clouds."""

    output_device: str | None = None
    """Optional device to move the :class:`~oracle_rri.rri_metrics.types.RriResult` to.

    Set to ``"cpu"`` for dashboard-style plotting, or leave as ``None`` to keep
    results on the compute device (useful for training-time label generation).
    """


class OracleRriLabeler:
    """Compute oracle RRI labels for candidates in a single snippet."""

    def __init__(self, config: OracleRriLabelerConfig) -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

        self._generator = self.config.generator.setup_target()
        self._depth_renderer = self.config.depth.setup_target()
        self._oracle = self.config.oracle.setup_target()

    def run(self, sample: EfmSnippetView) -> OracleRriLabelBatch:
        """Run the full candidate→render→RRI pipeline for one snippet.

        Args:
            sample: Input snippet including semi-dense points and GT mesh tensors.

        Returns:
            A batch containing candidates, renders, backprojected point clouds,
            and oracle RRI values.
        """

        if sample.mesh_verts is None or sample.mesh_faces is None:
            raise ValueError("OracleRriLabeler requires mesh_verts/mesh_faces on the sample (enable load_meshes).")

        self.console.log(f"Running label pipeline for scene={sample.scene_id} snippet={sample.snippet_id}")

        candidates = self._generator.generate_from_typed_sample(sample)
        self.console.log(f"Generated {int(candidates.views.tensor().shape[0])} valid candidates.")

        depths = self._depth_renderer.render(sample=sample, candidates=candidates)
        self.console.log(f"Rendered depths for {int(depths.depths.shape[0])} candidates.")

        candidate_pcs = build_candidate_pointclouds(sample, depths, stride=int(self.config.backprojection_stride))
        device = candidate_pcs.points.device
        dtype = candidate_pcs.points.dtype

        self.console.log(
            "Backprojected candidate PCs: "
            f"C={int(candidate_pcs.points.shape[0])} Pmax={int(candidate_pcs.points.shape[1])} "
            f"| semidense={int(candidate_pcs.semidense_points.shape[0])}"
        )

        rri = self._oracle.score(
            points_t=candidate_pcs.semidense_points.to(device=device, dtype=dtype),
            points_q=candidate_pcs.points,
            lengths_q=candidate_pcs.lengths,
            gt_verts=sample.mesh_verts.to(device=device, dtype=dtype),
            gt_faces=sample.mesh_faces.to(device=device),
            extend=candidate_pcs.occupancy_bounds.to(device=device, dtype=dtype),
        )
        if self.config.output_device is not None:
            rri = rri.to(device=torch.device(self.config.output_device))

        return OracleRriLabelBatch(
            sample=sample,
            candidates=candidates,
            depths=depths,
            candidate_pcs=candidate_pcs,
            rri=rri,
        )


__all__ = [
    "OracleRriLabelBatch",
    "OracleRriLabeler",
    "OracleRriLabelerConfig",
]
