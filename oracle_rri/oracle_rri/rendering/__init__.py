"""Rendering utilities (depth + point clouds) used by oracle RRI."""

from .candidate_depth_renderer import (
    CandidateDepthBatch,
    CandidateDepthRenderer,
    CandidateDepthRendererConfig,
)
from .efm3d_depth_renderer import Efm3dDepthRenderer, Efm3dDepthRendererConfig
from .plotting import depth_grid, depth_histogram, hit_ratio_bar, RenderingPlotBuilder
from .pytorch3d_depth_renderer import Pytorch3DDepthRenderer, Pytorch3DDepthRendererConfig

__all__ = [
    "CandidateDepthBatch",
    "CandidateDepthRenderer",
    "CandidateDepthRendererConfig",
    "Efm3dDepthRenderer",
    "Efm3dDepthRendererConfig",
    "depth_grid",
    "depth_histogram",
    "hit_ratio_bar",
    "RenderingPlotBuilder",
    "Pytorch3DDepthRenderer",
    "Pytorch3DDepthRendererConfig",
]
