"""Rendering utilities (depth + point clouds) used by oracle RRI."""

from .candidate_depth_renderer import (
    CandidateDepthRenderer,
    CandidateDepthRendererConfig,
    CandidateDepths,
)
from .efm3d_depth_renderer import Efm3dDepthRenderer, Efm3dDepthRendererConfig
from .plotting import RenderingPlotBuilder, depth_grid, depth_histogram
from .pytorch3d_depth_renderer import Pytorch3DDepthRenderer, Pytorch3DDepthRendererConfig

__all__ = [
    "CandidateDepths",
    "CandidateDepthRenderer",
    "CandidateDepthRendererConfig",
    "Efm3dDepthRenderer",
    "Efm3dDepthRendererConfig",
    "depth_grid",
    "depth_histogram",
    "RenderingPlotBuilder",
    "Pytorch3DDepthRenderer",
    "Pytorch3DDepthRendererConfig",
]
