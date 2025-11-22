"""Rendering utilities (depth + point clouds) used by oracle RRI."""

from .candidate_depth_renderer import (
    CandidateDepthBatch,
    CandidateDepthRenderer,
    CandidateDepthRendererConfig,
)
from .plotting import depth_grid
from .pytorch3d_depth_renderer import Pytorch3DDepthRenderer, Pytorch3DDepthRendererConfig

__all__ = [
    "CandidateDepthBatch",
    "CandidateDepthRenderer",
    "CandidateDepthRendererConfig",
    "depth_grid",
    "Pytorch3DDepthRenderer",
    "Pytorch3DDepthRendererConfig",
]
