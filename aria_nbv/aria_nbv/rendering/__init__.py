"""Rendering utilities (depth + point clouds) used by oracle RRI."""

from .candidate_depth_renderer import (
    CandidateDepthRenderer,
    CandidateDepthRendererConfig,
    CandidateDepths,
    DepthRendererBackend,
)
from .camera_batches import (
    CameraBatchBackend,
    CameraBatchLike,
    NativeCameraBatch,
)
from .candidate_pointclouds import (
    CandidatePointCloudBuilder,
    CandidatePointCloudBuilderConfig,
    CandidatePointClouds,
    PointCloudBackend,
    build_candidate_pointclouds,
)
from .efm3d_depth_renderer import Efm3dDepthRenderer, Efm3dDepthRendererConfig
from .mojo_depth_renderer import MojoDepthRenderer, MojoDepthRendererConfig
from .pytorch3d_depth_renderer import Pytorch3DDepthRenderer, Pytorch3DDepthRendererConfig

__all__ = [
    "CandidateDepths",
    "CandidateDepthRenderer",
    "CandidateDepthRendererConfig",
    "DepthRendererBackend",
    "CameraBatchBackend",
    "CameraBatchLike",
    "NativeCameraBatch",
    "CandidatePointCloudBuilder",
    "CandidatePointCloudBuilderConfig",
    "CandidatePointClouds",
    "PointCloudBackend",
    "build_candidate_pointclouds",
    "Efm3dDepthRenderer",
    "Efm3dDepthRendererConfig",
    "MojoDepthRenderer",
    "MojoDepthRendererConfig",
    "Pytorch3DDepthRenderer",
    "Pytorch3DDepthRendererConfig",
]
