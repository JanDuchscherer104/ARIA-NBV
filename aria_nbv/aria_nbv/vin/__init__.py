"""Public VIN package root with explicit imports and exports."""

from __future__ import annotations

from .backbone_evl import EvlBackbone, EvlBackboneConfig
from .model_v3 import VinModelV3, VinModelV3Config
from .pose_encoders import (
    PoseEncoder,
    PoseEncodingOutput,
    R6dLffPoseEncoder,
    R6dLffPoseEncoderConfig,
)
from .pose_encoding import LearnableFourierFeatures, LearnableFourierFeaturesConfig
from .traj_encoder import (
    TrajectoryEncoder,
    TrajectoryEncoderConfig,
    TrajectoryEncodingOutput,
)
from .types import EvlBackboneOutput, VinPrediction, VinV3ForwardDiagnostics

__all__ = [
    "EvlBackbone",
    "EvlBackboneConfig",
    "EvlBackboneOutput",
    "LearnableFourierFeatures",
    "LearnableFourierFeaturesConfig",
    "PoseEncoder",
    "PoseEncodingOutput",
    "R6dLffPoseEncoder",
    "R6dLffPoseEncoderConfig",
    "TrajectoryEncoder",
    "TrajectoryEncoderConfig",
    "TrajectoryEncodingOutput",
    "VinModelV3",
    "VinModelV3Config",
    "VinPrediction",
    "VinV3ForwardDiagnostics",
]
