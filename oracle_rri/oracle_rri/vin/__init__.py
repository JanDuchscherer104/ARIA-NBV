"""View Introspection Network (VIN) for RRI prediction.

This package implements a lightweight RRI predictor head on top of a frozen EVL
backbone (EFM3D) for next-best-view scoring.
"""

from .backbone_evl import EvlBackbone, EvlBackboneConfig
from .model_v3 import VinModelV3, VinModelV3Config
from .pose_encoders import (
    PoseEncoder,
    PoseEncodingOutput,
    R6dLffPoseEncoder,
    R6dLffPoseEncoderConfig,
)
from .pose_encoding import LearnableFourierFeatures, LearnableFourierFeaturesConfig
from .traj_encoder import TrajectoryEncoder, TrajectoryEncoderConfig, TrajectoryEncodingOutput
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
    "VinModelV3",
    "VinModelV3Config",
    "VinV3ForwardDiagnostics",
    "VinPrediction",
    "TrajectoryEncoder",
    "TrajectoryEncoderConfig",
    "TrajectoryEncodingOutput",
]
