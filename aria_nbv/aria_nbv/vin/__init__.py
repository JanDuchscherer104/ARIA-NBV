"""Public VIN package root with lazy exports.

Keep heavyweight model imports lazy so optional backends do not break unrelated
surfaces at package import time.
"""

from __future__ import annotations

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


def __getattr__(name: str):
    if name in {"EvlBackbone", "EvlBackboneConfig"}:
        from .backbone_evl import EvlBackbone, EvlBackboneConfig

        return {"EvlBackbone": EvlBackbone, "EvlBackboneConfig": EvlBackboneConfig}[name]
    if name in {"VinModelV3", "VinModelV3Config"}:
        from .model_v3 import VinModelV3, VinModelV3Config

        return {"VinModelV3": VinModelV3, "VinModelV3Config": VinModelV3Config}[name]
    if name in {"PoseEncoder", "PoseEncodingOutput", "R6dLffPoseEncoder", "R6dLffPoseEncoderConfig"}:
        from .pose_encoders import PoseEncoder, PoseEncodingOutput, R6dLffPoseEncoder, R6dLffPoseEncoderConfig

        return {
            "PoseEncoder": PoseEncoder,
            "PoseEncodingOutput": PoseEncodingOutput,
            "R6dLffPoseEncoder": R6dLffPoseEncoder,
            "R6dLffPoseEncoderConfig": R6dLffPoseEncoderConfig,
        }[name]
    if name in {"LearnableFourierFeatures", "LearnableFourierFeaturesConfig"}:
        from .pose_encoding import LearnableFourierFeatures, LearnableFourierFeaturesConfig

        return {
            "LearnableFourierFeatures": LearnableFourierFeatures,
            "LearnableFourierFeaturesConfig": LearnableFourierFeaturesConfig,
        }[name]
    if name in {"TrajectoryEncoder", "TrajectoryEncoderConfig", "TrajectoryEncodingOutput"}:
        from .traj_encoder import TrajectoryEncoder, TrajectoryEncoderConfig, TrajectoryEncodingOutput

        return {
            "TrajectoryEncoder": TrajectoryEncoder,
            "TrajectoryEncoderConfig": TrajectoryEncoderConfig,
            "TrajectoryEncodingOutput": TrajectoryEncodingOutput,
        }[name]
    if name in {"EvlBackboneOutput", "VinPrediction", "VinV3ForwardDiagnostics"}:
        from .types import EvlBackboneOutput, VinPrediction, VinV3ForwardDiagnostics

        return {
            "EvlBackboneOutput": EvlBackboneOutput,
            "VinPrediction": VinPrediction,
            "VinV3ForwardDiagnostics": VinV3ForwardDiagnostics,
        }[name]
    raise AttributeError(name)
