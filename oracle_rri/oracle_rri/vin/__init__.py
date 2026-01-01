"""View Introspection Network (VIN) for RRI prediction.

This package implements a lightweight RRI predictor head on top of a frozen EVL
backbone (EFM3D) for next-best-view scoring.
"""

from .backbone_evl import EvlBackbone, EvlBackboneConfig
from .model import VinModel, VinModelConfig
from .model_v2 import VinModelV2, VinModelV2Config
from .plotting import (
    build_candidate_encoding_figures,
    build_geometry_overview_figure,
    build_scene_field_evidence_figures,
    build_vin_encoding_figures,
    plot_vin_encodings_from_debug,
    save_vin_encoding_figures,
)
from .pose_encoders import (
    PoseEncoder,
    PoseEncoderConfig,
    PoseEncodingOutput,
    R6dLffPoseEncoder,
    R6dLffPoseEncoderConfig,
    ShellLffPoseEncoder,
    ShellLffPoseEncoderConfig,
    ShellShPoseEncoderAdapter,
    ShellShPoseEncoderAdapterConfig,
)
from .pose_encoding import (
    FourierFeatures,
    FourierFeaturesConfig,
    LearnableFourierFeatures,
    LearnableFourierFeaturesConfig,
)
from .spherical_encoding import ShellShPoseEncoder, ShellShPoseEncoderConfig
from .types import EvlBackboneOutput, VinForwardDiagnostics, VinPrediction

__all__ = [
    "CoralLayer",
    "EvlBackbone",
    "EvlBackboneConfig",
    "EvlBackboneOutput",
    "FourierFeatures",
    "FourierFeaturesConfig",
    "LearnableFourierFeatures",
    "LearnableFourierFeaturesConfig",
    "PoseEncoder",
    "PoseEncoderConfig",
    "PoseEncodingOutput",
    "R6dLffPoseEncoder",
    "R6dLffPoseEncoderConfig",
    "ShellLffPoseEncoder",
    "ShellLffPoseEncoderConfig",
    "ShellShPoseEncoderAdapter",
    "ShellShPoseEncoderAdapterConfig",
    "ShellShPoseEncoder",
    "ShellShPoseEncoderConfig",
    "VinModel",
    "VinModelConfig",
    "VinModelV2",
    "VinModelV2Config",
    "VinForwardDiagnostics",
    "VinPrediction",
    "build_candidate_encoding_figures",
    "build_geometry_overview_figure",
    "build_scene_field_evidence_figures",
    "build_vin_encoding_figures",
    "plot_vin_encodings_from_debug",
    "RriOrdinalBinner",
    "save_vin_encoding_figures",
    "coral_expected_from_logits",
    "coral_logits_to_prob",
    "coral_loss",
]
