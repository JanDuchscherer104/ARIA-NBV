"""View Introspection Network (VIN) for RRI prediction.

This package implements a lightweight RRI predictor head on top of a frozen EVL
backbone (EFM3D) for next-best-view scoring.
"""

from .backbone_evl import EvlBackbone, EvlBackboneConfig
from .coral import CoralLayer, coral_expected_from_logits, coral_logits_to_prob, coral_loss, ordinal_label_to_levels
from .model import VinModel, VinModelConfig
from .pose_encoding import (
    FourierFeatures,
    FourierFeaturesConfig,
    LearnableFourierFeatures,
    LearnableFourierFeaturesConfig,
)
from .rri_binning import RriOrdinalBinner
from .spherical_encoding import ShellShPoseEncoder, ShellShPoseEncoderConfig
from .types import EvlBackboneOutput, VinPrediction

__all__ = [
    "CoralLayer",
    "EvlBackbone",
    "EvlBackboneConfig",
    "EvlBackboneOutput",
    "FourierFeatures",
    "FourierFeaturesConfig",
    "LearnableFourierFeatures",
    "LearnableFourierFeaturesConfig",
    "ShellShPoseEncoder",
    "ShellShPoseEncoderConfig",
    "VinModel",
    "VinModelConfig",
    "VinPrediction",
    "RriOrdinalBinner",
    "coral_expected_from_logits",
    "coral_logits_to_prob",
    "coral_loss",
    "ordinal_label_to_levels",
]
