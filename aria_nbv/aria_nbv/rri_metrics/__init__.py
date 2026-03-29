"""RRI metric utilities and VIN logging metrics."""

from .coral import (
    CoralLayer,
    coral_expected_from_logits,
    coral_logits_to_prob,
    coral_loss,
    coral_random_loss,
)
from .logging import (
    LabelHistogram,
    Loss,
    Metric,
    RriErrorStats,
    VinMetrics,
    VinMetricsConfig,
    loss_key,
    metric_key,
    topk_accuracy_from_probs,
)
from .metrics import chamfer_point_mesh, chamfer_point_mesh_batched
from .rri_binning import RriOrdinalBinner, ordinal_labels_to_levels
from .types import DistanceAggregation, DistanceBreakdown, RriResult

__all__ = [
    "CoralLayer",
    "LabelHistogram",
    "Loss",
    "Metric",
    "RriErrorStats",
    "RriOrdinalBinner",
    "VinMetrics",
    "VinMetricsConfig",
    "loss_key",
    "metric_key",
    "topk_accuracy_from_probs",
    "chamfer_point_mesh",
    "chamfer_point_mesh_batched",
    "DistanceAggregation",
    "DistanceBreakdown",
    "RriResult",
    "coral_expected_from_logits",
    "coral_logits_to_prob",
    "coral_loss",
    "coral_random_loss",
    "ordinal_labels_to_levels",
]
