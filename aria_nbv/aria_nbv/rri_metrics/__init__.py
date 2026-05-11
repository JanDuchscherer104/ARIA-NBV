"""Oracle RRI metrics, ordinal labels, and VIN logging utilities.

RRI is computed from point-mesh error before and after adding a candidate view.
The directional components are accuracy (point to mesh) and completeness (mesh
to point); their scalarized sum is used by the current oracle implementation.
For target-aware labels, callers crop points and mesh to the matched target and
must mark empty or ambiguous crops invalid rather than assigning low RRI.

CORAL utilities convert continuous RRI-derived supervision into ordered bins for
VIN-style one-step scoring. Ordinal predictions are ranking/calibration signals,
not geometry metrics by themselves.
"""

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
