"""View Introspection Network (VIN) package root.

This module exposes the public VIN surface lazily so lightweight consumers can
import VIN types without triggering the full model stack during package
initialization. That keeps ``aria_nbv.data_handling`` free of avoidable import
cycles while preserving the existing public names under ``aria_nbv.vin``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_TO_MODULE = {
    "EvlBackbone": ".backbone_evl",
    "EvlBackboneConfig": ".backbone_evl",
    "EvlBackboneOutput": ".types",
    "LearnableFourierFeatures": ".pose_encoding",
    "LearnableFourierFeaturesConfig": ".pose_encoding",
    "PoseEncoder": ".pose_encoders",
    "PoseEncodingOutput": ".pose_encoders",
    "R6dLffPoseEncoder": ".pose_encoders",
    "R6dLffPoseEncoderConfig": ".pose_encoders",
    "TrajectoryEncoder": ".traj_encoder",
    "TrajectoryEncoderConfig": ".traj_encoder",
    "TrajectoryEncodingOutput": ".traj_encoder",
    "VinModelV3": ".model_v3",
    "VinModelV3Config": ".model_v3",
    "VinPrediction": ".types",
    "VinV3ForwardDiagnostics": ".types",
}

__all__ = sorted(_EXPORT_TO_MODULE)


def __getattr__(name: str) -> Any:
    """Lazily resolve public VIN exports on first access.

    Args:
        name: Requested attribute name.

    Returns:
        Resolved public object.

    Raises:
        AttributeError: If ``name`` is not a public VIN export.
    """

    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
