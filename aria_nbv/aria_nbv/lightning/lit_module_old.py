"""Legacy compatibility wrapper for the old Lightning module path.

The canonical VIN Lightning module implementation now lives in
:mod:`aria_nbv.lightning.lit_module`, while optimizer configs live in
:mod:`aria_nbv.lightning.optimizers`. This module remains only as the historical
import surface for callers that still import ``aria_nbv.lightning.lit_module_old``.
"""

from __future__ import annotations

from .lit_module import VinLightningModule, VinLightningModuleConfig
from .optimizers import AdamWConfig, OneCycleSchedulerConfig, ReduceLrOnPlateauConfig

__all__ = [
    "AdamWConfig",
    "OneCycleSchedulerConfig",
    "ReduceLrOnPlateauConfig",
    "VinLightningModule",
    "VinLightningModuleConfig",
]
