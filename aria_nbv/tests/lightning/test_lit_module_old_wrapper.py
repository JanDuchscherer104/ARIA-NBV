"""Compatibility tests for the legacy ``lit_module_old`` import path."""

from __future__ import annotations

from aria_nbv.lightning.lit_module import VinLightningModule, VinLightningModuleConfig
from aria_nbv.lightning.lit_module_old import (
    AdamWConfig,
    OneCycleSchedulerConfig,
    ReduceLrOnPlateauConfig,
)
from aria_nbv.lightning.lit_module_old import (
    VinLightningModule as LegacyVinLightningModule,
)
from aria_nbv.lightning.lit_module_old import (
    VinLightningModuleConfig as LegacyVinLightningModuleConfig,
)
from aria_nbv.lightning.optimizers import (
    AdamWConfig as CanonicalAdamWConfig,
)
from aria_nbv.lightning.optimizers import (
    OneCycleSchedulerConfig as CanonicalOneCycleSchedulerConfig,
)
from aria_nbv.lightning.optimizers import (
    ReduceLrOnPlateauConfig as CanonicalReduceLrOnPlateauConfig,
)


def test_lit_module_old_reexports_canonical_symbols() -> None:
    """Keep the legacy module path as a thin alias to the canonical owners."""

    assert LegacyVinLightningModule is VinLightningModule  # noqa: S101
    assert LegacyVinLightningModuleConfig is VinLightningModuleConfig  # noqa: S101
    assert AdamWConfig is CanonicalAdamWConfig  # noqa: S101
    assert OneCycleSchedulerConfig is CanonicalOneCycleSchedulerConfig  # noqa: S101
    assert ReduceLrOnPlateauConfig is CanonicalReduceLrOnPlateauConfig  # noqa: S101
