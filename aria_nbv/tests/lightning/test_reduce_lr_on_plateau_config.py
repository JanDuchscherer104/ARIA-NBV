"""Tests for ReduceLROnPlateau config exposure."""

# ruff: noqa: S101

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

PATIENCE_STEPS = 4
COOLDOWN_STEPS = 2


def _load_optimizers_module() -> types.ModuleType:
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "oracle_rri"
    lightning_root = pkg_root / "lightning"
    module_path = lightning_root / "optimizers.py"

    if "oracle_rri" not in sys.modules:
        pkg = types.ModuleType("oracle_rri")
        pkg.__path__ = [str(pkg_root)]
        sys.modules["oracle_rri"] = pkg
    if "oracle_rri.lightning" not in sys.modules:
        subpkg = types.ModuleType("oracle_rri.lightning")
        subpkg.__path__ = [str(lightning_root)]
        sys.modules["oracle_rri.lightning"] = subpkg

    spec = importlib.util.spec_from_file_location(
        "oracle_rri.lightning.optimizers",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load optimizers module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reduce_lr_on_plateau_config_params() -> None:
    """ReduceLROnPlateau config forwards fields into the scheduler."""
    module = _load_optimizers_module()
    cfg_cls = module.ReduceLrOnPlateauConfig

    cfg = cfg_cls(
        mode="max",
        factor=0.3,
        patience=PATIENCE_STEPS,
        threshold=0.01,
        threshold_mode="abs",
        cooldown=COOLDOWN_STEPS,
        min_lr=1e-6,
        eps=1e-9,
    )
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = cfg.setup_target(optimizer)

    assert scheduler.mode == "max"
    assert scheduler.factor == pytest.approx(0.3)
    assert scheduler.patience == PATIENCE_STEPS
    assert scheduler.threshold == pytest.approx(0.01)
    assert scheduler.threshold_mode == "abs"
    assert scheduler.cooldown == COOLDOWN_STEPS
    assert scheduler.min_lrs == [pytest.approx(1e-6)]
    assert scheduler.eps == pytest.approx(1e-9)
