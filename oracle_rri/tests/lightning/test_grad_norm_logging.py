"""Tests for gradient-norm logging configuration."""

# ruff: noqa: S101

from __future__ import annotations

import pytest
import torch
from torch import nn

from oracle_rri.utils.grad_norms import (
    GradNormLoggingConfig,
    _collect_grad_norm_targets,
    _grad_norm_from_params,
)


class _DummyGlobalPooler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pos_grid_encoder = nn.Linear(2, 2)
        self.q_proj = nn.Linear(2, 2)


class _DummyPoseEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pose_encoder_lff = nn.Linear(2, 2)


class _DummyTrajEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pose_encoder = _DummyPoseEncoder()


class _DummyVin(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pose_encoder = nn.Linear(2, 2)
        self.global_pooler = _DummyGlobalPooler()
        self.traj_encoder = _DummyTrajEncoder()
        self.empty_block = nn.Module()


def _target_names(vin: nn.Module, cfg: GradNormLoggingConfig) -> list[str]:
    return [name for name, _ in _collect_grad_norm_targets(vin, cfg)]


def test_grad_norm_targets_depth_default() -> None:
    """Default logging targets only depth-2 modules."""
    vin = _DummyVin()
    cfg = GradNormLoggingConfig(group_depth=2)
    names = _target_names(vin, cfg)
    assert "pose_encoder" in names
    assert "global_pooler" in names
    assert "traj_encoder" in names
    assert "global_pooler.pos_grid_encoder" not in names
    assert "traj_encoder.pose_encoder.pose_encoder_lff" not in names
    assert "empty_block" not in names


def test_grad_norm_targets_with_includes() -> None:
    """Include patterns add deeper modules without removing depth-2 defaults."""
    vin = _DummyVin()
    cfg = GradNormLoggingConfig(
        group_depth=2,
        include=[
            "global_pooler.pos_grid_encoder",
            "traj_encoder.pose_encoder.pose_encoder_lff",
        ],
    )
    names = _target_names(vin, cfg)
    assert "global_pooler" in names
    assert "global_pooler.pos_grid_encoder" in names
    assert "traj_encoder" in names
    assert "traj_encoder.pose_encoder.pose_encoder_lff" in names


def test_grad_norm_values() -> None:
    """Grad-norm calculations follow the selected norm type."""
    param = nn.Parameter(torch.zeros(2))
    param.grad = torch.tensor([3.0, 4.0])
    params = [param]
    assert _grad_norm_from_params(params, "L2") == pytest.approx(5.0)
    assert _grad_norm_from_params(params, "L1") == pytest.approx(7.0)
    assert _grad_norm_from_params(params, "Linf") == pytest.approx(4.0)
