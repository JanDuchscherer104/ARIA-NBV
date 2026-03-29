"""Tests for candidate panel diagnostics."""

# ruff: noqa: S101, D103, SLF001

import torch
from efm3d.aria.pose import PoseTW
from aria_nbv.app.panels import candidates as candidates_panel

ORTHO_TOL = 1e-6
ORTHO_ERR = 1e-3


def test_pose_orthonormality_stats_identity() -> None:
    pose = PoseTW.from_Rt(torch.eye(3), torch.zeros(3))
    stats = candidates_panel._pose_orthonormality_stats(pose)
    assert stats["orth_max"] < ORTHO_TOL
    assert stats["orth_mean"] < ORTHO_TOL
    assert stats["axis_norm_max"] < ORTHO_TOL
    assert abs(stats["det_mean"] - 1.0) < ORTHO_TOL


def test_pose_orthonormality_stats_detects_scale() -> None:
    r = torch.eye(3)
    r[0, 0] = 2.0
    pose = PoseTW.from_Rt(r, torch.zeros(3))
    stats = candidates_panel._pose_orthonormality_stats(pose)
    assert stats["orth_max"] > ORTHO_ERR
    assert stats["axis_norm_max"] > ORTHO_ERR
    assert stats["det_mean"] > 1.0
