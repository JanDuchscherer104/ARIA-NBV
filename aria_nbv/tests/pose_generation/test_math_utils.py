"""Tests for shared pose-generation math helpers."""

from __future__ import annotations

import torch

from aria_nbv.pose_generation.math_utils import normalize_last_dim
from aria_nbv.pose_generation.orientations import _normalise as orientation_normalise
from aria_nbv.pose_generation.plotting import _normalise as plotting_normalise


def test_normalize_last_dim_returns_unit_vectors() -> None:
    vectors = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])

    normalized = normalize_last_dim(vectors)

    lengths = normalized.norm(dim=-1)
    assert torch.allclose(lengths, torch.ones_like(lengths))


def test_pose_generation_modules_share_normalize_helper() -> None:
    assert orientation_normalise is normalize_last_dim
    assert plotting_normalise is normalize_last_dim
