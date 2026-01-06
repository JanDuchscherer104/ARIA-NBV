"""Shared utilities for building minimal VIN snippet views."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
from efm3d.aria.pose import PoseTW

from .efm_views import EfmSnippetView, VinSnippetView


def vin_snippet_cache_config_hash(
    *,
    dataset_config: dict[str, Any] | None,
    include_inv_dist_std: bool,
    include_obs_count: bool,
    semidense_max_points: int | None,
) -> str:
    """Compute a stable hash for VIN snippet cache compatibility checks."""
    payload = {
        "dataset_config": dataset_config,
        "include_inv_dist_std": include_inv_dist_std,
        "include_obs_count": include_obs_count,
        "semidense_max_points": semidense_max_points,
    }
    serial = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(serial.encode("utf-8")).hexdigest()


def build_vin_snippet_view(
    efm_snippet: EfmSnippetView,
    *,
    device: torch.device,
    max_points: int | None,
    include_inv_dist_std: bool = True,
    include_obs_count: bool = False,
) -> VinSnippetView:
    """Build a minimal VIN snippet view from an EFM snippet."""
    extra_dim = int(include_inv_dist_std) + int(include_obs_count)
    points_world = torch.zeros((0, 3 + extra_dim), dtype=torch.float32, device=device)
    try:
        semidense = efm_snippet.semidense
    except Exception:
        semidense = None
    if semidense is not None:
        points_world = semidense.collapse_points(
            max_points=max_points,
            include_inv_dist_std=include_inv_dist_std,
            include_obs_count=include_obs_count,
        ).to(device=device, dtype=torch.float32)

    traj_world_rig = PoseTW(torch.zeros((0, 12), dtype=torch.float32, device=device))
    try:
        traj_view = efm_snippet.trajectory.to(device=device, dtype=torch.float32)
        traj_world_rig = traj_view.t_world_rig
    except Exception:
        pass

    return VinSnippetView(points_world=points_world, t_world_rig=traj_world_rig)


def empty_vin_snippet(device: torch.device, *, extra_dim: int = 1) -> VinSnippetView:
    """Return an empty VIN snippet view on the given device.

    Args:
        extra_dim: Number of extra point channels beyond XYZ (defaults to 1 for inv_dist_std).
    """
    return VinSnippetView(
        points_world=torch.zeros((0, 3 + int(extra_dim)), dtype=torch.float32, device=device),
        t_world_rig=PoseTW(torch.zeros((0, 12), dtype=torch.float32, device=device)),
    )


__all__ = [
    "build_vin_snippet_view",
    "empty_vin_snippet",
    "vin_snippet_cache_config_hash",
]
