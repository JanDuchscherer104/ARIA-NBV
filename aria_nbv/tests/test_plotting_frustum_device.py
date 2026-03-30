"""Regression tests for frustum plotting device alignment."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make vendored efm3d importable
sys.path.append(str(Path(__file__).resolve().parents[2] / "external" / "efm3d"))

from efm3d.aria import PoseTW  # noqa: E402
from efm3d.aria.camera import get_aria_camera  # noqa: E402

from aria_nbv.utils.data_plotting import get_frustum_segments  # noqa: E402


def _dummy_camera(device: torch.device):
    """Create a minimal pinhole-like camera for plotting tests."""
    return get_aria_camera().to(device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required to reproduce mixed-device bug.")
def test_frustum_segments_allow_pose_on_cuda_and_cam_on_cpu() -> None:
    """Ensure frustum plotting handles pose on GPU with camera on CPU."""

    cam_cpu = _dummy_camera(torch.device("cpu"))
    pose_cuda = PoseTW.from_Rt(torch.eye(3, device="cuda"), torch.zeros(3, device="cuda"))

    segments = get_frustum_segments(cam_cpu, pose_cuda, scale=1.0)

    assert len(segments) == 5
    assert all(seg.shape == (3, 3) for seg in segments)


def test_frustum_segments_cpu_only() -> None:
    """Baseline: frustum segments render on CPU-only inputs."""

    device = torch.device("cpu")
    cam_cpu = _dummy_camera(device)
    pose_cpu = PoseTW.from_Rt(torch.eye(3, device=device), torch.zeros(3, device=device))

    segments = get_frustum_segments(cam_cpu, pose_cpu, scale=1.0)

    assert len(segments) == 5
    assert all(seg.shape == (3, 3) for seg in segments)
