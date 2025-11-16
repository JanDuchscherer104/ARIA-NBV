from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure the ATEK utilities are importable during tests.
REPO_ROOT = Path(__file__).resolve().parents[2]
ATEK_PATH = REPO_ROOT / "external" / "ATEK"
if str(ATEK_PATH) not in sys.path:
    sys.path.insert(0, str(ATEK_PATH))

from oracle_rri.metrics import OracleRRIMetric  # noqa: E402


def test_metric_identical_point_clouds_return_zero(mesh_path: Path) -> None:
    metric = OracleRRIMetric(mesh_path, mesh_samples=2048, max_points=10_000, face_batch=2000)
    pts = _sphere_vertices()
    metric.update(pts, pts.clone())
    value = metric.compute().item()
    assert pytest.approx(value, abs=1e-6) == 0.0


def test_metric_improvement_positive(mesh_path: Path) -> None:
    metric = OracleRRIMetric(mesh_path, mesh_samples=2048, max_points=20_000, face_batch=2000)
    clean = _sphere_vertices()
    noisy = clean + 0.05 * torch.randn_like(clean)
    candidate = torch.cat([noisy, clean], dim=0)

    metric.update(noisy, candidate)
    value = metric.compute().item()
    assert 0.0 <= value <= 1.0
    assert value > 0.0
