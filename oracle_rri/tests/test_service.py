from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ATEK_PATH = REPO_ROOT / "external" / "ATEK"
if str(ATEK_PATH) not in sys.path:
    sys.path.insert(0, str(ATEK_PATH))

import numpy as np
import pytest
import torch
import trimesh

from oracle_rri.config import DatasetPaths, OracleConfig
from oracle_rri.data.structures import FrameSlice, SnippetSample
from oracle_rri.services.oracle import OracleRRIService


@pytest.fixture()
def simple_mesh(tmp_path: Path) -> Path:
    mesh = trimesh.creation.icosphere(radius=1.0, subdivisions=1)
    mesh_path = tmp_path / "sphere.ply"
    mesh.export(mesh_path)
    return mesh_path


@pytest.fixture()
def config_stub(tmp_path: Path) -> OracleConfig:
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    (manifests_dir / "atek.json").write_text("{}")
    (manifests_dir / "mesh.json").write_text("{}")

    dataset = DatasetPaths(
        atek_manifest=manifests_dir / "atek.json",
        mesh_manifest=manifests_dir / "mesh.json",
        wds_root=tmp_path / "wds",
        mesh_root=tmp_path / "meshes",
    )
    return OracleConfig(dataset=dataset, output_root=tmp_path / "outputs")


def _rig_pose_identity() -> torch.Tensor:
    return torch.cat([torch.eye(3), torch.zeros(3, 1)], dim=1)


def _sphere_points(scale: float = 1.0) -> torch.Tensor:
    mesh = trimesh.creation.icosphere(radius=scale, subdivisions=1)
    return torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32)


def test_oracle_service_returns_positive_scores(
    simple_mesh: Path, config_stub: OracleConfig
) -> None:
    service = OracleRRIService(config_stub)
    service.load_mesh(simple_mesh)

    base_points = _sphere_points(scale=1.05)
    improved_points = torch.cat([base_points, _sphere_points()], dim=0)

    frame0 = FrameSlice(
        timestamp_ns=0,
        rig_pose_world=_rig_pose_identity(),
        points_world=base_points,
        points_std=torch.ones(base_points.shape[0]),
    )
    frame1 = FrameSlice(
        timestamp_ns=1,
        rig_pose_world=_rig_pose_identity(),
        points_world=improved_points,
        points_std=torch.ones(improved_points.shape[0]),
    )

    snippet = SnippetSample(
        sequence_name="seq-1",
        snippet_id=0,
        capture_timestamps_ns=torch.tensor([0, 1], dtype=torch.int64),
        frames=[frame0, frame1],
    )

    scores = list(service.compute_rris(snippet, baseline_frames=1))
    assert len(scores) == 1
    assert scores[0] > 0.0
