"""Tests for typed ASE dataset wrapper."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import trimesh
from torch.utils.data import DataLoader

from oracle_rri.data_handling.dataset import (
    ASEDataset,
    ASEDatasetConfig,
    ASESample,
    CameraLabel,
    ase_collate,
)


def _mock_flat_sample(scene: str = "81022", snippet: str = "shards-0000") -> dict:
    """Construct a minimal flattened ATEK sample."""
    sequence_name = f"AsePublicRelease2023_{scene}_{snippet}"
    return {
        "sequence_name": sequence_name,
        "mfcd#camera-rgb+images": torch.randn(2, 3, 4, 4),
        "mfcd#camera-rgb+projection_params": torch.tensor([100.0, 100.0, 2.0, 2.0]),
        "mfcd#camera-rgb+t_device_camera": torch.randn(2, 3, 4),
        "mfcd#camera-rgb+capture_timestamps_ns": torch.tensor([0, 1], dtype=torch.int64),
        "mfcd#camera-rgb+frame_ids": torch.tensor([0, 1], dtype=torch.int64),
        "mfcd#camera-rgb+exposure_durations_s": torch.tensor([0.01, 0.01]),
        "mfcd#camera-rgb+gains": torch.tensor([1.0, 1.0]),
        "mfcd#camera-rgb+camera_model_name": "fisheye624",
        "mfcd#camera-rgb+camera_valid_radius": torch.tensor([1.0]),
        "mtd#ts_world_device": torch.eye(3, 4).unsqueeze(0).repeat(2, 1, 1),
        "mtd#capture_timestamps_ns": torch.tensor([0, 1], dtype=torch.int64),
        "mtd#gravity_in_world": torch.tensor([0.0, 0.0, -9.81]),
        "msdpd#points_world": [torch.zeros(1, 3), torch.ones(1, 3)],
        "msdpd#points_dist_std": [torch.ones(1), torch.ones(1)],
        "msdpd#points_inv_dist_std": [torch.ones(1), torch.ones(1)],
        "msdpd#capture_timestamps_ns": torch.tensor([0, 1], dtype=torch.int64),
        "msdpd#points_volumn_min": torch.zeros(3),
        "msdpd#points_volumn_max": torch.ones(3),
        "gt_data": {"efm_gt": {}},
    }


class TestASEDatasetConfig:
    def test_requires_tar_urls(self):
        with pytest.raises(ValueError):
            ASEDatasetConfig()


class TestASEDataset:
    @pytest.fixture
    def dataset(self) -> ASEDataset:
        flat = _mock_flat_sample()
        mock_loader = MagicMock()
        mock_loader.__iter__.return_value = iter([flat])

        with patch("oracle_rri.data_handling.dataset.load_atek_wds_dataset", return_value=mock_loader):
            config = ASEDatasetConfig(
                tar_urls=["/tmp/fake.tar"],
                scene_to_mesh={},
                load_meshes=False,
                batch_size=None,
                verbose=False,
            )
            yield ASEDataset(config)

    def test_iter_yields_typed_sample(self, dataset: ASEDataset):
        sample = next(iter(dataset))
        assert isinstance(sample, ASESample)
        assert sample.scene_id == "81022"
        assert sample.snippet_id == "shards-0000"
        assert sample.has_rgb is True
        assert sample.has_slam_points is True
        assert sample.has_mesh is False

        rgb = sample.atek.camera_rgb
        assert rgb is not None
        assert rgb.label == CameraLabel.RGB
        assert rgb.images.shape == (2, 3, 4, 4)

        traj = sample.atek.trajectory
        assert traj is not None
        assert traj.ts_world_device.shape == (2, 3, 4)

    def test_to_efm_dict_remaps_keys(self, dataset: ASEDataset):
        sample = next(iter(dataset))
        efm = sample.to_efm_dict()

        assert "rgb/img" in efm
        assert "pose/t_world_rig" in efm
        assert "points/p3s_world" in efm
        assert efm["scene_id"] == "81022"

    def test_dataloader_collation(self):
        flat1 = _mock_flat_sample(scene="81022", snippet="shards-0000")
        flat2 = _mock_flat_sample(scene="81048", snippet="shards-0001")
        mock_loader = MagicMock()
        mock_loader.__iter__.return_value = iter([flat1, flat2])

        with patch("oracle_rri.data_handling.dataset.load_atek_wds_dataset", return_value=mock_loader):
            config = ASEDatasetConfig(
                tar_urls=["/tmp/fake.tar"],
                scene_to_mesh={},
                load_meshes=False,
                batch_size=None,
                verbose=False,
            )
            dataset = ASEDataset(config)
            loader = DataLoader(dataset, batch_size=2, collate_fn=ase_collate)
            batch = next(iter(loader))

        assert batch["scene_id"] == ["81022", "81048"]
        assert len(batch["atek"]) == 2
        assert batch["gt_mesh"] == [None, None]

    def test_mesh_loading_and_caching(self, tmp_path: Path):
        mesh_path = tmp_path / "scene_ply_81022.ply"
        trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]]).export(mesh_path)

        flat = _mock_flat_sample()
        mock_loader = MagicMock()
        mock_loader.__iter__.return_value = iter([flat, flat])

        with patch("oracle_rri.data_handling.dataset.load_atek_wds_dataset", return_value=mock_loader):
            config = ASEDatasetConfig(
                tar_urls=["/tmp/fake.tar"],
                scene_to_mesh={"81022": mesh_path},
                load_meshes=True,
                cache_meshes=True,
                batch_size=None,
                verbose=False,
            )
            dataset = ASEDataset(config)
            samples = list(dataset)

        assert samples[0].gt_mesh is samples[1].gt_mesh
        assert samples[0].has_mesh is True
