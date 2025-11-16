"""Tests for typed ASE dataset wrapper."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import trimesh
from torch.utils.data import DataLoader

import oracle_rri.data.views
from oracle_rri.data.dataset import ASEDataset, ASEDatasetConfig, TypedSample, ase_collate


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
        "mfcd#camera-slam-left+images": torch.randn(2, 1, 4, 4),
        "mfcd#camera-slam-left+projection_params": torch.tensor([90.0, 90.0, 2.0, 2.0]),
        "mfcd#camera-slam-left+t_device_camera": torch.randn(2, 3, 4),
        "mfcd#camera-slam-left+capture_timestamps_ns": torch.tensor([0, 1], dtype=torch.int64),
        "mfcd#camera-slam-left+frame_ids": torch.tensor([0, 1], dtype=torch.int64),
        "mfcd#camera-slam-left+exposure_durations_s": torch.tensor([0.01, 0.01]),
        "mfcd#camera-slam-left+gains": torch.tensor([1.0, 1.0]),
        "mfcd#camera-slam-left+camera_model_name": "fisheye624",
        "mfcd#camera-slam-left+camera_valid_radius": torch.tensor([1.0]),
        "mfcd#camera-slam-right+images": torch.randn(2, 1, 4, 4),
        "mfcd#camera-slam-right+projection_params": torch.tensor([90.0, 90.0, 2.0, 2.0]),
        "mfcd#camera-slam-right+t_device_camera": torch.randn(2, 3, 4),
        "mfcd#camera-slam-right+capture_timestamps_ns": torch.tensor([0, 1], dtype=torch.int64),
        "mfcd#camera-slam-right+frame_ids": torch.tensor([0, 1], dtype=torch.int64),
        "mfcd#camera-slam-right+exposure_durations_s": torch.tensor([0.01, 0.01]),
        "mfcd#camera-slam-right+gains": torch.tensor([1.0, 1.0]),
        "mfcd#camera-slam-right+camera_model_name": "fisheye624",
        "mfcd#camera-slam-right+camera_valid_radius": torch.tensor([1.0]),
        "mfcd#camera-rgb-depth+images": torch.randn(2, 1, 4, 4),
        "mfcd#camera-rgb-depth+projection_params": torch.tensor([100.0, 100.0, 2.0, 2.0]),
        "mfcd#camera-rgb-depth+t_device_camera": torch.randn(2, 3, 4),
        "mfcd#camera-rgb-depth+capture_timestamps_ns": torch.tensor([0, 1], dtype=torch.int64),
        "mfcd#camera-rgb-depth+frame_ids": torch.tensor([0, 1], dtype=torch.int64),
        "mfcd#camera-rgb-depth+exposure_durations_s": torch.tensor([0.01, 0.01]),
        "mfcd#camera-rgb-depth+gains": torch.tensor([1.0, 1.0]),
        "mfcd#camera-rgb-depth+camera_model_name": "fisheye624",
        "mfcd#camera-rgb-depth+camera_valid_radius": torch.tensor([1.0]),
        "mtd#ts_world_device": torch.eye(3, 4).unsqueeze(0).repeat(2, 1, 1),
        "mtd#capture_timestamps_ns": torch.tensor([0, 1], dtype=torch.int64),
        "mtd#gravity_in_world": torch.tensor([0.0, 0.0, -9.81]),
        "msdpd#points_world": [torch.zeros(1, 3), torch.ones(1, 3)],
        "msdpd#points_dist_std": [torch.ones(1), torch.ones(1)],
        "msdpd#points_inv_dist_std": [torch.ones(1), torch.ones(1)],
        "msdpd#capture_timestamps_ns": torch.tensor([0, 1], dtype=torch.int64),
        "msdpd#points_volume_min": torch.zeros(3),
        "msdpd#points_volume_max": torch.ones(3),
        "gt_data": {"obb3_gt": {}},
    }


class TestASEDatasetConfig:
    def test_requires_tar_urls(self):
        with pytest.raises(ValueError):
            ASEDatasetConfig()


class TestASEDataset:
    @pytest.fixture
    def tar_file(self, tmp_path: Path) -> Path:
        tar = tmp_path / "fake.tar"
        tar.write_bytes(b"")
        return tar

    @pytest.fixture
    def dataset(self, tar_file: Path) -> Generator[ASEDataset, None, None]:
        flat = _mock_flat_sample()
        mock_loader = MagicMock()
        mock_loader.__iter__.return_value = iter([flat])
        mock_loader.__len__.return_value = 1

        with patch("oracle_rri.data.dataset.load_atek_wds_dataset", return_value=mock_loader):
            config = ASEDatasetConfig(
                tar_urls=[tar_file],
                scene_to_mesh={},
                load_meshes=False,
                batch_size=None,
                verbose=False,
            )
            yield ASEDataset(config)

    def test_iter_yields_typed_sample(self, dataset: ASEDataset):
        sample = next(iter(dataset))
        assert isinstance(sample, TypedSample)
        assert sample.scene_id == "81022"
        assert sample.snippet_id == "shards-0000"
        cam = sample.camera_rgb
        traj = sample.trajectory
        assert cam.images is not None
        assert traj.ts_world_device is not None
        assert cam.images.shape == (2, 3, 4, 4)
        assert traj.ts_world_device.shape == (2, 3, 4)
        # GT data exists as raw dict (may be empty)
        assert isinstance(sample.gt.raw, dict)

    def test_to_efm_dict_remaps_keys(self, dataset: ASEDataset):
        sample = next(iter(dataset))
        efm = sample.to_efm_dict()
        assert "rgb/img" in efm
        assert "pose/t_world_rig" in efm
        assert "points/p3s_world" in efm
        assert efm["scene_id"] == "81022"

    def test_dataloader_collation(self, tar_file: Path):
        flat1 = _mock_flat_sample(scene="81022", snippet="shards-0000")
        flat2 = _mock_flat_sample(scene="81048", snippet="shards-0001")
        mock_loader = MagicMock()
        mock_loader.__iter__.return_value = iter([flat1, flat2])
        mock_loader.__len__.return_value = 2

        with patch("oracle_rri.data.dataset.load_atek_wds_dataset", return_value=mock_loader):
            config = ASEDatasetConfig(
                tar_urls=[tar_file],
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

    def test_missing_camera_key_raises(self, tar_file: Path):
        flat = _mock_flat_sample()
        flat.pop("mfcd#camera-rgb+images")
        mock_loader = MagicMock()
        mock_loader.__iter__.return_value = iter([flat])
        mock_loader.__len__.return_value = 1

        with patch("oracle_rri.data.dataset.load_atek_wds_dataset", return_value=mock_loader):
            config = ASEDatasetConfig(
                tar_urls=[tar_file],
                scene_to_mesh={},
                load_meshes=False,
                batch_size=None,
                verbose=False,
            )
            dataset = ASEDataset(config)
            sample = next(iter(dataset))

        with pytest.raises(KeyError):
            _ = sample.camera_rgb

    def test_repr_is_multiline_and_summarized(self, tar_file: Path):
        flat = _mock_flat_sample()
        mock_loader = MagicMock()
        mock_loader.__iter__.return_value = iter([flat])
        mock_loader.__len__.return_value = 1

        with patch("oracle_rri.data.dataset.load_atek_wds_dataset", return_value=mock_loader):
            config = ASEDatasetConfig(
                tar_urls=[tar_file],
                scene_to_mesh={},
                load_meshes=False,
                batch_size=None,
                verbose=False,
            )
            sample = next(iter(ASEDataset(config)))

        rep = repr(sample)
        assert "scene_id" in rep and "cameras" in rep
        assert "shape" in rep
        assert "\n" in rep  # multiline formatting

    def test_camera_docstring_has_shapes(self):
        doc = oracle_rri.data.views.CameraView.__doc__
        assert doc is not None and "(F, C, H, W)" in doc

    def test_to_moves_tensors(self, tar_file: Path):
        flat = _mock_flat_sample()
        mock_loader = MagicMock()
        mock_loader.__iter__.return_value = iter([flat])
        mock_loader.__len__.return_value = 1

        with patch("oracle_rri.data.dataset.load_atek_wds_dataset", return_value=mock_loader):
            config = ASEDatasetConfig(
                tar_urls=[tar_file],
                scene_to_mesh={},
                load_meshes=False,
                batch_size=None,
                verbose=False,
            )
            sample = next(iter(ASEDataset(config)))

        cam = sample.camera_rgb
        moved = cam.to("cpu")
        assert moved.images.device.type == "cpu"
        assert moved is not cam  # copy

    def test_mesh_loading_and_caching(self, tmp_path: Path):
        mesh_path = tmp_path / "scene_ply_81022.ply"
        trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]]).export(mesh_path)

        flat = _mock_flat_sample()
        mock_loader = MagicMock()
        mock_loader.__iter__.return_value = iter([flat, flat])

        with patch("oracle_rri.data.dataset.load_atek_wds_dataset", return_value=mock_loader):
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

        assert samples[0].mesh is samples[1].mesh
        assert samples[0].mesh is not None


class TestGTViewRealData:
    def test_parses_efm_gt_obb3(self):
        tar = Path(".data/ase_efm_eval/81286/shards-0000.tar")
        if not tar.exists():
            pytest.skip("real ASE shard missing locally")

        config = ASEDatasetConfig(
            tar_urls=[tar],
            scene_to_mesh={},
            load_meshes=False,
            batch_size=None,
            shuffle=False,
            repeat=False,
            verbose=False,
        )
        sample = next(iter(ASEDataset(config)))
        gt = sample.gt

        assert isinstance(gt.efm_gt, dict) and gt.efm_gt, "efm_gt should be populated from real shard"

        ts_key, cam_dict = next(iter(gt.efm_gt.items()))
        assert {"camera-rgb"} <= set(cam_dict), "camera-rgb entries expected in efm_gt"

        rgb = cam_dict["camera-rgb"]
        k = rgb["category_ids"].shape[0]

        assert rgb["object_dimensions"].shape == (k, 3)
        assert rgb["ts_world_object"].shape == (k, 3, 4)
        assert rgb["instance_ids"].shape == (k,)
        assert rgb["category_ids"].shape == (k,)
        assert len(rgb["category_names"]) == k
