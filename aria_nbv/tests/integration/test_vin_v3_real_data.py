"""Integration test for VIN v3 on real ASE snippets."""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3] / "external" / "efm3d"))

import pytest
import torch
from efm3d.aria.pose import PoseTW
from pytorch3d.renderer.cameras import PerspectiveCameras  # type: ignore[import-untyped]

if "seaborn" not in sys.modules:
    seaborn = types.ModuleType("seaborn")

    def _noop(*_args, **_kwargs):  # pragma: no cover - import shim only
        return None

    seaborn.set_theme = _noop
    seaborn.color_palette = lambda *args, **kwargs: []  # pragma: no cover - import shim only
    sys.modules["seaborn"] = seaborn

if "power_spherical" not in sys.modules:
    power_spherical = types.ModuleType("power_spherical")

    class DummyPowerSpherical:  # pragma: no cover - import shim only
        pass

    power_spherical.HypersphericalUniform = DummyPowerSpherical
    power_spherical.PowerSpherical = DummyPowerSpherical
    sys.modules["power_spherical"] = power_spherical

if "coral_pytorch" not in sys.modules:
    coral_pytorch = types.ModuleType("coral_pytorch")
    layers = types.ModuleType("coral_pytorch.layers")
    losses = types.ModuleType("coral_pytorch.losses")

    class DummyCoralLayer(torch.nn.Module):  # pragma: no cover - import shim only
        def __init__(self, size_in: int, num_classes: int, **kwargs) -> None:
            super().__init__()
            out_dim = max(int(num_classes) - 1, 1)
            self.proj = torch.nn.Linear(int(size_in), out_dim, bias=True)

        def forward(self, x):  # pragma: no cover - import shim only
            return self.proj(x)

    def dummy_coral_loss(*args, **kwargs):  # pragma: no cover - import shim only
        raise RuntimeError("coral_pytorch is not installed")

    layers.CoralLayer = DummyCoralLayer
    losses.coral_loss = dummy_coral_loss
    coral_pytorch.layers = layers
    coral_pytorch.losses = losses
    sys.modules["coral_pytorch"] = coral_pytorch
    sys.modules["coral_pytorch.layers"] = layers
    sys.modules["coral_pytorch.losses"] = losses

if "e3nn" not in sys.modules:
    e3nn = types.ModuleType("e3nn")
    o3 = types.ModuleType("e3nn.o3")
    e3nn.o3 = o3
    sys.modules["e3nn"] = e3nn
    sys.modules["e3nn.o3"] = o3

from oracle_rri.configs import PathConfig
from oracle_rri.data import AseEfmDatasetConfig
from oracle_rri.utils import Verbosity
from oracle_rri.vin import EvlBackboneConfig, VinModelV3Config


def _find_first_scene_with_shards() -> str | None:
    paths = PathConfig()
    tar_root = paths.resolve_atek_data_dir("efm")
    if not tar_root.exists():
        return None
    for scene_dir in sorted(tar_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        if any(scene_dir.glob("shards-*.tar")):
            return scene_dir.name
    return None


def _skip_if_missing_assets() -> None:
    paths = PathConfig()
    model_cfg = paths.root / ".configs" / "evl_inf_desktop.yaml"
    model_ckpt = paths.root / ".logs" / "ckpts" / "model_lite.pth"
    if not model_cfg.exists():
        pytest.skip(f"Missing EVL config: {model_cfg}", allow_module_level=True)
    if not model_ckpt.exists():
        pytest.skip(f"Missing EVL checkpoint: {model_ckpt}", allow_module_level=True)

    scene_id = _find_first_scene_with_shards()
    if scene_id is None:
        pytest.skip("No ASE shards found under .data/ase_efm", allow_module_level=True)


_skip_if_missing_assets()


@pytest.mark.integration
def test_vin_v3_forward_real_snippet_produces_scores() -> None:
    scene_id = _find_first_scene_with_shards()
    assert scene_id is not None

    cfg = AseEfmDatasetConfig(
        scene_ids=[scene_id],
        load_meshes=False,
        batch_size=None,
        verbosity=Verbosity.QUIET,
        is_debug=True,
    )
    ds = cfg.setup_target()
    sample = next(iter(ds))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vin = VinModelV3Config(backbone=EvlBackboneConfig(device=device)).setup_target().eval()

    t_world_rig = sample.trajectory.t_world_rig[-1]
    t_camera_rig = sample.camera_rgb.calib.T_camera_rig[-1]
    t_world_cam = t_world_rig @ t_camera_rig.inverse()

    rot = t_world_cam.R
    trans = t_world_cam.t
    cand_rot = rot.unsqueeze(0).repeat(2, 1, 1)
    cand_trans = trans.unsqueeze(0).repeat(2, 1)
    cand_trans[1] = cand_trans[1] + torch.tensor([100.0, 0.0, 0.0], dtype=cand_trans.dtype)
    candidates = PoseTW.from_Rt(cand_rot, cand_trans)

    poses_cw = candidates.inverse().to(device=device)
    rotations = poses_cw.R.transpose(-1, -2).contiguous()
    translations = poses_cw.t

    cam = sample.camera_rgb.calib.to(device=device)
    size_all = cam.size.reshape(-1, 2).to(dtype=torch.float32)
    size_base = size_all[0]
    width = int(size_base[0].item())
    height = int(size_base[1].item())
    image_size = torch.tensor([[height, width]], device=device, dtype=torch.float32).expand(rotations.shape[0], -1)

    focal_all = cam.f.reshape(-1, 2).to(dtype=torch.float32)
    principal_all = cam.c.reshape(-1, 2).to(dtype=torch.float32)
    focal_length = focal_all[0].expand(rotations.shape[0], -1)
    principal_point = principal_all[0].expand(rotations.shape[0], -1)

    cameras = PerspectiveCameras(
        device=device,
        R=rotations,
        T=translations,
        focal_length=focal_length,
        principal_point=principal_point,
        image_size=image_size,
        in_ndc=False,
    )

    with torch.no_grad():
        pred = vin(
            sample,
            candidates,
            reference_pose_world_rig=t_world_rig,
            p3d_cameras=cameras,
        )

    assert pred.logits.shape == (1, 2, 14)
    assert pred.prob.shape == (1, 2, 15)
    assert pred.expected.shape == (1, 2)
    assert pred.expected_normalized.shape == (1, 2)

    assert torch.isfinite(pred.logits).all()
    assert torch.isfinite(pred.expected_normalized).all()
    assert torch.allclose(pred.prob.sum(dim=-1), torch.ones((1, 2), device=pred.prob.device), atol=1e-4)
