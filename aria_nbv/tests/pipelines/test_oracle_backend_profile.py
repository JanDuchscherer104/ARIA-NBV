import pytest
import torch

from aria_nbv.pipelines import OracleBackendProfile, OracleBackendProfileError, OracleRriLabelerConfig
from aria_nbv.pipelines import oracle_backend_profile as profile_mod
from aria_nbv.pose_generation import CandidateViewGeneratorConfig
from aria_nbv.pose_generation.types import CollisionBackend
from aria_nbv.rendering import DepthRendererBackend
from aria_nbv.rendering.candidate_pointclouds import PointCloudBackend
from aria_nbv.rri_metrics.oracle_rri import OracleDistanceBackend
from aria_nbv.utils import TorchAccelerator


def test_oracle_labeler_default_profile_is_pytorch3d_cuda() -> None:
    cfg = OracleRriLabelerConfig()

    assert cfg.backend_profile == OracleBackendProfile.PYTORCH3D_CUDA


def test_pytorch3d_cuda_profile_resolves_all_stage_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = OracleRriLabelerConfig()
    original_device = cfg.device

    monkeypatch.setattr(profile_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(profile_mod, "_append_pytorch3d_errors", lambda errors: None)

    resolved = cfg.resolved()

    assert resolved is not cfg
    assert resolved.device == torch.device("cuda")
    assert resolved.generator.collision_backend == CollisionBackend.P3D
    assert resolved.depth.backend == DepthRendererBackend.PYTORCH3D
    assert resolved.pointcloud.backend == PointCloudBackend.PYTORCH3D
    assert resolved.oracle.backend == OracleDistanceBackend.PYTORCH3D
    assert cfg.device == original_device


def test_apple_mps_mojo_profile_resolves_without_pytorch3d(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_mod, "is_mps_available", lambda: True)
    monkeypatch.setattr(profile_mod, "_append_mojo_errors", lambda errors: None)

    def _fail_pytorch3d_check(errors: list[str]) -> None:
        raise AssertionError("Apple profile must not check PyTorch3D availability.")

    monkeypatch.setattr(profile_mod, "_append_pytorch3d_errors", _fail_pytorch3d_check)

    cfg = OracleRriLabelerConfig(
        backend_profile=OracleBackendProfile.APPLE_MPS_MOJO,
        torch_accelerator=TorchAccelerator.MPS,
    )
    resolved = cfg.resolved()

    assert resolved.device == torch.device("mps")
    assert resolved.generator.device == torch.device("cpu")
    assert resolved.depth.mojo.device == torch.device("mps")
    assert resolved.generator.collision_backend == CollisionBackend.MOJO
    assert resolved.depth.backend == DepthRendererBackend.MOJO
    assert resolved.pointcloud.backend == PointCloudBackend.MOJO
    assert resolved.oracle.backend == OracleDistanceBackend.MOJO


def test_mixed_production_backend_combination_is_rejected() -> None:
    cfg = OracleRriLabelerConfig(
        backend_profile=OracleBackendProfile.APPLE_MPS_MOJO,
        generator=CandidateViewGeneratorConfig(collision_backend=CollisionBackend.P3D),
    )

    with pytest.raises(OracleBackendProfileError, match="generator.collision_backend"):
        cfg.resolved(require_available=False)


def test_diagnostic_overrides_preserve_explicit_stage_backends() -> None:
    cfg = OracleRriLabelerConfig(
        backend_profile=OracleBackendProfile.APPLE_MPS_MOJO,
        torch_accelerator=TorchAccelerator.MPS,
        allow_backend_overrides=True,
        generator=CandidateViewGeneratorConfig(collision_backend=CollisionBackend.P3D),
    )

    resolved = cfg.resolved(require_available=False)

    assert resolved.generator.collision_backend == CollisionBackend.P3D
    assert resolved.depth.backend == DepthRendererBackend.MOJO
    assert resolved.pointcloud.backend == PointCloudBackend.MOJO
    assert resolved.oracle.backend == OracleDistanceBackend.MOJO
