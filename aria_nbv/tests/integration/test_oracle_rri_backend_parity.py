import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[2] / "external" / "efm3d"))

try:
    import pytorch3d.renderer as _pytorch3d_renderer  # noqa: F401
except Exception:
    PYTORCH3D_AVAILABLE = False
else:
    PYTORCH3D_AVAILABLE = True

from aria_nbv.configs import PathConfig  # noqa: E402
from aria_nbv.data_handling import AseEfmDatasetConfig  # noqa: E402
from aria_nbv.pipelines import OracleBackendProfile, OracleRriLabelerConfig  # noqa: E402
from aria_nbv.pose_generation import CandidateViewGeneratorConfig  # noqa: E402
from aria_nbv.pose_generation.types import (
    CollisionBackend,  # noqa: E402
    ViewDirectionMode,  # noqa: E402
)
from aria_nbv.rendering.candidate_depth_renderer import (  # noqa: E402
    CandidateDepthRendererConfig,
    DepthRendererBackend,
)
from aria_nbv.rendering.candidate_pointclouds import (  # noqa: E402
    CandidatePointCloudBuilderConfig,
    PointCloudBackend,
)
from aria_nbv.rri_metrics.oracle_rri import OracleDistanceBackend, OracleRRIConfig  # noqa: E402
from aria_nbv.utils import TorchAccelerator, Verbosity  # noqa: E402


def _require_mojo_backends() -> None:
    from aria_nbv.rendering.mojo_backend import is_mojo_available as rendering_mojo_available
    from aria_nbv.rri_metrics.mojo_backend import is_mojo_available as rri_mojo_available

    if not rendering_mojo_available() or not rri_mojo_available():
        pytest.skip("Mojo oracle backends not available locally.")


def _skip_if_missing_data() -> None:
    paths = PathConfig()
    atek_dir = paths.resolve_atek_data_dir("efm")
    if not atek_dir.exists():
        pytest.skip(f"ATEK data dir missing: {atek_dir}", allow_module_level=True)
    if not any(atek_dir.glob("**/*.tar")):
        pytest.skip(f"No ATEK shards found under {atek_dir}", allow_module_level=True)
    mesh_dir = paths.ase_meshes
    if not any(mesh_dir.glob("scene_ply_*.ply")):
        pytest.skip(f"No ASE meshes found under {mesh_dir}", allow_module_level=True)


_skip_if_missing_data()


@pytest.fixture(scope="session")
def efm_sample():
    cfg = AseEfmDatasetConfig(
        scene_ids=["81283"],
        batch_size=None,
        load_meshes=True,
        require_mesh=True,
        mesh_simplify_ratio=0.02,
        device="cpu",
        verbosity=Verbosity.QUIET,
        is_debug=False,
    )
    ds = cfg.setup_target()
    return next(iter(ds))


def _small_sample(efm_sample):
    mesh = efm_sample.mesh
    assert mesh is not None
    keep_faces = min(int(mesh.faces.shape[0]), 2_000)
    keep_idx = np.linspace(0, int(mesh.faces.shape[0]) - 1, num=keep_faces, dtype=int)
    mesh_small = mesh.submesh([keep_idx], append=True)
    verts_small = torch.as_tensor(mesh_small.vertices, dtype=torch.float32)
    faces_small = torch.as_tensor(mesh_small.faces, dtype=torch.int64)
    return type(efm_sample)(
        efm=efm_sample.efm,
        scene_id=efm_sample.scene_id,
        snippet_id=efm_sample.snippet_id,
        mesh=mesh_small,
        crop_bounds=efm_sample.crop_bounds,
        mesh_verts=verts_small,
        mesh_faces=faces_small,
    )


def _labeler_config(
    *,
    depth_backend: DepthRendererBackend,
    pointcloud_backend: PointCloudBackend,
    distance_backend: OracleDistanceBackend,
    collision_backend: CollisionBackend | None = None,
    ensure_collision_free: bool = False,
    ensure_free_space: bool = False,
    min_distance_to_mesh: float = 0.0,
) -> OracleRriLabelerConfig:
    generator_cfg = CandidateViewGeneratorConfig(
        num_samples=3,
        oversample_factor=1.0,
        max_resamples=0,
        min_radius=0.0,
        max_radius=0.0,
        view_direction_mode=ViewDirectionMode.FORWARD_RIG,
        min_distance_to_mesh=min_distance_to_mesh,
        ensure_collision_free=ensure_collision_free,
        ensure_free_space=ensure_free_space,
        collision_backend=collision_backend or CollisionBackend.P3D,
        device="cpu",
        verbosity=Verbosity.QUIET,
        is_debug=False,
    )
    depth_cfg = CandidateDepthRendererConfig(
        backend=depth_backend,
        max_candidates_final=3,
        oversample_factor=1.0,
        resolution_scale=0.1,
        verbosity=Verbosity.QUIET,
        is_debug=False,
    )
    pointcloud_cfg = CandidatePointCloudBuilderConfig(
        backend=pointcloud_backend,
        backprojection_stride=32,
    )
    oracle_cfg = OracleRRIConfig(backend=distance_backend)
    return OracleRriLabelerConfig(
        allow_backend_overrides=True,
        generator=generator_cfg,
        depth=depth_cfg,
        pointcloud=pointcloud_cfg,
        oracle=oracle_cfg,
        device="cpu",
    )


def test_oracle_rri_labeler_mojo_matches_baseline(efm_sample) -> None:
    _require_mojo_backends()
    if not PYTORCH3D_AVAILABLE:
        pytest.skip("PyTorch3D is required for Mojo/PyTorch3D backend parity.")

    sample = _small_sample(efm_sample)
    baseline = _labeler_config(
        depth_backend=DepthRendererBackend.PYTORCH3D,
        pointcloud_backend=PointCloudBackend.PYTORCH3D,
        distance_backend=OracleDistanceBackend.PYTORCH3D,
    ).setup_target()
    mojo = _labeler_config(
        depth_backend=DepthRendererBackend.MOJO,
        pointcloud_backend=PointCloudBackend.MOJO,
        distance_backend=OracleDistanceBackend.MOJO,
    ).setup_target()

    baseline_batch = baseline.run(sample)
    mojo_batch = mojo.run(sample)

    assert torch.equal(baseline_batch.depths.candidate_indices, mojo_batch.depths.candidate_indices)
    assert torch.equal(baseline_batch.depths.depths_valid_mask, mojo_batch.depths.depths_valid_mask)
    assert torch.allclose(baseline_batch.depths.depths, mojo_batch.depths.depths, atol=1e-4, rtol=1e-4)
    assert torch.allclose(
        baseline_batch.candidate_pcs.occupancy_bounds,
        mojo_batch.candidate_pcs.occupancy_bounds,
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.allclose(
        baseline_batch.rri.pm_dist_after,
        mojo_batch.rri.pm_dist_after,
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.allclose(baseline_batch.rri.rri, mojo_batch.rri.rri, atol=1e-4, rtol=1e-4)
    assert torch.equal(torch.argsort(baseline_batch.rri.rri), torch.argsort(mojo_batch.rri.rri))


def test_oracle_rri_labeler_full_mojo_path_runs_real_data(efm_sample) -> None:
    """Run candidate generation, depth, point clouds, and RRI all on the Mojo path."""

    _require_mojo_backends()

    sample = _small_sample(efm_sample)
    mojo = _labeler_config(
        depth_backend=DepthRendererBackend.MOJO,
        pointcloud_backend=PointCloudBackend.MOJO,
        distance_backend=OracleDistanceBackend.MOJO,
        collision_backend=CollisionBackend.MOJO,
        ensure_collision_free=True,
        ensure_free_space=True,
        min_distance_to_mesh=0.05,
    ).setup_target()

    batch = mojo.run(sample)

    assert int(batch.depths.depths.shape[0]) > 0
    assert int(batch.candidate_pcs.lengths.sum().item()) > 0
    assert torch.isfinite(batch.rri.rri).all()
    assert torch.isfinite(batch.rri.pm_dist_after).all()


def test_oracle_rri_labeler_pytorch3d_cuda_profile_runs_real_data(efm_sample) -> None:
    if not PYTORCH3D_AVAILABLE:
        pytest.skip("PyTorch3D is required for the pytorch3d_cuda profile integration.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the pytorch3d_cuda profile integration.")

    sample = _small_sample(efm_sample)
    generator_cfg = CandidateViewGeneratorConfig(
        num_samples=3,
        oversample_factor=1.0,
        max_resamples=0,
        min_radius=0.0,
        max_radius=0.0,
        view_direction_mode=ViewDirectionMode.FORWARD_RIG,
        min_distance_to_mesh=0.0,
        ensure_collision_free=False,
        ensure_free_space=False,
        verbosity=Verbosity.QUIET,
        is_debug=False,
    )
    depth_cfg = CandidateDepthRendererConfig(
        max_candidates_final=3,
        oversample_factor=1.0,
        resolution_scale=0.1,
        verbosity=Verbosity.QUIET,
        is_debug=False,
    )
    pointcloud_cfg = CandidatePointCloudBuilderConfig(backprojection_stride=32)
    cfg = OracleRriLabelerConfig(
        backend_profile=OracleBackendProfile.PYTORCH3D_CUDA,
        torch_accelerator=TorchAccelerator.CUDA,
        generator=generator_cfg,
        depth=depth_cfg,
        pointcloud=pointcloud_cfg,
        oracle=OracleRRIConfig(),
        verbosity=Verbosity.QUIET,
    )

    batch = cfg.setup_target().run(sample)

    assert batch.depths.depths.shape[0] == batch.rri.rri.shape[0]
    assert torch.isfinite(batch.rri.rri).all()
