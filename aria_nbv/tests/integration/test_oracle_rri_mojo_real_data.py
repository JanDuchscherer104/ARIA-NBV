import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[2] / "external" / "efm3d"))

from aria_nbv.configs import PathConfig  # noqa: E402
from aria_nbv.data_handling import AseEfmDatasetConfig  # noqa: E402
from aria_nbv.pipelines import OracleBackendProfile, OracleRriLabelerConfig  # noqa: E402
from aria_nbv.pose_generation import CandidateViewGeneratorConfig  # noqa: E402
from aria_nbv.pose_generation.types import ViewDirectionMode  # noqa: E402
from aria_nbv.rendering.candidate_depth_renderer import (  # noqa: E402
    CandidateDepthRendererConfig,
)
from aria_nbv.rendering.candidate_pointclouds import (  # noqa: E402
    CandidatePointCloudBuilderConfig,
)
from aria_nbv.rri_metrics.oracle_rri import OracleRRIConfig  # noqa: E402
from aria_nbv.utils import TorchAccelerator, Verbosity  # noqa: E402


def _require_mojo_backends() -> None:
    from aria_nbv.pose_generation.mojo_backend import is_mojo_available as pose_mojo_available
    from aria_nbv.rendering.mojo_backend import is_mojo_available as rendering_mojo_available
    from aria_nbv.rri_metrics.mojo_backend import is_mojo_available as rri_mojo_available

    if not torch.backends.mps.is_available():
        pytest.skip("Torch MPS is not available locally.")
    if not pose_mojo_available() or not rendering_mojo_available() or not rri_mojo_available():
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
    keep_faces = min(int(mesh.faces.shape[0]), 500)
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


def test_oracle_rri_labeler_full_mojo_path_runs_real_data(efm_sample) -> None:
    _require_mojo_backends()

    sample = _small_sample(efm_sample)
    generator_cfg = CandidateViewGeneratorConfig(
        num_samples=3,
        oversample_factor=1.0,
        max_resamples=0,
        min_radius=0.3,
        max_radius=0.6,
        view_direction_mode=ViewDirectionMode.FORWARD_RIG,
        min_distance_to_mesh=0.05,
        ensure_collision_free=True,
        ensure_free_space=True,
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
    pointcloud_cfg = CandidatePointCloudBuilderConfig(
        backprojection_stride=16,
    )
    oracle_cfg = OracleRRIConfig()
    labeler_cfg = OracleRriLabelerConfig(
        backend_profile=OracleBackendProfile.APPLE_MPS_MOJO,
        torch_accelerator=TorchAccelerator.MPS,
        generator=generator_cfg,
        depth=depth_cfg,
        pointcloud=pointcloud_cfg,
        oracle=oracle_cfg,
        verbosity=Verbosity.QUIET,
    )

    batch = labeler_cfg.setup_target().run(sample)

    assert int(batch.depths.depths.shape[0]) > 0
    assert int(batch.candidate_pcs.lengths.sum().item()) > 0
    assert torch.isfinite(batch.rri.rri).all()
    assert torch.isfinite(batch.rri.pm_dist_after).all()
