import sys
from pathlib import Path

import torch

# Make vendored efm3d importable
sys.path.append(str(Path(__file__).resolve().parents[2] / "external" / "efm3d"))

import pytest  # isort: split

from aria_nbv.configs import PathConfig
from aria_nbv.data import AseEfmDatasetConfig
from aria_nbv.pose_generation import CandidateViewGeneratorConfig
from aria_nbv.utils import Verbosity


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


def _load_sample():
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


def test_candidate_generation_seed_is_deterministic_on_real_data():
    sample = _load_sample()

    cfg = CandidateViewGeneratorConfig(
        num_samples=64,
        oversample_factor=1.0,
        max_resamples=0,
        min_radius=0.6,
        max_radius=0.8,
        min_elev_deg=-15.0,
        max_elev_deg=15.0,
        delta_azimuth_deg=360.0,
        ensure_collision_free=False,
        ensure_free_space=False,
        min_distance_to_mesh=0.0,
        device="cpu",
        verbosity=Verbosity.QUIET,
        seed=123,
        is_debug=False,
    )

    gen = cfg.setup_target()
    out1 = gen.generate_from_typed_sample(sample)
    out2 = gen.generate_from_typed_sample(sample)

    assert out1.shell_offsets_ref is not None
    assert out2.shell_offsets_ref is not None
    assert torch.allclose(out1.shell_offsets_ref, out2.shell_offsets_ref)

    out3 = cfg.model_copy(update={"seed": 124}).setup_target().generate_from_typed_sample(sample)
    assert out3.shell_offsets_ref is not None
    assert not torch.allclose(out1.shell_offsets_ref, out3.shell_offsets_ref)
