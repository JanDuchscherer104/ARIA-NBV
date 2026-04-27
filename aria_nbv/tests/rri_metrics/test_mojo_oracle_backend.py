import pytest
import torch

from aria_nbv.rri_metrics.oracle_rri import OracleRRIConfig
from aria_nbv.utils.pytorch3d_compat import is_pytorch3d_available


def _require_mojo_backend() -> None:
    from aria_nbv.rri_metrics.mojo_backend import is_mojo_available

    if not is_mojo_available():
        pytest.skip("Mojo RRI backend not available locally.")


def _unit_square_mesh(device: torch.device, *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    verts = torch.tensor(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device=device, dtype=torch.int64)
    return verts, faces


def test_oracle_distance_backend_contract() -> None:
    from aria_nbv.rri_metrics.oracle_rri import OracleDistanceBackend

    assert OracleDistanceBackend.PYTORCH3D.value == "pytorch3d"
    assert OracleDistanceBackend.MOJO.value == "mojo"


def test_oracle_rri_config_defaults_to_pytorch3d() -> None:
    from aria_nbv.rri_metrics.oracle_rri import OracleDistanceBackend

    cfg = OracleRRIConfig()
    assert cfg.backend == OracleDistanceBackend.PYTORCH3D
    assert cfg.mojo is not None


def test_oracle_rri_mojo_matches_pytorch3d() -> None:
    _require_mojo_backend()
    if not is_pytorch3d_available():
        pytest.skip("PyTorch3D is required for Mojo/PyTorch3D RRI parity.")

    from aria_nbv.rri_metrics.oracle_rri import OracleDistanceBackend

    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32

    gt_verts, gt_faces = _unit_square_mesh(device, dtype=dtype)
    points_t = torch.randn((64, 3), device=device, dtype=dtype)
    points_q = torch.randn((4, 16, 3), device=device, dtype=dtype)
    lengths_q = torch.tensor([16, 7, 0, 16], device=device, dtype=torch.long)
    extend = torch.tensor([-2, 2, -2, 2, -2, 2], device=device, dtype=dtype)

    baseline = (
        OracleRRIConfig()
        .setup_target()
        .score(
            points_t=points_t,
            points_q=points_q,
            lengths_q=lengths_q,
            gt_verts=gt_verts,
            gt_faces=gt_faces,
            extend=extend,
        )
    )
    mojo = (
        OracleRRIConfig(backend=OracleDistanceBackend.MOJO)
        .setup_target()
        .score(
            points_t=points_t,
            points_q=points_q,
            lengths_q=lengths_q,
            gt_verts=gt_verts,
            gt_faces=gt_faces,
            extend=extend,
        )
    )

    assert torch.allclose(baseline.pm_acc_after, mojo.pm_acc_after, atol=1e-4, rtol=1e-4)
    assert torch.allclose(baseline.pm_comp_after, mojo.pm_comp_after, atol=1e-4, rtol=1e-4)
    assert torch.allclose(baseline.pm_dist_after, mojo.pm_dist_after, atol=1e-4, rtol=1e-4)
    assert torch.allclose(baseline.rri, mojo.rri, atol=1e-4, rtol=1e-4)
