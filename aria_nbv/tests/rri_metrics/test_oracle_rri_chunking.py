import pytest
import torch

from aria_nbv.rri_metrics.oracle_rri import OracleDistanceBackend, OracleRRIConfig
from aria_nbv.utils.pytorch3d_compat import is_pytorch3d_available


def _require_pytorch3d() -> None:
    if not is_pytorch3d_available():
        pytest.skip("PyTorch3D is required for the default oracle distance backend.")


def _require_mojo_backend() -> None:
    from aria_nbv.rri_metrics.mojo_backend import is_mojo_available

    if not is_mojo_available():
        pytest.skip("Mojo RRI backend not available locally.")


def _unit_square_mesh(device: torch.device, *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a small, deterministic mesh for distance tests."""

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


def test_oracle_rri_score_matches_score_batch_alias():
    _require_pytorch3d()
    _assert_score_matches_score_batch_alias(OracleRRIConfig())


def test_oracle_rri_score_matches_score_batch_alias_mojo():
    _require_mojo_backend()
    _assert_score_matches_score_batch_alias(OracleRRIConfig(backend=OracleDistanceBackend.MOJO))


def _assert_score_matches_score_batch_alias(cfg: OracleRRIConfig) -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32

    gt_verts, gt_faces = _unit_square_mesh(device, dtype=dtype)
    points_t = torch.randn((128, 3), device=device, dtype=dtype)

    num_candidates = 7
    max_points_q = 32
    points_q = torch.randn((num_candidates, max_points_q, 3), device=device, dtype=dtype)
    lengths_q = torch.full((num_candidates,), max_points_q, device=device, dtype=torch.long)
    extend = torch.tensor([-2, 2, -2, 2, -2, 2], device=device, dtype=dtype)

    oracle = cfg.setup_target()
    out_score = oracle.score(
        points_t=points_t,
        points_q=points_q,
        lengths_q=lengths_q,
        gt_verts=gt_verts,
        gt_faces=gt_faces,
        extend=extend,
    )
    out_batch = oracle.score_batch(
        points_t=points_t,
        points_q=points_q,
        lengths_q=lengths_q,
        gt_verts=gt_verts,
        gt_faces=gt_faces,
        extend=extend,
    )

    assert torch.allclose(out_score.rri, out_batch.rri, atol=1e-6)
    assert torch.allclose(out_score.pm_dist_after, out_batch.pm_dist_after, atol=1e-6)
    assert torch.allclose(out_score.pm_acc_after, out_batch.pm_acc_after, atol=1e-6)
    assert torch.allclose(out_score.pm_comp_after, out_batch.pm_comp_after, atol=1e-6)


def test_oracle_rri_handles_empty_candidate_pointclouds():
    """If a candidate contributes zero points, then P_{t∪q} == P_t and RRI==0."""
    _require_pytorch3d()
    _assert_oracle_rri_handles_empty_candidate_pointclouds(OracleRRIConfig())


def test_oracle_rri_handles_empty_candidate_pointclouds_mojo():
    _require_mojo_backend()
    _assert_oracle_rri_handles_empty_candidate_pointclouds(OracleRRIConfig(backend=OracleDistanceBackend.MOJO))


def _assert_oracle_rri_handles_empty_candidate_pointclouds(cfg: OracleRRIConfig) -> None:
    """If a candidate contributes zero points, then P_{t∪q} == P_t and RRI==0."""

    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32

    gt_verts, gt_faces = _unit_square_mesh(device, dtype=dtype)
    points_t = torch.randn((64, 3), device=device, dtype=dtype)

    num_candidates = 3
    max_points_q = 16
    points_q = torch.randn((num_candidates, max_points_q, 3), device=device, dtype=dtype)
    lengths_q = torch.tensor([max_points_q, 0, max_points_q], device=device, dtype=torch.long)
    extend = torch.tensor([-2, 2, -2, 2, -2, 2], device=device, dtype=dtype)

    out = cfg.setup_target().score(
        points_t=points_t,
        points_q=points_q,
        lengths_q=lengths_q,
        gt_verts=gt_verts,
        gt_faces=gt_faces,
        extend=extend,
    )

    # Candidate 1 has no points; distances after must equal before and RRI must be 0.
    assert float(out.rri[1].item()) == 0.0
    assert torch.allclose(out.pm_dist_after[1], out.pm_dist_before[1], atol=1e-6)
    assert torch.allclose(out.pm_acc_after[1], out.pm_acc_before[1], atol=1e-6)
    assert torch.allclose(out.pm_comp_after[1], out.pm_comp_before[1], atol=1e-6)
