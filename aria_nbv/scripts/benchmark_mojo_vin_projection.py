"""Benchmark the optional Mojo VIN semidense reducer against the Torch baseline.

This compares the reduction-heavy projection helpers on identical projected data
and verifies numeric equivalence before reporting timing numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import torch
from pytorch3d.renderer.cameras import PerspectiveCameras  # type: ignore[import-untyped]

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
EXTERNAL_EFM3D = REPO_ROOT / "external" / "efm3d"
if str(EXTERNAL_EFM3D) not in sys.path:
    sys.path.append(str(EXTERNAL_EFM3D))
if str(REPO_ROOT / "aria_nbv") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "aria_nbv"))


def _install_import_shims() -> None:
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

    if "power_spherical" not in sys.modules:
        power_spherical = types.ModuleType("power_spherical")

        class DummyPowerSpherical:  # pragma: no cover - import shim only
            pass

        power_spherical.HypersphericalUniform = DummyPowerSpherical
        power_spherical.PowerSpherical = DummyPowerSpherical
        sys.modules["power_spherical"] = power_spherical

    if "e3nn" not in sys.modules:
        e3nn = types.ModuleType("e3nn")
        o3 = types.ModuleType("e3nn.o3")
        e3nn.o3 = o3
        sys.modules["e3nn"] = e3nn
        sys.modules["e3nn.o3"] = o3

    if "seaborn" not in sys.modules:
        seaborn = types.ModuleType("seaborn")

        def _noop(*_args, **_kwargs):  # pragma: no cover - import shim only
            return None

        seaborn.set_theme = _noop
        seaborn.color_palette = lambda *args, **kwargs: []  # pragma: no cover - import shim only
        sys.modules["seaborn"] = seaborn


_install_import_shims()

from aria_nbv.vin.model_v3 import SemidenseProjectionBackend, VinModelV3, VinModelV3Config  # noqa: E402
from aria_nbv.vin.mojo_backend import is_mojo_available  # noqa: E402


@dataclass(frozen=True)
class BenchmarkCaseResult:
    """Numeric summary for one VIN semidense benchmark case."""

    backend: str
    case: str
    batch_size: int
    num_candidates: int
    num_points: int
    grid_size: int
    repeats: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    timings_ms: list[float]


def _make_model(*, backend: SemidenseProjectionBackend, grid_size: int) -> VinModelV3:
    config = VinModelV3Config(
        field_dim=4,
        field_gn_groups=2,
        global_pool_grid_size=2,
        semidense_proj_grid_size=grid_size,
        semidense_proj_max_points=8192,
        semidense_projection_backend=backend,
        head_hidden_dim=8,
        head_num_layers=1,
        head_dropout=0.0,
        num_classes=5,
        use_voxel_valid_frac_gate=False,
        backbone=None,
    )
    return VinModelV3(config)


def _make_points(*, batch_size: int, num_points: int, device: torch.device) -> torch.Tensor:
    xyz = torch.randn((batch_size, num_points, 3), device=device, dtype=torch.float32)
    xyz[..., 0] = xyz[..., 0] * 0.3
    xyz[..., 1] = xyz[..., 1] * 0.25
    xyz[..., 2] = xyz[..., 2].abs() * 0.8 + 1.0
    inv_sigma = torch.rand((batch_size, num_points, 1), device=device, dtype=torch.float32) * 0.02
    n_obs = torch.randint(1, 12, (batch_size, num_points, 1), device=device).to(dtype=torch.float32)
    return torch.cat([xyz, inv_sigma, n_obs], dim=-1)


def _make_cameras(*, num_cams: int, device: torch.device) -> PerspectiveCameras:
    rot = torch.eye(3, device=device, dtype=torch.float32).expand(num_cams, 3, 3).contiguous()
    trans = torch.zeros((num_cams, 3), device=device, dtype=torch.float32)
    trans[:, 0] = torch.linspace(0.0, 0.3, num_cams, device=device, dtype=torch.float32)
    trans[:, 1] = torch.linspace(0.0, -0.1, num_cams, device=device, dtype=torch.float32)
    return PerspectiveCameras(
        device=device,
        R=rot,
        T=trans,
        focal_length=torch.tensor([[56.0, 56.0]], device=device, dtype=torch.float32).expand(num_cams, -1),
        principal_point=torch.tensor([[48.0, 48.0]], device=device, dtype=torch.float32).expand(num_cams, -1),
        image_size=torch.tensor([[96.0, 96.0]], device=device, dtype=torch.float32).expand(num_cams, -1),
        in_ndc=False,
    )


def _clone_proj_data(proj_data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in proj_data.items()}


def _run_case(
    *,
    model: VinModelV3,
    proj_data: dict[str, torch.Tensor],
    batch_size: int,
    num_candidates: int,
    repeats: int,
    case: str,
) -> tuple[list[float], Any]:
    timings_ms: list[float] = []
    last_output: Any = None
    for _ in range(repeats):
        proj_iter = _clone_proj_data(proj_data)
        start = time.perf_counter()
        if case == "accumulation":
            last_output = model._get_semidense_projection_accumulation(proj_iter, device=torch.device("cpu"))
        elif case == "projection_features":
            last_output = model._encode_semidense_projection_features(
                proj_iter,
                batch_size=batch_size,
                num_candidates=num_candidates,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        elif case == "grid_features":
            last_output = model._encode_semidense_grid_features(
                proj_iter,
                batch_size=batch_size,
                num_candidates=num_candidates,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        elif case == "projection_combo":
            last_output = (
                model._encode_semidense_projection_features(
                    proj_iter,
                    batch_size=batch_size,
                    num_candidates=num_candidates,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                ),
                model._encode_semidense_grid_features(
                    proj_iter,
                    batch_size=batch_size,
                    num_candidates=num_candidates,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                ),
            )
        else:  # pragma: no cover - guarded by parser
            raise ValueError(f"Unknown benchmark case '{case}'.")
        timings_ms.append((time.perf_counter() - start) * 1000.0)
    return timings_ms, last_output


def _assert_equivalent(case: str, torch_output: Any, mojo_output: Any) -> None:
    if case == "accumulation":
        assert isinstance(torch_output, dict)
        assert isinstance(mojo_output, dict)
        for key in torch_output:
            if not torch.allclose(torch_output[key], mojo_output[key], atol=1e-5, rtol=1e-5):
                raise RuntimeError(f"VIN semidense equivalence failed for '{case}' key '{key}'.")
        return
    if case == "projection_combo":
        torch_proj, torch_grid = torch_output
        mojo_proj, mojo_grid = mojo_output
        if not torch.allclose(torch_proj, mojo_proj, atol=1e-5, rtol=1e-5):
            raise RuntimeError("VIN semidense equivalence failed for projection features.")
        if not torch.allclose(torch_grid, mojo_grid, atol=1e-5, rtol=1e-5):
            raise RuntimeError("VIN semidense equivalence failed for grid features.")
        return
    if not torch.allclose(torch_output, mojo_output, atol=1e-5, rtol=1e-5):
        raise RuntimeError(f"VIN semidense equivalence failed for case '{case}'.")


def run_benchmark_case(
    *,
    case: str,
    repeats: int,
    batch_size: int,
    num_candidates: int,
    num_points: int,
    grid_size: int,
) -> dict[str, BenchmarkCaseResult]:
    """Run one equivalence-checked VIN semidense benchmark case for both backends."""

    if not is_mojo_available():
        raise RuntimeError(
            "Mojo backend is not available. Install Mojo into `<repo>/.mojo-venv` or set "
            "`ARIA_NBV_MOJO_SITE_PACKAGES` before running this benchmark."
        )

    device = torch.device("cpu")
    torch.manual_seed(0)
    model_torch = _make_model(backend=SemidenseProjectionBackend.TORCH, grid_size=grid_size)
    model_mojo = _make_model(backend=SemidenseProjectionBackend.MOJO, grid_size=grid_size)
    model_mojo.load_state_dict(model_torch.state_dict())

    points_world = _make_points(batch_size=batch_size, num_points=num_points, device=device)
    cameras = _make_cameras(num_cams=batch_size * num_candidates, device=device)
    proj_data = model_torch._project_semidense_points(
        points_world,
        cameras,
        batch_size=batch_size,
        num_candidates=num_candidates,
        device=device,
    )
    if proj_data is None:
        raise RuntimeError("Projected semidense data is missing.")

    _, torch_output = _run_case(
        model=model_torch,
        proj_data=proj_data,
        batch_size=batch_size,
        num_candidates=num_candidates,
        repeats=1,
        case=case,
    )
    _, mojo_output = _run_case(
        model=model_mojo,
        proj_data=proj_data,
        batch_size=batch_size,
        num_candidates=num_candidates,
        repeats=1,
        case=case,
    )
    _assert_equivalent(case, torch_output, mojo_output)

    torch_times, _ = _run_case(
        model=model_torch,
        proj_data=proj_data,
        batch_size=batch_size,
        num_candidates=num_candidates,
        repeats=repeats,
        case=case,
    )
    mojo_times, _ = _run_case(
        model=model_mojo,
        proj_data=proj_data,
        batch_size=batch_size,
        num_candidates=num_candidates,
        repeats=repeats,
        case=case,
    )

    def _summary(backend: str, timings_ms: list[float]) -> BenchmarkCaseResult:
        return BenchmarkCaseResult(
            backend=backend,
            case=case,
            batch_size=batch_size,
            num_candidates=num_candidates,
            num_points=num_points,
            grid_size=grid_size,
            repeats=repeats,
            mean_ms=mean(timings_ms),
            median_ms=median(timings_ms),
            min_ms=min(timings_ms),
            max_ms=max(timings_ms),
            std_ms=pstdev(timings_ms) if len(timings_ms) > 1 else 0.0,
            timings_ms=timings_ms,
        )

    return {
        SemidenseProjectionBackend.TORCH.value: _summary(SemidenseProjectionBackend.TORCH.value, torch_times),
        SemidenseProjectionBackend.MOJO.value: _summary(SemidenseProjectionBackend.MOJO.value, mojo_times),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=["accumulation", "projection_features", "grid_features", "projection_combo", "all"],
        default="all",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-candidates", type=int, default=48)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    case_names = (
        ["accumulation", "projection_features", "grid_features", "projection_combo"]
        if args.case == "all"
        else [args.case]
    )
    summary: dict[str, Any] = {}
    for case_name in case_names:
        results = run_benchmark_case(
            case=case_name,
            repeats=args.repeats,
            batch_size=args.batch_size,
            num_candidates=args.num_candidates,
            num_points=args.num_points,
            grid_size=args.grid_size,
        )
        torch_result = results[SemidenseProjectionBackend.TORCH.value]
        mojo_result = results[SemidenseProjectionBackend.MOJO.value]
        summary[case_name] = {
            "torch": asdict(torch_result),
            "mojo": asdict(mojo_result),
            "speedup_vs_torch": torch_result.mean_ms / mojo_result.mean_ms,
        }

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
