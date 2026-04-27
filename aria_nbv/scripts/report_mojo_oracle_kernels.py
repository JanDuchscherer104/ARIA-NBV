"""Benchmark Mojo oracle kernel modules against Python baselines and plot results.

This script does not depend on the repo Python package graph. It talks directly
to the compiled Python-importable Mojo modules and compares them against local
Python reference implementations so it can run even when the vendored EFM stack
is unavailable in the workspace.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MOJO_SITE_PACKAGES = _REPO_ROOT / ".mojo-venv" / "lib" / "python3.12" / "site-packages"


def _ensure_mojo_paths() -> None:
    candidate = os.environ.get("ARIA_NBV_MOJO_SITE_PACKAGES", str(_MOJO_SITE_PACKAGES))
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
    import mojo.importer  # noqa: F401

    for path in [
        _REPO_ROOT / "aria_nbv" / "aria_nbv" / "pose_generation" / "mojo",
        _REPO_ROOT / "aria_nbv" / "aria_nbv" / "rendering" / "mojo",
        _REPO_ROOT / "aria_nbv" / "aria_nbv" / "rri_metrics" / "mojo",
    ]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _load_modules():
    _ensure_mojo_paths()
    return (
        importlib.import_module("mesh_collision_kernels"),
        importlib.import_module("oracle_render_kernels"),
        importlib.import_module("oracle_distance_kernels"),
    )


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _point_triangle_distance_sq(point: np.ndarray, tri: np.ndarray) -> float:
    a, b, c = tri
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = _dot(ab, ap)
    d2 = _dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return _dot(ap, ap)

    bp = point - b
    d3 = _dot(ab, bp)
    d4 = _dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return _dot(bp, bp)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        proj = a + v * ab
        diff = point - proj
        return _dot(diff, diff)

    cp = point - c
    d5 = _dot(ab, cp)
    d6 = _dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return _dot(cp, cp)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        proj = a + w * ac
        diff = point - proj
        return _dot(diff, diff)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        proj = b + w * (c - b)
        diff = point - proj
        return _dot(diff, diff)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    proj = a + ab * v + ac * w
    diff = point - proj
    return _dot(diff, diff)


def _ray_triangle_t(direction: np.ndarray, tri: np.ndarray) -> float:
    a, b, c = tri
    e1 = b - a
    e2 = c - a
    pvec = np.cross(direction, e2)
    det = _dot(e1, pvec)
    if -1e-6 < det < 1e-6:
        return -1.0
    inv_det = 1.0 / det
    tvec = -a
    u = _dot(tvec, pvec) * inv_det
    if u < 0.0 or u > 1.0:
        return -1.0
    qvec = np.cross(tvec, e1)
    v = _dot(direction, qvec) * inv_det
    if v < 0.0 or (u + v) > 1.0:
        return -1.0
    t = _dot(e2, qvec) * inv_det
    return t if t > 0.0 else -1.0


def _python_point_mesh_distance_sq(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    out = np.empty((points.shape[0],), dtype=np.float32)
    for i, point in enumerate(points):
        best = np.float32(1e30)
        for tri in triangles:
            dist_sq = np.float32(_point_triangle_distance_sq(point, tri))
            if dist_sq < best:
                best = dist_sq
        out[i] = best
    return out


def _python_triangle_point_distance_sq(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    out = np.empty((triangles.shape[0],), dtype=np.float32)
    for i, tri in enumerate(triangles):
        best = np.float32(1e30)
        for point in points:
            dist_sq = np.float32(_point_triangle_distance_sq(point, tri))
            if dist_sq < best:
                best = dist_sq
        out[i] = best
    return out


def _python_render_depth_map(
    triangles: np.ndarray,
    *,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    znear: float,
    zfar: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.full((height * width,), zfar, dtype=np.float32)
    hit = np.zeros((height * width,), dtype=np.uint8)
    for ray_idx in range(height * width):
        u_idx = ray_idx % width
        v_idx = ray_idx // width
        u = np.float32(u_idx + 0.5)
        v = np.float32(v_idx + 0.5)
        direction = np.array([-(u - cx) / fx, -(v - cy) / fy, 1.0], dtype=np.float32)
        best = np.float32(zfar)
        did_hit = False
        for tri in triangles:
            t = _ray_triangle_t(direction, tri)
            if znear <= t < best:
                best = np.float32(t)
                did_hit = True
        depth[ray_idx] = best
        hit[ray_idx] = np.uint8(1 if did_hit else 0)
    return depth, hit


def _python_unproject_valid_points(
    depth: np.ndarray,
    valid: np.ndarray,
    pose_3x4: np.ndarray,
    *,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    for v_idx in range(0, height, stride):
        for u_idx in range(0, width, stride):
            idx = v_idx * width + u_idx
            if valid[idx] == 0:
                continue
            z = depth[idx]
            if z <= 0.0:
                continue
            u = np.float32(u_idx + 0.5)
            v = np.float32(v_idx + 0.5)
            x_cam = -((u - cx) / fx) * z
            y_cam = -((v - cy) / fy) * z
            cam = np.array([x_cam, y_cam, z, 1.0], dtype=np.float32)
            world = pose_3x4 @ cam
            points.append(world.astype(np.float32))
    if not points:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((6,), dtype=np.float32)
    stacked = np.stack(points, axis=0)
    bounds = np.array(
        [
            stacked[:, 0].min(),
            stacked[:, 0].max(),
            stacked[:, 1].min(),
            stacked[:, 1].max(),
            stacked[:, 2].min(),
            stacked[:, 2].max(),
        ],
        dtype=np.float32,
    )
    return stacked, bounds


def _time_call(fn, repeats: int) -> list[float]:
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return times


@dataclass(frozen=True)
class BenchRow:
    kernel: str
    size_label: str
    baseline_mean_ms: float
    mojo_mean_ms: float
    speedup: float


def _bench_collision(mesh_mod, *, repeats: int) -> list[BenchRow]:
    rows: list[BenchRow] = []
    for num_points, num_tris in [(128, 64), (512, 256), (1024, 512)]:
        rng = np.random.default_rng(0)
        points = rng.normal(size=(num_points, 3)).astype(np.float32)
        triangles = rng.normal(size=(num_tris, 3, 3)).astype(np.float32)
        out = np.empty((num_points,), dtype=np.float32)

        baseline = _time_call(
            lambda points=points, triangles=triangles: _python_point_mesh_distance_sq(points, triangles),
            repeats,
        )
        mojo = _time_call(
            lambda points=points, num_points=num_points, triangles=triangles, num_tris=num_tris, out=out: mesh_mod.point_mesh_distance_sq_f32(
                points.ctypes.data,
                num_points,
                triangles.ctypes.data,
                num_tris,
                out.ctypes.data,
                min(os.cpu_count() or 1, num_points),
            ),
            repeats,
        )
        expected = _python_point_mesh_distance_sq(points[:32], triangles[:32])
        test_out = np.empty((32,), dtype=np.float32)
        mesh_mod.point_mesh_distance_sq_f32(
            points[:32].ctypes.data,
            32,
            triangles[:32].ctypes.data,
            32,
            test_out.ctypes.data,
            min(os.cpu_count() or 1, 32),
        )
        assert np.allclose(expected, test_out, atol=1e-5)
        rows.append(
            BenchRow(
                kernel="point->mesh",
                size_label=f"P={num_points}, T={num_tris}",
                baseline_mean_ms=mean(baseline),
                mojo_mean_ms=mean(mojo),
                speedup=mean(baseline) / mean(mojo),
            )
        )
    return rows


def _bench_render(render_mod, *, repeats: int) -> list[BenchRow]:
    rows: list[BenchRow] = []
    rng = np.random.default_rng(1)
    for width, height, num_tris in [(16, 16, 96), (32, 32, 256), (48, 48, 512)]:
        triangles = rng.normal(size=(num_tris, 3, 3)).astype(np.float32)
        triangles[:, :, 2] += 5.0
        depth = np.empty((width * height,), dtype=np.float32)
        hit = np.empty((width * height,), dtype=np.uint8)

        baseline = _time_call(
            lambda triangles=triangles, width=width, height=height: _python_render_depth_map(
                triangles,
                width=width,
                height=height,
                fx=60.0,
                fy=60.0,
                cx=width / 2.0,
                cy=height / 2.0,
                znear=1e-3,
                zfar=20.0,
            ),
            repeats,
        )
        mojo = _time_call(
            lambda triangles=triangles, num_tris=num_tris, width=width, height=height, depth=depth, hit=hit: render_mod.render_depth_map_f32(
                triangles.ctypes.data,
                num_tris,
                (width, height, 60.0, 60.0, width / 2.0, height / 2.0, 1e-3, 20.0, min(os.cpu_count() or 1, width * height)),
                depth.ctypes.data,
                hit.ctypes.data,
            ),
            repeats,
        )
        exp_depth, exp_hit = _python_render_depth_map(
            triangles[:64],
            width=16,
            height=16,
            fx=60.0,
            fy=60.0,
            cx=8.0,
            cy=8.0,
            znear=1e-3,
            zfar=20.0,
        )
        test_depth = np.empty((16 * 16,), dtype=np.float32)
        test_hit = np.empty((16 * 16,), dtype=np.uint8)
        render_mod.render_depth_map_f32(
            triangles[:64].ctypes.data,
            64,
            (16, 16, 60.0, 60.0, 8.0, 8.0, 1e-3, 20.0, 16),
            test_depth.ctypes.data,
            test_hit.ctypes.data,
        )
        assert np.allclose(exp_depth, test_depth, atol=1e-4)
        assert np.array_equal(exp_hit, test_hit)
        rows.append(
            BenchRow(
                kernel="depth render",
                size_label=f"{width}x{height}, T={num_tris}",
                baseline_mean_ms=mean(baseline),
                mojo_mean_ms=mean(mojo),
                speedup=mean(baseline) / mean(mojo),
            )
        )
    return rows


def _bench_unproject(render_mod, *, repeats: int) -> list[BenchRow]:
    rows: list[BenchRow] = []
    rng = np.random.default_rng(2)
    pose = np.array([[1, 0, 0, 0.1], [0, 1, 0, -0.2], [0, 0, 1, 0.3]], dtype=np.float32)
    for width, height, stride in [(48, 48, 1), (64, 64, 2), (96, 96, 4)]:
        depth = rng.uniform(2.0, 6.0, size=(width * height,)).astype(np.float32)
        valid = (rng.random(size=(width * height,)) > 0.2).astype(np.uint8)
        max_points = ((height + stride - 1) // stride) * ((width + stride - 1) // stride)
        out_points = np.full((max_points, 3), np.nan, dtype=np.float32)
        out_count = np.zeros((1,), dtype=np.int32)
        out_bounds = np.zeros((6,), dtype=np.float32)

        baseline = _time_call(
            lambda depth=depth, valid=valid, pose=pose, width=width, height=height, stride=stride: _python_unproject_valid_points(
                depth,
                valid,
                pose,
                width=width,
                height=height,
                fx=80.0,
                fy=80.0,
                cx=width / 2.0,
                cy=height / 2.0,
                stride=stride,
            ),
            repeats,
        )
        mojo = _time_call(
            lambda depth=depth, valid=valid, pose=pose, width=width, height=height, stride=stride, out_points=out_points, out_count=out_count, out_bounds=out_bounds: render_mod.unproject_valid_points_f32(
                depth.ctypes.data,
                valid.ctypes.data,
                pose.reshape(-1).ctypes.data,
                (width, height, 80.0, 80.0, width / 2.0, height / 2.0, stride),
                (out_points.ctypes.data, out_count.ctypes.data, out_bounds.ctypes.data),
            ),
            repeats,
        )
        exp_points, exp_bounds = _python_unproject_valid_points(
            depth[: 32 * 32],
            valid[: 32 * 32],
            pose,
            width=32,
            height=32,
            fx=80.0,
            fy=80.0,
            cx=16.0,
            cy=16.0,
            stride=2,
        )
        test_points = np.full((((32 + 1) // 2) * ((32 + 1) // 2), 3), np.nan, dtype=np.float32)
        test_count = np.zeros((1,), dtype=np.int32)
        test_bounds = np.zeros((6,), dtype=np.float32)
        render_mod.unproject_valid_points_f32(
            depth[: 32 * 32].ctypes.data,
            valid[: 32 * 32].ctypes.data,
            pose.reshape(-1).ctypes.data,
            (32, 32, 80.0, 80.0, 16.0, 16.0, 2),
            (test_points.ctypes.data, test_count.ctypes.data, test_bounds.ctypes.data),
        )
        assert test_count[0] == exp_points.shape[0]
        assert np.allclose(test_points[: test_count[0]], exp_points, atol=1e-5)
        assert np.allclose(test_bounds, exp_bounds, atol=1e-5)
        rows.append(
            BenchRow(
                kernel="unproject",
                size_label=f"{width}x{height}, s={stride}",
                baseline_mean_ms=mean(baseline),
                mojo_mean_ms=mean(mojo),
                speedup=mean(baseline) / mean(mojo),
            )
        )
    return rows


def _bench_distance(dist_mod, *, repeats: int) -> list[BenchRow]:
    rows: list[BenchRow] = []
    for num_points, num_tris in [(128, 64), (512, 256), (1024, 512)]:
        rng = np.random.default_rng(3)
        points = rng.normal(size=(num_points, 3)).astype(np.float32)
        triangles = rng.normal(size=(num_tris, 3, 3)).astype(np.float32)
        out_points = np.empty((num_points,), dtype=np.float32)
        out_tris = np.empty((num_tris,), dtype=np.float32)

        baseline = _time_call(
            lambda points=points, triangles=triangles: (
                _python_point_mesh_distance_sq(points, triangles),
                _python_triangle_point_distance_sq(points, triangles),
            ),
            repeats,
        )
        mojo = _time_call(
            lambda points=points, num_points=num_points, triangles=triangles, num_tris=num_tris, out_points=out_points, out_tris=out_tris: (
                dist_mod.point_mesh_distance_sq_f32(
                    points.ctypes.data,
                    num_points,
                    triangles.ctypes.data,
                    num_tris,
                    out_points.ctypes.data,
                    min(os.cpu_count() or 1, num_points),
                ),
                dist_mod.triangle_point_distance_sq_f32(
                    points.ctypes.data,
                    num_points,
                    triangles.ctypes.data,
                    num_tris,
                    out_tris.ctypes.data,
                    min(os.cpu_count() or 1, num_tris),
                ),
            ),
            repeats,
        )
        exp_points = _python_point_mesh_distance_sq(points[:32], triangles[:32])
        exp_tris = _python_triangle_point_distance_sq(points[:32], triangles[:32])
        test_points = np.empty((32,), dtype=np.float32)
        test_tris = np.empty((32,), dtype=np.float32)
        dist_mod.point_mesh_distance_sq_f32(
            points[:32].ctypes.data,
            32,
            triangles[:32].ctypes.data,
            32,
            test_points.ctypes.data,
            min(os.cpu_count() or 1, 32),
        )
        dist_mod.triangle_point_distance_sq_f32(
            points[:32].ctypes.data,
            32,
            triangles[:32].ctypes.data,
            32,
            test_tris.ctypes.data,
            min(os.cpu_count() or 1, 32),
        )
        assert np.allclose(exp_points, test_points, atol=1e-5)
        assert np.allclose(exp_tris, test_tris, atol=1e-5)
        rows.append(
            BenchRow(
                kernel="full distance",
                size_label=f"P={num_points}, T={num_tris}",
                baseline_mean_ms=mean(baseline),
                mojo_mean_ms=mean(mojo),
                speedup=mean(baseline) / mean(mojo),
            )
        )
    return rows


def _plot(rows: list[BenchRow], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    kernels = sorted({row.kernel for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, kernel in zip(axes, kernels, strict=False):
        subset = [row for row in rows if row.kernel == kernel]
        labels = [row.size_label for row in subset]
        baseline = [row.baseline_mean_ms for row in subset]
        mojo = [row.mojo_mean_ms for row in subset]
        x = np.arange(len(subset))
        ax.bar(x - 0.18, baseline, width=0.36, label="Python baseline", color="#9c6644")
        ax.bar(x + 0.18, mojo, width=0.36, label="Mojo", color="#2a9d8f")
        ax.set_title(kernel)
        ax.set_ylabel("Mean runtime [ms]")
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Mojo Oracle Kernel Benchmarks", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    bar_path = out_dir / "mojo_oracle_kernel_runtime.png"
    fig.savefig(bar_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(bar_path)

    fig2, ax2 = plt.subplots(figsize=(11, 5))
    speedups = [row.speedup for row in rows]
    labels2 = [f"{row.kernel}\n{row.size_label}" for row in rows]
    x2 = np.arange(len(rows))
    colors = ["#1d3557" if s >= 2.0 else "#457b9d" if s >= 1.2 else "#e76f51" for s in speedups]
    ax2.bar(x2, speedups, color=colors)
    ax2.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax2.axhline(2.0, color="#2a9d8f", linewidth=1.0, linestyle=":")
    ax2.set_ylabel("Speedup vs Python baseline [x]")
    ax2.set_title("Mojo speedup by kernel and problem size")
    ax2.set_xticks(x2, labels2, rotation=25, ha="right")
    ax2.grid(axis="y", alpha=0.25)
    fig2.tight_layout()
    speed_path = out_dir / "mojo_oracle_kernel_speedup.png"
    fig2.savefig(speed_path, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    paths.append(speed_path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5, help="Timed repetitions per benchmark case.")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/nbv-mojo-oracle-report"), help="Output directory for plots.")
    args = parser.parse_args()

    mesh_mod, render_mod, dist_mod = _load_modules()
    rows: list[BenchRow] = []
    print("running collision benchmarks...", flush=True)
    rows.extend(_bench_collision(mesh_mod, repeats=args.repeats))
    print("running render benchmarks...", flush=True)
    rows.extend(_bench_render(render_mod, repeats=args.repeats))
    print("running unprojection benchmarks...", flush=True)
    rows.extend(_bench_unproject(render_mod, repeats=args.repeats))
    print("running distance benchmarks...", flush=True)
    rows.extend(_bench_distance(dist_mod, repeats=args.repeats))
    plot_paths = _plot(rows, args.out_dir)

    print("Kernel benchmark summary")
    for row in rows:
        print(
            f"{row.kernel:>13} | {row.size_label:<18} | baseline={row.baseline_mean_ms:8.3f} ms | "
            f"mojo={row.mojo_mean_ms:8.3f} ms | speedup={row.speedup:5.2f}x"
        )
    print("Plots:")
    for path in plot_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
