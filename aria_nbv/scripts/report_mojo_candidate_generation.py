"""Generate benchmark plots and numeric summaries for the experimental Mojo backend."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parents[1]))

from benchmark_mojo_candidate_generation import asdict, run_benchmark_case  # noqa: E402


def _count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _count_exported_functions(path: Path) -> int:
    return path.read_text(encoding="utf-8").count("module.def_function[")


def _implementation_metrics(repo_root: Path) -> dict[str, int]:
    python_bridge = repo_root / "aria_nbv" / "aria_nbv" / "pose_generation" / "mojo_backend.py"
    mojo_kernel = repo_root / "aria_nbv" / "aria_nbv" / "pose_generation" / "mojo" / "mesh_collision_kernels.mojo"
    return {
        "runtime_python_files": 1,
        "runtime_mojo_files": 1,
        "python_bridge_loc": _count_lines(python_bridge),
        "mojo_kernel_loc": _count_lines(mojo_kernel),
        "mojo_exported_functions": _count_exported_functions(mojo_kernel),
        "active_accelerated_rules": 1,
        "fallback_rules": 1,
    }


def _plot_benchmarks(summary: dict[str, object], out_path: Path) -> None:
    cases = summary["cases"]
    case_labels = ["clearance only", "full pipeline"]
    trimesh_means = [cases["clearance_only"]["trimesh"]["mean_ms"], cases["full_pipeline"]["trimesh"]["mean_ms"]]
    mojo_means = [cases["clearance_only"]["mojo"]["mean_ms"], cases["full_pipeline"]["mojo"]["mean_ms"]]
    speedups = [cases["clearance_only"]["speedup_vs_trimesh"], cases["full_pipeline"]["speedup_vs_trimesh"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    ax = axes[0]
    x = [0, 1]
    width = 0.34
    ax.bar([i - width / 2 for i in x], trimesh_means, width=width, label="Python/Trimesh", color="#d9a441")
    ax.bar([i + width / 2 for i in x], mojo_means, width=width, label="Mojo backend", color="#2d7dd2")
    for idx, speedup in enumerate(speedups):
        ymax = max(trimesh_means[idx], mojo_means[idx])
        ax.text(idx, ymax + 4.0, f"{speedup:.2f}x", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x, case_labels)
    ax.set_ylabel("Mean runtime (ms)")
    ax.set_title("Candidate Generation Runtime")
    ax.legend(frameon=False)

    ax = axes[1]
    distributions = [
        cases["clearance_only"]["trimesh"]["timings_ms"],
        cases["clearance_only"]["mojo"]["timings_ms"],
        cases["full_pipeline"]["trimesh"]["timings_ms"],
        cases["full_pipeline"]["mojo"]["timings_ms"],
    ]
    ax.boxplot(
        distributions,
        patch_artist=True,
        tick_labels=[
            "clear\npy",
            "clear\nmojo",
            "full\npy",
            "full\nmojo",
        ],
        boxprops={"facecolor": "#d7e3f4", "edgecolor": "#284b63"},
        medianprops={"color": "#c1121f", "linewidth": 2},
    )
    ax.set_ylabel("Per-run runtime (ms)")
    ax.set_title("Per-run timing distribution")

    fig.suptitle("Mojo Candidate-Generation Benchmark", fontsize=15, fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_impl_overview(metrics: dict[str, int], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, face: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.2,rounding_size=0.08",
            linewidth=1.5,
            edgecolor="#1f2933",
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=16, linewidth=1.5))

    box(0.4, 3.9, 2.1, 1.0, "Config\nCollisionBackend.MOJO", "#e7f0fd")
    box(3.1, 4.0, 2.2, 0.9, "MinDistanceToMeshRule", "#d9f0d8")
    box(3.1, 1.6, 2.2, 0.9, "PathCollisionRule", "#f7e6c4")
    box(5.9, 4.0, 2.2, 0.9, "mojo_backend.py", "#e7f0fd")
    box(5.9, 1.6, 2.2, 0.9, "Trimesh ray engine\n(fallback kept)", "#fce8d6")
    box(8.6, 4.0, 2.3, 0.9, "mesh_collision_kernels.mojo", "#dbeafe")
    box(8.6, 1.6, 2.3, 0.9, "Ground-truth mesh", "#eef2f7")

    arrow(2.5, 4.4, 3.1, 4.45)
    arrow(2.5, 4.3, 3.1, 2.05)
    arrow(5.3, 4.45, 5.9, 4.45)
    arrow(8.1, 4.45, 8.6, 4.45)
    arrow(5.3, 2.05, 5.9, 2.05)
    arrow(8.1, 2.05, 8.6, 2.05)

    ax.text(
        0.4,
        0.45,
        (
            f"Runtime files: {metrics['runtime_python_files']} Python + {metrics['runtime_mojo_files']} Mojo\n"
            f"Bridge LOC: {metrics['python_bridge_loc']} | Kernel LOC: {metrics['mojo_kernel_loc']}\n"
            f"Exported Mojo functions: {metrics['mojo_exported_functions']}\n"
            f"Live accelerated rules: {metrics['active_accelerated_rules']} | Runtime fallbacks: {metrics['fallback_rules']}"
        ),
        fontsize=11,
        family="monospace",
    )
    ax.text(6.0, 5.35, "Current runtime topology", fontsize=15, fontweight="bold", ha="center")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_timings_csv(summary: dict[str, object], out_path: Path) -> None:
    cases = summary["cases"]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "backend", "run_index", "timing_ms"],
        )
        writer.writeheader()
        for case_name, case_summary in cases.items():
            for backend in ("trimesh", "mojo"):
                for idx, timing in enumerate(case_summary[backend]["timings_ms"], start=1):
                    writer.writerow(
                        {
                            "case": case_name,
                            "backend": backend,
                            "run_index": idx,
                            "timing_ms": timing,
                        }
                    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=THIS_DIR.parents[1] / ".logs" / "benchmarks" / "mojo_candidate_generation",
        help="Directory for PNG/JSON/CSV report artifacts.",
    )
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--mesh-subdivisions", type=int, default=3)
    parser.add_argument("--min-distance", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clearance = run_benchmark_case(
        ensure_collision_free=False,
        repeats=args.repeats,
        num_samples=args.num_samples,
        mesh_subdivisions=args.mesh_subdivisions,
        min_distance=args.min_distance,
    )
    full = run_benchmark_case(
        ensure_collision_free=True,
        repeats=args.repeats,
        num_samples=args.num_samples,
        mesh_subdivisions=args.mesh_subdivisions,
        min_distance=args.min_distance,
    )

    summary = {
        "implementation": _implementation_metrics(THIS_DIR.parents[1]),
        "cases": {
            "clearance_only": {
                "trimesh": asdict(clearance["trimesh"]),
                "mojo": asdict(clearance["mojo"]),
                "speedup_vs_trimesh": clearance["trimesh"].mean_ms / clearance["mojo"].mean_ms,
            },
            "full_pipeline": {
                "trimesh": asdict(full["trimesh"]),
                "mojo": asdict(full["mojo"]),
                "speedup_vs_trimesh": full["trimesh"].mean_ms / full["mojo"].mean_ms,
            },
        },
    }

    summary_path = out_dir / "benchmark_summary.json"
    timings_path = out_dir / "benchmark_timings.csv"
    benchmark_plot_path = out_dir / "benchmark_runtime.png"
    impl_plot_path = out_dir / "mojo_impl_overview.png"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_timings_csv(summary, timings_path)
    _plot_benchmarks(summary, benchmark_plot_path)
    _plot_impl_overview(summary["implementation"], impl_plot_path)

    print(summary_path)
    print(timings_path)
    print(benchmark_plot_path)
    print(impl_plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
