"""Plot ordinal binning thresholds for RRI on a small real-data sample.

This script runs the oracle label pipeline for a limited number of snippets and
visualizes:

1) raw RRI distribution,
2) fitted quantile edges on raw RRI values,
3) resulting ordinal label histogram (K classes, approximately balanced).

Outputs are written under `docs/figures/impl/vin/` by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def _ensure_oracle_rri_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _device_from_arg(value: str) -> torch.device:
    if value.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _load_binner_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing binner JSON at {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_from_logs(logs_dir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    rri_path = logs_dir / "rri_binner_rri.npy"
    edges_path = logs_dir / "rri_binner_edges.npy"
    json_path = logs_dir / "rri_binner.json"

    rri = np.load(rri_path)
    edges = np.load(edges_path)
    num_classes = int(edges.shape[0]) + 1
    if json_path.exists():
        data = _load_binner_json(json_path)
        num_classes = int(data.get("num_classes", num_classes))
    return rri, edges, num_classes


def main() -> None:
    _ensure_oracle_rri_importable()

    from oracle_rri.data import AseEfmDatasetConfig
    from oracle_rri.pipelines.oracle_rri_labeler import OracleRriLabelerConfig
    from oracle_rri.pose_generation import CandidateViewGeneratorConfig
    from oracle_rri.rendering import CandidateDepthRendererConfig
    from oracle_rri.rendering.pytorch3d_depth_renderer import Pytorch3DDepthRendererConfig
    from oracle_rri.rri_metrics.rri_binning import RriOrdinalBinner
    from oracle_rri.utils import Verbosity

    parser = argparse.ArgumentParser(description="Plot RRI binning thresholds from a small oracle sample.")
    parser.add_argument("--scene-id", type=str, default="81283")
    parser.add_argument(
        "--atek-variant",
        type=str,
        default="efm",
        choices=["efm", "efm_eval", "cubercnn", "cubercnn_eval"],
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-snippets", type=int, default=6)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--backprojection-stride", type=int, default=16)
    parser.add_argument("--mesh-simplify-ratio", type=float, default=0.02)
    parser.add_argument("--num-classes", type=int, default=15)
    parser.add_argument("--out-dir", type=str, default="docs/figures/impl/vin")
    parser.add_argument("--logs-dir", type=str, default=".logs/vin")
    parser.add_argument("--use-logs", action="store_true", help="Use precomputed RRI/edges from logs-dir.")
    parser.add_argument("--rri-npy", type=str, default="")
    parser.add_argument("--edges-npy", type=str, default="")
    parser.add_argument("--binner-json", type=str, default="")
    args = parser.parse_args()

    device = _device_from_arg(str(args.device))
    out_dir = Path(args.out_dir).resolve()

    sns.set_theme(style="whitegrid", context="talk")

    rri_np: np.ndarray | None = None
    edges_np: np.ndarray | None = None
    num_classes: int | None = None

    if args.use_logs or args.rri_npy or args.edges_npy or args.binner_json:
        logs_dir = Path(args.logs_dir)
        if args.use_logs:
            rri_np, edges_np, num_classes = _load_from_logs(logs_dir)
        else:
            if args.rri_npy:
                rri_np = np.load(args.rri_npy)
            if args.edges_npy:
                edges_np = np.load(args.edges_npy)
            if args.binner_json:
                data = _load_binner_json(Path(args.binner_json))
                num_classes = int(data["num_classes"])
                if edges_np is None:
                    edges_np = np.array(data["edges"], dtype=np.float32)
        if rri_np is None or edges_np is None:
            raise RuntimeError("Need rri_npy and edges_npy (or use --use-logs) to plot from logs.")
        if num_classes is None:
            num_classes = int(edges_np.shape[0]) + 1
        rri = torch.tensor(rri_np, dtype=torch.float32)
        edges = torch.tensor(edges_np, dtype=torch.float32)
        mask = torch.isfinite(rri)
        rri = rri[mask]
        binner = RriOrdinalBinner(num_classes=int(num_classes), edges=edges.detach().cpu())
        labels = binner.transform(rri)
    else:
        ds_cfg = AseEfmDatasetConfig(
            atek_variant=str(args.atek_variant),
            scene_ids=[str(args.scene_id)],
            load_meshes=True,
            require_mesh=True,
            mesh_simplify_ratio=float(args.mesh_simplify_ratio),
            device=str(device),
            batch_size=None,
            verbosity=Verbosity.QUIET,
            is_debug=False,
        )
        ds = ds_cfg.setup_target()

        labeler_cfg = OracleRriLabelerConfig(
            generator=CandidateViewGeneratorConfig(
                num_samples=128,
                device=str(device),
                verbosity=Verbosity.QUIET,
                is_debug=False,
            ),
            depth=CandidateDepthRendererConfig(
                renderer=Pytorch3DDepthRendererConfig(
                    # Avoid PyTorch3D bin overflows for dense indoor meshes.
                    # For the small sample sizes in this script, the naive path is fast enough.
                    bin_size=0,
                    verbosity=Verbosity.QUIET,
                ),
                max_candidates_final=int(args.max_candidates),
                oversample_factor=1.0,
                verbosity=Verbosity.QUIET,
                is_debug=False,
            ),
            backprojection_stride=int(args.backprojection_stride),
            out_device="cpu",
        )
        labeler = labeler_cfg.setup_target()

        rri_all: list[torch.Tensor] = []
        for i, sample in enumerate(ds):
            if i >= int(args.num_snippets):
                break
            batch = labeler.run(sample)
            rri_all.append(batch.rri.rri.detach().cpu().to(dtype=torch.float32))

        if not rri_all:
            raise RuntimeError("No RRI samples collected (dataset iteration produced no snippets).")

        rri = torch.cat(rri_all, dim=0).reshape(-1)
        mask = torch.isfinite(rri)
        rri = rri[mask]

        if rri.numel() < int(args.num_classes):
            raise RuntimeError(
                f"Need at least K={int(args.num_classes)} samples for quantile edges, got {rri.numel()}."
            )

        binner = RriOrdinalBinner.fit_from_iterable(
            [rri],
            num_classes=int(args.num_classes),
        )

        labels = binner.transform(rri)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0))

    sns.histplot(rri.numpy(), bins=50, color="#285f82", alpha=0.9, ax=axes[0])
    for edge in binner.edges.detach().cpu().numpy().tolist():
        axes[0].axvline(float(edge), color="black", linewidth=1.0, alpha=0.25)
    axes[0].set_title("Raw oracle RRI + quantile edges")
    axes[0].set_xlabel("rri")
    axes[0].set_ylabel("count")

    k = int(binner.num_classes)
    counts = torch.bincount(labels, minlength=k).numpy()
    sns.barplot(x=np.arange(k), y=counts, color="#285f82", ax=axes[1])
    axes[1].set_title("Ordinal labels (K classes)")
    axes[1].set_xlabel("label")
    axes[1].set_ylabel("count")

    fig.suptitle(
        f"RRI ordinal binning (scene={args.scene_id}, snippets={int(args.num_snippets)}, "
        f"samples={int(rri.numel())}, K={k})",
        fontsize=13,
    )

    path = out_dir / "vin_rri_binning.png"
    _save_fig(fig, path)
    print(f"Saved: {path}")

    fig_edges, ax_edges = plt.subplots(1, 1, figsize=(6.5, 4.0))
    edges = binner.edges.detach().cpu().numpy()
    sns.lineplot(x=np.arange(1, k), y=edges, marker="o", ax=ax_edges, color="#285f82")
    ax_edges.set_title("RRI quantile thresholds")
    ax_edges.set_xlabel("quantile index j")
    ax_edges.set_ylabel("edge e_j")
    ax_edges.set_xlim(1, k - 1)

    edges_path = out_dir / "vin_rri_thresholds.png"
    _save_fig(fig_edges, edges_path)
    print(f"Saved: {edges_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
