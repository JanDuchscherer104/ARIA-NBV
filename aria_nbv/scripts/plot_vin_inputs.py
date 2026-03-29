"""Generate plots for VIN input modalities and features on a real ASE snippet.

This script saves a small set of figures under `docs/figures/impl/vin/` by default:

1) Input images (RGB + SLAM-L) with shapes
2) Candidate pose descriptor distributions (r, <f,-u>, az/el for u and f)
3) EVL neck feature visualizations (magnitude slices + histograms)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def _ensure_aria_nbv_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _device_from_arg(value: str) -> torch.device:
    if value.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _to_uint8_image(img: torch.Tensor) -> np.ndarray:
    """Convert CHW float/uint8 to HWC uint8 for plotting."""

    if img.ndim != 3:
        raise ValueError(f"Expected CHW image, got {tuple(img.shape)}")
    if img.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        arr = img.to(dtype=torch.uint8).permute(1, 2, 0).cpu().numpy()
        return arr
    arr_f = img.to(dtype=torch.float32).clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (arr_f * 255.0).round().astype(np.uint8)


def _az_el_luf(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute azimuth/elevation in LUF coordinates (x=left, y=up, z=fwd)."""

    if v.shape[-1] != 3:
        raise ValueError(f"Expected (...,3), got {tuple(v.shape)}")
    v = v / (v.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    az = torch.atan2(v[..., 0], v[..., 2])
    el = torch.asin(v[..., 1].clamp(-1.0, 1.0))
    return az, el


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_input_images(*, sample: Any, out_dir: Path) -> Path:
    rgb = sample.efm["rgb/img"][0]  # C H W
    slaml = sample.efm["slaml/img"][0]  # 1 H W

    rgb_u8 = _to_uint8_image(rgb)
    slaml_u8 = _to_uint8_image(slaml.repeat(3, 1, 1))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(rgb_u8)
    axes[0].set_title(f"rgb/img[0]  {tuple(rgb.shape)}")
    axes[0].axis("off")

    axes[1].imshow(slaml_u8)
    axes[1].set_title(f"slaml/img[0]  {tuple(slaml.shape)}")
    axes[1].axis("off")

    path = out_dir / "vin_input_images.png"
    _save_fig(fig, path)
    return path


def _plot_pose_descriptor(*, candidates: Any, out_dir: Path) -> Path:
    # candidates.views.T_camera_rig: camera <- reference (PoseTW)
    t_cam_ref = candidates.views.T_camera_rig
    t_ref_cam = t_cam_ref.inverse()  # reference <- camera

    t = t_ref_cam.t.view(-1, 3).to(dtype=torch.float32)  # N 3
    r = torch.linalg.vector_norm(t, dim=-1).clamp_min(1e-8)
    u = t / r[:, None]

    z_cam = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    r_rc = t_ref_cam.R.to(dtype=torch.float32)
    f = torch.einsum("nij,j->ni", r_rc, z_cam)
    f = f / (f.norm(dim=-1, keepdim=True).clamp_min(1e-8))

    s = (f * (-u)).sum(dim=-1)  # <f, -u>
    dot_fu = (f * u).sum(dim=-1)  # <f, u>

    az_u, el_u = _az_el_luf(u)
    az_f, el_f = _az_el_luf(f)
    az_neg_u, el_neg_u = _az_el_luf(-u)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].hist(r.cpu().numpy(), bins=40, color="#285f82", alpha=0.9)
    axes[0, 0].set_title("radius r = ||t|| (m)")
    axes[0, 0].set_xlabel("r")
    axes[0, 0].set_ylabel("count")

    axes[0, 1].hist(
        s.cpu().numpy(),
        bins=40,
        range=(-1.0, 1.0),
        color="#fc5555",
        alpha=0.9,
        label="s=<f,-u>",
    )
    axes[0, 1].hist(
        dot_fu.cpu().numpy(),
        bins=40,
        range=(-1.0, 1.0),
        histtype="step",
        color="#285f82",
        linewidth=2.0,
        label="<f,u>",
    )
    axes[0, 1].set_title("view alignment scalars")
    axes[0, 1].set_xlabel("value")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].set_xlim(-1.0, 1.0)
    axes[0, 1].ticklabel_format(useOffset=False, axis="x")
    axes[0, 1].legend(loc="upper left", frameon=False, fontsize=9)

    axes[1, 0].scatter(np.rad2deg(az_u.cpu().numpy()), np.rad2deg(el_u.cpu().numpy()), s=6, alpha=0.6)
    axes[1, 0].set_title("u direction (az/el, deg) in reference (LUF)")
    axes[1, 0].set_xlabel("azimuth (deg)")
    axes[1, 0].set_ylabel("elevation (deg)")
    axes[1, 0].set_xlim(-180, 180)
    axes[1, 0].set_ylim(-90, 90)
    axes[1, 0].grid(True, linewidth=0.3, alpha=0.5)

    axes[1, 1].scatter(
        np.rad2deg(az_f.cpu().numpy()),
        np.rad2deg(el_f.cpu().numpy()),
        s=10,
        alpha=0.55,
        label="f",
    )
    axes[1, 1].scatter(
        np.rad2deg(az_neg_u.cpu().numpy()),
        np.rad2deg(el_neg_u.cpu().numpy()),
        s=6,
        alpha=0.35,
        label="-u",
    )
    axes[1, 1].set_title("forward f vs direction to reference (-u)")
    axes[1, 1].set_xlabel("azimuth (deg)")
    axes[1, 1].set_ylabel("elevation (deg)")
    axes[1, 1].set_xlim(-180, 180)
    axes[1, 1].set_ylim(-90, 90)
    axes[1, 1].grid(True, linewidth=0.3, alpha=0.5)
    axes[1, 1].legend(loc="upper left", frameon=False, fontsize=9)

    fig.suptitle(f"Candidate pose descriptor (N_c={int(t.shape[0])})", fontsize=14)

    path = out_dir / "vin_pose_descriptor.png"
    _save_fig(fig, path)
    return path


def _plot_evl_features(*, backbone_out: Any, out_dir: Path, max_hist: int = 200_000) -> Path:
    occ = backbone_out.occ_feat  # 1 C D H W
    obb = backbone_out.obb_feat

    # Compute magnitude volume: mean abs across channels.
    occ_mag = occ.abs().mean(dim=1).squeeze(0)  # D H W
    obb_mag = obb.abs().mean(dim=1).squeeze(0)  # D H W
    d = int(occ_mag.shape[0])
    mid = d // 2

    occ_slice = occ_mag[mid].detach().float().cpu().numpy()
    obb_slice = obb_mag[mid].detach().float().cpu().numpy()

    # Histograms: sample values to keep it fast.
    rng = torch.Generator(device=occ.device).manual_seed(0)
    occ_flat = occ_mag.flatten()
    obb_flat = obb_mag.flatten()
    if occ_flat.numel() > max_hist:
        idx = torch.randint(0, occ_flat.numel(), (max_hist,), generator=rng, device=occ.device)
        occ_samp = occ_flat[idx]
        obb_samp = obb_flat[idx]
    else:
        occ_samp = occ_flat
        obb_samp = obb_flat
    occ_np = occ_samp.detach().float().cpu().numpy()
    obb_np = obb_samp.detach().float().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    im0 = axes[0, 0].imshow(occ_slice, cmap="viridis")
    axes[0, 0].set_title(f"occ_feat | mean|abs| slice D={mid}  ({tuple(occ_mag.shape)})")
    axes[0, 0].axis("off")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(obb_slice, cmap="viridis")
    axes[0, 1].set_title(f"obb_feat | mean|abs| slice D={mid}  ({tuple(obb_mag.shape)})")
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[1, 0].hist(occ_np, bins=60, color="#285f82", alpha=0.9)
    axes[1, 0].set_title("occ_feat mean|abs| histogram (sampled)")
    axes[1, 0].set_xlabel("value")
    axes[1, 0].set_ylabel("count")

    axes[1, 1].hist(obb_np, bins=60, color="#fc5555", alpha=0.9)
    axes[1, 1].set_title("obb_feat mean|abs| histogram (sampled)")
    axes[1, 1].set_xlabel("value")
    axes[1, 1].set_ylabel("count")

    fig.suptitle("EVL neck feature magnitudes (frozen backbone)", fontsize=14)

    path = out_dir / "vin_evl_features.png"
    _save_fig(fig, path)
    return path


def main() -> None:
    _ensure_aria_nbv_importable()

    from aria_nbv.data import AseEfmDatasetConfig
    from aria_nbv.pose_generation import CandidateViewGeneratorConfig
    from aria_nbv.utils import Verbosity
    from aria_nbv.vin import EvlBackboneConfig, VinModelConfig

    parser = argparse.ArgumentParser(description="Plot VIN input features on a real snippet.")
    parser.add_argument("--scene-id", type=str, default="81283")
    parser.add_argument(
        "--atek-variant",
        type=str,
        default="efm",
        choices=["efm", "efm_eval", "cubercnn", "cubercnn_eval"],
    )
    parser.add_argument("--device", type=str, default="auto", help="Device for EVL inference (plots computed on CPU).")
    parser.add_argument("--num-samples", type=int, default=256, help="Candidate samples for descriptor plots.")
    parser.add_argument("--mesh-simplify-ratio", type=float, default=0.02)
    parser.add_argument("--out-dir", type=str, default="docs/figures/impl/vin")
    args = parser.parse_args()

    device = _device_from_arg(str(args.device))
    out_dir = Path(args.out_dir).resolve()

    # Load sample on CPU (candidate generation uses mesh).
    ds_cfg = AseEfmDatasetConfig(
        atek_variant=str(args.atek_variant),
        scene_ids=[str(args.scene_id)],
        load_meshes=True,
        require_mesh=True,
        mesh_simplify_ratio=float(args.mesh_simplify_ratio),
        device="cpu",
        batch_size=None,
        verbosity=Verbosity.QUIET,
        is_debug=False,
    )
    ds = ds_cfg.setup_target()
    sample = next(iter(ds))

    # Candidate generation for descriptor plots.
    cand_cfg = CandidateViewGeneratorConfig(
        num_samples=int(args.num_samples),
        device="cpu",
        verbosity=Verbosity.QUIET,
        is_debug=False,
    )
    candidates = cand_cfg.setup_target().generate_from_typed_sample(sample)

    # EVL features for magnitude plots (runs on GPU if available).
    vin = VinModelConfig(backbone=EvlBackboneConfig(device=device)).setup_target().eval()
    with torch.no_grad():
        backbone_out = vin.backbone.forward(sample.efm)

    p0 = _plot_input_images(sample=sample, out_dir=out_dir)
    p1 = _plot_pose_descriptor(candidates=candidates, out_dir=out_dir)
    p2 = _plot_evl_features(backbone_out=backbone_out, out_dir=out_dir)

    print("Saved:")
    print(f"  {p0}")
    print(f"  {p1}")
    print(f"  {p2}")


if __name__ == "__main__":  # pragma: no cover
    main()
