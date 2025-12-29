"""Plotting utilities for VIN encodings and pose descriptors."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from e3nn import o3  # type: ignore[import-untyped]

from .types import VinForwardDiagnostics


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _unit_dir_from_az_el(*, az: torch.Tensor, el: torch.Tensor) -> torch.Tensor:
    """Convert az/el (LUF: az=atan2(x,z), el=asin(y)) to unit vectors."""

    x = torch.cos(el) * torch.sin(az)
    y = torch.sin(el)
    z = torch.cos(el) * torch.cos(az)
    v = torch.stack([x, y, z], dim=-1)
    return v / (v.norm(dim=-1, keepdim=True).clamp_min(1e-8))


def _plot_shell_descriptor_concept(
    *,
    u: torch.Tensor,
    f: torch.Tensor,
    r: torch.Tensor,
    out_dir: Path,
    stem: str,
) -> Path:
    """Plot a simple 3D diagram for one candidate descriptor."""

    u0 = u[0].detach().cpu().numpy()
    f0 = f[0].detach().cpu().numpy()
    r0 = float(r[0].detach().cpu().item())
    t0 = u0 * r0

    fig = plt.figure(figsize=(7.0, 6.2))
    ax = fig.add_subplot(111, projection="3d")

    # Reference axes (LUF)
    axis_len = 1.2
    ax.quiver(0, 0, 0, axis_len, 0, 0, color="#285f82", linewidth=2, arrow_length_ratio=0.06)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color="#285f82", linewidth=2, arrow_length_ratio=0.06)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color="#285f82", linewidth=2, arrow_length_ratio=0.06)
    ax.text(axis_len, 0, 0, "x (left)", color="#285f82")
    ax.text(0, axis_len, 0, "y (up)", color="#285f82")
    ax.text(0, 0, axis_len, "z (fwd)", color="#285f82")

    # Candidate center and vectors.
    ax.scatter([t0[0]], [t0[1]], [t0[2]], color="#fc5555", s=60, label="candidate center")
    ax.quiver(0, 0, 0, t0[0], t0[1], t0[2], color="#fc5555", linewidth=2, arrow_length_ratio=0.06, label="t=r·u")

    # Unit vectors shown from the candidate center for readability.
    ax.quiver(
        t0[0],
        t0[1],
        t0[2],
        u0[0],
        u0[1],
        u0[2],
        color="#285f82",
        linewidth=2,
        arrow_length_ratio=0.10,
        label="u (pos dir)",
    )
    ax.quiver(
        t0[0],
        t0[1],
        t0[2],
        f0[0],
        f0[1],
        f0[2],
        color="#2a9d8f",
        linewidth=2,
        arrow_length_ratio=0.10,
        label="f (forward)",
    )

    lim = 2.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_title("Shell descriptor (one candidate): t, r, u, f", pad=14)
    ax.legend(loc="upper left", frameon=False)
    ax.view_init(elev=18, azim=35)

    path = out_dir / f"{stem}_shell_descriptor_concept.png"
    _save_fig(fig, path)
    return path


def _plot_sh_components(
    *,
    lmax: int,
    normalization: str,
    out_dir: Path,
    stem: str,
    n_az: int = 220,
    n_el: int = 110,
) -> Path:
    """Plot a few real SH components as heatmaps over az/el."""

    az = torch.linspace(-torch.pi, torch.pi, steps=int(n_az))
    el = torch.linspace(-0.5 * torch.pi, 0.5 * torch.pi, steps=int(n_el))
    az_grid, el_grid = torch.meshgrid(az, el, indexing="xy")  # (n_az,n_el)
    dirs = _unit_dir_from_az_el(az=az_grid, el=el_grid)  # (n_az,n_el,3)

    irreps = o3.Irreps.spherical_harmonics(int(lmax))
    y = o3.spherical_harmonics(irreps, dirs, normalize=True, normalization=str(normalization))  # (n_az,n_el,dim)

    # Map component index to (l, m) assuming e3nn ordering (l=0..L, m=-l..l).
    lm: list[tuple[int, int]] = []
    for degree in range(int(lmax) + 1):
        for order in range(-degree, degree + 1):
            lm.append((degree, order))

    wanted = [(0, 0), (1, -1), (1, 0), (1, 1)]
    idxs: list[int] = []
    titles: list[str] = []
    for degree, order in wanted:
        try:
            idx = lm.index((degree, order))
        except ValueError:  # pragma: no cover
            continue
        idxs.append(idx)
        titles.append(f"Y_{degree}^{order} (component {idx})")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2))
    for ax, idx, title in zip(axes.ravel(), idxs, titles, strict=True):
        comp = y[..., idx].detach().cpu().numpy().T  # (n_el,n_az) for imshow
        im = ax.imshow(
            comp,
            origin="lower",
            cmap="coolwarm",
            extent=[-180, 180, -90, 90],
            aspect="auto",
        )
        ax.set_title(title)
        ax.set_xlabel("azimuth (deg)")
        ax.set_ylabel("elevation (deg)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(f"Real spherical harmonics over directions on S^2 (lmax={lmax})", fontsize=14)

    path = out_dir / f"{stem}_sh_components.png"
    _save_fig(fig, path)
    return path


def _plot_radius_fourier_features(
    *,
    r_min: float,
    r_max: float,
    freqs: Iterable[float],
    out_dir: Path,
    stem: str,
) -> Path:
    """Plot sin/cos radius Fourier features for linear r (meters)."""

    r = torch.linspace(float(r_min), float(r_max), steps=500).view(-1, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))

    axes[0].hist(r.squeeze(1).numpy(), bins=50, color="#285f82", alpha=0.9)
    axes[0].set_title("radius r (meters)")
    axes[0].set_xlabel("r")
    axes[0].set_ylabel("count")

    freq_list = list(freqs)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(freq_list)))
    for c, w in zip(colors, freq_list, strict=True):
        axes[1].plot(
            r.squeeze(1).numpy(),
            torch.sin(2.0 * torch.pi * float(w) * r).squeeze(1).numpy(),
            color=c,
            label=f"{w:g}",
        )

    axes[1].set_title("example Fourier features: sin(2π ω r)")
    axes[1].set_xlabel("r (m)")
    axes[1].set_ylabel("value")
    axes[1].grid(True, linewidth=0.3, alpha=0.5)
    axes[1].legend(title="ω", frameon=False, fontsize=9)

    fig.suptitle("1D Fourier features for radius (linear input)", fontsize=14)

    path = out_dir / f"{stem}_radius_fourier_features.png"
    _save_fig(fig, path)
    return path


def plot_vin_encodings_from_debug(
    debug: VinForwardDiagnostics,
    *,
    out_dir: Path,
    lmax: int,
    sh_normalization: str,
    radius_freqs: Iterable[float],
    file_stem_prefix: str,
) -> dict[str, Path]:
    """Generate VIN encoding plots using the VIN forward diagnostics.

    Args:
        debug: Diagnostics from :meth:`VinModel.forward_with_debug`.
        out_dir: Output directory for saved figures.
        lmax: Maximum SH degree to visualize.
        sh_normalization: Spherical harmonics normalization mode.
        radius_freqs: Frequencies for the radius Fourier feature plot.
        file_stem_prefix: Prefix used for output filenames.

    Returns:
        Mapping of figure labels to saved paths.
    """

    out_dir = Path(out_dir)
    u = debug.candidate_center_dir_rig.reshape(-1, 3)
    f = debug.candidate_forward_dir_rig.reshape(-1, 3)
    r = debug.candidate_radius_m.reshape(-1, 1)

    r_min = float(r.min().item())
    r_max = float(r.max().item())
    if r_min == r_max:
        r_max = r_min + 1e-3

    plots: dict[str, Path] = {}
    plots["shell_descriptor"] = _plot_shell_descriptor_concept(u=u, f=f, r=r, out_dir=out_dir, stem=file_stem_prefix)
    plots["sh_components"] = _plot_sh_components(
        lmax=int(lmax),
        normalization=str(sh_normalization),
        out_dir=out_dir,
        stem=file_stem_prefix,
    )
    plots["radius_fourier_features"] = _plot_radius_fourier_features(
        r_min=r_min,
        r_max=r_max,
        freqs=radius_freqs,
        out_dir=out_dir,
        stem=file_stem_prefix,
    )
    return plots
