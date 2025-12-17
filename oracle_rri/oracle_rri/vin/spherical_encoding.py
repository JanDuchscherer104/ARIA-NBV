"""Spherical-harmonics pose encoding for VIN.

VIN candidate generation samples camera centers on a (possibly biased) spherical shell around a
reference pose. A natural descriptor for such candidates is therefore expressed in terms of
directions on $\\mathbb{S}^2$ and a scalar radius.

This module provides a lightweight pose encoder based on real spherical harmonics computed via
`e3nn`, plus a 1D Fourier-feature encoding for the radius and a small MLP for additional scalar
pose terms.
"""

from __future__ import annotations

from typing import Literal

import torch
from e3nn import o3  # type: ignore[import-untyped]
from pydantic import Field
from torch import Tensor, nn

from ..utils import BaseConfig
from .pose_encoding import FourierFeatures


class ShellShPoseEncoder(nn.Module):
    """Encode shell-based pose descriptors using spherical harmonics.

    Inputs are unit vectors on $\\mathbb{S}^2$:

    - $u$: candidate position direction in the reference frame,
    - $f$: candidate forward direction in the reference frame,

    plus a radius $r=\\lVert t\\rVert$ and optional scalar pose terms (e.g. $\\langle f, -u \\rangle$).

    The encoder computes real spherical harmonics up to degree ``lmax`` for $u$ and $f$ and projects
    them into a learnable embedding space. The radius is encoded via **1D Fourier features** (on
    $r$ by default) and projected to a learnable embedding.

    Args:
        lmax: Maximum spherical harmonics degree.
        sh_out_dim: Output dimensionality after projecting each SH vector.
        radius_num_frequencies: Number of Fourier frequencies for the radius encoding.
        radius_out_dim: Output dimensionality after projecting the radius Fourier features.
        radius_include_input: Concatenate the raw radius input to Fourier features when True.
        radius_learnable: Make the radius Fourier frequency matrix learnable when True.
        radius_init_scale: Stddev used to initialize the radius Fourier frequency matrix.
        radius_log_input: Encode $\\log(r+\\varepsilon)$ when True, else encode $r$ directly.
        include_radius: If ``True``, include the radius embedding in the output.
        scalar_in_dim: Number of additional scalar pose features.
        scalar_out_dim: Output dimensionality of the scalar MLP.
        scalar_hidden_dim: Hidden size for the scalar MLP.
        normalization: e3nn SH normalization mode.
        include_scalars: If ``True``, include the scalar MLP output in the embedding.
    """

    def __init__(
        self,
        *,
        lmax: int,
        sh_out_dim: int,
        radius_num_frequencies: int,
        radius_out_dim: int,
        radius_include_input: bool,
        radius_learnable: bool,
        radius_init_scale: float,
        radius_log_input: bool,
        include_radius: bool,
        scalar_in_dim: int,
        scalar_out_dim: int,
        scalar_hidden_dim: int,
        normalization: Literal["component", "norm"] = "component",
        include_scalars: bool = True,
    ) -> None:
        super().__init__()
        if lmax < 0:
            raise ValueError("lmax must be >= 0.")
        if sh_out_dim <= 0:
            raise ValueError("sh_out_dim must be > 0.")
        if radius_num_frequencies <= 0:
            raise ValueError("radius_num_frequencies must be > 0.")
        if radius_out_dim <= 0:
            raise ValueError("radius_out_dim must be > 0.")
        if radius_init_scale <= 0:
            raise ValueError("radius_init_scale must be > 0.")
        if scalar_in_dim <= 0:
            raise ValueError("scalar_in_dim must be > 0.")
        if scalar_out_dim <= 0:
            raise ValueError("scalar_out_dim must be > 0.")
        if scalar_hidden_dim <= 0:
            raise ValueError("scalar_hidden_dim must be > 0.")

        self.lmax = lmax
        self.normalization = str(normalization)
        self.include_scalars = bool(include_scalars)
        self.include_radius = bool(include_radius)
        self.radius_log_input = bool(radius_log_input)

        irreps_sh = o3.Irreps.spherical_harmonics(self.lmax)
        sh_in_dim = int(irreps_sh.dim)

        def _proj(in_dim: int, out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(int(in_dim), int(out_dim)),
                nn.GELU(),
                nn.Linear(int(out_dim), int(out_dim)),
            )

        self._irreps_sh = irreps_sh
        self._proj_u = _proj(sh_in_dim, sh_out_dim)
        self._proj_f = _proj(sh_in_dim, sh_out_dim)

        self._radius_ff = FourierFeatures(
            input_dim=1,
            num_frequencies=int(radius_num_frequencies),
            include_input=bool(radius_include_input),
            learnable=bool(radius_learnable),
            init_scale=float(radius_init_scale),
        )
        self._proj_r = _proj(self._radius_ff.output_dim, radius_out_dim)

        self._scalar_mlp = nn.Sequential(
            nn.Linear(int(scalar_in_dim), int(scalar_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(scalar_hidden_dim), int(scalar_out_dim)),
        )

        self._sh_out_dim = int(sh_out_dim)
        self._radius_out_dim = int(radius_out_dim)
        self._scalar_out_dim = int(scalar_out_dim)

    @property
    def out_dim(self) -> int:
        """Output embedding dimensionality."""

        base = 2 * self._sh_out_dim
        if self.include_radius:
            base += self._radius_out_dim
        if self.include_scalars:
            base += self._scalar_out_dim
        return base

    def forward(self, u: Tensor, f: Tensor, *, r: Tensor, scalars: Tensor | None = None) -> Tensor:
        """Encode directions and scalar pose features.

        Args:
            u: ``Tensor["... 3", float32]`` unit direction from reference to candidate center.
            f: ``Tensor["... 3", float32]`` unit forward direction of the candidate camera.
            r: ``Tensor["... 1", float32]`` radius $\\lVert t\\rVert$ of the candidate center in the reference frame.
            scalars: Optional ``Tensor["... scalar_in_dim", float32]`` scalar pose features.

        Returns:
            ``Tensor["... out_dim", float32]`` pose embedding.
        """

        if u.shape[-1] != 3 or f.shape[-1] != 3:
            raise ValueError(f"Expected u,f with last dim 3, got {u.shape} and {f.shape}.")
        if r.shape[-1] != 1:
            raise ValueError(f"Expected r[..., 1], got {tuple(r.shape)}.")
        if self.include_scalars and scalars is None:
            raise ValueError("scalars must be provided when include_scalars=True.")

        u_f32 = u.to(dtype=torch.float32)
        f_f32 = f.to(dtype=torch.float32)

        y_u = o3.spherical_harmonics(
            self._irreps_sh,
            u_f32,
            normalize=True,
            normalization=self.normalization,
        )
        y_f = o3.spherical_harmonics(
            self._irreps_sh,
            f_f32,
            normalize=True,
            normalization=self.normalization,
        )

        parts: list[Tensor] = [self._proj_u(y_u), self._proj_f(y_f)]
        if self.include_radius:
            r_f32 = r.to(dtype=torch.float32)
            if self.radius_log_input:
                r_f32 = torch.log(r_f32 + 1e-8)
            r_ff = self._radius_ff(r_f32)
            parts.append(self._proj_r(r_ff))
        if self.include_scalars:
            scalars_f32 = scalars.to(dtype=torch.float32)  # type: ignore[union-attr]
            parts.append(self._scalar_mlp(scalars_f32))
        return torch.cat(parts, dim=-1)


class ShellShPoseEncoderConfig(BaseConfig[ShellShPoseEncoder]):
    """Config-as-factory wrapper for :class:`ShellShPoseEncoder`."""

    target: type[ShellShPoseEncoder] = Field(default=ShellShPoseEncoder, exclude=True)

    lmax: int = 2
    """Maximum spherical harmonics degree."""

    sh_out_dim: int = 16
    """Projection dimension for each SH vector (for $u$ and $f$)."""

    radius_num_frequencies: int = 6
    """Number of Fourier frequencies for the 1D radius encoding."""

    radius_out_dim: int = 16
    """Projection dimension for the encoded radius."""

    radius_include_input: bool = True
    """Concatenate the raw radius input to Fourier features when True."""

    radius_learnable: bool = True
    """Make the radius Fourier frequency matrix learnable when True."""

    radius_init_scale: float = 1.0
    """Stddev used to initialize the radius Fourier frequency matrix."""

    radius_log_input: bool = False
    """Encode $r$ directly by default; set to ``True`` to encode $\\log(r+\\varepsilon)`` instead."""

    include_radius: bool = True
    """Include radius features in the embedding when True."""

    scalar_in_dim: int = 1
    """Number of additional scalar pose features (e.g. $\\langle f, -u \\rangle$)."""

    scalar_out_dim: int = 16
    """Output dimension of the scalar MLP."""

    scalar_hidden_dim: int = 32
    """Hidden dimension of the scalar MLP."""

    normalization: Literal["component", "norm"] = "component"
    """e3nn spherical harmonics normalization mode."""

    include_scalars: bool = True
    """Include scalar features in the embedding when True."""

    def setup_target(self) -> ShellShPoseEncoder:  # type: ignore[override]
        return self.target(
            lmax=int(self.lmax),
            sh_out_dim=int(self.sh_out_dim),
            radius_num_frequencies=int(self.radius_num_frequencies),
            radius_out_dim=int(self.radius_out_dim),
            radius_include_input=bool(self.radius_include_input),
            radius_learnable=bool(self.radius_learnable),
            radius_init_scale=float(self.radius_init_scale),
            radius_log_input=bool(self.radius_log_input),
            include_radius=bool(self.include_radius),
            scalar_in_dim=int(self.scalar_in_dim),
            scalar_out_dim=int(self.scalar_out_dim),
            scalar_hidden_dim=int(self.scalar_hidden_dim),
            normalization=str(self.normalization),
            include_scalars=bool(self.include_scalars),
        )


__all__ = [
    "ShellShPoseEncoder",
    "ShellShPoseEncoderConfig",
]
