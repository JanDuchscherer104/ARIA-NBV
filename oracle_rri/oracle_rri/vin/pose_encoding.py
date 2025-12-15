"""Pose encoding modules for VIN candidate poses.

For a learnable positional encoding, we implement Learnable Fourier Features
as introduced in:

    *Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding*
    (NeurIPS 2021)

The implementation follows the paper's formulation (learned projection +
sin/cos features + small MLP). For reference, see the public implementation at:
https://github.com/JHLew/Learnable-Fourier-Features
"""

from __future__ import annotations

import math

import torch
from pydantic import Field
from torch import nn

from ..utils import BaseConfig

Tensor = torch.Tensor


class FourierFeatures(nn.Module):
    """Learnable Fourier features for continuous inputs.

    Given an input ``x ∈ R^D`` we compute features:

        [x, sin(2π B x), cos(2π B x)]

    where B is a learnable matrix of shape ``(F, D)``.
    """

    def __init__(
        self,
        input_dim: int,
        num_frequencies: int,
        *,
        include_input: bool = True,
        learnable: bool = True,
        init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0.")
        if num_frequencies <= 0:
            raise ValueError("num_frequencies must be > 0.")

        self.input_dim = int(input_dim)
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        self.learnable = bool(learnable)

        b_init = torch.randn((self.num_frequencies, self.input_dim)) * float(init_scale)
        if learnable:
            self.B = nn.Parameter(b_init)
        else:
            self.register_buffer("B", b_init, persistent=False)

    @property
    def output_dim(self) -> int:
        base = self.input_dim if self.include_input else 0
        return base + 2 * self.num_frequencies

    def forward(self, x: Tensor) -> Tensor:
        """Encode inputs.

        Args:
            x: ``Tensor["... input_dim", float32]``.

        Returns:
            ``Tensor["... output_dim", float32]``.
        """

        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected x[..., {self.input_dim}], got {tuple(x.shape)}.")

        proj = (x @ self.B.T) * (2.0 * math.pi)
        emb = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        if not self.include_input:
            return emb
        return torch.cat([x, emb], dim=-1)


class LearnableFourierFeatures(nn.Module):
    """Learnable Fourier Features (LFF) positional encoding.

    This module maps continuous inputs $x \\in \\mathbb{R}^D$ into a learned
    feature space by:

    1) learning a projection matrix $W_r$,
    2) applying sinusoidal features, and
    3) mapping them through a small MLP.

    Compared to fixed random Fourier features, the learned projection and the
    MLP allow the encoding to adapt to the downstream task.
    """

    def __init__(
        self,
        input_dim: int,
        fourier_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        gamma: float = 1.0,
        include_input: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0.")
        if fourier_dim <= 0 or fourier_dim % 2 != 0:
            raise ValueError("fourier_dim must be a positive even integer.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0.")
        if output_dim <= 0:
            raise ValueError("output_dim must be > 0.")
        if gamma <= 0:
            raise ValueError("gamma must be > 0.")

        self.input_dim = int(input_dim)
        self.fourier_dim = int(fourier_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.include_input = bool(include_input)

        half = self.fourier_dim // 2
        self.Wr = nn.Parameter(torch.randn((half, self.input_dim)) * float(gamma**2))
        self.mlp = nn.Sequential(
            nn.Linear(self.fourier_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        self._div_term = math.sqrt(self.fourier_dim)

    @property
    def out_dim(self) -> int:
        return (self.input_dim if self.include_input else 0) + self.output_dim

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected x[..., {self.input_dim}], got {tuple(x.shape)}.")

        xwr = x @ self.Wr.T
        fourier = torch.cat([torch.cos(xwr), torch.sin(xwr)], dim=-1) / float(self._div_term)
        enc = self.mlp(fourier)
        if not self.include_input:
            return enc
        return torch.cat([x, enc], dim=-1)


class FourierFeaturesConfig(BaseConfig[FourierFeatures]):
    """Config-as-factory wrapper for :class:`FourierFeatures`."""

    target: type[FourierFeatures] = Field(default=FourierFeatures, exclude=True)

    input_dim: int = 6
    """Input dimensionality (default: 6 for SE(3) relative pose as translation + so(3) log)."""

    num_frequencies: int = 8
    """Number of Fourier frequencies."""

    include_input: bool = True
    """Concatenate raw inputs to Fourier features when True."""

    learnable: bool = True
    """Make the frequency matrix learnable when True."""

    init_scale: float = 1.0
    """Stddev for initializing the frequency matrix."""

    def setup_target(self) -> FourierFeatures:  # type: ignore[override]
        return self.target(
            input_dim=int(self.input_dim),
            num_frequencies=int(self.num_frequencies),
            include_input=bool(self.include_input),
            learnable=bool(self.learnable),
            init_scale=float(self.init_scale),
        )


class LearnableFourierFeaturesConfig(BaseConfig[LearnableFourierFeatures]):
    """Config-as-factory wrapper for :class:`LearnableFourierFeatures`."""

    target: type[LearnableFourierFeatures] = Field(default=LearnableFourierFeatures, exclude=True)

    input_dim: int = 6
    """Input dimensionality (default: 6 for SE(3) relative pose as translation + so(3) log)."""

    fourier_dim: int = 128
    """Fourier feature dimensionality (must be even)."""

    hidden_dim: int = 256
    """Hidden dimension of the internal MLP."""

    output_dim: int = 64
    """Output positional encoding dimensionality (per input vector)."""

    gamma: float = 1.0
    """Initialization scale (paper parameter $\\gamma$)."""

    include_input: bool = False
    """Concatenate raw inputs to the learned encoding when True."""

    def setup_target(self) -> LearnableFourierFeatures:  # type: ignore[override]
        return self.target(
            input_dim=int(self.input_dim),
            fourier_dim=int(self.fourier_dim),
            hidden_dim=int(self.hidden_dim),
            output_dim=int(self.output_dim),
            gamma=float(self.gamma),
            include_input=bool(self.include_input),
        )
