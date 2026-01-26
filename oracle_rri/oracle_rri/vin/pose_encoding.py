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
from pydantic import Field, field_validator
from torch import Tensor, nn

from ..utils import BaseConfig


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

    def __init__(self, config: "LearnableFourierFeaturesConfig") -> None:
        super().__init__()
        self.config = config
        self.input_dim = self.config.input_dim
        self.fourier_dim = self.config.fourier_dim
        self.hidden_dim = self.config.hidden_dim
        self.output_dim = self.config.output_dim
        self.include_input = self.config.include_input

        half = self.fourier_dim // 2
        self.Wr = nn.Parameter(torch.randn((half, self.input_dim)) * self.config.gamma)
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
        """

        TODO: Ensure same signature as other pose encoders - i.e. ShellShPoseEncoder.forward (derive both from a common base class!)
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected x[..., {self.input_dim}], got {tuple(x.shape)}.")

        xwr = x @ self.Wr.T
        fourier = torch.cat([torch.cos(xwr), torch.sin(xwr)], dim=-1) / self._div_term
        enc = self.mlp(fourier)
        if not self.include_input:
            return enc
        return torch.cat([x, enc], dim=-1)


class LearnableFourierFeaturesConfig(BaseConfig[LearnableFourierFeatures]):
    """Config-as-factory wrapper for :class:`LearnableFourierFeatures`."""

    target: type[LearnableFourierFeatures] = Field(default=LearnableFourierFeatures, exclude=True)

    input_dim: int = Field(default=6, gt=0)
    """Input dimensionality (default: 6 for SE(3) relative pose as translation + so(3) log)."""

    fourier_dim: int = Field(default=128, gt=0)
    """Fourier feature dimensionality (must be even)."""

    hidden_dim: int = Field(default=256, gt=0)
    """Hidden dimension of the internal MLP."""

    output_dim: int = Field(default=64, gt=0)
    """Output positional encoding dimensionality (per input vector)."""

    gamma: float = Field(default=1.0, gt=0.0)
    """Initialization scale (paper parameter $\\gamma$)."""

    include_input: bool = False
    """Concatenate raw inputs to the learned encoding when True."""

    @field_validator("fourier_dim")
    @classmethod
    def _validate_fourier_dim_is_even(cls, value: int) -> int:
        if value % 2 != 0:
            raise ValueError("fourier_dim must be even.")
        return value
