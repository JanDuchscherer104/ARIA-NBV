"""Typed wrappers around :mod:`power_spherical` distributions."""

from __future__ import annotations

from collections.abc import Iterable

import torch

try:  # pragma: no cover
    from power_spherical import HypersphericalUniform as _HypersphericalUniform
    from power_spherical import PowerSpherical as _PowerSpherical
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError("Install 'power_spherical' to use pose generation samplers") from exc


class HypersphericalUniform(_HypersphericalUniform):
    """Uniform distribution on the unit hypersphere with device-aware sampling."""

    def __init__(self, dim: int, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None):
        super().__init__(dim=dim, device=device, dtype=dtype)

    def sample(self, sample_shape: Iterable[int] = ()):  # type: ignore[override]
        return super().sample(sample_shape)

    def rsample(self, sample_shape: Iterable[int] = ()):  # type: ignore[override]
        return super().rsample(sample_shape)


class PowerSpherical(_PowerSpherical):
    """PowerSpherical distribution with explicit dtype/device handling."""

    def __init__(
        self,
        mu: torch.Tensor,
        kappa: torch.Tensor | float,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        mu = mu.to(device=device, dtype=dtype) if device or dtype else mu
        if not isinstance(kappa, torch.Tensor):
            kappa = torch.tensor(kappa, device=mu.device, dtype=mu.dtype)
        else:
            kappa = kappa.to(device=mu.device, dtype=mu.dtype)
        super().__init__(mu, kappa)

    def sample(self, sample_shape: Iterable[int] = ()):  # type: ignore[override]
        return super().sample(sample_shape)

    def rsample(self, sample_shape: Iterable[int] = ()):  # type: ignore[override]
        return super().rsample(sample_shape)


__all__ = ["HypersphericalUniform", "PowerSpherical"]
