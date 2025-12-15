"""RRI normalization and binning utilities (VIN-NBV style).

VIN-NBV does not regress RRI directly. Instead it:
  1) groups training samples by *capture stage* (e.g. number of base views),
  2) z-normalizes RRI within each stage,
  3) soft-clips z-scores with ``tanh``, and
  4) maps clipped z-scores to ordinal bins with approximately equal mass.

This module implements that label construction so training code can produce
ordinal labels compatible with the CORAL loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

Tensor = torch.Tensor


@dataclass(slots=True)
class RriOrdinalBinner:
    """Stage-aware RRI → ordinal label mapping.

    Attributes:
        num_classes: Number of ordinal classes ``K``.
        tanh_scale: Scale applied before ``tanh`` (z / tanh_scale).
        stage_mean: Mean RRI per stage id.
        stage_std: Stddev RRI per stage id (clamped to ``>= eps``).
        edges: ``Tensor["K-1", float32]`` monotonically increasing bin edges in
            clipped z-score space.
    """

    num_classes: int
    tanh_scale: float
    stage_mean: dict[int, float]
    stage_std: dict[int, float]
    edges: Tensor

    @staticmethod
    def _validate_inputs(rri: Tensor, stage: Tensor) -> tuple[Tensor, Tensor]:
        if rri.ndim != 1:
            rri = rri.reshape(-1)
        if stage.ndim != 1:
            stage = stage.reshape(-1)
        if rri.numel() != stage.numel():
            raise ValueError(f"rri and stage must have same numel, got {rri.numel()} and {stage.numel()}.")
        if stage.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"stage must be integer dtype, got {stage.dtype}.")
        return rri.to(dtype=torch.float32), stage.to(dtype=torch.int64)

    @classmethod
    def fit(
        cls,
        rri: Tensor,
        stage: Tensor,
        *,
        num_classes: int = 15,
        tanh_scale: float = 1.0,
        eps: float = 1e-6,
    ) -> RriOrdinalBinner:
        """Fit stage stats and global equal-mass bin edges.

        Args:
            rri: Oracle RRI values. Shape ``(...,)``.
            stage: Capture stage id per sample (e.g., number of base views). Shape ``(...,)``.
            num_classes: Number of ordinal classes ``K``.
            tanh_scale: Scale applied before tanh clipping.
            eps: Lower bound for stddev to avoid division by zero.

        Returns:
            A fitted :class:`RriOrdinalBinner`.
        """

        if num_classes < 2:
            raise ValueError("num_classes must be >= 2.")
        if tanh_scale <= 0:
            raise ValueError("tanh_scale must be > 0.")

        rri_f, stage_i = cls._validate_inputs(rri, stage)
        unique = torch.unique(stage_i)

        stage_mean: dict[int, float] = {}
        stage_std: dict[int, float] = {}
        z_all: list[Tensor] = []

        for sid in unique.tolist():
            mask = stage_i == int(sid)
            vals = rri_f[mask]
            mean = vals.mean()
            std = vals.std(unbiased=False).clamp_min(float(eps))
            stage_mean[int(sid)] = float(mean.item())
            stage_std[int(sid)] = float(std.item())
            z_all.append((vals - mean) / std)

        z = torch.cat(z_all, dim=0)
        z_clip = torch.tanh(z / float(tanh_scale))

        qs = torch.linspace(1.0 / num_classes, (num_classes - 1) / num_classes, steps=num_classes - 1, device=z.device)
        edges = torch.quantile(z_clip, qs).to(dtype=torch.float32)

        return cls(
            num_classes=int(num_classes),
            tanh_scale=float(tanh_scale),
            stage_mean=stage_mean,
            stage_std=stage_std,
            edges=edges,
        )

    def transform(self, rri: Tensor, stage: Tensor) -> Tensor:
        """Convert RRI values to ordinal labels.

        Args:
            rri: Oracle RRI values. Shape ``(...,)``.
            stage: Capture stage ids. Shape ``(...,)``.

        Returns:
            ``Tensor["...", int64]`` labels in ``[0, K-1]``.
        """

        rri_f, stage_i = self._validate_inputs(rri, stage)
        z = torch.empty_like(rri_f)
        for sid, mean in self.stage_mean.items():
            mask = stage_i == int(sid)
            if not mask.any():
                continue
            std = float(self.stage_std.get(int(sid), 1.0))
            z[mask] = (rri_f[mask] - float(mean)) / max(std, 1e-6)
        z_clip = torch.tanh(z / float(self.tanh_scale))

        # labels = number of edges below value
        labels = torch.bucketize(z_clip, self.edges.to(device=z_clip.device), right=False)
        return labels.to(dtype=torch.int64)

    # --------------------------------------------------------------------- IO helpers
    def to_dict(self) -> dict:
        return {
            "num_classes": int(self.num_classes),
            "tanh_scale": float(self.tanh_scale),
            "stage_mean": dict(self.stage_mean),
            "stage_std": dict(self.stage_std),
            "edges": self.edges.detach().cpu().tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> RriOrdinalBinner:
        return cls(
            num_classes=int(data["num_classes"]),
            tanh_scale=float(data.get("tanh_scale", 1.0)),
            stage_mean={int(k): float(v) for k, v in dict(data["stage_mean"]).items()},
            stage_std={int(k): float(v) for k, v in dict(data["stage_std"]).items()},
            edges=torch.tensor(data["edges"], dtype=torch.float32),
        )

    def save(self, path: str | Path) -> Path:
        """Save binner to JSON."""
        import json

        out_path = Path(path)
        out_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> RriOrdinalBinner:
        """Load binner from JSON."""
        import json

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
