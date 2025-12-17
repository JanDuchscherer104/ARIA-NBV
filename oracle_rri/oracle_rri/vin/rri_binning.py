"""RRI → ordinal label binning (VIN-NBV style).

We discretize oracle RRI values into ordinal bins:

1) z-normalize within a *stage*,
2) soft-clip z via ``tanh``, and
3) fit global equal-mass quantile edges.

The binner can optionally store (and save) the raw fit data on CPU so you can
refit edges for different ``K`` without rerunning the oracle.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

Tensor = torch.Tensor


def _unique_path(path: Path, *, overwrite: bool) -> Path:
    """Return a non-overwriting path by appending ``-<n>`` before the suffix."""

    path = Path(path)
    if overwrite or not path.exists():
        return path
    idx = 1
    while True:  # pragma: no cover - should terminate quickly
        candidate = path.with_name(f"{path.stem}-{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


@dataclass(slots=True)
class RriOrdinalBinner:
    """Stage-aware RRI → ordinal label mapping.

    Attributes:
        num_classes: Number of ordinal classes ``K``.
        tanh_scale: Scale applied before tanh clipping (z / tanh_scale).
        stage_mean: Mean RRI per stage id.
        stage_std: Stddev RRI per stage id (clamped to ``>= eps``).
        edges: ``Tensor["K-1", float32]`` monotonically increasing bin edges in
            clipped z-score space.
    """

    num_classes: int = 0
    tanh_scale: float = 1.0
    stage_mean: dict[int, float] = field(default_factory=dict)
    stage_std: dict[int, float] = field(default_factory=dict)
    edges: Tensor = field(default_factory=lambda: torch.empty((0,), dtype=torch.float32))

    eps: float = 1e-6
    """Lower bound for per-stage stddev (also used during fitting)."""

    _rri_chunks: list[Tensor] = field(default_factory=list, repr=False)
    _stage_chunks: list[Tensor] = field(default_factory=list, repr=False)

    # --------------------------------------------------------------------- validation / IO
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_classes": int(self.num_classes),
            "tanh_scale": float(self.tanh_scale),
            "stage_mean": dict(self.stage_mean),
            "stage_std": dict(self.stage_std),
            "edges": self.edges.detach().cpu().tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RriOrdinalBinner:
        return cls(
            num_classes=int(data["num_classes"]),
            tanh_scale=float(data.get("tanh_scale", 1.0)),
            stage_mean={int(k): float(v) for k, v in dict(data["stage_mean"]).items()},
            stage_std={int(k): float(v) for k, v in dict(data["stage_std"]).items()},
            edges=torch.tensor(data["edges"], dtype=torch.float32),
        )

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Save fitted binner to JSON (non-overwriting by default)."""

        import json

        out_path = _unique_path(Path(path), overwrite=overwrite)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(out_path, json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> RriOrdinalBinner:
        """Load fitted binner from JSON (no fit data)."""

        import json

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def save_fit_data(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Save accumulated fit data (CPU tensors) as a torch file."""

        out_path = _unique_path(Path(path), overwrite=overwrite)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rri, stage = self._fit_tensors()
        state = {"tanh_scale": float(self.tanh_scale), "eps": float(self.eps), "rri": rri, "stage": stage}
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, out_path)
        return out_path

    @classmethod
    def load_fit_data(cls, path: str | Path) -> RriOrdinalBinner:
        """Load fit data (CPU tensors) and return an *unfitted* binner."""

        load_path = Path(path)
        try:
            state = torch.load(load_path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older torch
            state = torch.load(load_path, map_location="cpu")
        if not isinstance(state, dict):  # pragma: no cover - defensive
            raise TypeError(f"Expected dict state for fit data, got {type(state)}.")
        binner = cls(tanh_scale=float(state.get("tanh_scale", 1.0)), eps=float(state.get("eps", 1e-6)))
        binner.append(state["rri"], state["stage"])
        return binner

    # --------------------------------------------------------------------- fit data (CPU)
    def append(self, rri: Tensor, stage: Tensor) -> None:
        """Append new fit samples (stored on CPU)."""

        rri_f, stage_i = self._validate_inputs(rri, stage)
        self._rri_chunks.append(rri_f.detach().to(device="cpu"))
        self._stage_chunks.append(stage_i.detach().to(device="cpu"))

    def _fit_tensors(self) -> tuple[Tensor, Tensor]:
        if not self._rri_chunks:
            return torch.empty((0,), dtype=torch.float32), torch.empty((0,), dtype=torch.int64)
        return torch.cat(self._rri_chunks, dim=0), torch.cat(self._stage_chunks, dim=0)

    # --------------------------------------------------------------------- fitting / refitting
    def fit_edges(self, *, num_classes: int) -> RriOrdinalBinner:
        """Fit stage stats + quantile edges from the accumulated fit data."""

        if int(num_classes) < 2:
            raise ValueError("num_classes must be >= 2.")
        if float(self.tanh_scale) <= 0:
            raise ValueError("tanh_scale must be > 0.")

        rri, stage = self._fit_tensors()
        if rri.numel() == 0:
            raise ValueError("No fit data available. Call append(...) or load_fit_data(...) first.")

        unique = torch.unique(stage)

        stage_mean: dict[int, float] = {}
        stage_std: dict[int, float] = {}
        z_all: list[Tensor] = []

        for sid in unique.tolist():
            mask = stage == int(sid)
            vals = rri[mask]
            mean = vals.mean()
            std = vals.std(unbiased=False).clamp_min(float(self.eps))
            stage_mean[int(sid)] = float(mean.item())
            stage_std[int(sid)] = float(std.item())
            z_all.append((vals - mean) / std)

        z = torch.cat(z_all, dim=0)
        z_clip = torch.tanh(z / float(self.tanh_scale))

        qs = torch.linspace(
            1.0 / float(num_classes),
            float(num_classes - 1) / float(num_classes),
            steps=int(num_classes) - 1,
            device=z_clip.device,
        )
        edges = torch.quantile(z_clip, qs).to(dtype=torch.float32)

        self.num_classes = int(num_classes)
        self.stage_mean = stage_mean
        self.stage_std = stage_std
        self.edges = edges
        return self

    def refit_edges(self, *, num_classes: int) -> RriOrdinalBinner:
        """Recompute edges for a different ``K`` using the stored fit data."""

        return self.fit_edges(num_classes=int(num_classes))

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
        """Fit binner from tensors (fit data is stored on CPU for refitting)."""

        binner = cls(tanh_scale=float(tanh_scale), eps=float(eps))
        binner.append(rri, stage)
        return binner.fit_edges(num_classes=int(num_classes))

    @classmethod
    def fit_from_iterable(
        cls,
        iterable: Iterable[tuple[Tensor, Tensor]],
        *,
        num_classes: int = 15,
        tanh_scale: float = 1.0,
        eps: float = 1e-6,
        save_fit_data_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> RriOrdinalBinner:
        """Fit binner from an iterable, saving partial fit data on errors/Ctrl-C."""

        binner = cls(tanh_scale=float(tanh_scale), eps=float(eps))
        try:
            for rri, stage in iterable:
                binner.append(rri, stage)
        except (KeyboardInterrupt, Exception):
            if save_fit_data_path is not None:
                binner.save_fit_data(save_fit_data_path, overwrite=overwrite)
            raise
        return binner.fit_edges(num_classes=int(num_classes))

    # --------------------------------------------------------------------- label mapping
    def transform(self, rri: Tensor, stage: Tensor) -> Tensor:
        """Convert RRI values to ordinal labels.

        Args:
            rri: Oracle RRI values. Shape ``(...,)``.
            stage: Capture stage ids. Shape ``(...,)``.

        Returns:
            ``Tensor["...", int64]`` labels in ``[0, K-1]``.
        """

        if int(self.num_classes) < 2 or self.edges.numel() != int(self.num_classes) - 1:
            raise RuntimeError("Binner not fitted. Call fit(...) or fit_edges(...) first.")

        rri_f, stage_i = self._validate_inputs(rri, stage)
        unique = torch.unique(stage_i).tolist()
        missing = [int(s) for s in unique if int(s) not in self.stage_mean]
        if missing:
            raise ValueError(f"Stage ids missing from binner stats: {missing}.")

        z = torch.empty_like(rri_f)
        for sid in unique:
            mask = stage_i == int(sid)
            mean = float(self.stage_mean[int(sid)])
            std = float(self.stage_std[int(sid)])
            z[mask] = (rri_f[mask] - mean) / max(std, float(self.eps))
        z_clip = torch.tanh(z / float(self.tanh_scale))

        labels = torch.bucketize(z_clip, self.edges.to(device=z_clip.device), right=False)
        return labels.to(dtype=torch.int64)


__all__ = ["RriOrdinalBinner"]
