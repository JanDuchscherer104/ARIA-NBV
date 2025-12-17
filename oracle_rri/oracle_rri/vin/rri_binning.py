"""RRI → ordinal label binning for CORAL training.

VIN trains an ordinal regressor (CORAL) rather than directly regressing oracle
RRI values. We therefore map continuous RRIs to ordinal labels via *empirical
quantile edges* (equal-mass bins):

- fit $K-1$ quantiles on observed oracle RRIs,
- label = number of edges below the value (``torch.bucketize``).

Design goals:
    - Accumulate fit data on CPU.
    - Persist fit data to resume after Ctrl-C / crashes.
    - Avoid overwriting JSON outputs by default.

File formats:
    - Fit data: ``.pt`` (torch.save state with ``rri_chunks``)
    - Fitted binner: ``.json`` (num_classes + edges)
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..configs import PathConfig


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


def _atomic_torch_save(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


@dataclass(slots=True)
class RriOrdinalBinner:
    """RRI → ordinal label mapping (CORAL-compatible)."""

    num_classes: int = 0
    """Number of ordinal classes $K$."""

    edges: Tensor = field(default_factory=lambda: torch.empty((0,), dtype=torch.float32))
    """Quantile edges. Shape ``(K-1,)``."""

    _rri_chunks: list[Tensor] = field(default_factory=list, repr=False)

    @property
    def is_fitted(self) -> bool:
        return int(self.num_classes) >= 2 and int(self.edges.numel()) == int(self.num_classes) - 1

    def transform(self, rri: Tensor) -> Tensor:
        """Convert oracle RRI values to ordinal labels.

        Args:
            rri: Oracle RRI values. Shape ``(...,)``.

        Returns:
            ``Tensor["...", int64]`` labels in ``[0, K-1]``.
        """

        if not self.is_fitted:
            raise RuntimeError("Binner not fitted. Call fit_from_iterable(...) or load a fitted JSON binner.")

        rri_f = rri.reshape(-1).to(dtype=torch.float32)
        labels = torch.bucketize(rri_f, self.edges.to(device=rri_f.device), right=False)
        return labels.to(dtype=torch.int64)

    # --------------------------------------------------------------------- JSON (checkpoint + artifact)
    def to_dict(self) -> dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Binner not fitted; only fitted binners can be serialized.")
        return {"num_classes": int(self.num_classes), "edges": self.edges.detach().cpu().tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RriOrdinalBinner":
        return cls(
            num_classes=int(data["num_classes"]),
            edges=torch.tensor(data["edges"], dtype=torch.float32),
        )

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Save a fitted binner as JSON."""

        if not self.is_fitted:
            raise RuntimeError("Binner not fitted; call fit_from_iterable(...) before save().")

        out_path = PathConfig().resolve_artifact_path(path, expected_suffix=".json")
        out_path = _unique_path(out_path, overwrite=overwrite)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(out_path, json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "RriOrdinalBinner":
        """Load a fitted binner from JSON."""

        in_path = PathConfig().resolve_artifact_path(path, expected_suffix=".json", create_parent=False)
        data = json.loads(in_path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    # --------------------------------------------------------------------- fitting (single entry point)
    @classmethod
    def fit_from_iterable(
        cls,
        iterable: Iterable[Tensor | tuple[Tensor, Any]],
        *,
        num_classes: int = 15,
        target_items: int | None = None,
        max_skips: int = 0,
        fit_data_path: str | Path | None = None,
        resume: bool = False,
        save_every: int = 1,
        on_progress: Callable[[int, int, Tensor | None, Any | None], None] | None = None,
    ) -> "RriOrdinalBinner":
        """Fit a binner from a stream of RRIs, optionally resumable via ``fit_data_path``.

        Notes:
            - All fit data is stored on CPU.
            - Fit data is saved on Ctrl-C / exceptions when ``fit_data_path`` is provided.
            - The iterable may yield either ``rri`` tensors or ``(rri, meta)`` tuples.
        """

        path = None
        if fit_data_path is not None:
            path = PathConfig().resolve_artifact_path(fit_data_path, expected_suffix=".pt")

        binner = cls()
        if path is not None and path.exists():
            if not resume:
                raise FileExistsError(f"Fit data already exists at {path}. Delete it or pass resume=True.")
            state = torch.load(path, map_location="cpu", weights_only=True)
            binner._rri_chunks = [t.reshape(-1).to(dtype=torch.float32) for t in state["rri_chunks"]]  # noqa: SLF001

        successes = len(binner._rri_chunks)  # noqa: SLF001
        skipped = 0

        def _save_fit_data() -> None:
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            state = {"rri_chunks": [t.detach().to(device="cpu", dtype=torch.float32) for t in binner._rri_chunks]}  # noqa: SLF001
            _atomic_torch_save(path, state)

        try:
            for item in iterable:
                if target_items is not None and successes >= int(target_items):
                    break

                if isinstance(item, tuple):
                    rri_raw, meta = item
                else:
                    rri_raw, meta = item, None

                rri_f = rri_raw.reshape(-1).to(dtype=torch.float32)
                rri_f = rri_f[torch.isfinite(rri_f)]
                if rri_f.numel() == 0:
                    skipped += 1
                    if on_progress is not None:
                        on_progress(successes, skipped, None, meta)
                    if int(max_skips) > 0 and skipped >= int(max_skips):
                        raise RuntimeError(
                            f"Unable to fit binner: only {successes}/{target_items or '∞'} items after {skipped} skips."
                        )
                    continue

                binner._rri_chunks.append(rri_f.detach().cpu())  # noqa: SLF001
                successes = len(binner._rri_chunks)  # noqa: SLF001

                if on_progress is not None:
                    on_progress(successes, skipped, rri_f.detach(), meta)

                if path is not None and int(save_every) > 0 and (successes % int(save_every) == 0):
                    _save_fit_data()
        except (KeyboardInterrupt, Exception):
            _save_fit_data()
            raise

        if target_items is not None and successes < int(target_items):
            raise RuntimeError(
                f"Dataset exhausted while fitting binner ({successes}/{int(target_items)} successful items collected)."
            )

        _save_fit_data()
        return binner._finalize(num_classes=int(num_classes))  # noqa: SLF001

    # --------------------------------------------------------------------- internals
    def _finalize(self, *, num_classes: int) -> "RriOrdinalBinner":
        if int(num_classes) < 2:
            raise ValueError("num_classes must be >= 2.")
        if not self._rri_chunks:
            raise ValueError("No fit data available.")

        rri = torch.cat(self._rri_chunks, dim=0).to(dtype=torch.float32)
        qs = torch.linspace(
            1.0 / float(num_classes),
            float(num_classes - 1) / float(num_classes),
            steps=int(num_classes) - 1,
            device=rri.device,
        )
        self.num_classes = int(num_classes)
        self.edges = torch.quantile(rri, qs).to(dtype=torch.float32).detach().cpu()
        return self


__all__ = ["RriOrdinalBinner"]
