"""Metric names and torchmetrics bundles for VIN training.

This module centralizes logging keys (``Metric``) and stateful torchmetrics
objects used in Lightning. We prefer a single ``Metric`` container that owns
all sub-metrics because VIN needs *different* inputs per metric
(``pred_scores`` vs. ``pred_class`` vs. ``labels``), which is awkward to
represent with a plain ``MetricCollection``. The custom wrapper still follows
torchmetrics best practices: ``add_state`` for distributed reduction, explicit
``update``/``compute``/``reset`` methods, and ``full_state_update=False`` to
avoid unnecessary synchronization overhead.
"""

from __future__ import annotations

from enum import StrEnum

import torch
from pydantic import Field
from torch import Tensor
from torchmetrics import Metric as MetricBase
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.regression import SpearmanCorrCoef

from ..utils import BaseConfig, Stage


class Metric(StrEnum):
    """Metric suffixes composed with Stage as ``{stage}/{metric}``."""

    LOSS = "loss"
    RRI_MEAN = "rri_mean"
    PRED_RRI_MEAN = "pred_rri_mean"
    VALID_FRAC_MEAN = "valid_frac_mean"
    CANDIDATE_VALID_FRACTION = "candidate_valid_fraction"
    TOP3_ACCURACY = "top3_accuracy"

    SPEARMAN = "spearman"
    SPEARMAN_STEP = "spearman_step"
    CONFUSION_MATRIX = "confusion_matrix"
    CONFUSION_MATRIX_STEP = "confusion_matrix_step"
    LABEL_HISTOGRAM = "label_histogram"
    LABEL_HISTOGRAM_STEP = "label_histogram_step"

    def __str__(self) -> str:
        return self.value


class LabelHistogram(MetricBase):
    """Accumulate label counts for ordinal classes."""

    full_state_update = False

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.add_state(
            "counts",
            default=torch.zeros(self.num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, target: Tensor) -> None:
        if target.numel() == 0:
            return
        labels = target.to(dtype=torch.int64).reshape(-1)
        counts = torch.bincount(labels, minlength=self.num_classes)
        self.counts = self.counts + counts.to(device=self.counts.device)

    def compute(self) -> Tensor:
        return self.counts


class VinMetrics(MetricBase):
    """Container for VIN metrics computed from candidate rankings."""

    full_state_update = False

    def __init__(self, *, num_classes: int) -> None:
        super().__init__()
        self.spearman = SpearmanCorrCoef()
        self.confusion = MulticlassConfusionMatrix(num_classes=int(num_classes))
        self.label_hist = LabelHistogram(num_classes=int(num_classes))
        self.add_state("has_updates", default=torch.zeros((), dtype=torch.bool), dist_reduce_fx="max")

    def update(
        self,
        *,
        pred_scores: Tensor,
        rri: Tensor,
        pred_class: Tensor,
        labels: Tensor,
    ) -> None:
        if pred_scores.numel() == 0:
            return
        self.spearman.update(pred_scores, rri)
        self.confusion.update(pred_class, labels)
        self.label_hist.update(labels)
        self.has_updates.fill_(True)

    def compute(self) -> dict[str, Tensor]:
        if not bool(self.has_updates.item()):
            return {}
        return {
            "spearman": self.spearman.compute(),
            "confusion": self.confusion.compute(),
            "label_hist": self.label_hist.compute(),
        }

    def reset(self) -> None:  # type: ignore[override]
        self.spearman.reset()
        self.confusion.reset()
        self.label_hist.reset()
        self.has_updates.fill_(False)


class VinMetricsConfig(BaseConfig[VinMetrics]):
    """Configuration for VIN torchmetrics bundles."""

    target: type[VinMetrics] = Field(default_factory=lambda: VinMetrics, exclude=True)
    """Factory target for :meth:`~oracle_rri.utils.base_config.BaseConfig.setup_target`."""

    num_classes: int
    """Number of ordinal classes used for confusion/histogram metrics."""

    def setup_target(self) -> VinMetrics:  # type: ignore[override]
        return self.target(num_classes=int(self.num_classes))


def metric_key(stage: Stage, metric: Metric) -> str:
    """Compose a logging key using the stage prefix."""
    return f"{stage.value}/{metric.value}"


def topk_accuracy_from_probs(probs: Tensor, labels: Tensor, *, top_k: int) -> Tensor:
    """Compute top-k accuracy from class probabilities.

    Args:
        probs: ``Tensor["N K"]`` class probabilities.
        labels: ``Tensor["N"]`` integer class labels.
        top_k: Number of highest-probability classes to consider.

    Returns:
        ``Tensor[""]`` scalar accuracy in ``[0, 1]``.
    """
    if probs.numel() == 0 or labels.numel() == 0:
        return torch.tensor(float("nan"), device=probs.device)
    if probs.ndim != 2:
        raise ValueError(f"Expected probs with shape (N, K), got {tuple(probs.shape)}.")
    labels = labels.reshape(-1)
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(
            "Expected probs and labels to have matching first dimension, "
            f"got {probs.shape[0]} and {labels.shape[0]}.",
        )
    k = min(int(top_k), probs.shape[-1])
    if k < 1:
        raise ValueError("top_k must be >= 1.")
    topk = probs.topk(k=k, dim=-1).indices
    correct = (topk == labels.unsqueeze(-1)).any(dim=-1)
    return correct.to(dtype=torch.float32).mean()


__all__ = [
    "LabelHistogram",
    "Metric",
    "VinMetrics",
    "VinMetricsConfig",
    "metric_key",
    "topk_accuracy_from_probs",
]
