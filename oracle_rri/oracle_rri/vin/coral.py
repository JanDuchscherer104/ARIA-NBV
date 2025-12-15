"""CORAL ordinal regression utilities.

This module uses the MIT-licensed reference implementation from
`coral-pytorch <https://raschka-research-group.github.io/coral-pytorch/>`_.

VIN-NBV trains the RRI predictor via ordinal classification and uses a ranking-aware
loss (CORAL) to penalize distant misclassifications more strongly than nearby ones.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

try:  # Prefer the upstream reference implementation.
    from coral_pytorch.layers import CoralLayer as _CoralLayer
    from coral_pytorch.losses import coral_loss as _coral_loss
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise ModuleNotFoundError(
        "Missing optional dependency 'coral-pytorch'. Install it via `uv sync --extra dev` "
        "or add it to your environment."
    ) from exc

Tensor = torch.Tensor


def ordinal_label_to_levels(labels: Tensor, *, num_classes: int) -> Tensor:
    """Convert ordinal labels to CORAL level targets.

    CORAL represents a K-class ordinal label ``y ∈ {0, …, K-1}`` as ``K-1`` binary
    targets:

        levels[k] = 1  if y > k
                  = 0  otherwise

    Args:
        labels: ``Tensor["...", int64]`` with values in ``[0, num_classes-1]``.
        num_classes: Number of ordinal classes ``K``.

    Returns:
        ``Tensor["... K-1", float32]`` of binary level targets.
    """

    if num_classes < 2:
        raise ValueError("num_classes must be >= 2.")
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"labels must be integer dtype, got {labels.dtype}.")

    thresholds = torch.arange(num_classes - 1, device=labels.device, dtype=labels.dtype)
    levels = labels.unsqueeze(-1) > thresholds  # ... x (K-1)
    return levels.to(dtype=torch.float32)


def coral_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    num_classes: int,
    importance_weights: Tensor | None = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """Compute CORAL loss (sum of binary cross-entropies over thresholds).

    Args:
        logits: ``Tensor["... K-1", float32]`` threshold logits.
        labels: ``Tensor["...", int64]`` ordinal labels in ``[0, K-1]``.
        num_classes: Number of ordinal classes ``K``.
        importance_weights: Optional per-threshold weights ``Tensor["K-1"]``.
        reduction: Reduction mode.

    Returns:
        Loss tensor. If reduction="none": ``Tensor["..."]``.
    """

    levels = ordinal_label_to_levels(labels, num_classes=num_classes)

    # `coral_pytorch.losses.coral_loss` expects logits and levels with identical shapes.
    loss = _coral_loss(
        logits.reshape(-1, logits.shape[-1]),
        levels.reshape(-1, levels.shape[-1]),
        importance_weights=importance_weights,
        reduction=None if reduction == "none" else reduction,
    )
    if reduction == "none":
        return loss.reshape(labels.shape)
    return loss


def coral_logits_to_prob(logits: Tensor) -> Tensor:
    """Convert CORAL logits to a proper class distribution.

    Args:
        logits: ``Tensor["... K-1"]``

    Returns:
        ``Tensor["... K"]`` probabilities that sum to 1 along the last dim.
    """

    if logits.shape[-1] < 1:
        raise ValueError("Expected logits with last dim K-1 >= 1.")

    p_gt = torch.sigmoid(logits)  # P(y > k), shape (..., K-1)
    k_minus_1 = p_gt.shape[-1]
    num_classes = k_minus_1 + 1

    probs: list[Tensor] = []
    probs.append(1.0 - p_gt[..., 0])
    for k in range(1, k_minus_1):
        probs.append(p_gt[..., k - 1] - p_gt[..., k])
    probs.append(p_gt[..., -1])

    prob = torch.stack(probs, dim=-1)
    prob = torch.clamp(prob, min=0.0, max=1.0)
    prob = prob / (prob.sum(dim=-1, keepdim=True) + 1e-8)

    assert prob.shape[-1] == num_classes
    return prob


def coral_expected_from_logits(logits: Tensor) -> tuple[Tensor, Tensor]:
    """Compute expected ordinal value from CORAL logits.

    Args:
        logits: ``Tensor["... K-1"]``.

    Returns:
        Tuple of ``(expected, expected_normalized)`` where:
          - expected is in ``[0, K-1]``
          - expected_normalized is in ``[0, 1]``
    """

    # In CORAL, the logits parameterize the probabilities P(y > k).
    # The expected rank label is E[y] = sum_{k=0}^{K-2} P(y > k).
    prob_gt = torch.sigmoid(logits)
    expected = prob_gt.sum(dim=-1)
    expected_normalized = expected / max(1.0, float(prob_gt.shape[-1]))
    return expected, expected_normalized


class CoralLayer(nn.Module):
    """CORAL output layer with shared weights and per-threshold biases.

    This implements logits:
        logit_k = w^T x + b_k,  k = 0..K-2
    """

    def __init__(self, in_dim: int, num_classes: int, *, preinit_bias: bool = True) -> None:
        super().__init__()
        self.layer = _CoralLayer(size_in=int(in_dim), num_classes=int(num_classes), preinit_bias=bool(preinit_bias))

    def forward(self, x: Tensor) -> Tensor:
        """Compute threshold logits.

        Args:
            x: ``Tensor["... in_dim"]``.

        Returns:
            ``Tensor["... K-1"]``.
        """

        return self.layer(x)
