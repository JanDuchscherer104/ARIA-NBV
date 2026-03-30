"""Offline-cache payload serialization helpers.

NBV_LEGACY_OFFLINE_CACHE_REMOVE_AFTER_FULL_MIGRATION:
This module exists for the legacy oracle/VIN cache payload format and its
tests. Remove it after the immutable-store cutover.

The active serialization logic lives on the shared data models themselves via
``to_serializable`` / ``from_serializable`` methods backed by
``aria_nbv.utils.typed_payloads``. This module exposes the canonical helper
functions used by the legacy oracle-cache and VIN-cache tests.
"""

from __future__ import annotations

from typing import Any

import torch

from ..pose_generation.types import CandidateSamplingResult
from ..rendering.candidate_depth_renderer import CandidateDepths
from ..rendering.candidate_pointclouds import CandidatePointClouds
from ..rri_metrics.types import RriResult
from ..vin.types import EvlBackboneOutput


def encode_candidates(candidates: CandidateSamplingResult) -> dict[str, Any]:
    """Serialize candidate sampling results for legacy cache payloads."""

    return candidates.to_serializable()


def decode_candidates(payload: dict[str, Any]) -> CandidateSamplingResult:
    """Deserialize candidate sampling results from a legacy cache payload."""

    return CandidateSamplingResult.from_serializable(payload, device=None)


def encode_depths(depths: CandidateDepths) -> dict[str, Any]:
    """Serialize candidate depths for legacy cache payloads."""

    return depths.to_serializable()


def decode_depths(payload: dict[str, Any], *, device: torch.device) -> CandidateDepths:
    """Deserialize candidate depths from a legacy cache payload."""

    return CandidateDepths.from_serializable(payload, device=device)


def encode_candidate_pcs(pcs: CandidatePointClouds) -> dict[str, Any]:
    """Serialize candidate point clouds for legacy cache payloads."""

    return pcs.to_serializable()


def decode_candidate_pcs(payload: dict[str, Any], *, device: torch.device) -> CandidatePointClouds:
    """Deserialize candidate point clouds from a legacy cache payload."""

    return CandidatePointClouds.from_serializable(payload, device=device)


def encode_rri(rri: RriResult) -> dict[str, Any]:
    """Serialize RRI results for legacy cache payloads."""

    return rri.to_serializable()


def decode_rri(payload: dict[str, Any], *, device: torch.device) -> RriResult:
    """Deserialize RRI results from a legacy cache payload."""

    return RriResult.from_serializable(payload, device=device)


def encode_backbone(backbone: EvlBackboneOutput) -> dict[str, Any]:
    """Serialize EVL backbone outputs for legacy cache payloads."""

    return backbone.to_serializable()


def decode_backbone(
    payload: dict[str, Any],
    *,
    device: torch.device,
    include_fields: set[str] | None = None,
) -> EvlBackboneOutput:
    """Deserialize EVL backbone outputs from a legacy cache payload."""

    return EvlBackboneOutput.from_serializable(
        payload,
        device=device,
        include_fields=include_fields,
    )


__all__ = [
    "decode_backbone",
    "decode_candidate_pcs",
    "decode_candidates",
    "decode_depths",
    "decode_rri",
    "encode_backbone",
    "encode_candidate_pcs",
    "encode_candidates",
    "encode_depths",
    "encode_rri",
]
