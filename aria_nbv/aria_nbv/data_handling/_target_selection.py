"""Actor-visible target selection for target-conditioned ARIA-NBV.

This module implements the V1 `OBS-SEL / PRED-Q / GT-EVAL` target contract.
V1 ranks observed or predicted OBB records by actor-visible evidence only: OBB
confidence, projected area, semidense support, EVL support, and support deficit.
GT OBBs are allowed only in explicit V0 sanity/upper-bound mode or after
selection for label/evaluation matching.

The selector stores both actor-facing descriptors and oracle-side audit fields.
Target matching is accepted only when the best GT match passes configured IoU
or score thresholds and the top-1/top-2 gap is unambiguous. Unmatched,
ambiguous, empty-crop, or no-source cases are target-invalid states, not low
target-RRI examples.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import torch
from efm3d.aria.aria_constants import ARIA_SNIPPET_T_WORLD_SNIPPET
from efm3d.aria.obb import ObbTW, obb_iou3d
from efm3d.aria.pose import PoseTW
from pydantic import field_validator
from torch import Tensor

from ..utils import BaseConfig
from ..vin.types import EvlBackboneOutput
from .efm_views import EfmSnippetView, VinSnippetView


class TargetSelectionPolicy(StrEnum):
    """Supported top-K target selection policies."""

    GREEDY_TOP_K = "greedy_top_k"
    TEMPERATURE_SOFTMAX_TOP_K = "temperature_softmax_top_k"


class TargetSourceMode(StrEnum):
    """Selector source protocol."""

    V1_ACTOR_VISIBLE = "v1_actor_visible"
    V0_GT_SANITY = "v0_gt_sanity"


TARGET_INVALID_REASON_CODES: dict[str, int] = {
    "VALID": 0,
    "NO_ACTOR_VISIBLE_SOURCE": 1,
    "GT_SOURCE_DISALLOWED": 2,
    "PADDED_OBB": 3,
    "OBB_NONFINITE": 4,
    "OBB_EXTENT_INVALID": 5,
    "CONFIDENCE_TOO_LOW": 6,
    "NO_PROJECTED_VISIBILITY": 7,
    "PROJECTED_AREA_TOO_SMALL": 8,
    "TARGET_SUPPORT_TOO_LOW": 9,
    "TARGET_GT_UNMATCHED": 10,
    "TARGET_GT_AMBIGUOUS": 11,
}
"""Version-1 target invalidity reason bit positions."""

TARGET_INVALID_REASON_VERSION = "target-selection-invalidity-v1"
"""Version label for `TARGET_INVALID_REASON_CODES`."""


@dataclass(frozen=True, slots=True)
class TargetCandidateRow:
    """One actor-visible target candidate and its oracle audit fields.

    Geometry is stored in ARIA world coordinates. ``relative_pose_reference``
    is ``T_reference_rig_target = T_world_reference_rig^{-1} @
    T_world_target`` flattened to 12 values. GT match fields are labels and
    diagnostics only; they do not participate in V1 ranking.
    """

    scene_id: str | None
    snippet_id: str | None
    source: str
    source_index: int
    target_row_id: int
    target_id: str
    sem_id: int
    inst_id: int
    class_name: str
    confidence: float
    center_world: tuple[float, float, float]
    extents: tuple[float, float, float]
    pose_world_object: tuple[float, ...]
    relative_pose_reference_object: tuple[float, ...]
    projected_area_pixels: float
    projected_area_fraction: float
    semidense_support_count: int
    evl_support_count: int
    visibility_score: float
    support_score: float
    deficit_score: float
    score: float
    eligible: bool
    invalid_reason_bitset: int
    primary_invalid_reason: int
    selected_rank: int | None = None
    selection_probability: float | None = None
    selection_log_probability: float | None = None
    selection_entropy: float | None = None
    gt_label_valid: bool = False
    gt_target_row_id: int | None = None
    gt_target_id: str | None = None
    gt_match_iou: float | None = None
    gt_match_score: float | None = None
    gt_match_status: str = "not_requested"


@dataclass(frozen=True, slots=True)
class TargetSelectionResult:
    """Ranked target table and selected top-K rows for one snippet."""

    rows: tuple[TargetCandidateRow, ...]
    ranked_rows: tuple[TargetCandidateRow, ...]
    selected_rows: tuple[TargetCandidateRow, ...]
    k: int
    policy: str
    source_mode: str
    source: str | None
    seed: int | None
    temperature: float | None
    warnings: tuple[str, ...] = ()
    reason_code_version: str = TARGET_INVALID_REASON_VERSION


@dataclass(slots=True)
class _TargetSource:
    source: str
    obbs: ObbTW
    sem_id_to_name: list[str] | None = None


class TargetSelectorConfig(BaseConfig):
    """Configuration for `ActorVisibleTargetSelector`."""

    @property
    def target(self) -> type["ActorVisibleTargetSelector"]:
        """Factory target for `BaseConfig.setup_target`."""

        return ActorVisibleTargetSelector

    k: int = 3
    """Number of selected targets to materialize."""

    policy: TargetSelectionPolicy = TargetSelectionPolicy.GREEDY_TOP_K
    """Top-K policy applied after hard target eligibility masking."""

    source_mode: TargetSourceMode = TargetSourceMode.V1_ACTOR_VISIBLE
    """Whether to use V1 actor-visible sources or the V0 GT sanity source."""

    seed: int | None = 0
    """Seed for stochastic target selection policies."""

    temperature: float = 1.0
    """Softmax temperature for ``temperature_softmax_top_k``."""

    min_confidence: float = 0.2
    """Minimum observed/predicted OBB confidence for V1 eligibility."""

    min_projected_area_pixels: float = 16.0
    """Minimum max projected 2D area over RGB/SLAM OBB boxes."""

    projected_area_normalizer_pixels: float = 240.0 * 240.0
    """Image-area normalizer used for projected-area fractions."""

    projected_area_full_score_fraction: float = 0.05
    """Projected-area fraction that saturates the visibility score."""

    min_support_points: int = 1
    """Minimum semidense plus EVL points inside the target OBB."""

    support_saturation_points: int = 128
    """Support count at which the deficit score reaches zero."""

    obb_support_scale: float = 1.0
    """OBB scale used when counting semidense/EVL points inside a target."""

    max_support_points: int = 20000
    """Maximum support points inspected per snippet, using deterministic prefix truncation."""

    match_gt: bool = True
    """Match selected V1 targets to GT OBBs after ranking when GT is available."""

    min_gt_iou: float = 0.1
    """Minimum sampled 3D OBB IoU for a GT target match."""

    gt_iou_samples: int = 8
    """Samples per dimension for EFM's sampled OBB IoU fallback."""

    gt_ambiguity_margin: float = 0.02
    """IoU margin below which competing GT matches are considered ambiguous."""

    @field_validator("k", "min_support_points", "support_saturation_points", "max_support_points", "gt_iou_samples")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        value = int(value)
        if value < 1:
            raise ValueError("Target selector integer thresholds must be >= 1.")
        return value

    @field_validator(
        "temperature",
        "min_projected_area_pixels",
        "projected_area_normalizer_pixels",
        "projected_area_full_score_fraction",
        "obb_support_scale",
    )
    @classmethod
    def _positive_float(cls, value: float) -> float:
        value = float(value)
        if value <= 0.0:
            raise ValueError("Target selector float thresholds must be > 0.")
        return value

    @field_validator("min_confidence", "min_gt_iou", "gt_ambiguity_margin")
    @classmethod
    def _non_negative_float(cls, value: float) -> float:
        value = float(value)
        if value < 0.0:
            raise ValueError("Target selector thresholds must be >= 0.")
        return value


class ActorVisibleTargetSelector:
    """Select top-K target OBBs from actor-visible snippet evidence."""

    def __init__(self, config: TargetSelectorConfig) -> None:
        """Initialize the selector.

        Args:
            config: Selection thresholds, policy, and source mode.
        """

        self.config = config

    def select(self, sample: Any) -> TargetSelectionResult:
        """Select top-K targets from a VIN offline sample or oracle batch.

        Args:
            sample: Object carrying ``detected_obbs`` or ``backbone_out`` for
                V1, plus optional ``gt_obbs`` for post-selection matching.

        Returns:
            Ranked target table and selected top-K rows.
        """

        warnings: list[str] = []
        source = self._resolve_source(sample, warnings=warnings)
        if source is None:
            return TargetSelectionResult(
                rows=(),
                ranked_rows=(),
                selected_rows=(),
                k=self.config.k,
                policy=self.config.policy.value,
                source_mode=self.config.source_mode.value,
                source=None,
                seed=self.config.seed,
                temperature=self._policy_temperature(),
                warnings=tuple(warnings),
            )

        world_obbs = _world_obbs_for_sample(source.obbs, sample)
        rows = self._build_rows(sample, source=source, world_obbs=world_obbs)
        ranked = tuple(sorted((row for row in rows if row.eligible), key=_ranking_key))
        selected = self._select_rows(ranked)
        selected = self._match_selected_to_gt(selected, sample=sample, pred_obbs_world=world_obbs)
        selected_by_id = {row.target_id: row for row in selected}
        rows = tuple(selected_by_id.get(row.target_id, row) for row in rows)
        ranked = tuple(selected_by_id.get(row.target_id, row) for row in ranked)

        return TargetSelectionResult(
            rows=rows,
            ranked_rows=ranked,
            selected_rows=selected,
            k=self.config.k,
            policy=self.config.policy.value,
            source_mode=self.config.source_mode.value,
            source=source.source,
            seed=self.config.seed,
            temperature=self._policy_temperature(),
            warnings=tuple(warnings),
        )

    def _resolve_source(self, sample: Any, *, warnings: list[str]) -> _TargetSource | None:
        if self.config.source_mode == TargetSourceMode.V0_GT_SANITY:
            gt = _compact_obb_block(getattr(sample, "gt_obbs", None))
            if gt is None:
                warnings.append("V0 GT target selection requested, but sample has no GT OBB block.")
                return None
            return _TargetSource(source="gt_obbs_v0_sanity", obbs=gt[0], sem_id_to_name=gt[1])

        detected = _compact_obb_block(getattr(sample, "detected_obbs", None))
        if detected is not None:
            return _TargetSource(source="detected_obbs", obbs=detected[0], sem_id_to_name=detected[1])

        backbone = getattr(sample, "backbone_out", None)
        if isinstance(backbone, EvlBackboneOutput):
            obb = backbone.obb_pred_viz if backbone.obb_pred_viz is not None else backbone.obb_pred
            if obb is not None:
                source_name = "backbone.obb_pred_viz" if backbone.obb_pred_viz is not None else "backbone.obb_pred"
                return _TargetSource(source=source_name, obbs=obb, sem_id_to_name=backbone.obb_pred_sem_id_to_name)

        if getattr(sample, "gt_obbs", None) is not None:
            warnings.append("V1 target selection refused GT OBBs because they are oracle-only labels/evaluation data.")
        warnings.append("No actor-visible detected/predicted OBB source was available for target selection.")
        return None

    def _build_rows(self, sample: Any, *, source: _TargetSource, world_obbs: ObbTW) -> tuple[TargetCandidateRow, ...]:
        valid_data, source_indices = _valid_obb_data_with_source_indices(world_obbs)
        if valid_data.numel() == 0:
            return ()
        obbs = ObbTW(valid_data)
        semidense_points = _semidense_points(sample, max_points=self.config.max_support_points)
        evl_points, evl_counts = _evl_support_points(sample, max_points=self.config.max_support_points)
        reference_pose = _reference_pose_world_rig(sample)
        scene_id = _string_or_none(getattr(sample, "scene_id", None))
        snippet_id = _string_or_none(getattr(sample, "snippet_id", None))

        rows: list[TargetCandidateRow] = []
        for row_index in range(int(obbs.shape[0])):
            obb = ObbTW(obbs._data[row_index])
            sem_id = int(obb.sem_id.reshape(-1)[0].item())
            inst_id = int(obb.inst_id.reshape(-1)[0].item())
            confidence = float(obb.prob.reshape(-1)[0].item())
            extents_t = obb.bb3_diagonal.detach().cpu().reshape(-1).to(dtype=torch.float32)
            center_t = obb.bb3_center_world.detach().cpu().reshape(-1).to(dtype=torch.float32)
            pose_world = obb.T_world_object.tensor().detach().cpu().reshape(-1).to(dtype=torch.float32)
            relative_pose = (reference_pose.inverse() @ obb.T_world_object).tensor().detach().cpu().reshape(-1)
            projected_area = _max_projected_area(obb)
            projected_fraction = projected_area / float(self.config.projected_area_normalizer_pixels)
            visibility_score = _clamp01(projected_fraction / self.config.projected_area_full_score_fraction)
            semidense_count = _points_inside_count(obb, semidense_points, scale=self.config.obb_support_scale)
            evl_count = _points_inside_count(
                obb,
                evl_points,
                scale=self.config.obb_support_scale,
                positive_counts=evl_counts,
            )
            total_support = semidense_count + evl_count
            support_score = _clamp01(total_support / float(self.config.min_support_points))
            deficit_score = 1.0 - _clamp01(total_support / float(self.config.support_saturation_points))
            reason_bitset = _target_reason_bitset(
                obb=obb,
                confidence=confidence,
                projected_area=projected_area,
                total_support=total_support,
                config=self.config,
            )
            eligible = reason_bitset == (1 << TARGET_INVALID_REASON_CODES["VALID"])
            score = confidence * visibility_score * support_score * deficit_score if eligible else float("nan")
            source_index = int(source_indices[row_index])
            target_id = _target_id(
                scene_id=scene_id,
                snippet_id=snippet_id,
                source=source.source,
                sem_id=sem_id,
                inst_id=inst_id,
                source_index=source_index,
            )
            rows.append(
                TargetCandidateRow(
                    scene_id=scene_id,
                    snippet_id=snippet_id,
                    source=source.source,
                    source_index=source_index,
                    target_row_id=source_index,
                    target_id=target_id,
                    sem_id=sem_id,
                    inst_id=inst_id,
                    class_name=_class_name(sem_id, source.sem_id_to_name),
                    confidence=confidence,
                    center_world=_float_tuple(center_t, length=3),  # type: ignore[assignment]
                    extents=_float_tuple(extents_t, length=3),  # type: ignore[assignment]
                    pose_world_object=_float_tuple(pose_world),
                    relative_pose_reference_object=_float_tuple(relative_pose),
                    projected_area_pixels=float(projected_area),
                    projected_area_fraction=float(projected_fraction),
                    semidense_support_count=int(semidense_count),
                    evl_support_count=int(evl_count),
                    visibility_score=float(visibility_score),
                    support_score=float(support_score),
                    deficit_score=float(deficit_score),
                    score=float(score),
                    eligible=bool(eligible),
                    invalid_reason_bitset=int(reason_bitset),
                    primary_invalid_reason=_primary_reason(reason_bitset),
                )
            )
        return tuple(rows)

    def _select_rows(self, ranked_rows: tuple[TargetCandidateRow, ...]) -> tuple[TargetCandidateRow, ...]:
        if not ranked_rows:
            return ()
        if self.config.policy == TargetSelectionPolicy.GREEDY_TOP_K:
            return tuple(
                replace(row, selected_rank=rank, selection_probability=1.0, selection_log_probability=0.0)
                for rank, row in enumerate(ranked_rows[: self.config.k])
            )
        return self._sample_rows_without_replacement(ranked_rows)

    def _sample_rows_without_replacement(
        self, ranked_rows: tuple[TargetCandidateRow, ...]
    ) -> tuple[TargetCandidateRow, ...]:
        scores = torch.tensor([row.score for row in ranked_rows], dtype=torch.float32)
        remaining = torch.ones(scores.shape[0], dtype=torch.bool)
        selected: list[TargetCandidateRow] = []
        generator = torch.Generator(device="cpu")
        if self.config.seed is not None:
            generator.manual_seed(int(self.config.seed))

        for rank in range(min(self.config.k, len(ranked_rows))):
            logits = scores / float(self.config.temperature)
            masked_logits = torch.where(remaining, logits, torch.full_like(logits, float("-inf")))
            probabilities = torch.softmax(masked_logits, dim=0)
            entropy = -torch.sum(probabilities[remaining] * torch.log(probabilities[remaining].clamp_min(1e-12)))
            sampled = int(torch.multinomial(probabilities, 1, replacement=False, generator=generator).item())
            selected.append(
                replace(
                    ranked_rows[sampled],
                    selected_rank=rank,
                    selection_probability=float(probabilities[sampled].item()),
                    selection_log_probability=float(torch.log(probabilities[sampled].clamp_min(1e-12)).item()),
                    selection_entropy=float(entropy.item()),
                )
            )
            remaining[sampled] = False
            if not bool(remaining.any().item()):
                break
        return tuple(selected)

    def _match_selected_to_gt(
        self,
        selected_rows: tuple[TargetCandidateRow, ...],
        *,
        sample: Any,
        pred_obbs_world: ObbTW,
    ) -> tuple[TargetCandidateRow, ...]:
        if not selected_rows:
            return ()
        if self.config.source_mode == TargetSourceMode.V0_GT_SANITY:
            return tuple(
                replace(
                    row,
                    gt_label_valid=True,
                    gt_target_row_id=row.target_row_id,
                    gt_target_id=row.target_id,
                    gt_match_iou=1.0,
                    gt_match_score=1.0,
                    gt_match_status="v0_gt_input",
                )
                for row in selected_rows
            )
        if not self.config.match_gt:
            return selected_rows
        gt_block = _compact_obb_block(getattr(sample, "gt_obbs", None))
        if gt_block is None:
            return tuple(replace(row, gt_match_status="missing_gt") for row in selected_rows)

        gt_world = _world_obbs_for_sample(gt_block[0], sample)
        gt_data, gt_source_indices = _valid_obb_data_with_source_indices(gt_world)
        pred_data, pred_source_indices = _valid_obb_data_with_source_indices(pred_obbs_world)
        if gt_data.numel() == 0:
            return tuple(replace(row, gt_match_status="missing_gt") for row in selected_rows)

        gt_obbs = ObbTW(gt_data)
        pred_obbs = ObbTW(pred_data)
        pred_index_by_source = {int(source_index): index for index, source_index in enumerate(pred_source_indices)}
        matched: list[TargetCandidateRow] = []
        for row in selected_rows:
            pred_index = pred_index_by_source.get(row.source_index)
            if pred_index is None:
                matched.append(replace(row, gt_match_status="unmatched_gt"))
                continue
            pred = ObbTW(pred_obbs._data[pred_index].unsqueeze(0))
            candidate_matches: list[tuple[int, float]] = []
            for gt_index in range(int(gt_obbs.shape[0])):
                gt = ObbTW(gt_obbs._data[gt_index].unsqueeze(0))
                if int(gt.sem_id.reshape(-1)[0].item()) != row.sem_id:
                    continue
                iou = _safe_obb_iou(pred, gt, samples=self.config.gt_iou_samples)
                if iou >= self.config.min_gt_iou:
                    candidate_matches.append((gt_index, iou))
            candidate_matches.sort(key=lambda item: item[1], reverse=True)
            if not candidate_matches:
                matched.append(
                    replace(
                        row,
                        gt_label_valid=False,
                        gt_match_status="unmatched_gt",
                        invalid_reason_bitset=_target_failure_bitset(
                            row.invalid_reason_bitset,
                            TARGET_INVALID_REASON_CODES["TARGET_GT_UNMATCHED"],
                        ),
                        primary_invalid_reason=TARGET_INVALID_REASON_CODES["TARGET_GT_UNMATCHED"],
                    )
                )
                continue
            if len(candidate_matches) > 1 and (
                candidate_matches[0][1] - candidate_matches[1][1] <= self.config.gt_ambiguity_margin
            ):
                matched.append(
                    replace(
                        row,
                        gt_label_valid=False,
                        gt_match_iou=float(candidate_matches[0][1]),
                        gt_match_score=float(candidate_matches[0][1]),
                        gt_match_status="ambiguous_gt",
                        invalid_reason_bitset=_target_failure_bitset(
                            row.invalid_reason_bitset,
                            TARGET_INVALID_REASON_CODES["TARGET_GT_AMBIGUOUS"],
                        ),
                        primary_invalid_reason=TARGET_INVALID_REASON_CODES["TARGET_GT_AMBIGUOUS"],
                    )
                )
                continue
            gt_index, best_iou = candidate_matches[0]
            gt_source_index = int(gt_source_indices[gt_index])
            matched.append(
                replace(
                    row,
                    gt_label_valid=True,
                    gt_target_row_id=gt_source_index,
                    gt_target_id=_target_id(
                        scene_id=row.scene_id,
                        snippet_id=row.snippet_id,
                        source="gt_obbs",
                        sem_id=row.sem_id,
                        inst_id=int(gt_obbs.inst_id.reshape(-1)[gt_index].item()),
                        source_index=gt_source_index,
                    ),
                    gt_match_iou=float(best_iou),
                    gt_match_score=float(best_iou),
                    gt_match_status="matched",
                )
            )

        return _mark_duplicate_gt_matches(tuple(matched))

    def _policy_temperature(self) -> float | None:
        return (
            float(self.config.temperature)
            if self.config.policy == TargetSelectionPolicy.TEMPERATURE_SOFTMAX_TOP_K
            else None
        )


def _compact_obb_block(value: Any) -> tuple[ObbTW, list[str] | None] | None:
    if value is None:
        return None
    obbs = getattr(value, "obbs", value)
    if obbs is None:
        return None
    sem_id_to_name = getattr(value, "sem_id_to_name", None)
    if isinstance(obbs, ObbTW):
        return obbs, sem_id_to_name
    return ObbTW(torch.as_tensor(obbs, dtype=torch.float32)), sem_id_to_name


def target_gt_aabb_world(
    row: TargetCandidateRow,
    sample: Any,
    *,
    margin_m: float = 0.0,
) -> Tensor:
    """Resolve the matched GT target OBB to a world-frame AABB.

    Args:
        row: Actor-visible target row after GT matching.
        sample: VIN offline sample carrying ``gt_obbs`` and snippet transform.
        margin_m: Optional symmetric world-space margin in metres.

    Returns:
        ``Tensor[6]`` in ``[xmin, xmax, ymin, ymax, zmin, zmax]`` order.

    Raises:
        ValueError: If the row is not label-valid or the matched GT row cannot
            be resolved.
    """

    obb = target_gt_obb_world(row, sample)
    corners = obb.bb3corners_world.detach().cpu().reshape(-1, 3).to(dtype=torch.float32)
    lower = corners.min(dim=0).values - float(margin_m)
    upper = corners.max(dim=0).values + float(margin_m)
    return torch.stack([lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]]).to(dtype=torch.float32)


def target_gt_obb_world(row: TargetCandidateRow, sample: Any) -> ObbTW:
    """Resolve the matched GT target OBB in world coordinates.

    Args:
        row: Actor-visible target row after GT matching.
        sample: VIN offline sample carrying ``gt_obbs`` and snippet transform.

    Returns:
        A single-row `ObbTW` in world coordinates.

    Raises:
        ValueError: If the row is not label-valid or the matched GT row cannot
            be resolved.
    """

    if not row.gt_label_valid or row.gt_target_row_id is None:
        raise ValueError("Target row is not GT-label valid; refusing to build target RRI crop.")
    gt_block = _compact_obb_block(getattr(sample, "gt_obbs", None))
    if gt_block is None:
        raise ValueError("Target RRI crop requires sample.gt_obbs.")
    gt_world = _world_obbs_for_sample(gt_block[0], sample)
    gt_data, gt_source_indices = _valid_obb_data_with_source_indices(gt_world)
    try:
        gt_index = gt_source_indices.index(int(row.gt_target_row_id))
    except ValueError as exc:
        raise ValueError(f"Matched GT target row {row.gt_target_row_id} is not present in sample.gt_obbs.") from exc
    return ObbTW(gt_data[gt_index].unsqueeze(0))


def _world_obbs_for_sample(obbs: ObbTW, sample: Any) -> ObbTW:
    selected = _latest_valid_obb_slice(obbs)
    transform = _snippet_t_world_snippet(sample)
    if transform is None:
        return selected
    return selected.transform(transform)


def _latest_valid_obb_slice(obbs: ObbTW) -> ObbTW:
    data = obbs.tensor().detach().cpu().to(dtype=torch.float32)
    if data.ndim == 1:
        data = data.unsqueeze(0)
    if data.ndim == 2:
        return ObbTW(data)
    rows = data.reshape(-1, data.shape[-2], data.shape[-1])
    for index in range(rows.shape[0] - 1, -1, -1):
        candidate = ObbTW(rows[index])
        if bool((~candidate.get_padding_mask()).any().item()):
            return candidate
    return ObbTW(rows[-1])


def _valid_obb_data_with_source_indices(obbs: ObbTW) -> tuple[Tensor, list[int]]:
    data = obbs.tensor().detach().cpu().to(dtype=torch.float32)
    if data.ndim == 1:
        data = data.unsqueeze(0)
    flat = data.reshape(-1, data.shape[-1])
    flat_obbs = ObbTW(flat)
    valid = (~flat_obbs.get_padding_mask()).reshape(-1)
    source_indices = torch.nonzero(valid, as_tuple=False).reshape(-1).tolist()
    return flat[valid], [int(index) for index in source_indices]


def _snippet_t_world_snippet(sample: Any) -> PoseTW | None:
    snippet = getattr(sample, "efm_snippet_view", None)
    if isinstance(snippet, EfmSnippetView):
        value = snippet.efm.get(ARIA_SNIPPET_T_WORLD_SNIPPET)
        if isinstance(value, PoseTW):
            return PoseTW(value.tensor().reshape(-1, 12)[:1])
        if torch.is_tensor(value):
            return PoseTW(value.reshape(-1, 12)[:1])
    vin_snippet = getattr(sample, "vin_snippet", None)
    if isinstance(vin_snippet, VinSnippetView):
        poses = vin_snippet.t_world_rig.tensor().reshape(-1, 12)
        if poses.shape[0] > 0:
            return PoseTW(poses[:1])
    efm_or_vin = getattr(sample, "efm_snippet_view", None)
    if isinstance(efm_or_vin, VinSnippetView):
        poses = efm_or_vin.t_world_rig.tensor().reshape(-1, 12)
        if poses.shape[0] > 0:
            return PoseTW(poses[:1])
    return None


def _reference_pose_world_rig(sample: Any) -> PoseTW:
    reference = getattr(sample, "reference_pose_world_rig", None)
    if isinstance(reference, PoseTW):
        return PoseTW(reference.tensor().reshape(-1, 12)[:1])
    oracle = getattr(sample, "oracle", None)
    reference = getattr(oracle, "reference_pose_world_rig", None)
    if isinstance(reference, PoseTW):
        return PoseTW(reference.tensor().reshape(-1, 12)[:1])
    transform = _snippet_t_world_snippet(sample)
    if transform is not None:
        return transform
    return PoseTW()


def _semidense_points(sample: Any, *, max_points: int) -> Tensor:
    vin_snippet = getattr(sample, "vin_snippet", None)
    if isinstance(vin_snippet, VinSnippetView):
        return _valid_prefix_points(vin_snippet.points_world, vin_snippet.lengths, max_points=max_points)
    efm_or_vin = getattr(sample, "efm_snippet_view", None)
    if isinstance(efm_or_vin, VinSnippetView):
        return _valid_prefix_points(efm_or_vin.points_world, efm_or_vin.lengths, max_points=max_points)
    if isinstance(efm_or_vin, EfmSnippetView):
        semidense = efm_or_vin.semidense
        points = semidense.points_world
        lengths = semidense.lengths.to(device=points.device)
        max_len = points.shape[1]
        mask = torch.arange(max_len, device=points.device).unsqueeze(0) < lengths.clamp_max(max_len).unsqueeze(-1)
        flat = points[..., :3][mask]
        finite = torch.isfinite(flat).all(dim=-1)
        return flat[finite][:max_points].detach().cpu().to(dtype=torch.float32)
    return torch.zeros((0, 3), dtype=torch.float32)


def _valid_prefix_points(points: Tensor, lengths: Tensor, *, max_points: int) -> Tensor:
    pts = points.detach().cpu().to(dtype=torch.float32)
    length = int(torch.as_tensor(lengths).reshape(-1)[0].item()) if torch.as_tensor(lengths).numel() else pts.shape[0]
    if pts.ndim != 2 or pts.shape[-1] < 3:
        return torch.zeros((0, 3), dtype=torch.float32)
    pts = pts[: max(0, min(length, pts.shape[0])), :3]
    finite = torch.isfinite(pts).all(dim=-1)
    return pts[finite][:max_points]


def _evl_support_points(sample: Any, *, max_points: int) -> tuple[Tensor, Tensor | None]:
    backbone = getattr(sample, "backbone_out", None)
    if not isinstance(backbone, EvlBackboneOutput) or backbone.pts_world is None:
        return torch.zeros((0, 3), dtype=torch.float32), None
    points = backbone.pts_world.detach().cpu().to(dtype=torch.float32)
    if points.ndim == 3:
        points = points[0]
    points = points.reshape(-1, points.shape[-1])[:, :3]
    finite = torch.isfinite(points).all(dim=-1)
    counts = None
    if backbone.counts is not None:
        count_values = backbone.counts.detach().cpu().reshape(-1)
        if count_values.shape[0] == points.shape[0]:
            counts = count_values[finite][:max_points]
    return points[finite][:max_points], counts


def _points_inside_count(obb: ObbTW, points: Tensor, *, scale: float, positive_counts: Tensor | None = None) -> int:
    if points.numel() == 0:
        return 0
    inside = obb.points_inside_bb3(points.to(dtype=torch.float32), scale_obb=float(scale))
    if positive_counts is not None:
        inside = inside & (positive_counts.to(dtype=torch.float32) > 0)
    return int(inside.sum().item())


def _max_projected_area(obb: ObbTW) -> float:
    areas: list[float] = []
    for camera_id in range(3):
        bb2 = obb.bb2(camera_id).detach().cpu().reshape(-1, 4).to(dtype=torch.float32)
        visible = torch.all(bb2 > 0, dim=-1)
        width = (bb2[:, 1] - bb2[:, 0]).clamp_min(0)
        height = (bb2[:, 3] - bb2[:, 2]).clamp_min(0)
        area = torch.where(visible, width * height, torch.zeros_like(width))
        if area.numel():
            areas.append(float(area.max().item()))
    return max(areas) if areas else 0.0


def _target_reason_bitset(
    *,
    obb: ObbTW,
    confidence: float,
    projected_area: float,
    total_support: int,
    config: TargetSelectorConfig,
) -> int:
    bitset = 0
    data = obb.tensor().reshape(-1)
    extents = obb.bb3_diagonal.reshape(-1)
    if not torch.isfinite(data).all():
        bitset |= 1 << TARGET_INVALID_REASON_CODES["OBB_NONFINITE"]
    if not torch.isfinite(extents).all() or bool((extents <= 0).any().item()):
        bitset |= 1 << TARGET_INVALID_REASON_CODES["OBB_EXTENT_INVALID"]
    if confidence < config.min_confidence:
        bitset |= 1 << TARGET_INVALID_REASON_CODES["CONFIDENCE_TOO_LOW"]
    if projected_area <= 0.0:
        bitset |= 1 << TARGET_INVALID_REASON_CODES["NO_PROJECTED_VISIBILITY"]
    if projected_area < config.min_projected_area_pixels:
        bitset |= 1 << TARGET_INVALID_REASON_CODES["PROJECTED_AREA_TOO_SMALL"]
    if total_support < config.min_support_points:
        bitset |= 1 << TARGET_INVALID_REASON_CODES["TARGET_SUPPORT_TOO_LOW"]
    return (1 << TARGET_INVALID_REASON_CODES["VALID"]) if bitset == 0 else bitset


def _primary_reason(bitset: int) -> int:
    if bitset == (1 << TARGET_INVALID_REASON_CODES["VALID"]):
        return TARGET_INVALID_REASON_CODES["VALID"]
    for _name, bit in sorted(TARGET_INVALID_REASON_CODES.items(), key=lambda item: item[1]):
        if bit != TARGET_INVALID_REASON_CODES["VALID"] and bitset & (1 << bit):
            return bit
    return TARGET_INVALID_REASON_CODES["VALID"]


def _safe_obb_iou(pred: ObbTW, gt: ObbTW, *, samples: int) -> float:
    try:
        value = obb_iou3d(pred, gt, samp_per_dim=int(samples))
        return float(value.reshape(-1)[0].item())
    except Exception:
        pred_center = pred.bb3_center_world.reshape(-1, 3)[0]
        gt_center = gt.bb3_center_world.reshape(-1, 3)[0]
        pred_diag = torch.linalg.norm(pred.bb3_diagonal.reshape(-1, 3)[0]).clamp_min(1e-6)
        gt_diag = torch.linalg.norm(gt.bb3_diagonal.reshape(-1, 3)[0]).clamp_min(1e-6)
        return float((1.0 - torch.linalg.norm(pred_center - gt_center) / (pred_diag + gt_diag)).clamp(0.0, 1.0))


def _mark_duplicate_gt_matches(rows: tuple[TargetCandidateRow, ...]) -> tuple[TargetCandidateRow, ...]:
    counts: dict[int, int] = {}
    for row in rows:
        if row.gt_label_valid and row.gt_target_row_id is not None:
            counts[row.gt_target_row_id] = counts.get(row.gt_target_row_id, 0) + 1
    output: list[TargetCandidateRow] = []
    for row in rows:
        if row.gt_label_valid and row.gt_target_row_id is not None and counts.get(row.gt_target_row_id, 0) > 1:
            output.append(
                replace(
                    row,
                    gt_label_valid=False,
                    gt_match_status="ambiguous_pred_to_gt",
                    invalid_reason_bitset=_target_failure_bitset(
                        row.invalid_reason_bitset,
                        TARGET_INVALID_REASON_CODES["TARGET_GT_AMBIGUOUS"],
                    ),
                    primary_invalid_reason=TARGET_INVALID_REASON_CODES["TARGET_GT_AMBIGUOUS"],
                )
            )
        else:
            output.append(row)
    return tuple(output)


def _target_failure_bitset(current: int, reason: int) -> int:
    valid = 1 << TARGET_INVALID_REASON_CODES["VALID"]
    return (int(current) & ~valid) | (1 << int(reason))


def _ranking_key(row: TargetCandidateRow) -> tuple[float, int, str]:
    return (-float(row.score), int(row.source_index), row.target_id)


def _class_name(sem_id: int, sem_id_to_name: list[str] | None) -> str:
    if sem_id_to_name is None or sem_id < 0 or sem_id >= len(sem_id_to_name):
        return "<unknown>"
    name = str(sem_id_to_name[sem_id])
    return name if name else "<unknown>"


def _target_id(
    *,
    scene_id: str | None,
    snippet_id: str | None,
    source: str,
    sem_id: int,
    inst_id: int,
    source_index: int,
) -> str:
    return f"{scene_id or 'scene'}:{snippet_id or 'snippet'}:{source}:sem={sem_id}:inst={inst_id}:idx={source_index}"


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _float_tuple(values: Tensor, *, length: int | None = None) -> tuple[float, ...]:
    flat = values.detach().cpu().reshape(-1).to(dtype=torch.float32)
    if length is not None:
        flat = flat[:length]
    return tuple(float(value.item()) for value in flat)


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


__all__ = [
    "ActorVisibleTargetSelector",
    "TARGET_INVALID_REASON_CODES",
    "TARGET_INVALID_REASON_VERSION",
    "TargetCandidateRow",
    "TargetSelectionPolicy",
    "TargetSelectionResult",
    "TargetSelectorConfig",
    "TargetSourceMode",
    "target_gt_aabb_world",
    "target_gt_obb_world",
]
