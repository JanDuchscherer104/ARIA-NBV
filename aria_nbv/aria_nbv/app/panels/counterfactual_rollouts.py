"""Streamlit panel for live and stored counterfactual rollout inspection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import streamlit as st
import torch

from ...data_handling import (
    ActorVisibleTargetSelector,
    TargetCandidateRow,
    TargetSelectionPolicy,
    TargetSelectorConfig,
    TargetSourceMode,
    VinOfflineDatasetConfig,
    VinOfflineSample,
    VinOfflineStoreConfig,
)
from ...pose_generation import (
    CandidateGenerationRuntimeContext,
    CandidateMixtureComponentConfig,
    CandidateMixtureViewGeneratorConfig,
    CandidateViewGeneratorConfig,
    CounterfactualCandidateEvaluation,
    CounterfactualMetricBundle,
    CounterfactualOracleRriScorerConfig,
    CounterfactualPoseGeneratorConfig,
    CounterfactualRolloutResult,
    CounterfactualSelectionPolicy,
    CounterfactualTargetOracleRriScorerConfig,
    TargetRriInvalidError,
    ViewDirectionMode,
)
from ...pose_generation.plotting import CounterfactualPlotBuilder, plot_counterfactual_paths_simple
from ...rendering import CandidateDepthRendererConfig
from ...rollouts import RolloutZarrStoreReader
from ...utils import Console, Verbosity
from ..rerun_launch import (
    build_rerun_rollout_spawn_command,
    format_command,
    repo_root,
    spawn_background_command,
)
from ..state_types import config_signature
from .common import _info_popover, _pretty_label, _report_exception, _strip_ansi

if TYPE_CHECKING:
    from ...pose_generation.counterfactuals import CounterfactualEvaluatorFn


_SOURCE_TARGET_INFO = """
This block chooses the immutable VIN offline root and the actor-visible target candidates.

- `VIN offline store`: source rows with cached EFM/backbone state and attached mesh assets.
- `Split` / `Split-local sample index`: which source row is inspected.
- `Target source mode`: V1 should use actor-visible target records; GT-only modes are sanity/evaluation paths.
- `Target top-k` / `Target policy`: how many eligible actor-visible targets are retained and how they are ranked.
- `Min target confidence` / `Min target support`: actor-side quality filters before rollout generation.
- `Min GT IoU` / `GT ambiguity gap`: GT matching gates for labels and evaluation crops only.
- `Target softmax temperature`: stochastic target-selection temperature when a sampling policy is used.
"""

_LOADED_SAMPLE_INFO = """
Loaded sample metrics:

- `Scene`: ASE scene id for the source row.
- `Snippet`: source snippet/window id.
- `Source`: target-record source used by the selector.
- `Selected targets`: number of actor-visible targets retained for rollout generation.

Target table fields:

- `target_row_id`: split-local target row id used by rollouts and stored targets.
- `selected_rank`: rank among selected targets; `None` means retained for inspection but not selected.
- `class`: human-readable class name when available.
- `sem_id` / `inst_id`: semantic and instance ids from the actor-visible target record.
- `confidence`: detector/source confidence.
- `score`: target-selection score, not target-RRI.
- `support`: semidense plus EVL support count used as actor-visible evidence.
- `eligible`: whether actor-side gates allow the target.
- `gt_label_valid`: whether GT matching produced a valid oracle/evaluation label.
- `gt_match_status`: GT matching outcome such as `matched`, `not_requested`, or ambiguity/invalid status.
- `gt_iou`: IoU of the accepted GT match when available.
"""

_ACTIVE_TARGET_INFO = """
The active target is the object conditioned into target-RRI rollout generation.

Label format: `target 0 · 28 · sem=28 inst=51297 · score=0.000 · valid`.

- `target 0`: target row id.
- `28`: class label as exposed by the current target record; numeric labels appear when no class name is resolved.
- `sem=... inst=...`: semantic and instance ids used to identify the actor-visible target.
- `score=...`: target-selection score; it is not an RRI reward.
- `valid`: GT-only matching succeeded, so target-RRI labels/evaluation crops can be computed.

The actor sees the target descriptor and support, not the matched GT crop. GT fields stay in GT-EVAL.
"""

_ROLLOUT_GENERATION_INFO = """
This block defines the finite-candidate rollout tree.

- `Scoring mode`: `target_rri` is thesis-core; `scene_rri` and `geometry` are diagnostics.
- `Candidates per step`: requested valid candidate budget regenerated at each rollout step.
- `Generator device`: local live rendering uses CPU because PyTorch3D GPU support is not guaranteed here.
- `Horizon` (`H`): maximum rollout length.
- `Branch factor` (`B`): number of child actions retained per expanded state.
- `Beam width`: optional cap on retained partial trajectories.
- `Selection policy`: how candidates are selected from the scored valid set.
- `Softmax temperature`: stochastic-policy temperature.
- `Seed`: controls repeatable sampling.
- `Min history distance` / `Min sibling distance`: geometric guards against duplicate or near-duplicate actions.
- `Log rollout/scorer timing`: emits timing diagnostics in the Logs tab.
"""

_TARGET_MIXTURE_INFO = """
Target-RRI rollouts use a mixed finite candidate set.

- `TARGET_POINT`: view directions aimed at the active actor-visible target center.
- `RADIAL_TOWARDS`: radial poses oriented toward the target region.
- `RADIAL_AWAY`: radial poses that expand support from the opposite side.
- `FORWARD_RIG`: forward-facing rig poses for exploration/coverage.

The default budget `16` maps to `6/4/3/3`. These are candidate-set sampling counts, not rollout branch counts.
"""

_SCORER_CONTROLS_INFO = """
These controls affect oracle scoring cost and validity.

- `Backprojection stride`: depth-to-point-cloud stride used after rendering candidate views.
- `Target crop margin`: margin around the matched GT target OBB for target-RRI evaluation.
- `Min current target points`: minimum current target support required before target-RRI is meaningful.
- `Also compute scene RRI audit`: optional diagnostic scene-RRI pass; off by default for speed.
"""

_ROLLOUT_RESULT_INFO = """
Result table and plots:

- `cumulative_score`: cumulative score used by the selected policy.
- `cumulative_rri`: cumulative RRI when an RRI scorer is attached; intentionally empty in geometry mode.
- `terminated_early`: rollout stopped before `H`, usually because no valid successor action remained.
- `final_x/y/z`: final selected rig pose translation in world coordinates.
- `Paths`: trajectory-level visualization.
- `Step Shell`: per-step candidate shell and selected candidate view.
- `Logs`: captured Console output from generation/scoring.
"""

_STORED_ROLLOUTS_INFO = """
Stored rollouts inspect a standalone `rollouts.zarr` shard without recomputation.

- Validation metadata checks table/mask consistency before inspection.
- The manifest records source/config lineage and source coverage.
- Target summary separates actor target validity from GT-label validity.
- Candidate rows show `actor_action`, `q_train`, target/scene RRI labels, and strategy/mixture provenance.
- `q_train` marks rows usable for finite-candidate `Q_H` training views.
- Rerun launch opens the selected rollout row in the 3D inspector.
"""


class LiveRolloutScoringMode(StrEnum):
    """Available scoring modes for live rollout generation."""

    TARGET_RRI = "target_rri"
    SCENE_RRI = "scene_rri"
    GEOMETRY = "geometry"


@dataclass(slots=True)
class LiveRolloutScoreContext:
    """Evaluator and candidate-runtime state for one live rollout run."""

    score_label: str
    evaluator: "CounterfactualEvaluatorFn | None"
    runtime_context: CandidateGenerationRuntimeContext | None


def _live_rollout_device_options() -> list[str]:
    """Return UI device choices after verifying PyTorch3D CUDA rasterization."""

    options = ["cpu"]
    if _pytorch3d_cuda_rasterization_available():
        options.append("cuda")
    return options


@lru_cache(maxsize=1)
def _pytorch3d_cuda_rasterization_available() -> bool:
    """Return whether the installed PyTorch3D extension can rasterize on CUDA."""

    if not torch.cuda.is_available():
        return False
    try:
        from pytorch3d.renderer import FoVPerspectiveCameras, MeshRasterizer, RasterizationSettings
        from pytorch3d.structures import Meshes

        device = torch.device("cuda")
        verts = torch.tensor(
            [[-0.5, -0.5, 2.0], [0.5, -0.5, 2.0], [0.0, 0.5, 2.0]],
            dtype=torch.float32,
            device=device,
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Meshes(verts=[verts], faces=[faces])
        cameras = FoVPerspectiveCameras(device=device)
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(image_size=8, blur_radius=0.0, faces_per_pixel=1),
        )
        fragments = rasterizer(mesh)
        return bool(fragments.pix_to_face.numel())
    except Exception:
        return False


def _live_depth_config(*, max_candidates: int, device: str) -> CandidateDepthRendererConfig:
    """Build the scorer depth config on the same explicit device as rollout generation."""

    return CandidateDepthRendererConfig(
        device=torch.device(device),
        max_candidates_final=int(max_candidates),
    )


class _SceneRriScoreAdapter:
    """Rename scene-level oracle RRI so the UI does not imply target scoring."""

    def __init__(self, scorer: object) -> None:
        self.scorer = scorer

    def __call__(self, candidates, trajectory, step_index) -> CounterfactualCandidateEvaluation:
        evaluation = self.scorer(candidates, trajectory, step_index)
        metrics = CounterfactualMetricBundle.from_vectors(evaluation.metric_vectors)
        if metrics.scene_rri is None and metrics.rri is not None:
            metrics.scene_rri = metrics.rri
        return CounterfactualCandidateEvaluation(
            scores=evaluation.scores,
            score_label=LiveRolloutScoringMode.SCENE_RRI.value,
            metrics=metrics,
            candidate_point_clouds_world=evaluation.candidate_point_clouds_world,
            candidate_point_cloud_lengths=evaluation.candidate_point_cloud_lengths,
        )


def _build_live_dataset_config(*, store_dir: Path, split: str) -> VinOfflineDatasetConfig:
    """Return the VIN offline reader config required by live target-RRI rollouts."""

    return VinOfflineDatasetConfig(
        store=VinOfflineStoreConfig(store_dir=store_dir),
        split=split,  # type: ignore[arg-type]
        return_format="sample",
        include_efm_snippet=True,
        include_gt_mesh=True,
        load_backbone=True,
        load_candidates=False,
        load_depths=False,
        load_candidate_pcs=False,
        load_gt_obbs=True,
        load_detected_obbs=True,
        load_trajectory_metadata=True,
    )


def _load_vin_offline_sample(*, store_dir: Path, split: str, sample_index: int) -> VinOfflineSample:
    """Load one split-local `VinOfflineSample` with live snippet and target assets."""

    dataset = _build_live_dataset_config(store_dir=store_dir, split=split).setup_target()
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"Sample index {sample_index} is outside split '{split}' length {len(dataset)}.")
    sample = dataset[int(sample_index)]
    if not isinstance(sample, VinOfflineSample):
        raise TypeError("Live rollout inspector requires VinOfflineDatasetConfig(return_format='sample').")
    if sample.efm_snippet_view is None or sample.efm_snippet_view.mesh is None:
        raise ValueError("Live rollout sample must include an attached EFM snippet and GT mesh.")
    return sample


def _target_mixture_counts_from_budget(candidate_budget: int) -> dict[ViewDirectionMode, int]:
    """Allocate a 6/4/3/3 target-aware mixture for a requested candidate budget."""

    budget = int(candidate_budget)
    if budget < 4:
        raise ValueError("Target-aware rollout mixtures require at least 4 candidates.")
    weights = {
        ViewDirectionMode.TARGET_POINT: 6.0,
        ViewDirectionMode.RADIAL_TOWARDS: 4.0,
        ViewDirectionMode.RADIAL_AWAY: 3.0,
        ViewDirectionMode.FORWARD_RIG: 3.0,
    }
    total = sum(weights.values())
    counts = {mode: max(1, int(np.floor(budget * weight / total))) for mode, weight in weights.items()}
    while sum(counts.values()) < budget:
        deficits = {mode: (budget * weights[mode] / total) - counts[mode] for mode in counts}
        mode = max(deficits, key=deficits.get)
        counts[mode] += 1
    while sum(counts.values()) > budget:
        mode = max((mode for mode in counts if counts[mode] > 1), key=counts.get)
        counts[mode] -= 1
    return counts


def _target_mixture_config(
    base: CandidateViewGeneratorConfig,
    *,
    counts: dict[ViewDirectionMode, int],
) -> CandidateMixtureViewGeneratorConfig:
    """Build a target-aware mixed candidate generator from per-family counts."""

    components = [
        CandidateMixtureComponentConfig(
            name="target_point",
            count=int(counts[ViewDirectionMode.TARGET_POINT]),
            strategy=ViewDirectionMode.TARGET_POINT,
            view_max_azimuth_deg=0.0,
            view_max_elevation_deg=0.0,
        ),
        CandidateMixtureComponentConfig(
            name="radial_towards",
            count=int(counts[ViewDirectionMode.RADIAL_TOWARDS]),
            strategy=ViewDirectionMode.RADIAL_TOWARDS,
            view_max_azimuth_deg=0.0,
            view_max_elevation_deg=0.0,
        ),
        CandidateMixtureComponentConfig(
            name="radial_away",
            count=int(counts[ViewDirectionMode.RADIAL_AWAY]),
            strategy=ViewDirectionMode.RADIAL_AWAY,
            view_max_azimuth_deg=0.0,
            view_max_elevation_deg=0.0,
        ),
        CandidateMixtureComponentConfig(
            name="forward_rig",
            count=int(counts[ViewDirectionMode.FORWARD_RIG]),
            strategy=ViewDirectionMode.FORWARD_RIG,
            view_max_azimuth_deg=0.0,
            view_max_elevation_deg=0.0,
        ),
    ]
    return CandidateMixtureViewGeneratorConfig(base=base, components=components)


def _candidate_config_for_live_rollout(
    *,
    scoring_mode: LiveRolloutScoringMode,
    candidate_budget: int,
    seed: int | None,
    device: str,
    counts: dict[ViewDirectionMode, int] | None = None,
) -> CandidateViewGeneratorConfig | CandidateMixtureViewGeneratorConfig:
    """Return the candidate generator used by one live rollout run."""

    base = CandidateViewGeneratorConfig(
        num_samples=int(candidate_budget),
        oversample_factor=2.0,
        seed=seed,
        device=device,
        collect_rule_masks=True,
        collect_debug_stats=True,
        verbosity=Verbosity.NORMAL,
    )
    if scoring_mode is LiveRolloutScoringMode.TARGET_RRI:
        resolved_counts = counts or _target_mixture_counts_from_budget(candidate_budget)
        return _target_mixture_config(base, counts=resolved_counts)
    return base


def _validate_policy_for_scoring_mode(
    *,
    scoring_mode: LiveRolloutScoringMode,
    selection_policy: CounterfactualSelectionPolicy,
) -> None:
    """Reject policy/scoring combinations that would mislabel geometric scores."""

    if (
        scoring_mode is LiveRolloutScoringMode.GEOMETRY
        and selection_policy is CounterfactualSelectionPolicy.ORACLE_GREEDY
    ):
        raise ValueError("oracle_greedy requires an RRI scorer; choose target_rri, scene_rri, or a geometry policy.")


def _score_context_for_mode(
    *,
    scoring_mode: LiveRolloutScoringMode,
    sample: VinOfflineSample,
    target: TargetCandidateRow | None,
    target_scorer_config: CounterfactualTargetOracleRriScorerConfig,
    scene_scorer_config: CounterfactualOracleRriScorerConfig,
) -> LiveRolloutScoreContext:
    """Create the scorer and target runtime context for a live rollout."""

    if scoring_mode is LiveRolloutScoringMode.GEOMETRY:
        return LiveRolloutScoreContext(
            score_label=LiveRolloutScoringMode.GEOMETRY.value,
            evaluator=None,
            runtime_context=None,
        )

    if sample.efm_snippet_view is None:
        raise ValueError("Live RRI scoring requires sample.efm_snippet_view.")

    if scoring_mode is LiveRolloutScoringMode.SCENE_RRI:
        scorer = scene_scorer_config.setup_target(sample=sample.efm_snippet_view)
        return LiveRolloutScoreContext(
            score_label=LiveRolloutScoringMode.SCENE_RRI.value,
            evaluator=_SceneRriScoreAdapter(scorer),
            runtime_context=None,
        )

    if target is None:
        raise ValueError("target_rri scoring requires a selected target row.")
    if not target.gt_label_valid:
        raise ValueError(f"Selected target is not GT-label valid: status={target.gt_match_status}.")
    scorer = target_scorer_config.setup_target(
        sample=sample.efm_snippet_view,
        target_sample=sample,
        target_row=target,
    )
    return LiveRolloutScoreContext(
        score_label=LiveRolloutScoringMode.TARGET_RRI.value,
        evaluator=scorer,
        runtime_context=CandidateGenerationRuntimeContext(
            target_center_world=torch.tensor(target.center_world, dtype=torch.float32),
            target_id=target.target_id,
        ),
    )


def _run_live_rollout(
    *,
    sample: VinOfflineSample,
    scoring_mode: LiveRolloutScoringMode,
    target: TargetCandidateRow | None,
    candidate_config: CandidateViewGeneratorConfig | CandidateMixtureViewGeneratorConfig,
    rollout_config: CounterfactualPoseGeneratorConfig,
    target_scorer_config: CounterfactualTargetOracleRriScorerConfig,
    scene_scorer_config: CounterfactualOracleRriScorerConfig,
) -> tuple[CounterfactualRolloutResult, str]:
    """Generate one live rollout result and capture Console logs for display."""

    if sample.efm_snippet_view is None:
        raise ValueError("Live rollout generation requires sample.efm_snippet_view.")
    _validate_policy_for_scoring_mode(
        scoring_mode=scoring_mode,
        selection_policy=rollout_config.selection_policy,
    )
    context = _score_context_for_mode(
        scoring_mode=scoring_mode,
        sample=sample,
        target=target,
        target_scorer_config=target_scorer_config,
        scene_scorer_config=scene_scorer_config,
    )
    resolved_rollout_config = rollout_config.model_copy(update={"candidate_config": candidate_config})

    lines: list[str] = []

    def _sink(message: str) -> None:
        lines.append(_strip_ansi(message))

    Console.set_sink(_sink)
    try:
        rollouts = resolved_rollout_config.setup_target().generate_from_typed_sample(
            sample.efm_snippet_view,
            score_candidates=context.evaluator,
            candidate_runtime_context=context.runtime_context,
        )
    finally:
        Console.set_sink(None)
    if context.score_label == LiveRolloutScoringMode.GEOMETRY.value:
        rollouts.score_label = LiveRolloutScoringMode.GEOMETRY.value
    return rollouts, "\n".join(lines)


def _counterfactual_trajectory_rows(
    rollouts: CounterfactualRolloutResult,
) -> list[dict[str, int | float | bool | None]]:
    """Summarize rollout trajectories for compact panel tables."""

    rows: list[dict[str, int | float | bool | None]] = []
    for traj_idx, trajectory in enumerate(rollouts.trajectories):
        final_pos = trajectory.final_pose_world().t.detach().cpu().reshape(-1).tolist()
        rows.append(
            {
                "trajectory": traj_idx,
                "steps": len(trajectory.steps),
                "cumulative_score": float(trajectory.cumulative_score),
                "cumulative_rri": (None if trajectory.cumulative_rri is None else float(trajectory.cumulative_rri)),
                "terminated_early": bool(trajectory.terminated_early),
                "final_x": float(final_pos[0]),
                "final_y": float(final_pos[1]),
                "final_z": float(final_pos[2]),
            }
        )
    return rows


def _target_rows_table(rows: tuple[TargetCandidateRow, ...]) -> list[dict[str, object]]:
    """Return a compact dataframe payload for target rows."""

    return [
        {
            "target_row_id": int(row.target_row_id),
            "selected_rank": row.selected_rank,
            "class": row.class_name,
            "sem_id": int(row.sem_id),
            "inst_id": int(row.inst_id),
            "confidence": float(row.confidence),
            "score": None if not np.isfinite(row.score) else float(row.score),
            "support": int(row.semidense_support_count + row.evl_support_count),
            "eligible": bool(row.eligible),
            "gt_label_valid": bool(row.gt_label_valid),
            "gt_match_status": row.gt_match_status,
            "gt_iou": row.gt_match_iou,
        }
        for row in rows
    ]


def _target_detail_row(row: TargetCandidateRow) -> dict[str, object]:
    """Return one target row with pose/crop-relevant fields."""

    return {
        "target_id": row.target_id,
        "source": row.source,
        "source_index": int(row.source_index),
        "center_world": tuple(float(v) for v in row.center_world),
        "extents": tuple(float(v) for v in row.extents),
        "pose_world_object": tuple(float(v) for v in row.pose_world_object),
        "relative_pose_reference_object": tuple(float(v) for v in row.relative_pose_reference_object),
        "gt_target_id": row.gt_target_id,
        "gt_target_row_id": row.gt_target_row_id,
        "gt_match_status": row.gt_match_status,
        "gt_match_iou": row.gt_match_iou,
        "invalid_reason_bitset": int(row.invalid_reason_bitset),
        "primary_invalid_reason": int(row.primary_invalid_reason),
    }


def _format_target_option(row: TargetCandidateRow) -> str:
    status = "valid" if row.gt_label_valid else row.gt_match_status
    return (
        f"target {row.target_row_id} · {row.class_name} · sem={row.sem_id} inst={row.inst_id} · "
        f"score={row.score:.3f} · {status}"
    )


def _render_live_rollouts_tab() -> None:
    st.header("Live Target-RRI Counterfactual Rollouts")
    st.caption(
        "Generate multi-step rollouts from VIN offline roots. Target-RRI mode uses V1 actor-visible target "
        "selection and GT-only evaluation crops; scene and geometry modes are diagnostics."
    )

    default_store = VinOfflineStoreConfig().store_dir
    with st.expander("Source sample and target selection", expanded=True):
        _info_popover("source sample and target selection", _SOURCE_TARGET_INFO)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            store_dir = Path(
                st.text_input("VIN offline store", value=str(default_store), key="cf_store_dir")
            ).expanduser()
            split = st.selectbox("Split", options=["all", "train", "val"], index=0, key="cf_split")
            sample_index = int(
                st.number_input("Split-local sample index", min_value=0, value=0, step=1, key="cf_sample_index")
            )
        with col_b:
            source_mode = st.selectbox(
                "Target source mode",
                options=list(TargetSourceMode),
                index=list(TargetSourceMode).index(TargetSourceMode.V1_ACTOR_VISIBLE),
                format_func=lambda mode: mode.value,
                key="cf_target_source_mode",
            )
            target_k = int(st.slider("Target top-k", min_value=1, max_value=12, value=3, step=1, key="cf_target_k"))
            target_policy = st.selectbox(
                "Target policy",
                options=list(TargetSelectionPolicy),
                index=0,
                format_func=lambda policy: policy.value,
                key="cf_target_policy",
            )
        with col_c:
            min_conf = float(st.slider("Min target confidence", 0.0, 1.0, 0.2, step=0.05, key="cf_min_conf"))
            min_support = int(st.slider("Min target support", 1, 256, 1, step=1, key="cf_min_support"))
            min_gt_iou = float(st.slider("Min GT IoU", 0.0, 1.0, 0.1, step=0.05, key="cf_min_gt_iou"))
            gt_gap = float(st.slider("GT ambiguity gap", 0.0, 0.5, 0.02, step=0.01, key="cf_gt_gap"))
            target_temperature = float(
                st.slider("Target softmax temperature", 0.05, 5.0, 1.0, step=0.05, key="cf_target_temperature")
            )

    selector_cfg = TargetSelectorConfig(
        k=int(target_k),
        policy=target_policy,
        source_mode=source_mode,
        min_confidence=float(min_conf),
        min_support_points=int(min_support),
        min_gt_iou=float(min_gt_iou),
        gt_ambiguity_margin=float(gt_gap),
        temperature=float(target_temperature),
    )
    load_key = f"{store_dir.resolve() if store_dir.exists() else store_dir}|{split}|{sample_index}|{config_signature(selector_cfg)}"
    cache = st.session_state.setdefault("cf_live_source_cache", {})
    if st.button("Load sample and targets", key="cf_load_sample_targets"):
        try:
            sample = _load_vin_offline_sample(store_dir=store_dir, split=str(split), sample_index=int(sample_index))
            target_result = ActorVisibleTargetSelector(selector_cfg).select(sample)
            cache[load_key] = {"sample": sample, "target_result": target_result}
        except Exception as exc:  # pragma: no cover - UI guard
            _report_exception(exc, context="Failed to load VIN offline sample and targets")
            cache.pop(load_key, None)

    payload = cache.get(load_key)
    if payload is None:
        st.info("Load a VIN offline sample to inspect actor-visible targets and generate live rollouts.")
        return

    sample = payload["sample"]
    target_result = payload["target_result"]
    st.subheader("Loaded Sample")
    _info_popover("loaded sample fields", _LOADED_SAMPLE_INFO)
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Scene", sample.scene_id)
    col_s2.metric("Snippet", sample.snippet_id)
    col_s3.metric("Source", target_result.source or "none")
    col_s4.metric("Selected targets", len(target_result.selected_rows))
    if target_result.warnings:
        st.warning("\n".join(target_result.warnings))

    st.dataframe(_target_rows_table(target_result.rows), width="stretch", hide_index=True)
    selected_target = None
    if target_result.selected_rows:
        _info_popover("active target label", _ACTIVE_TARGET_INFO)
        selected_target = st.selectbox(
            "Active target",
            options=list(target_result.selected_rows),
            format_func=_format_target_option,
            key="cf_active_target",
        )
        st.json(_target_detail_row(selected_target), expanded=False)

    with st.expander("Rollout generation", expanded=True):
        _info_popover("rollout generation controls", _ROLLOUT_GENERATION_INFO)
        cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
        with cfg_col1:
            scoring_mode = st.selectbox(
                "Scoring mode",
                options=list(LiveRolloutScoringMode),
                index=0,
                format_func=lambda mode: mode.value,
                key="cf_scoring_mode",
            )
            candidate_budget = int(st.slider("Candidates per step", 4, 128, 16, step=1, key="cf_candidate_budget"))
            device = st.selectbox(
                "Generator device",
                options=_live_rollout_device_options(),
                index=0,
                key="cf_generator_device",
            )
            if torch.cuda.is_available() and "cuda" not in _live_rollout_device_options():
                st.caption("CUDA is hidden because the PyTorch3D rasterization smoke check failed.")
        with cfg_col2:
            horizon = int(st.slider("Horizon", min_value=1, max_value=5, value=3, step=1, key="cf_horizon"))
            branch_factor = int(
                st.slider("Branch factor", min_value=1, max_value=6, value=2, step=1, key="cf_branch_factor")
            )
            cap_beam = st.checkbox("Cap beam width", value=True, key="cf_beam_enabled")
            beam_width = (
                int(st.slider("Beam width", min_value=1, max_value=12, value=4, step=1, key="cf_beam_width"))
                if cap_beam
                else None
            )
        with cfg_col3:
            policy_options = list(CounterfactualSelectionPolicy)
            if scoring_mode is LiveRolloutScoringMode.GEOMETRY:
                policy_options = [
                    policy for policy in policy_options if policy is not CounterfactualSelectionPolicy.ORACLE_GREEDY
                ]
            selection_policy = st.selectbox(
                "Selection policy",
                options=policy_options,
                index=0,
                format_func=lambda policy: policy.value,
                key="cf_selection_policy",
            )
            temperature = float(st.slider("Softmax temperature", 0.05, 5.0, 1.0, step=0.05, key="cf_temperature"))
            seed = int(st.number_input("Seed", min_value=0, value=0, step=1, key="cf_seed"))

        guard_col1, guard_col2, guard_col3 = st.columns(3)
        with guard_col1:
            min_history_distance = float(
                st.slider("Min history distance (m)", 0.0, 2.0, 0.0, step=0.05, key="cf_min_history_distance")
            )
        with guard_col2:
            min_sibling_distance = float(
                st.slider("Min sibling distance (m)", 0.0, 2.0, 0.15, step=0.05, key="cf_min_sibling_distance")
            )
        with guard_col3:
            log_timing = st.checkbox("Log rollout/scorer timing", value=False, key="cf_log_timing")

        target_counts = None
        if scoring_mode is LiveRolloutScoringMode.TARGET_RRI:
            _info_popover("target mixture families", _TARGET_MIXTURE_INFO)
            advanced_counts = st.checkbox(
                "Advanced target-mixture counts", value=False, key="cf_advanced_mixture_counts"
            )
            if advanced_counts:
                mix_cols = st.columns(4)
                target_counts = {
                    ViewDirectionMode.TARGET_POINT: int(
                        mix_cols[0].number_input("TARGET_POINT", min_value=1, value=6, step=1)
                    ),
                    ViewDirectionMode.RADIAL_TOWARDS: int(
                        mix_cols[1].number_input("RADIAL_TOWARDS", min_value=1, value=4, step=1)
                    ),
                    ViewDirectionMode.RADIAL_AWAY: int(
                        mix_cols[2].number_input("RADIAL_AWAY", min_value=1, value=3, step=1)
                    ),
                    ViewDirectionMode.FORWARD_RIG: int(
                        mix_cols[3].number_input("FORWARD_RIG", min_value=1, value=3, step=1)
                    ),
                }
                st.caption(f"Advanced mixture total: {sum(target_counts.values())} candidates per step.")
            else:
                target_counts = _target_mixture_counts_from_budget(candidate_budget)
                st.caption(
                    "Default target mixture: "
                    + ", ".join(f"{mode.value}={count}" for mode, count in target_counts.items())
                )

    with st.expander("Scorer controls", expanded=False):
        _info_popover("scorer controls", _SCORER_CONTROLS_INFO)
        score_col1, score_col2, score_col3 = st.columns(3)
        with score_col1:
            backprojection_stride = int(
                st.slider("Backprojection stride", 1, 16, 1, step=1, key="cf_backprojection_stride")
            )
        with score_col2:
            target_crop_margin = float(
                st.slider("Target crop margin (m)", 0.0, 0.5, 0.0, step=0.01, key="cf_target_crop_margin")
            )
        with score_col3:
            min_current_target_points = int(
                st.slider("Min current target points", 1, 512, 1, step=1, key="cf_min_current_target_points")
            )
        include_scene_audit = st.checkbox(
            "Also compute scene RRI audit in target mode", value=False, key="cf_include_scene_audit"
        )

    candidate_config = _candidate_config_for_live_rollout(
        scoring_mode=scoring_mode,
        candidate_budget=int(candidate_budget if target_counts is None else sum(target_counts.values())),
        seed=int(seed),
        device=str(device),
        counts=target_counts,
    )
    rollout_cfg = CounterfactualPoseGeneratorConfig(
        candidate_config=candidate_config,
        horizon=int(horizon),
        branch_factor=int(branch_factor),
        beam_width=beam_width,
        selection_policy=selection_policy,
        selection_temperature=float(temperature),
        min_history_distance_m=float(min_history_distance),
        min_sibling_distance_m=float(min_sibling_distance),
        seed=int(seed),
        log_timing=bool(log_timing),
        verbosity=Verbosity.NORMAL,
    )
    live_candidate_count = int(candidate_budget if target_counts is None else sum(target_counts.values()))
    depth_cfg = _live_depth_config(max_candidates=live_candidate_count, device=str(device))
    target_scorer_cfg = CounterfactualTargetOracleRriScorerConfig(
        depth=depth_cfg,
        backprojection_stride=int(backprojection_stride),
        target_crop_margin_m=float(target_crop_margin),
        min_current_target_points=int(min_current_target_points),
        include_scene_rri=bool(include_scene_audit),
        log_timing=bool(log_timing),
    )
    scene_scorer_cfg = CounterfactualOracleRriScorerConfig(
        depth=depth_cfg,
        backprojection_stride=int(backprojection_stride),
    )

    run_key = "|".join(
        [
            load_key,
            scoring_mode.value,
            "" if selected_target is None else selected_target.target_id,
            config_signature(candidate_config),
            config_signature(rollout_cfg),
            config_signature(target_scorer_cfg),
            config_signature(scene_scorer_cfg),
        ]
    )
    rollout_cache = st.session_state.setdefault("cf_live_rollout_cache", {})
    if st.button("Run / refresh live rollouts", key="cf_run_live_rollouts"):
        try:
            with st.spinner("Generating live counterfactual rollouts..."):
                rollouts, log_text = _run_live_rollout(
                    sample=sample,
                    scoring_mode=scoring_mode,
                    target=selected_target,
                    candidate_config=candidate_config,
                    rollout_config=rollout_cfg,
                    target_scorer_config=target_scorer_cfg,
                    scene_scorer_config=scene_scorer_cfg,
                )
            rollout_cache[run_key] = {"rollouts": rollouts, "logs": log_text}
        except TargetRriInvalidError as exc:
            st.error(f"Target-RRI invalid: {exc}")
            rollout_cache.pop(run_key, None)
        except Exception as exc:  # pragma: no cover - UI guard
            _report_exception(exc, context="Live rollout generation failed")
            rollout_cache.pop(run_key, None)

    cached_rollout = rollout_cache.get(run_key)
    if cached_rollout is None:
        st.caption("Configure the rollout, then click run to materialize trajectories.")
        return

    rollouts = cached_rollout["rollouts"]
    log_text = cached_rollout["logs"]
    _render_rollout_result(sample, rollouts, log_text=log_text, scoring_mode=scoring_mode)


def _render_rollout_result(
    sample: VinOfflineSample,
    rollouts: CounterfactualRolloutResult,
    *,
    log_text: str,
    scoring_mode: LiveRolloutScoringMode,
) -> None:
    """Render one live rollout result."""

    rows = _counterfactual_trajectory_rows(rollouts)
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Trajectories", len(rollouts.trajectories))
    metric_col2.metric("Horizon", rollouts.horizon)
    metric_col3.metric("Score label", rollouts.score_label)
    best_score = max((traj.cumulative_score for traj in rollouts.trajectories), default=0.0)
    metric_col4.metric("Best cumulative score", f"{best_score:.3f}")
    _info_popover("rollout result table and plots", _ROLLOUT_RESULT_INFO)
    if scoring_mode is LiveRolloutScoringMode.GEOMETRY:
        st.info("Geometry mode does not compute RRI; cumulative_rri is intentionally empty.")

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    plot_tab, step_tab, log_tab = st.tabs(["Paths", "Step Shell", "Logs"])
    snippet = sample.efm_snippet_view
    with plot_tab:
        show_selected_frusta = st.checkbox("Overlay selected frusta", value=True, key="cf_show_selected_frusta")
        if snippet is not None:
            builder = (
                CounterfactualPlotBuilder.from_rollouts(
                    snippet,
                    rollouts,
                    title=_pretty_label("Counterfactual rollout paths"),
                )
                .add_mesh()
                .add_counterfactual_paths(show_step_markers=True)
            )
            if show_selected_frusta:
                builder = builder.add_counterfactual_selected_frusta(scale=0.45)
            st.plotly_chart(builder.finalize(), width="stretch")
        else:
            st.plotly_chart(plot_counterfactual_paths_simple(rollouts), width="stretch")

    with step_tab:
        if not rollouts.trajectories:
            st.info("No trajectories were generated.")
        else:
            trajectory_index = st.selectbox(
                "Trajectory",
                options=list(range(len(rollouts.trajectories))),
                format_func=lambda idx: (
                    f"traj {idx} · steps={rows[idx]['steps']} · score={rows[idx]['cumulative_score']:.3f}"
                ),
                key="cf_step_traj_idx",
            )
            trajectory = rollouts.trajectories[int(trajectory_index)]
            if not trajectory.steps:
                st.info("Selected trajectory terminated before choosing any rollout step.")
            else:
                step_display_index = st.slider(
                    "Step",
                    min_value=1,
                    max_value=len(trajectory.steps),
                    value=1,
                    step=1,
                    key="cf_step_idx",
                )
                include_rejected = st.checkbox("Show rejected candidates", value=False, key="cf_step_include_rejected")
                if snippet is None:
                    st.info("Step-shell plot requires the attached EFM snippet.")
                else:
                    step_fig = (
                        CounterfactualPlotBuilder.from_rollouts(
                            snippet,
                            rollouts,
                            title=_pretty_label(f"Counterfactual step {step_display_index}"),
                        )
                        .add_mesh()
                        .add_counterfactual_step_shell(
                            trajectory_index=int(trajectory_index),
                            step_index=int(step_display_index - 1),
                            include_rejected=include_rejected,
                            show_frusta=True,
                        )
                        .finalize()
                    )
                    st.plotly_chart(step_fig, width="stretch")

    with log_tab:
        if log_text.strip():
            st.code(log_text, language="text")
        else:
            st.caption("No Console output was emitted during this rollout run.")


def _render_stored_rollouts_tab() -> None:
    st.header("Stored Rollout Zarr")
    st.caption("Load a standalone rollouts.zarr store, validate row contracts, and open selected rows in Rerun.")
    _info_popover("stored rollouts", _STORED_ROLLOUTS_INFO)
    default_store = repo_root() / ".data" / "offline_cache" / "rollouts_v1_smoke.zarr"
    default_config = repo_root() / ".configs" / "rerun_offline.toml"
    store_path = Path(
        st.text_input("rollouts.zarr path", value=str(default_store), key="rollout_store_path")
    ).expanduser()
    config_path = Path(
        st.text_input("Rerun inspector config", value=str(default_config), key="rollout_rerun_config_path")
    ).expanduser()

    if not store_path.exists():
        st.info("Enter an existing rollouts.zarr path to inspect generated rollout rows.")
        return

    try:
        reader = RolloutZarrStoreReader(store_path)
        manifest_bundle = reader.manifest()
        validation = reader.validate()
    except Exception as exc:  # pragma: no cover - UI guard
        _report_exception(exc, context="Failed to load rollout store")
        return

    root_attrs = dict(reader.root.attrs)
    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
    meta_col1.metric("Rollouts", validation.num_rollouts)
    meta_col2.metric("Steps", validation.num_steps)
    meta_col3.metric("Candidates", validation.num_candidates)
    meta_col4.metric("Validation", "OK" if validation.ok else "FAILED")
    if validation.ok:
        st.success("Store validation passed.")
    else:
        st.error("Store validation failed.")
        st.dataframe([{"error": error} for error in validation.errors], width="stretch", hide_index=True)

    with st.expander("Root metadata", expanded=False):
        st.json(root_attrs)
    with st.expander("Generation manifest", expanded=False):
        st.json(manifest_bundle["manifest"])

    _render_rollout_store_summaries(reader, manifest=manifest_bundle["manifest"])
    rollout_ids = reader.array("rollouts/rollout_row_id").astype(int).tolist()
    if not rollout_ids:
        st.info("No rollout rows are present.")
        return
    selected_rollout = int(
        st.selectbox(
            "Rollout row",
            options=rollout_ids,
            format_func=lambda row_id: _format_rollout_option(reader, row_id),
            key="rollout_row_selector",
        )
    )
    st.dataframe(_candidate_rows_for_rollout(reader, selected_rollout), width="stretch", hide_index=True)
    command = build_rerun_rollout_spawn_command(
        config_path=config_path,
        rollout_store=store_path,
        rollout_row_id=selected_rollout,
    )
    st.code(format_command(command), language="bash")
    if st.button("Open selected rollout in Rerun", key="rollout_open_rerun"):
        try:
            process = spawn_background_command(command)
        except Exception as exc:  # pragma: no cover - UI guard
            _report_exception(exc, context="Failed to spawn Rerun inspector")
        else:
            st.success(f"Spawned Rerun inspector with pid {process.pid}.")


def _render_rollout_store_summaries(reader: RolloutZarrStoreReader, *, manifest: dict[str, object]) -> None:
    target_rows = reader.array("targets/target_row_id")
    rollout_rows = reader.array("rollouts/rollout_row_id")
    step_rows = reader.array("steps/step_row_id")
    candidate_rows = reader.array("candidates/candidate_row_id")
    q_h = reader.q_h_view()
    q_train = q_h["q_train_mask"]
    valid_action = q_h["valid_action_mask"]
    actor_action = reader.array("candidates/actor_action_mask")
    oracle_label = reader.array("candidates/oracle_label_mask")
    summary = [
        {"table": "targets", "rows": int(target_rows.shape[0])},
        {"table": "rollouts", "rows": int(rollout_rows.shape[0])},
        {"table": "steps", "rows": int(step_rows.shape[0])},
        {"table": "candidates", "rows": int(candidate_rows.shape[0])},
        {"table": "actor_action candidates", "rows": int(actor_action.sum())},
        {"table": "oracle_label candidates", "rows": int(oracle_label.sum())},
        {"table": "q_h valid actions", "rows": int(valid_action.sum())},
        {"table": "q_h train cells", "rows": int(q_train.sum())},
    ]
    st.dataframe(summary, width="stretch", hide_index=True)
    coverage = manifest.get("source_coverage", {})
    if isinstance(coverage, dict) and coverage:
        st.markdown("**Source coverage**")
        st.json(coverage)
    target_summary = []
    target_valid = reader.array("targets/target_valid_mask")
    gt_valid = reader.array("targets/gt_label_valid_mask")
    for row_id, valid, gt_label in zip(target_rows, target_valid, gt_valid, strict=True):
        target_summary.append(
            {
                "target_row_id": int(row_id),
                "target_valid": bool(valid),
                "gt_label_valid": bool(gt_label),
            }
        )
    if target_summary:
        st.dataframe(target_summary, width="stretch", hide_index=True)


def _candidate_rows_for_rollout(reader: RolloutZarrStoreReader, rollout_row_id: int) -> list[dict[str, object]]:
    rollout_ids = reader.array("candidates/rollout_row_id")
    mask = rollout_ids == int(rollout_row_id)
    candidate_row_ids = reader.array("candidates/candidate_row_id")
    step_indices = reader.array("candidates/step_index")
    shell_indices = reader.array("candidates/shell_index")
    selected_mask = reader.array("candidates/selected_mask")
    actor_action_mask = reader.array("candidates/actor_action_mask")
    q_train_mask = reader.array("candidates/q_train_mask")
    target_rri = reader.array("candidates/target_rri")
    scene_rri = reader.array("candidates/scene_rri")
    strategy_id = reader.array("candidates/strategy_id")
    mixture_id = reader.array("candidates/mixture_id")
    rows: list[dict[str, object]] = []
    for index in np.nonzero(mask)[0].tolist():
        rows.append(
            {
                "candidate_row_id": int(candidate_row_ids[index]),
                "step_index": int(step_indices[index]),
                "shell_index": int(shell_indices[index]),
                "selected": bool(selected_mask[index]),
                "actor_action": bool(actor_action_mask[index]),
                "q_train": bool(q_train_mask[index]),
                "target_rri": _finite_or_none(target_rri[index]),
                "scene_rri": _finite_or_none(scene_rri[index]),
                "strategy_id": int(strategy_id[index]),
                "mixture_id": int(mixture_id[index]),
            }
        )
    return rows


def _format_rollout_option(reader: RolloutZarrStoreReader, rollout_row_id: int) -> str:
    rollout_rows = reader.array("rollouts/rollout_row_id")
    matches = np.nonzero(rollout_rows == int(rollout_row_id))[0]
    if matches.size != 1:
        return f"rollout {rollout_row_id}"
    index = int(matches[0])
    policies = _string_list(reader, "dictionaries/policy")
    scenes = _string_list(reader, "dictionaries/scene")
    policy = _dict_value(policies, int(reader.array("rollouts/policy_id")[index]))
    scene = _dict_value(scenes, int(reader.array("rollouts/scene_id")[index]))
    target_row = int(reader.array("rollouts/target_row_id")[index])
    chain = int(reader.array("rollouts/chain_id")[index])
    horizon = int(reader.array("rollouts/horizon")[index])
    branch_factor = int(reader.array("rollouts/branch_factor")[index])
    beam = _format_stored_beam_width(int(reader.array("rollouts/beam_width")[index]))
    return (
        f"{rollout_row_id} · scene {scene} · target {target_row} · {policy} · "
        f"chain {chain} · H={horizon} · B={branch_factor} · beam={beam}"
    )


def _format_stored_beam_width(value: int) -> str:
    return "NaN" if int(value) < 0 else str(int(value))


def _string_list(reader: RolloutZarrStoreReader, path: str) -> list[str]:
    try:
        return json.loads(bytes(reader.array(path).tolist()).decode("utf-8"))
    except Exception:
        return []


def _dict_value(values: list[str], index: int) -> str:
    if index < 0 or index >= len(values):
        return ""
    return values[index]


def _finite_or_none(value: object) -> float | None:
    value_float = float(value)
    return value_float if np.isfinite(value_float) else None


def render_counterfactual_rollouts_page() -> None:
    """Render live target-RRI generation and stored rollout inspection."""

    live_tab, stored_tab = st.tabs(["Live Rollouts", "Stored Rollouts"])
    with live_tab:
        _info_popover(
            "live target-rri rollouts",
            "Target-RRI mode loads a VIN offline sample, selects an actor-visible target, "
            "uses GT only for matching/evaluation crops, and scores selected rollout branches "
            "with target-specific oracle RRI.",
        )
        _render_live_rollouts_tab()
    with stored_tab:
        _render_stored_rollouts_tab()


__all__ = [
    "LiveRolloutScoringMode",
    "render_counterfactual_rollouts_page",
]
