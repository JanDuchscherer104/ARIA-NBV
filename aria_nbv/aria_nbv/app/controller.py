"""Pipeline controller for the refactored Streamlit app.

This module contains all heavy compute orchestration. Pages should call the
controller methods and remain responsible only for UI and plotting.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext

from ..data_handling import EfmSnippetView
from ..pipelines import OracleBackendProfile, OracleRriLabelerConfig
from ..pose_generation.types import CandidateSamplingResult
from ..rendering import CandidateDepths
from ..rendering.candidate_pointclouds import CandidatePointClouds
from ..rri_metrics.types import RriResult
from ..utils import Console
from .stage_subprocess import run_stage_subprocess
from .state_types import (
    AppState,
    candidates_key,
    config_signature,
    depths_key,
    pcs_key,
    sample_key,
)

ProgressCallback = Callable[[str], AbstractContextManager[None]]


def _noop_progress(_: str) -> AbstractContextManager[None]:
    return nullcontext()


class PipelineController:
    """Compute + cache pipeline stages in Streamlit session state."""

    def __init__(
        self,
        state: AppState,
        *,
        console: Console,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.state = state
        self.console = console
        self.progress = progress or _noop_progress

    # ------------------------------------------------------------------ public
    def get_sample(self, *, force: bool) -> EfmSnippetView:
        """Load and cache the selected dataset sample."""

        cfg = self.state.dataset_cfg
        cfg_sig = config_signature(cfg)
        cache = self.state.data

        if (
            not force
            and cache.sample is not None
            and cache.cfg_sig == cfg_sig
            and cache.sample_idx == int(self.state.sample_idx)
        ):
            return cache.sample

        ds_iter = cache.dataset_iter

        # Reset iterator when config changed or we need to go backwards.
        reset_iter = cache.cfg_sig != cfg_sig or ds_iter is None or cache.last_iter_idx is None
        if not reset_iter and self.state.sample_idx <= int(cache.last_iter_idx):
            reset_iter = True

        if reset_iter:
            self.console.log("Initialising dataset iterator")
            ds = cfg.setup_target()
            ds_iter = iter(ds)
            start_idx = 0
        else:
            assert cache.last_iter_idx is not None
            start_idx = int(cache.last_iter_idx) + 1

        target_idx = int(self.state.sample_idx)
        steps = max(target_idx - start_idx, 0)
        self.console.log(f"Advancing dataset iterator from {start_idx} to {target_idx} (steps={steps})")

        assert ds_iter is not None
        sample = None
        for _ in range(steps + 1):
            sample = next(ds_iter)
        assert sample is not None

        cache.cfg_sig = cfg_sig
        cache.sample_idx = target_idx
        cache.dataset_iter = ds_iter
        cache.last_iter_idx = target_idx
        cache.sample = sample

        # Invalidate downstream stages.
        self._invalidate_after_data()

        return sample

    def get_candidates(self, *, force: bool) -> CandidateSamplingResult:
        """Generate and cache candidates for the current sample."""

        sample = self.get_sample(force=False)
        resolved_labeler_cfg = self.state.labeler_cfg.resolved(require_available=True)
        cfg = resolved_labeler_cfg.generator
        cfg_sig = config_signature(cfg)
        skey = sample_key(sample)
        cache = self.state.candidates

        if not force and cache.candidates is not None and cache.cfg_sig == cfg_sig and cache.sample_key == skey:
            return cache.candidates

        with self.progress("Generating candidates..."):
            if self._use_stage_subprocess(resolved_labeler_cfg):
                result = run_stage_subprocess(
                    "generate_candidates",
                    self._stage_payload(resolved_labeler_cfg),
                )
                candidates = CandidateSamplingResult.from_serializable(
                    result["candidates"],
                    device=cfg.device,
                )
            else:
                candidates = cfg.setup_target().generate_from_typed_sample(sample)

        cache.cfg_sig = cfg_sig
        cache.sample_key = skey
        cache.candidates = candidates

        self._invalidate_after_candidates()
        return candidates

    def get_depths(self, *, force: bool) -> CandidateDepths:
        """Render and cache candidate depths for the current sample."""

        sample = self.get_sample(force=False)
        candidates = self.get_candidates(force=False)
        resolved_labeler_cfg = self.state.labeler_cfg.resolved(require_available=True)
        cfg = resolved_labeler_cfg.depth
        cfg_sig = config_signature(cfg)
        skey = sample_key(sample)
        ckey = candidates_key(candidates)
        cache = self.state.depth

        if (
            not force
            and cache.depths is not None
            and cache.cfg_sig == cfg_sig
            and cache.sample_key == skey
            and cache.candidates_key == ckey
        ):
            return cache.depths

        with self.progress("Rendering depth maps..."):
            if self._use_stage_subprocess(resolved_labeler_cfg):
                depths, pcs = self._compute_renders_subprocess(
                    resolved_labeler_cfg=resolved_labeler_cfg,
                    candidates=candidates,
                    stride=int(self.state.labeler_cfg.backprojection_stride),
                )
                self._store_pointcloud_cache(
                    depths=depths,
                    pcs=pcs,
                    pointcloud_cfg_sig=config_signature(
                        resolved_labeler_cfg.pointcloud.model_copy(
                            update={"backprojection_stride": int(self.state.labeler_cfg.backprojection_stride)},
                        ),
                    ),
                    stride=int(self.state.labeler_cfg.backprojection_stride),
                )
            else:
                depths = cfg.setup_target().render(sample=sample, candidates=candidates)

        cache.cfg_sig = cfg_sig
        cache.sample_key = skey
        cache.candidates_key = ckey
        cache.depths = depths

        if self._use_stage_subprocess(resolved_labeler_cfg):
            self.state.rri = type(self.state.rri)()
        else:
            self._invalidate_after_depths()
        return depths

    def get_renders(self, *, force: bool) -> tuple[CandidateDepths, CandidatePointClouds]:
        """Compute the "renders" stage: depth maps + depth-hit point clouds.

        The app treats candidate point clouds as part of the rendering stage so
        the Candidate Renders page can always show the depth-hit backprojection
        without requiring a separate toggle/button.
        """

        depths = self.get_depths(force=force)
        stride = int(self.state.labeler_cfg.backprojection_stride)
        pcs = self.get_candidate_pointclouds(stride=stride, force=force)
        return depths, pcs

    def get_candidate_pointclouds(self, *, stride: int, force: bool) -> CandidatePointClouds:
        """Backproject and cache candidate point clouds for the current depth batch."""

        sample = self.get_sample(force=False)
        depths = self.get_depths(force=False)
        depth_key = depths_key(depths)
        resolved_labeler_cfg = self.state.labeler_cfg.resolved(require_available=True)
        cfg = resolved_labeler_cfg.pointcloud.model_copy(
            update={"backprojection_stride": int(stride)},
        )
        cfg_sig = config_signature(cfg)
        pcs_cache = self.state.pcs
        if pcs_cache.by_stride is None or pcs_cache.depth_key != depth_key or pcs_cache.cfg_sig != cfg_sig:
            pcs_cache.depth_key = depth_key
            pcs_cache.cfg_sig = cfg_sig
            pcs_cache.by_stride = {}

        assert pcs_cache.by_stride is not None
        if not force and stride in pcs_cache.by_stride:
            return pcs_cache.by_stride[stride]

        with self.progress("Backprojecting depth maps..."):
            if self._use_stage_subprocess(resolved_labeler_cfg):
                candidates = self.get_candidates(force=False)
                depths_sub, pcs = self._compute_renders_subprocess(
                    resolved_labeler_cfg=resolved_labeler_cfg,
                    candidates=candidates,
                    stride=int(stride),
                )
                self.state.depth.depths = depths_sub
                self.state.depth.cfg_sig = config_signature(resolved_labeler_cfg.depth)
                self.state.depth.sample_key = sample_key(sample)
                self.state.depth.candidates_key = candidates_key(candidates)
            else:
                pcs = cfg.setup_target().build(sample, depths)

        pcs_cache.by_stride[stride] = pcs
        # Only invalidate RRI if we potentially overwrote the stride used by the oracle labeler.
        if force or stride == int(self.state.labeler_cfg.backprojection_stride):
            self._invalidate_after_pcs()
        return pcs

    def run_labeler(self, *, force: bool) -> tuple[CandidateDepths, CandidatePointClouds, RriResult]:
        """Run or refresh oracle RRI from the current resolved stage configs.

        The app reuses cached candidates, renders, and candidate point clouds
        when available so the RRI page does not unnecessarily rerun earlier
        stages that the user already inspected on the candidate or render pages.
        """

        sample = self.get_sample(force=False)
        cfg = self.state.labeler_cfg.resolved(require_available=True)
        cfg_sig = config_signature(cfg)

        cache = self.state.rri
        if (
            not force
            and cache.result is not None
            and cache.cfg_sig == cfg_sig
            and cache.pcs_key is not None
            and self.state.depth.depths is not None
        ):
            pcs_cached = None
            if self.state.pcs.by_stride is not None:
                pcs_cached = self.state.pcs.by_stride.get(int(cfg.backprojection_stride))
            if pcs_cached is not None:
                return self.state.depth.depths, pcs_cached, cache.result

        depths = self.get_depths(force=False)
        pcs = self.get_candidate_pointclouds(stride=int(cfg.backprojection_stride), force=False)
        if sample.mesh_verts is None or sample.mesh_faces is None:
            raise ValueError("Oracle RRI scoring requires mesh_verts and mesh_faces on the sample.")
        with self.progress("Running oracle label pipeline..."):
            if self._use_stage_subprocess(cfg):
                result = run_stage_subprocess(
                    "score_rri",
                    {
                        **self._stage_payload(cfg),
                        "pcs": pcs.to_serializable(),
                    },
                )
                rri = RriResult.from_serializable(result["rri"], device=cfg.device)
            else:
                rri = cfg.oracle.setup_target().score(
                    points_t=pcs.semidense_points,
                    points_q=pcs.points,
                    lengths_q=pcs.lengths,
                    gt_verts=sample.mesh_verts.to(device=pcs.points.device, dtype=pcs.points.dtype),
                    gt_faces=sample.mesh_faces.to(device=pcs.points.device),
                    extend=pcs.occupancy_bounds,
                )

        dkey = depths_key(depths)
        pointcloud_cfg_sig = config_signature(cfg.pointcloud)
        if (
            self.state.pcs.by_stride is None
            or self.state.pcs.depth_key != dkey
            or self.state.pcs.cfg_sig != pointcloud_cfg_sig
        ):
            self.state.pcs.depth_key = dkey
            self.state.pcs.cfg_sig = pointcloud_cfg_sig
            self.state.pcs.by_stride = {}
        assert self.state.pcs.by_stride is not None
        self.state.pcs.by_stride[int(cfg.backprojection_stride)] = pcs

        cache.cfg_sig = cfg_sig
        cache.pcs_key = pcs_key(pcs)
        cache.result = rri

        return depths, pcs, rri

    # ------------------------------------------------------------------ invalidation
    def _invalidate_after_data(self) -> None:
        self.state.candidates = type(self.state.candidates)()
        self.state.depth = type(self.state.depth)()
        self.state.pcs = type(self.state.pcs)()
        self.state.rri = type(self.state.rri)()

    def _invalidate_after_candidates(self) -> None:
        self.state.depth = type(self.state.depth)()
        self.state.pcs = type(self.state.pcs)()
        self.state.rri = type(self.state.rri)()

    def _invalidate_after_depths(self) -> None:
        self.state.pcs = type(self.state.pcs)()
        self.state.rri = type(self.state.rri)()

    def _invalidate_after_pcs(self) -> None:
        self.state.rri = type(self.state.rri)()

    def _use_stage_subprocess(self, resolved_labeler_cfg: OracleRriLabelerConfig) -> bool:
        return (
            resolved_labeler_cfg.backend_profile == OracleBackendProfile.APPLE_MPS_MOJO
            and threading.current_thread() is not threading.main_thread()
        )

    def _stage_payload(self, resolved_labeler_cfg: OracleRriLabelerConfig) -> dict[str, object]:
        dataset_payload = self.state.dataset_cfg.model_dump(mode="python", round_trip=True)
        dataset_payload["device"] = str(resolved_labeler_cfg.device)
        return {
            "dataset_cfg": dataset_payload,
            "sample_idx": int(self.state.sample_idx),
            "labeler_cfg": self.state.labeler_cfg.model_dump(mode="python", round_trip=True),
        }

    def _compute_renders_subprocess(
        self,
        *,
        resolved_labeler_cfg: OracleRriLabelerConfig,
        candidates: CandidateSamplingResult,
        stride: int,
    ) -> tuple[CandidateDepths, CandidatePointClouds]:
        result = run_stage_subprocess(
            "render_depths_and_pcs",
            {
                **self._stage_payload(resolved_labeler_cfg),
                "candidates": candidates.to_serializable(),
                "stride": int(stride),
            },
        )
        depths = CandidateDepths.from_serializable(result["depths"], device=resolved_labeler_cfg.depth.device)
        pcs = CandidatePointClouds.from_serializable(result["pcs"], device=resolved_labeler_cfg.device)
        return depths, pcs

    def _store_pointcloud_cache(
        self,
        *,
        depths: CandidateDepths,
        pcs: CandidatePointClouds,
        pointcloud_cfg_sig: str,
        stride: int,
    ) -> None:
        dkey = depths_key(depths)
        if (
            self.state.pcs.by_stride is None
            or self.state.pcs.depth_key != dkey
            or self.state.pcs.cfg_sig != pointcloud_cfg_sig
        ):
            self.state.pcs.depth_key = dkey
            self.state.pcs.cfg_sig = pointcloud_cfg_sig
            self.state.pcs.by_stride = {}
        assert self.state.pcs.by_stride is not None
        self.state.pcs.by_stride[int(stride)] = pcs


__all__ = ["PipelineController"]
