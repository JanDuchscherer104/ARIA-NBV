"""Main dashboard app logic (orchestrates pages and state)."""

from __future__ import annotations

import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

import streamlit as st
import torch

from ..data import AseEfmDatasetConfig, EfmSnippetView
from ..pose_generation import CandidateViewGeneratorConfig
from ..pose_generation.types import CandidateSamplingResult
from ..rendering import CandidateDepthRendererConfig, Pytorch3DDepthRendererConfig
from ..rendering.candidate_depth_renderer import CandidateDepths
from ..utils import Console, Verbosity
from .config import DashboardConfig
from .panels import render_candidates_page, render_data_page, render_depth_page, render_rri_page
from .services import get_executor, load_dataset
from .state import STATE_KEYS, get, init_task_state, safe_rerun, store
from .ui import candidate_config_ui, dataset_config_ui, renderer_config_ui


@dataclass(slots=True)
class DashboardApp:
    config: DashboardConfig

    def run(self) -> None:  # pragma: no cover - UI code
        """Render the Streamlit dashboard with fail-soft error handling.

        Any uncaught exception is surfaced inline with a full traceback rather
        than crashing the app, so the session remains usable for debugging.
        """
        console = Console.with_prefix("streamlit_app")
        try:
            self._render_body(console)
        except Exception as exc:  # pragma: no cover
            trace = traceback.format_exc()
            console.error(trace)
            st.error("Unexpected error encountered. The session stays alive; see full traceback below.")
            st.exception(exc)
            with st.expander("Full traceback", expanded=True):
                st.code(trace, language="text")
            st.stop()

    def _render_body(self, console: Console) -> None:  # pragma: no cover - UI code
        st.set_page_config(page_title="NBV Explorer", layout="wide")
        init_task_state()
        global_verbosity = st.sidebar.selectbox(
            "Verbosity (global)",
            options=[Verbosity.QUIET, Verbosity.NORMAL, Verbosity.VERBOSE],
            format_func=lambda v: v.name.title(),
            index=2,  # Default to VERBOSE
        )
        console = console.set_verbosity(global_verbosity)
        page = st.radio(
            "Select view",
            ("Data", "Candidate Poses", "Candidate Renders", "RRI"),
            horizontal=True,
            help="Switch between data inspection, candidate poses, render results, and RRI preview.",
        )

        sample = cast(EfmSnippetView | None, get(STATE_KEYS["sample"]))
        candidates = cast(CandidateSamplingResult | None, get(STATE_KEYS["candidates"]))
        depth_batch = cast(CandidateDepths | None, get(STATE_KEYS["depth"]))

        cfg_changed = {"sample": False, "cand": False, "depth": False}
        pipeline_order: tuple[str, ...] = ("data", "candidates", "depth")
        run_all = st.sidebar.button("Run ALL (data → candidates → renders)")

        def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
            return cfg.model_dump() if hasattr(cfg, "model_dump") else dict(cfg)

        def _cfg_from_state(key: str, cfg_cls: type[Any]) -> Any:
            stored = get(key)
            if isinstance(stored, cfg_cls):
                return stored
            if isinstance(stored, dict):
                return cfg_cls.model_validate(stored)
            return cfg_cls()

        def _cfg_from_state_optional(key: str, cfg_cls: type[Any]) -> Any | None:
            stored = get(key)
            if stored is None:
                return None
            if isinstance(stored, cfg_cls):
                return stored
            if isinstance(stored, dict):
                return cfg_cls.model_validate(stored)
            return None

        def _refresh_stage_vars() -> None:
            nonlocal sample, candidates, depth_batch
            sample = cast(EfmSnippetView | None, get(STATE_KEYS["sample"]))
            candidates = cast(CandidateSamplingResult | None, get(STATE_KEYS["candidates"]))
            depth_batch = cast(CandidateDepths | None, get(STATE_KEYS["depth"]))

        def _run_data_stage(cfg: AseEfmDatasetConfig | None, sample_idx: int, *, allow_ui: bool = True):
            if cfg is None:
                if allow_ui:
                    st.warning("No cached dataset config. Configure the dataset on the Data page first.")
                return None
            try:
                cached = cast(EfmSnippetView | None, get(STATE_KEYS["sample"]))
                cfg_same = get(STATE_KEYS["sample_cfg"]) == _cfg_to_dict(cfg)
                last_idx = cast(int | None, get(STATE_KEYS["sample_idx"]))
                if cached is not None and cfg_same and last_idx == sample_idx:
                    console.log("Using cached sample")
                    return cached

                ds = load_dataset(cfg)
                ds_iter = cast(Iterator[Any] | None, get(STATE_KEYS["dataset_iter"]))
                if not cfg_same or ds_iter is None:
                    console.log("Initializing new dataset iterator")
                    ds_iter = iter(ds)
                    store(STATE_KEYS["dataset_iter"], ds_iter)
                    start_idx = 0
                else:
                    console.log("Reusing cached dataset iterator")
                    start_idx = last_idx + 1 if last_idx is not None else 0

                steps = max(sample_idx - start_idx, 0)
                console.log(f"Advancing dataset iterator from {start_idx} to {sample_idx} (steps={steps})")
                for _ in range(steps + 1):
                    sample_local = next(ds_iter)

                store(STATE_KEYS["sample"], sample_local)
                store(STATE_KEYS["sample_cfg"], _cfg_to_dict(cfg))
                store(STATE_KEYS["sample_idx"], sample_idx)
                store(STATE_KEYS["candidates"], None)
                store(STATE_KEYS["cand_cfg"], None)
                store(STATE_KEYS["depth"], None)
                store(STATE_KEYS["depth_cfg"], None)
                st.session_state.pop("nbv_candidate_pcs", None)
                cfg_changed["sample"] = False
                if allow_ui:
                    safe_rerun()
                return sample_local
            except Exception as exc:  # pragma: no cover
                if allow_ui:
                    st.error(f"Failed to load dataset sample: {exc}")
                console.error(str(exc))
                return None

        def _run_candidates_stage(cfg: CandidateViewGeneratorConfig | None, *, allow_ui: bool = True):
            if cfg is None:
                if allow_ui:
                    st.warning("No cached candidate config. Run candidate generation once to cache settings.")
                return None
            local_sample = cast(EfmSnippetView | None, get(STATE_KEYS["sample"]))
            if local_sample is None:
                if allow_ui:
                    st.warning("Load data first on the Data page, then run candidates.")
                return None
            try:
                cached = cast(CandidateSamplingResult | None, get(STATE_KEYS["candidates"]))
                if cached is not None and get(STATE_KEYS["cand_cfg"]) == _cfg_to_dict(cfg):
                    console.log("Using cached candidates")
                    return cached

                generator = cfg.setup_target()
                with st.status("Generating candidates...", expanded=False):
                    candidates_local = generator.generate_from_typed_sample(local_sample)
                store(STATE_KEYS["candidates"], candidates_local)
                store(STATE_KEYS["cand_cfg"], _cfg_to_dict(cfg))
                store(STATE_KEYS["depth"], None)
                store(STATE_KEYS["depth_cfg"], None)
                st.session_state.pop("nbv_candidate_pcs", None)
                cfg_changed["cand"] = False
                if allow_ui:
                    safe_rerun()
                return candidates_local
            except Exception as exc:
                if allow_ui:
                    st.error(f"Candidate generation failed: {exc}")
                console.error(str(exc))
                raise exc

        def _run_depth_stage(
            cfg: CandidateDepthRendererConfig | None, render_depths: bool = True, *, allow_ui: bool = True
        ):
            if not render_depths:
                return None
            if cfg is None:
                if allow_ui:
                    st.warning("No cached renderer config. Configure renderer settings first.")
                return None
            # Force PyTorch3D on CUDA when available to avoid CPU efm3d fallback.
            if torch.cuda.is_available() and not isinstance(cfg.renderer, Pytorch3DDepthRendererConfig):
                console.log("Forcing depth renderer to Pytorch3D (cuda)")
                cfg = cfg.model_copy(update={"renderer": Pytorch3DDepthRendererConfig(device="cuda")})
                store(STATE_KEYS["depth_cfg"], _cfg_to_dict(cfg))
                store(STATE_KEYS["depth"], None)
            local_sample = cast(EfmSnippetView | None, get(STATE_KEYS["sample"]))
            local_candidates = cast(CandidateSamplingResult | None, get(STATE_KEYS["candidates"]))
            if local_sample is None or local_candidates is None:
                if allow_ui:
                    st.warning("Need data and candidates. Run previous stages before rendering.")
                return None
            cached = cast(CandidateDepths | None, get(STATE_KEYS["depth"]))
            if cached is not None and get(STATE_KEYS["depth_cfg"]) == _cfg_to_dict(cfg):
                console.log("Using cached depth renders")
                return cached
            renderer = cfg.setup_target()
            try:
                executor = get_executor()
                with st.status("Rendering depth maps (async)...", expanded=False):
                    future = executor.submit(renderer.render, local_sample, local_candidates)
                    depth_local = future.result()
                store(STATE_KEYS["depth"], depth_local)
                store(STATE_KEYS["depth_cfg"], _cfg_to_dict(cfg))
                st.session_state.pop("nbv_candidate_pcs", None)
                cfg_changed["depth"] = False
                if allow_ui:
                    safe_rerun()
                return depth_local
            except Exception as exc:  # pragma: no cover
                if allow_ui:
                    st.warning(f"Depth rendering failed: {exc}")
                console.warn(str(exc))
                return None

        def _run_previous_stages(current_stage: str, *, allow_ui: bool = False) -> None:
            try:
                target_idx = pipeline_order.index(current_stage)
            except ValueError:
                st.warning(f"Unknown stage: {current_stage}")
                return
            for stage in pipeline_order[:target_idx]:
                if stage == "data":
                    data_cfg_state = _cfg_from_state_optional(STATE_KEYS["sample_cfg"], AseEfmDatasetConfig)
                    sample_idx_state = int(get(STATE_KEYS["sample_idx"]) or 0)
                    _run_data_stage(data_cfg_state, sample_idx_state, allow_ui=allow_ui)
                elif stage == "candidates":
                    cand_cfg_state = _cfg_from_state_optional(STATE_KEYS["cand_cfg"], CandidateViewGeneratorConfig)
                    cand_cfg_state = cand_cfg_state or CandidateViewGeneratorConfig()
                    _run_candidates_stage(cand_cfg_state, allow_ui=allow_ui)
                elif stage == "depth":
                    renderer_cfg_state = _cfg_from_state_optional(STATE_KEYS["depth_cfg"], CandidateDepthRendererConfig)
                    _run_depth_stage(renderer_cfg_state, True, allow_ui=allow_ui)
            _refresh_stage_vars()

        # Auto-run prerequisite stages when switching tabs and results are missing
        if page == "Candidate Poses" and (sample is None or candidates is None):
            _run_previous_stages("candidates", allow_ui=False)
            _refresh_stage_vars()
        if page in ("Candidate Renders", "RRI") and depth_batch is None:
            _run_previous_stages("depth", allow_ui=False)
            _refresh_stage_vars()

        if page == "Data":
            dataset_cfg_prev = (
                _cfg_from_state_optional(STATE_KEYS["sample_cfg"], AseEfmDatasetConfig) or self.config.dataset
            )
            sample_cached = cast(EfmSnippetView | None, get(STATE_KEYS["sample"]))
            with st.sidebar.form("data_form"):
                dataset_cfg = dataset_config_ui(
                    st.sidebar,
                    verbosity=global_verbosity,
                    is_debug=dataset_cfg_prev.is_debug,
                )
                sample_idx = int(get(STATE_KEYS["sample_idx"]) or 0)
                next_sample = st.form_submit_button("Next sample")
                run_data = st.form_submit_button("Run / refresh data")
            if next_sample:
                sample_idx += 1
                run_data = True
            clear = st.sidebar.button("Clear cache")
            if clear:
                for key in STATE_KEYS.values():
                    st.session_state.pop(key, None)
                st.rerun()

            cfg_changed["sample"] = get(STATE_KEYS["sample_cfg"]) != _cfg_to_dict(dataset_cfg)
            if run_data or sample_cached is None or cfg_changed["sample"]:
                sample = _run_data_stage(dataset_cfg, sample_idx)
                _refresh_stage_vars()
            else:
                console.log("Using cached sample for Data page")
                sample = sample_cached
            if run_all:
                _run_previous_stages("depth", allow_ui=True)
                _refresh_stage_vars()

            if sample is None:
                st.info("No sample loaded. Configure dataset and click 'Run / refresh data' (or 'Next sample').")
            else:
                cfg_state = _cfg_from_state(STATE_KEYS["sample_cfg"], AseEfmDatasetConfig)
                render_data_page(sample, crop_margin=cfg_state.mesh_crop_margin_m)
            return

        if page == "Candidate Poses":
            cand_cfg_prev = (
                _cfg_from_state_optional(STATE_KEYS["cand_cfg"], CandidateViewGeneratorConfig) or self.config.generator
            )
            cand_cached = cast(CandidateSamplingResult | None, get(STATE_KEYS["candidates"]))
            with st.sidebar.form("cand_form"):
                candidate_cfg = candidate_config_ui(
                    cand_cfg_prev,
                    st.sidebar,
                    is_debug=cand_cfg_prev.is_debug,
                    verbosity=global_verbosity,
                )
                st.sidebar.subheader("Candidate plot options")
                frustum_scale = st.sidebar.slider("Frustum scale", 0.1, 1.0, 0.5, step=0.05)
                max_frustums = st.sidebar.slider("Max frustums", 1, 24, 6)
                plot_rejected_only = st.sidebar.checkbox("Plot rejected poses only (if any)", value=False)
                run_prev = st.form_submit_button("Run previous")
                run_cand = st.form_submit_button("Run / refresh candidates")
            cfg_changed["cand"] = get(STATE_KEYS["cand_cfg"]) != _cfg_to_dict(candidate_cfg)

            if run_prev:
                _run_previous_stages("candidates")
            if run_cand or cand_cached is None or cfg_changed["cand"]:
                _run_candidates_stage(candidate_cfg, allow_ui=False)
            else:
                console.log("Using cached candidates for Candidate Poses page")
                candidates = cand_cached
            _refresh_stage_vars()

            if run_all:
                _run_previous_stages("depth", allow_ui=True)
                _refresh_stage_vars()

            if candidates is None or sample is None:
                st.info("No candidates yet. Configure generator and click 'Run / refresh candidates'.")
            else:
                render_candidates_page(
                    sample, candidates, candidate_cfg, frustum_scale, max_frustums, plot_rejected_only
                )
            if cfg_changed["cand"]:
                st.info("Candidate settings changed; rerun to refresh results.")
            return

        with st.sidebar.form("depth_form"):
            depth_cfg_prev = (
                _cfg_from_state_optional(STATE_KEYS["depth_cfg"], CandidateDepthRendererConfig) or self.config.renderer
            )
            depth_cached = cast(CandidateDepths | None, get(STATE_KEYS["depth"]))
            renderer_cfg = renderer_config_ui(
                depth_cfg_prev,
                st.sidebar,
                is_debug=depth_cfg_prev.is_debug,
                verbosity=global_verbosity,
            )
            # Auto-upgrade UI result to PyTorch3D on CUDA (default) unless user explicitly picked cpu
            if torch.cuda.is_available() and not isinstance(renderer_cfg.renderer, Pytorch3DDepthRendererConfig):
                console.log("Upgrading depth renderer to Pytorch3D (cuda) for renders")
                renderer_cfg = renderer_cfg.model_copy(update={"renderer": Pytorch3DDepthRendererConfig(device="cuda")})
                store(STATE_KEYS["depth_cfg"], _cfg_to_dict(renderer_cfg))
                store(STATE_KEYS["depth"], None)
            render_depths = st.checkbox("Compute depth renders", value=True, key="compute_depths")
            run_prev = st.form_submit_button("Run previous")
            run_depth = st.form_submit_button("Run / refresh renders")
        cfg_changed["depth"] = get(STATE_KEYS["depth_cfg"]) != _cfg_to_dict(renderer_cfg)

        if run_prev:
            _run_previous_stages("depth")
        if run_depth or depth_cached is None or cfg_changed["depth"]:
            if render_depths:
                _run_depth_stage(renderer_cfg, render_depths, allow_ui=True)
            elif depth_batch is None:
                st.warning("No cached depths available; enable 'Compute depth renders' or run renders first.")
            else:
                st.info("Using cached depths to rebuild plots.")
            _refresh_stage_vars()
        else:
            console.log("Using cached depth renders for Candidate Renders page")
            depth_batch = depth_cached
        if run_all:
            _run_previous_stages("depth", allow_ui=True)
            _refresh_stage_vars()

        if depth_batch is None:
            if candidates is None:
                st.info("No renders yet. Run candidates first, then click 'Run / refresh renders'.")
            elif not render_depths:
                st.info("Enable 'Compute depth renders' to produce depth maps.")
            else:
                st.info("Click 'Run / refresh renders' to compute depth maps.")
        else:
            if page == "RRI":
                render_rri_page(sample, depth_batch)
            else:
                render_depth_page(depth_batch)

        stale_parts = [name for name, changed in cfg_changed.items() if changed]
        if stale_parts:
            st.info(f"Cached results are stale for: {', '.join(stale_parts)}. Click the page's run button to update.")


# Resolve forward refs now that DashboardApp is defined
DashboardConfig.model_rebuild(_types_namespace={"DashboardApp": DashboardApp})


__all__ = ["DashboardApp"]
