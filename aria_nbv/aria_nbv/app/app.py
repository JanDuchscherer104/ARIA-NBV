"""Refactored Streamlit app entrypoint."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from ..configs import PathConfig
from ..data_handling._legacy_cache_api import OracleRriCacheConfig, OracleRriCacheDatasetConfig
from ..pipelines import OracleBackendProfileError
from ..pose_generation import CandidateViewGeneratorConfig
from ..utils import Console, Verbosity
from .config import NbvStreamlitAppConfig
from .controller import PipelineController
from .state import clear_state, get_state, safe_rerun, store_state
from .state_types import config_signature
from .ui import (
    backend_profile_ui,
    candidate_config_ui,
    dataset_config_ui,
    oracle_config_ui,
    renderer_config_ui,
)


def _candidate_cfg_from_cache(labeler_payload: dict[str, Any] | None) -> CandidateViewGeneratorConfig | None:
    if not isinstance(labeler_payload, dict):
        return None
    generator_payload = labeler_payload.get("generator")
    if not isinstance(generator_payload, dict):
        return None
    try:
        return CandidateViewGeneratorConfig(**generator_payload)
    except Exception as exc:  # pragma: no cover - UI guard
        Console.with_prefix("cand_cache").warn(
            f"Failed to parse cached candidate config ({type(exc).__name__}: {exc}).",
        )
        return None


def _load_offline_cache_dataset(
    cache_cfg: OracleRriCacheDatasetConfig,
) -> tuple[Any | None, int, str | None]:
    # NBV_LEGACY_OFFLINE_CACHE_REMOVE_AFTER_FULL_MIGRATION: Streamlit adapter
    # for the legacy oracle-cache reader.
    cfg_sig = config_signature(cache_cfg)
    cache_state = st.session_state.setdefault("cand_offline_cache", {})
    cache_ds = cache_state.get("cache_ds")
    if cache_state.get("cfg_sig") != cfg_sig or cache_ds is None:
        try:
            cache_ds = cache_cfg.setup_target()
        except Exception as exc:  # pragma: no cover - IO guard
            return None, 0, f"Failed to load offline cache: {type(exc).__name__}: {exc}"
        cache_state["cache_ds"] = cache_ds
        cache_state["cfg_sig"] = cfg_sig
        cache_state["cache_len"] = len(cache_ds)
        cache_state["cache_idx"] = 0
        st.session_state["cand_cache_index"] = 0
    cache_len = int(cache_state.get("cache_len", 0) or 0)
    return cache_ds, cache_len, None


def _render_page_import_error(page_name: str, exc: Exception) -> None:
    """Render a page-local import/runtime error without crashing the whole app."""

    trace = traceback.format_exc()
    st.error(f"{page_name} is unavailable: {type(exc).__name__}: {exc}")
    with st.expander("Full traceback", expanded=False):
        st.code(trace, language="text")


def _run_safe_page(page_name: str, fn: Any) -> None:
    """Run one page callback and contain page-local failures."""

    try:
        fn()
    except Exception as exc:
        _render_page_import_error(page_name, exc)


@dataclass(slots=True)
class NbvStreamlitApp:
    config: NbvStreamlitAppConfig

    def run(self) -> None:  # pragma: no cover - UI code
        console = Console.with_prefix("nbv_streamlit_app")
        try:
            self._render(console)
        except Exception as exc:  # pragma: no cover
            trace = traceback.format_exc()
            print(trace, flush=True)
            console.error(trace)
            st.error(
                "Unexpected error encountered. The session stays alive; see full traceback below.",
            )
            st.exception(exc)
            with st.expander("Full traceback", expanded=True):
                st.code(trace, language="text")
            st.stop()

    def _render(self, console: Console) -> None:  # pragma: no cover - UI code
        st.set_page_config(page_title="NBV Explorer", layout="wide")

        state = get_state(self.config.dataset, self.config.labeler)
        console = console.set_verbosity(
            st.sidebar.selectbox(
                "Verbosity (global)",
                options=[Verbosity.QUIET, Verbosity.NORMAL, Verbosity.VERBOSE],
                format_func=lambda v: v.name.title(),
                index=2,
            ),
        )
        st.sidebar.divider()
        state.labeler_cfg = backend_profile_ui(state.labeler_cfg, st.sidebar)
        try:
            resolved_labeler_cfg = state.labeler_cfg.resolved(require_available=True)
        except OracleBackendProfileError as exc:
            store_state(state)
            st.error(str(exc))
            st.info(
                "Use `uv sync --extra dev --extra pytorch3d` on CUDA workstations, "
                "or select `apple_mps_mojo` after completing the Mojo setup on Apple Silicon.",
            )
            st.stop()

        controller = PipelineController(
            state,
            console=console,
            progress=lambda msg: st.status(msg, expanded=False),
        )

        # Global controls -------------------------------------------------
        st.sidebar.divider()
        st.sidebar.subheader("Run controls")
        run_all = st.sidebar.button("Run ALL (data → candidates → renders)")
        clear = st.sidebar.button("Clear session state")
        if clear:
            clear_state()
            safe_rerun()

        # Show cache status ----------------------------------------------
        st.sidebar.divider()
        st.sidebar.subheader("Cache status")
        st.sidebar.caption(f"sample_idx={state.sample_idx}")
        st.sidebar.write(
            {
                "data": state.data.sample is not None,
                "candidates": state.candidates.candidates is not None,
                "depth": state.depth.depths is not None,
                "pcs": bool(state.pcs.by_stride),
                "rri": state.rri.result is not None,
            },
        )

        if run_all:
            controller.get_sample(force=True)
            controller.get_candidates(force=True)
            controller.get_renders(force=True)
            store_state(state)
            safe_rerun()

        # Pages ----------------------------------------------------------
        def _page_data() -> None:
            try:
                from .panels import render_data_page
            except Exception as exc:
                _render_page_import_error("Data page", exc)
                return

            with st.sidebar.form("data_form"):
                dataset_cfg_prev = state.dataset_cfg
                dataset_cfg = dataset_config_ui(
                    st.sidebar,
                    device=str(resolved_labeler_cfg.device),
                    verbosity=console.verbosity,
                    is_debug=dataset_cfg_prev.is_debug,
                )
                sample_idx = st.number_input(
                    "Sample index",
                    min_value=0,
                    value=int(state.sample_idx),
                    step=1,
                )
                next_sample = st.form_submit_button("Next sample")
                refresh_data = st.form_submit_button("Run / refresh data")
            if next_sample:
                sample_idx += 1
                refresh_data = True
            state.sample_idx = int(sample_idx)
            state.dataset_cfg = dataset_cfg
            sample = controller.get_sample(force=refresh_data)
            store_state(state)

            render_data_page(sample, crop_margin=dataset_cfg.mesh_crop_margin_m)

        def _page_candidates() -> None:
            try:
                from .panels import render_candidates_page
            except Exception as exc:
                _render_page_import_error("Candidate Poses page", exc)
                return

            data_source = st.sidebar.selectbox(
                "Candidate source",
                options=["online (oracle labeler)", "offline cache"],
                index=0,
                key="cand_data_source",
            )
            # NBV_LEGACY_OFFLINE_CACHE_REMOVE_AFTER_FULL_MIGRATION: the "offline
            # cache" branch is the legacy oracle-cache UI path.
            use_offline_cache = data_source == "offline cache"

            cand_cfg_prev = state.labeler_cfg.generator
            cand_cfg = cand_cfg_prev
            source_caption = None
            source_note = None

            if use_offline_cache:
                expander = st.sidebar.expander("Candidate Generator", expanded=False)
                paths = PathConfig()
                cache_root = paths.offline_cache_dir or (paths.data_root / "oracle_rri_cache")
                cache_dir = expander.text_input(
                    "Offline cache dir",
                    value=str(cache_root),
                    key="cand_cache_dir",
                )
                cache_split = expander.selectbox(
                    "Cache split",
                    options=["all", "train", "val"],
                    index=0,
                    key="cand_cache_split",
                )
                include_snippet = expander.checkbox(
                    "Attach EFM snippet (3D plots)",
                    value=True,
                    key="cand_cache_attach_snippet",
                )
                include_gt_mesh = False
                if include_snippet:
                    include_gt_mesh = expander.checkbox(
                        "Include GT mesh",
                        value=False,
                        key="cand_cache_include_mesh",
                    )

                cache_dir_path = Path(cache_dir).expanduser()
                cache_ds = None
                cache_len = 0
                cache_error = None
                if not cache_dir_path.exists():
                    expander.warning("Offline cache directory does not exist.")
                else:
                    cache_cfg = OracleRriCacheDatasetConfig(
                        cache=OracleRriCacheConfig(cache_dir=cache_dir_path, paths=paths),
                        split=cache_split,
                        include_efm_snippet=include_snippet,
                        include_gt_mesh=include_gt_mesh,
                        load_backbone=False,
                        load_candidates=True,
                        load_depths=True,
                        load_candidate_pcs=True,
                        return_format="cache_sample",
                    )
                    cache_ds, cache_len, cache_error = _load_offline_cache_dataset(cache_cfg)
                    if cache_error is not None:
                        expander.warning(cache_error)

                cache_state = st.session_state.setdefault("cand_offline_cache", {})
                cache_idx_default = int(cache_state.get("cache_idx", 0) or 0)
                max_cache_idx = max(cache_len - 1, 0)
                cache_idx_default = min(cache_idx_default, max_cache_idx)
                next_cached = expander.button("Next cached sample", key="cand_cache_next")
                if next_cached and cache_len > 0:
                    cache_idx_default = (cache_idx_default + 1) % cache_len
                    st.session_state["cand_cache_index"] = cache_idx_default
                cache_idx = expander.number_input(
                    "Cache index",
                    min_value=0,
                    max_value=max_cache_idx,
                    value=int(cache_idx_default),
                    step=1,
                    key="cand_cache_index",
                )
                cache_state["cache_idx"] = int(cache_idx)
                if cache_len > 0:
                    expander.caption(f"Cached samples: {cache_len}")
                elif cache_ds is not None and cache_error is None and cache_dir_path.exists():
                    expander.warning("Offline cache is empty.")

                if cache_ds is None or cache_len == 0:
                    st.warning("Offline cache unavailable; showing online candidates.")
                    sample = controller.get_sample(force=False)
                    candidates = controller.get_candidates(force=False)
                else:
                    cache_sample = cache_ds[int(cache_idx)]
                    cache_cand_cfg = _candidate_cfg_from_cache(cache_ds.metadata.labeler_config)
                    sample = cache_sample.efm_snippet_view
                    candidates = cache_sample.candidates
                    cand_cfg = cache_cand_cfg or cand_cfg_prev
                    source_caption = f"Offline cache sample: {cache_sample.scene_id}:{cache_sample.snippet_id}"
                    source_note = "Online candidate config is ignored for offline cache samples."
            else:
                with st.sidebar.form("cand_form"):
                    cand_cfg = candidate_config_ui(
                        cand_cfg_prev,
                        st.sidebar,
                        is_debug=cand_cfg_prev.is_debug,
                        verbosity=console.verbosity,
                    )
                    refresh_cand = st.form_submit_button("Run / refresh candidates")
                state.labeler_cfg = state.labeler_cfg.model_copy(
                    update={"generator": cand_cfg},
                )
                sample = controller.get_sample(force=False)
                if refresh_cand or state.candidates.candidates is not None:
                    candidates = controller.get_candidates(force=refresh_cand)
                else:
                    store_state(state)
                    st.header("Candidate Poses")
                    st.info("No candidate batch in the session.")
                    return

            store_state(state)

            render_candidates_page(
                sample,
                candidates,
                cand_cfg,
                source_caption=source_caption,
                source_note=source_note,
            )

        def _page_renders() -> None:
            try:
                from .panels import render_depth_page
            except Exception as exc:
                _render_page_import_error("Candidate Renders page", exc)
                return

            with st.sidebar.form("depth_form"):
                depth_cfg_prev = state.labeler_cfg.depth
                depth_cfg = renderer_config_ui(
                    depth_cfg_prev,
                    st.sidebar,
                    is_debug=depth_cfg_prev.is_debug,
                    verbosity=console.verbosity,
                )
                stride_prev = int(state.labeler_cfg.backprojection_stride)
                stride = st.slider(
                    "Backprojection stride",
                    1,
                    32,
                    stride_prev,
                    step=1,
                    key="depth_stride",
                )
                refresh_depth = st.form_submit_button("Run / refresh renders")
            state.labeler_cfg = state.labeler_cfg.model_copy(
                update={
                    "depth": depth_cfg,
                    "backprojection_stride": int(stride),
                },
            )

            depth_batch, pcs = controller.get_renders(force=refresh_depth)
            sample = controller.get_sample(force=False)
            store_state(state)

            render_depth_page(sample, depth_batch, pcs=pcs)

        def _page_rri() -> None:
            try:
                from .panels import render_rri_page
            except Exception as exc:
                _render_page_import_error("RRI page", exc)
                return

            with st.sidebar.form("rri_form"):
                labeler_cfg_prev = state.labeler_cfg
                oracle_cfg_prev = labeler_cfg_prev.oracle
                oracle_cfg = oracle_config_ui(oracle_cfg_prev, st.sidebar)
                stride = st.slider(
                    "Backprojection stride",
                    1,
                    32,
                    int(labeler_cfg_prev.backprojection_stride),
                    step=1,
                    key="rri_stride",
                )
                refresh_rri = st.form_submit_button("Run / refresh RRI")

            # Always plot on CPU to keep Plotly conversions predictable.
            labeler_cfg = state.labeler_cfg.model_copy(
                update={
                    "oracle": oracle_cfg,
                    "backprojection_stride": int(stride),
                    "output_device": "cpu",
                },
            )
            state.labeler_cfg = labeler_cfg

            sample = controller.get_sample(force=False)
            depths, pcs, rri = controller.run_labeler(force=refresh_rri)
            store_state(state)

            render_rri_page(sample, depths, pcs, rri)

        def _page_vin() -> None:
            try:
                from .panels import render_vin_diagnostics_page
            except Exception as exc:
                _render_page_import_error("VIN Diagnostics page", exc)
                return

            render_vin_diagnostics_page()

        def _page_offline_stats() -> None:
            try:
                from .panels import render_offline_stats_page
            except Exception as exc:
                _render_page_import_error("Offline Stats page", exc)
                return

            render_offline_stats_page()

        def _page_rri_binning() -> None:
            try:
                from .panels import render_rri_binning_page
            except Exception as exc:
                _render_page_import_error("RRI Binning page", exc)
                return

            render_rri_binning_page()

        def _page_wandb() -> None:
            try:
                from .panels import render_wandb_analysis_page
            except Exception as exc:
                _render_page_import_error("W&B Analysis page", exc)
                return

            render_wandb_analysis_page()

        def _page_testing_attr() -> None:
            try:
                from .panels import render_testing_attribution_page
            except Exception as exc:
                _render_page_import_error("Testing & Attribution page", exc)
                return

            render_testing_attribution_page()

        def _page_optuna_sweep() -> None:
            try:
                from aria_nbv.app.panels.optuna_sweep import render_optuna_sweep_page
            except Exception as exc:
                _render_page_import_error("Optuna Sweep page", exc)
                return

            render_optuna_sweep_page()

        def _page_data_safe() -> None:
            _run_safe_page("Data page", _page_data)

        def _page_candidates_safe() -> None:
            _run_safe_page("Candidate Poses page", _page_candidates)

        def _page_renders_safe() -> None:
            _run_safe_page("Candidate Renders page", _page_renders)

        def _page_rri_safe() -> None:
            _run_safe_page("RRI page", _page_rri)

        def _page_vin_safe() -> None:
            _run_safe_page("VIN Diagnostics page", _page_vin)

        def _page_wandb_safe() -> None:
            _run_safe_page("W&B Analysis page", _page_wandb)

        def _page_optuna_sweep_safe() -> None:
            _run_safe_page("Optuna Sweep page", _page_optuna_sweep)

        def _page_testing_attr_safe() -> None:
            _run_safe_page("Testing & Attribution page", _page_testing_attr)

        def _page_rri_binning_safe() -> None:
            _run_safe_page("RRI Binning page", _page_rri_binning)

        def _page_offline_stats_safe() -> None:
            _run_safe_page("Offline Stats page", _page_offline_stats)

        st.navigation(
            [
                st.Page(_page_data_safe, title="Data", default=True),
                st.Page(
                    _page_candidates_safe,
                    title="Candidate Poses",
                    url_path="candidate-poses",
                ),
                st.Page(
                    _page_renders_safe,
                    title="Candidate Renders",
                    url_path="candidate-renders",
                ),
                st.Page(_page_rri_safe, title="RRI", url_path="rri"),
                st.Page(
                    _page_vin_safe,
                    title="VIN Diagnostics",
                    url_path="vin-diagnostics",
                ),
                st.Page(
                    _page_wandb_safe,
                    title="W&B Analysis",
                    url_path="wandb-analysis",
                ),
                st.Page(
                    _page_optuna_sweep_safe,
                    title="Optuna Sweep",
                    url_path="optuna-sweep",
                ),
                st.Page(
                    _page_testing_attr_safe,
                    title="Testing & Attribution",
                    url_path="testing-attribution",
                ),
                st.Page(
                    _page_rri_binning_safe,
                    title="RRI Binning",
                    url_path="rri-binning",
                ),
                st.Page(
                    _page_offline_stats_safe,
                    title="Offline Stats",
                    url_path="offline-stats",
                ),
            ],
            position="top",
        ).run()


# Resolve forward refs now that NbvStreamlitApp is defined
NbvStreamlitAppConfig.model_rebuild(
    _types_namespace={"NbvStreamlitApp": NbvStreamlitApp},
)


__all__ = ["NbvStreamlitApp"]
