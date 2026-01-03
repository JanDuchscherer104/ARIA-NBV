"""VIN diagnostics panel."""

from __future__ import annotations

import traceback

import streamlit as st

from ...configs import PathConfig
from ...utils import Stage
from ..state import VIN_DIAG_STATE_KEY, get_vin_state
from ..state_types import config_signature
from .offline_cache_utils import (
    _load_efm_snippet_for_cache,
    _prepare_offline_cache_dataset,
)
from .vin_diag_tabs import (
    VinDiagContext,
    render_coral_tab,
    render_encodings_tab,
    render_evidence_tab,
    render_field_tab,
    render_geometry_tab,
    render_pose_tab,
    render_summary_tab,
    render_tokens_tab,
    render_transforms_tab,
)
from .vin_utils import (
    _build_experiment_config,
    _load_vin_module_from_checkpoint,
    _run_vin_debug,
    _vin_oracle_batch_from_cache,
)


def render_vin_diagnostics_page() -> None:
    """Render VIN diagnostics using AriaNBVExperimentConfig (independent from app pipeline)."""
    st.header("VIN Diagnostics")
    st.caption(
        "Run VIN forward_with_debug on oracle batches and inspect internal tensors.",
    )

    state = get_vin_state()

    run = False
    use_offline_cache = False
    attach_snippet = True
    include_gt_mesh = False
    with st.sidebar.form("vin_diag_form"):
        st.subheader("VIN Diagnostics")
        paths = PathConfig()
        config_dir = paths.configs_dir
        config_paths = sorted(
            config_dir.glob("*.toml"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        toml_options = ["(none)"] + [path.name for path in config_paths]
        toml_choice = st.selectbox(
            "Experiment config TOML (optional)",
            options=toml_options,
            index=0,
        )
        toml_path = None if toml_choice == "(none)" else str(config_dir / toml_choice)

        ckpt_dir = paths.checkpoints
        ckpt_paths = sorted(
            ckpt_dir.glob("*.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        use_checkpoint = st.checkbox("Use checkpoint", value=False)
        ckpt_path = None
        if use_checkpoint:
            if ckpt_paths:
                ckpt_choice = st.selectbox(
                    "Checkpoint",
                    options=[path.name for path in ckpt_paths],
                    index=0,
                )
                ckpt_path = ckpt_dir / ckpt_choice
            else:
                st.info(f"No checkpoints found in {ckpt_dir}.")
        data_source = st.selectbox(
            "Data source",
            options=["online (oracle labeler)", "offline cache"],
            index=0,
        )
        use_offline_cache = data_source == "offline cache"
        cache_dir = None
        map_location = "cpu"
        if use_offline_cache:
            cache_dir = st.text_input(
                "Offline cache dir",
                value=str(PathConfig().offline_cache_dir),
            )
            map_location = st.selectbox(
                "Cache map_location",
                options=["cpu", "cuda"],
                index=0,
            )
            attach_snippet = st.checkbox(
                "Attach EFM snippet (geometry plots)",
                value=True,
            )
            if attach_snippet:
                include_gt_mesh = st.checkbox(
                    "Include GT mesh",
                    value=False,
                    key="vin_diag_include_mesh",
                )
        stage = st.selectbox(
            "Stage",
            options=[Stage.TRAIN, Stage.VAL, Stage.TEST],
            format_func=lambda s: s.value,
        )
        run = st.form_submit_button("Run / refresh VIN diagnostics")

    if st.sidebar.button("Clear VIN cache"):
        st.session_state.pop(VIN_DIAG_STATE_KEY, None)
        st.rerun()

    cache_ds = None
    if use_offline_cache:
        paths = PathConfig()
        try:
            cache_ds = _prepare_offline_cache_dataset(
                cache_dir=cache_dir,
                map_location=map_location,
                paths=paths,
                state=state,
                stage=stage,
                include_efm_snippet=attach_snippet,
                include_gt_mesh=include_gt_mesh,
            )
        except Exception as exc:  # pragma: no cover - UI guard
            trace = traceback.format_exc()
            print(trace, flush=True)
            state.offline_cache_len = 0
            state.offline_cache = None
            state.offline_cache_sig = None
            st.sidebar.error(f"{type(exc).__name__}: {exc}")
            with st.sidebar.expander("Full traceback", expanded=False):
                st.code(trace, language="text")
            cache_ds = None
        cache_len = int(state.offline_cache_len or 0)
        if cache_len > 0:
            advance = st.sidebar.button("Next cached sample")
            if advance:
                state.offline_cache_idx = (state.offline_cache_idx + 1) % cache_len
                st.session_state["vin_cache_index"] = state.offline_cache_idx
                run = True
            cache_idx = st.sidebar.number_input(
                "Cache index",
                min_value=0,
                max_value=max(0, cache_len - 1),
                value=int(state.offline_cache_idx),
                step=1,
                key="vin_cache_index",
            )
            state.offline_cache_idx = int(cache_idx)
            st.sidebar.caption(f"Cache samples: {cache_len}")
        else:
            st.sidebar.warning("Offline cache is empty or missing.")

    if run:
        try:
            resolved_ckpt = None
            if use_checkpoint and ckpt_path is not None:
                try:
                    resolved_ckpt = PathConfig().resolve_checkpoint_path(ckpt_path)
                except Exception as exc:  # pragma: no cover - IO guard
                    st.sidebar.error(f"{type(exc).__name__}: {exc}")
                    resolved_ckpt = None

            cfg = _build_experiment_config(
                toml_path=toml_path,
                stage=stage,
                use_offline_cache=data_source == "offline cache",
                cache_dir=cache_dir,
                map_location=map_location,
                include_efm_snippet=attach_snippet,
                include_gt_mesh=include_gt_mesh,
            )
            cfg_sig = config_signature(cfg)
            if resolved_ckpt is not None:
                cfg_sig = f"{cfg_sig}|ckpt:{resolved_ckpt}"

            if state.cfg_sig != cfg_sig or state.module is None or state.datamodule is None:
                trainer, module, datamodule = cfg.setup_target(setup_stage=stage)
                _ = trainer  # unused but kept for future diagnostics
                state.cfg_sig = cfg_sig
                state.experiment = cfg
                state.module = module
                state.datamodule = datamodule
            if resolved_ckpt is not None:
                state.module = _load_vin_module_from_checkpoint(
                    checkpoint_path=resolved_ckpt,
                    device="cpu",
                )

            assert state.module is not None and state.datamodule is not None
            with st.spinner("Running oracle labeler + VIN forward..."):
                if use_offline_cache:
                    if cache_ds is None:
                        raise RuntimeError("Offline cache dataset is not available.")
                    cache_len = int(state.offline_cache_len or 0)
                    if cache_len == 0:
                        raise RuntimeError("Offline cache is empty.")
                    cache_idx = min(
                        max(int(state.offline_cache_idx), 0),
                        cache_len - 1,
                    )
                    cache_sample = cache_ds[cache_idx]
                    efm_snippet = cache_sample.efm_snippet_view
                    if attach_snippet:
                        snippet_key = f"{cache_sample.scene_id}:{cache_sample.snippet_id}"
                        if efm_snippet is None and (
                            state.offline_snippet_key != snippet_key or state.offline_snippet is None
                        ):
                            try:
                                paths = cfg.paths if isinstance(cfg.paths, PathConfig) else PathConfig()
                                efm_snippet = _load_efm_snippet_for_cache(
                                    scene_id=cache_sample.scene_id,
                                    snippet_id=cache_sample.snippet_id,
                                    dataset_payload=cache_ds.metadata.dataset_config,
                                    device=map_location,
                                    paths=paths,
                                    include_gt_mesh=include_gt_mesh,
                                )
                                state.offline_snippet_key = snippet_key
                                state.offline_snippet = efm_snippet
                                state.offline_snippet_error = None
                            except Exception as exc:  # pragma: no cover - IO guard
                                state.offline_snippet_key = snippet_key
                                state.offline_snippet = None
                                state.offline_snippet_error = f"{type(exc).__name__}: {exc}"
                        if efm_snippet is None:
                            efm_snippet = state.offline_snippet
                        else:
                            state.offline_snippet_key = snippet_key
                            state.offline_snippet = efm_snippet
                            state.offline_snippet_error = None
                    else:
                        state.offline_snippet_error = None
                        state.offline_snippet = None
                        state.offline_snippet_key = None
                    batch = _vin_oracle_batch_from_cache(
                        cache_sample,
                        efm_snippet=efm_snippet,
                    )
                else:
                    batch = next(datamodule.iter_oracle_batches(stage=stage))

                pred, debug = _run_vin_debug(state.module, batch)

            state.batch = batch
            state.pred = pred
            state.debug = debug
            state.error = None
        except Exception:  # pragma: no cover - UI error guard
            trace = traceback.format_exc()
            print(trace, flush=True)
            state.error = trace

    if state.error:
        st.error("VIN diagnostics failed. See traceback below.")
        st.code(state.error, language="text")
        return

    if state.debug is None or state.pred is None or state.batch is None or state.experiment is None:
        st.info("Run the VIN diagnostics to load a batch.")
        return

    debug = state.debug
    pred = state.pred
    batch = state.batch
    cfg = state.experiment

    num_candidates = int(debug.candidate_valid.shape[-1])
    valid_mask = debug.candidate_valid.reshape(-1)
    valid_count = int(valid_mask.sum().item())
    has_tokens = hasattr(debug, "token_valid")
    has_semidense_frustum = getattr(debug, "semidense_frustum", None) is not None
    if has_tokens:
        valid_frac = debug.token_valid.float().mean(dim=-1).reshape(-1)
        mean_valid_frac = f"{float(valid_frac.mean().item()):.3f}"
    else:
        valid_frac = debug.candidate_valid.float().reshape(-1)
        mean_valid_frac = "n/a"

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Candidates", num_candidates)
    col_b.metric("Valid candidates", f"{valid_count}/{num_candidates}")
    col_c.metric("Mean token valid frac", mean_valid_frac)

    snippet_label = (
        f"**Scene:** `{batch.scene_id}` &nbsp;&nbsp; "
        f"**Snippet:** `{batch.snippet_id}` &nbsp;&nbsp; "
        f"**Device:** `{debug.candidate_center_rig_m.device!s}`"
    )
    if use_offline_cache and state.offline_cache_len:
        snippet_label += f" &nbsp;&nbsp; **Cache idx:** `{state.offline_cache_idx}`"
    st.markdown(snippet_label)

    (
        tab_summary,
        tab_pose,
        tab_geometry,
        tab_field,
        tab_tokens,
        tab_evidence,
        tab_transforms,
        tab_concept,
        tab_coral,
    ) = st.tabs(
        [
            "Summary",
            "Pose Descriptor",
            "Geometry",
            "Field Slices",
            "Frustum Tokens",
            "Backbone Evidence",
            "Transforms",
            "FF Encodings",
            "CORAL / Ordinal",
        ],
    )

    ctx = VinDiagContext(
        state=state,
        debug=debug,
        pred=pred,
        batch=batch,
        cfg=cfg,
        use_offline_cache=use_offline_cache,
        attach_snippet=attach_snippet,
        include_gt_mesh=include_gt_mesh,
        has_tokens=has_tokens,
        has_semidense_frustum=has_semidense_frustum,
        num_candidates=num_candidates,
    )

    with tab_summary:
        render_summary_tab(ctx)

    with tab_pose:
        render_pose_tab(ctx)

    with tab_geometry:
        render_geometry_tab(ctx)

    with tab_field:
        render_field_tab(ctx)

    with tab_tokens:
        render_tokens_tab(ctx)

    with tab_evidence:
        render_evidence_tab(ctx)

    with tab_transforms:
        render_transforms_tab(ctx)

    with tab_concept:
        render_encodings_tab(ctx)

    with tab_coral:
        render_coral_tab(ctx)
