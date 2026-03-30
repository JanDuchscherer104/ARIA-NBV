"""VIN diagnostics panel."""

from __future__ import annotations

import traceback
from pathlib import Path

import streamlit as st
import torch

from ...configs import PathConfig
from ...data_handling import (
    VinOracleBatch,
    VinSnippetCacheConfig,
    VinSnippetCacheDatasetConfig,
    empty_vin_snippet,
    read_vin_snippet_cache_metadata,
)
from ...utils import Stage
from ..state import VIN_DIAG_STATE_KEY, get_vin_state
from ..state_types import config_signature
from .offline_cache_utils import (
    _load_efm_snippet_for_cache,
    _prepare_offline_cache_dataset,
)
from .vin_diag_tabs import (
    VinDiagContext,
    render_bin_values_tab,
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
    _has_backbone_obbs,
    _run_vin_debug,
    _should_fetch_vin_snippet,
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
    batch_size = 1
    snippet_source = "VinSnippetCacheDataset"
    require_vin_snippet = False
    vin_snippet_cache_dir = None
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

        data_source = st.selectbox(
            "Data source",
            options=["offline cache", "online (oracle labeler)"],
            index=0,
        )
        use_offline_cache = data_source == "offline cache"
        cache_dir = None
        if use_offline_cache:
            cache_dir = st.text_input(
                "Offline cache dir",
                value=str(PathConfig().offline_cache_dir),
            )
            attach_snippet = st.checkbox(
                "Attach snippet (geometry plots)",
                value=True,
            )
            snippet_source = st.selectbox(
                "Snippet source",
                options=[
                    "OracleRriCacheDataset (EFM)",
                    "VinSnippetCacheDataset",
                ],
                index=1,
            )
            if snippet_source == "VinSnippetCacheDataset":
                default_vin_cache = (
                    paths.offline_cache_dir or (paths.data_root / "oracle_rri_cache")
                ) / "vin_snippet_cache"
                vin_snippet_cache_dir = st.text_input(
                    "VIN snippet cache dir",
                    value=str(default_vin_cache),
                )
                require_vin_snippet = st.checkbox(
                    "Require VIN snippet entries",
                    value=False,
                    help=("Validate that VIN snippet cache has entries even when snippets are not attached."),
                )
            if attach_snippet and snippet_source == "OracleRriCacheDataset (EFM)":
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
        batch_size = int(
            st.number_input(
                "Batch size",
                min_value=1,
                max_value=16,
                value=1,
                step=1,
            )
        )
        run = st.form_submit_button("Run / refresh VIN diagnostics")

    if st.sidebar.button("Clear VIN cache"):
        st.session_state.pop(VIN_DIAG_STATE_KEY, None)
        st.rerun()

    if use_offline_cache and snippet_source == "OracleRriCacheDataset (EFM)" and batch_size > 1:
        st.sidebar.warning(
            "Batching >1 requires VinSnippetView (VIN snippet cache) or detached snippets. "
            "Falling back to batch size 1.",
        )
        batch_size = 1
    if not use_offline_cache and batch_size > 1:
        st.sidebar.warning("Online diagnostics currently run with batch size 1.")
        batch_size = 1

    cache_ds = None
    if use_offline_cache:
        paths = PathConfig()
        try:
            cache_ds = _prepare_offline_cache_dataset(
                cache_dir=cache_dir,
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
            if "vin_cache_index" not in st.session_state:
                st.session_state["vin_cache_index"] = int(state.offline_cache_idx)
            elif int(st.session_state["vin_cache_index"]) > max(0, cache_len - 1):
                st.session_state["vin_cache_index"] = max(0, cache_len - 1)
            advance = st.sidebar.button("Next cached batch")
            if advance:
                step = max(int(batch_size), 1)
                state.offline_cache_idx = (state.offline_cache_idx + step) % cache_len
                st.session_state["vin_cache_index"] = state.offline_cache_idx
                run = True
            cache_idx = st.sidebar.number_input(
                "Cache index",
                min_value=0,
                max_value=max(0, cache_len - 1),
                step=1,
                key="vin_cache_index",
            )
            state.offline_cache_idx = int(cache_idx)
            st.sidebar.caption(f"Cache samples: {cache_len}")
        else:
            st.sidebar.warning("Offline cache is empty or missing.")

    if run:
        try:
            include_efm_snippet = snippet_source == "OracleRriCacheDataset (EFM)"
            cfg = _build_experiment_config(
                toml_path=toml_path,
                stage=stage,
                use_offline_cache=data_source == "offline cache",
                cache_dir=cache_dir,
                include_efm_snippet=include_efm_snippet,
                include_gt_mesh=include_gt_mesh,
            )
            cfg_sig = config_signature(cfg)

            if state.cfg_sig != cfg_sig or state.module is None or state.datamodule is None:
                trainer, module, datamodule = cfg.setup_target(setup_stage=stage)
                _ = trainer  # unused but kept for future diagnostics
                state.cfg_sig = cfg_sig
                state.experiment = cfg
                state.module = module
                state.datamodule = datamodule

            assert state.module is not None and state.datamodule is not None
            if getattr(state.module, "_binner", None) is None and hasattr(state.module, "_load_binner_from_config"):
                try:
                    state.module._binner = state.module._load_binner_from_config()  # type: ignore[attr-defined]
                except Exception as exc:  # pragma: no cover - diagnostics guard
                    st.sidebar.warning(f"Failed to load RRI binner: {type(exc).__name__}: {exc}")
            try:
                if hasattr(state.module, "_maybe_init_bin_values"):
                    state.module._maybe_init_bin_values()  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - diagnostics guard
                st.sidebar.warning(f"Failed to init CORAL bin values: {type(exc).__name__}: {exc}")
            with st.spinner("Running oracle labeler + VIN forward..."):
                if use_offline_cache:
                    if cache_ds is None:
                        raise RuntimeError("Offline cache dataset is not available.")
                    cache_len = int(state.offline_cache_len or 0)
                    if cache_len == 0:
                        raise RuntimeError("Offline cache is empty.")
                    cache_idx = min(max(int(state.offline_cache_idx), 0), cache_len - 1)
                    step = max(int(batch_size), 1)
                    cache_indices = [(cache_idx + offset) % cache_len for offset in range(step)]
                    batches: list[VinOracleBatch] = []
                    drop_backbone_obbs = batch_size > 1
                    stripped_obbs = False

                    vin_snippet_ds = None
                    vin_snippet_extra_dim: int | None = None
                    use_vin_snippet_cache = snippet_source == "VinSnippetCacheDataset"
                    should_fetch_vin_snippet = _should_fetch_vin_snippet(
                        use_vin_snippet_cache=use_vin_snippet_cache,
                        attach_snippet=attach_snippet,
                        require_vin_snippet=require_vin_snippet,
                    )
                    if should_fetch_vin_snippet:
                        paths = cfg.paths if isinstance(cfg.paths, PathConfig) else PathConfig()
                        vin_cache_root = vin_snippet_cache_dir
                        if not vin_cache_root:
                            vin_cache_root = str(
                                (paths.offline_cache_dir or (paths.data_root / "oracle_rri_cache"))
                                / "vin_snippet_cache",
                            )
                        vin_cache_path = Path(vin_cache_root)
                        if (vin_cache_path / "vin_snippet_cache").exists():
                            vin_cache_path = vin_cache_path / "vin_snippet_cache"
                        vin_cfg = VinSnippetCacheDatasetConfig(
                            cache=VinSnippetCacheConfig(cache_dir=vin_cache_path, paths=paths),
                            map_location="cpu",
                        )
                        try:
                            meta = read_vin_snippet_cache_metadata(vin_cfg.cache.metadata_path)
                            vin_snippet_extra_dim = int(meta.include_inv_dist_std) + int(meta.include_obs_count)
                        except FileNotFoundError:
                            vin_snippet_extra_dim = None
                        vin_sig = config_signature(vin_cfg)
                        if state.vin_snippet_cache_sig != vin_sig or state.vin_snippet_cache is None:
                            vin_snippet_ds = vin_cfg.setup_target()
                            state.vin_snippet_cache_sig = vin_sig
                            state.vin_snippet_cache = vin_snippet_ds
                            state.vin_snippet_cache_len = len(vin_snippet_ds)
                        else:
                            vin_snippet_ds = state.vin_snippet_cache

                    for idx in cache_indices:
                        cache_sample = cache_ds[idx]
                        snippet_view = cache_sample.efm_snippet_view
                        if use_vin_snippet_cache:
                            vin_snippet = None
                            missing_msg = None
                            if should_fetch_vin_snippet:
                                if vin_snippet_ds is None:
                                    raise RuntimeError("VIN snippet cache dataset is not available.")
                                vin_snippet = vin_snippet_ds.get_by_scene_snippet(
                                    scene_id=cache_sample.scene_id,
                                    snippet_id=cache_sample.snippet_id,
                                    map_location="cpu",
                                )
                                if vin_snippet is None:
                                    missing_msg = (
                                        "VIN snippet cache missing entry for "
                                        f"scene={cache_sample.scene_id} snippet={cache_sample.snippet_id}."
                                    )
                                    if require_vin_snippet:
                                        raise RuntimeError(missing_msg)
                            if vin_snippet is None:
                                msg = missing_msg or (
                                    "VIN snippet cache is unavailable for this sample; "
                                    "ensure the cache is built for the selected split."
                                )
                                if require_vin_snippet:
                                    raise RuntimeError(msg)
                                if vin_snippet_extra_dim is None:
                                    vin_snippet_extra_dim = 2
                                snippet_view = empty_vin_snippet(
                                    torch.device("cpu"),
                                    extra_dim=vin_snippet_extra_dim,
                                )
                                state.offline_snippet_error = f"{msg} (using empty VIN snippet)"
                            else:
                                snippet_view = vin_snippet
                                state.offline_snippet_error = None
                            state.offline_snippet = None
                            state.offline_snippet_key = None
                        elif attach_snippet:
                            snippet_key = f"{cache_sample.scene_id}:{cache_sample.snippet_id}"
                            if step == 1 and (
                                snippet_view is None
                                and (state.offline_snippet_key != snippet_key or state.offline_snippet is None)
                            ):
                                try:
                                    paths = cfg.paths if isinstance(cfg.paths, PathConfig) else PathConfig()
                                    snippet_view = _load_efm_snippet_for_cache(
                                        scene_id=cache_sample.scene_id,
                                        snippet_id=cache_sample.snippet_id,
                                        dataset_payload=cache_ds.metadata.dataset_config,
                                        device="cpu",
                                        paths=paths,
                                        include_gt_mesh=include_gt_mesh,
                                    )
                                    state.offline_snippet_key = snippet_key
                                    state.offline_snippet = snippet_view
                                    state.offline_snippet_error = None
                                except Exception as exc:  # pragma: no cover - IO guard
                                    state.offline_snippet_key = snippet_key
                                    state.offline_snippet = None
                                    state.offline_snippet_error = f"{type(exc).__name__}: {exc}"
                            if step == 1 and snippet_view is None:
                                snippet_view = state.offline_snippet
                            elif step == 1:
                                state.offline_snippet_key = snippet_key
                                state.offline_snippet = snippet_view
                                state.offline_snippet_error = None
                            if step > 1 and snippet_view is None:
                                paths = cfg.paths if isinstance(cfg.paths, PathConfig) else PathConfig()
                                snippet_view = _load_efm_snippet_for_cache(
                                    scene_id=cache_sample.scene_id,
                                    snippet_id=cache_sample.snippet_id,
                                    dataset_payload=cache_ds.metadata.dataset_config,
                                    device="cpu",
                                    paths=paths,
                                    include_gt_mesh=include_gt_mesh,
                                )
                        else:
                            snippet_view = None
                            state.offline_snippet_error = None
                            state.offline_snippet = None
                            state.offline_snippet_key = None
                        batches.append(
                            _vin_oracle_batch_from_cache(
                                cache_sample,
                                efm_snippet=snippet_view,
                                drop_backbone_obbs=drop_backbone_obbs,
                            ),
                        )
                        if drop_backbone_obbs and not stripped_obbs and _has_backbone_obbs(cache_sample.backbone_out):
                            stripped_obbs = True

                    if stripped_obbs:
                        st.sidebar.warning(
                            "OBB outputs are dropped for batched VIN diagnostics. "
                            "Use batch size 1 to keep OBB predictions.",
                        )

                    if len(batches) == 1:
                        batch = batches[0]
                    else:
                        batch = VinOracleBatch.collate(batches)
                else:
                    batch = next(datamodule.iter_oracle_batches(stage=stage))

                pred, debug = _run_vin_debug(state.module, batch)
                if use_offline_cache and not attach_snippet:
                    batch.efm_snippet_view = None

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
    elif getattr(debug, "voxel_valid_frac", None) is not None:
        valid_frac = debug.voxel_valid_frac.reshape(-1)
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
        tab_bin_values,
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
            "Bin Values",
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

    with tab_bin_values:
        render_bin_values_tab(ctx)
