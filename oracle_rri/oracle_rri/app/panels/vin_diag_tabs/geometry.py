"""Geometry tab for VIN diagnostics."""

from __future__ import annotations

import streamlit as st
import torch

from ....configs import PathConfig
from ....rri_metrics.coral import coral_loss
from ....vin.plotting import build_alignment_figures, build_geometry_overview_figure
from ..common import _info_popover
from ..data import scene_plot_options_ui
from ..offline_cache_utils import _load_efm_snippet_for_cache
from .context import VinDiagContext


def render_geometry_tab(ctx: VinDiagContext) -> None:
    """Render the Geometry tab.

    Args:
        ctx: Shared VIN diagnostics context.
    """
    state = ctx.state
    debug = ctx.debug
    pred = ctx.pred
    batch = ctx.batch
    cfg = ctx.cfg

    _info_popover(
        "geometry overview",
        "Combines candidate centers, trajectory, semidense points, and GT mesh "
        "in the same world frame. Frusta and GT OBBs help verify that the "
        "candidate poses align with scene geometry and annotations.",
    )
    snippet_view = batch.efm_snippet_view
    if snippet_view is None and ctx.use_offline_cache and ctx.attach_snippet:
        snippet_key = f"{batch.scene_id}:{batch.snippet_id}"
        if state.offline_snippet_key != snippet_key or state.offline_snippet is None:
            with st.spinner("Loading EFM snippet for geometry..."):
                try:
                    cache_ds = state.offline_cache
                    dataset_payload = cache_ds.metadata.dataset_config if cache_ds else None
                    paths = cfg.paths if isinstance(cfg.paths, PathConfig) else PathConfig()
                    snippet_view = _load_efm_snippet_for_cache(
                        scene_id=batch.scene_id,
                        snippet_id=batch.snippet_id,
                        dataset_payload=dataset_payload,
                        device="cpu",
                        paths=paths,
                        include_gt_mesh=ctx.include_gt_mesh,
                    )
                    state.offline_snippet_key = snippet_key
                    state.offline_snippet = snippet_view
                    state.offline_snippet_error = None
                except Exception as exc:  # pragma: no cover - IO guard
                    state.offline_snippet_key = snippet_key
                    state.offline_snippet = None
                    state.offline_snippet_error = f"{type(exc).__name__}: {exc}"
        else:
            snippet_view = state.offline_snippet
        if snippet_view is not None:
            batch.efm_snippet_view = snippet_view

    if snippet_view is None:
        if state.offline_snippet_error:
            st.warning(state.offline_snippet_error)
        st.info(
            "Geometry plots require raw EFM snippets; enable 'Attach EFM snippet' or use online data.",
        )
        return

    cam_choice, plot_opts = scene_plot_options_ui(
        snippet_view,
        key_prefix="vin_geom",
    )
    frustum_indices = plot_opts.frustum_frame_indices[-1:] if plot_opts.frustum_frame_indices else []
    axis_expander = st.expander("Axes & candidate settings", expanded=False)
    with axis_expander:
        show_reference_axes = st.checkbox(
            "Show reference axes",
            value=True,
            key="vin_geom_ref_axes",
        )
        show_voxel_axes = st.checkbox(
            "Show voxel axes",
            value=True,
            key="vin_geom_voxel_axes",
        )
        candidate_pose_mode = st.selectbox(
            "Candidate coordinates",
            options=["ref rig", "world cam", "rig (raw)"],
            index=0,
            key="vin_geom_candidate_pose_mode",
        )
        candidate_color_mode = st.selectbox(
            "Candidate color mode",
            options=["valid fraction", "solid", "loss"],
            index=0,
            key="vin_geom_candidate_color_mode",
        )
        candidate_color = "#ffd966"
        candidate_colorscale = "Viridis"
        if candidate_color_mode == "solid":
            candidate_color = st.color_picker(
                "Candidate color",
                value="#ffd966",
                key="vin_geom_candidate_color",
            )
        else:
            candidate_colorscale = st.selectbox(
                "Candidate colorscale",
                options=["Viridis", "Cividis", "Plasma", "Turbo", "Magma"],
                index=0,
                key="vin_geom_candidate_colorscale",
            )

    candidate_frusta_indices: list[int] = []
    candidate_frusta_scale = 0.5
    candidate_frusta_color = "#ff4d4d"
    candidate_frusta_show_axes = False
    candidate_frusta_show_center = False
    candidate_frusta_camera = cam_choice
    candidate_frusta_frame_index = frustum_indices[0] if frustum_indices else 0
    with st.expander("Candidate frusta", expanded=False):
        show_candidate_frusta = st.checkbox(
            "Show candidate frusta",
            value=False,
            key="vin_geom_candidate_frusta",
        )
        if show_candidate_frusta:
            options = list(range(ctx.num_candidates))
            default = options[: min(4, len(options))] if options else []
            candidate_frusta_indices = st.multiselect(
                "Candidate indices",
                options=options,
                default=default,
                key="vin_geom_candidate_frusta_indices",
            )
            candidate_frusta_camera = st.selectbox(
                "Candidate frusta camera",
                options=["rgb", "slam-l", "slam-r"],
                index=0,
                key="vin_geom_candidate_frusta_camera",
            )
            candidate_frusta_scale = float(
                st.slider(
                    "Candidate frusta scale",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    step=0.05,
                    key="vin_geom_candidate_frusta_scale",
                ),
            )
            candidate_frusta_color = st.color_picker(
                "Candidate frusta color",
                value="#ff4d4d",
                key="vin_geom_candidate_frusta_color",
            )
            candidate_frusta_show_axes = st.checkbox(
                "Show candidate axes",
                value=False,
                key="vin_geom_candidate_frusta_axes",
            )
            candidate_frusta_show_center = st.checkbox(
                "Show candidate center",
                value=False,
                key="vin_geom_candidate_frusta_center",
            )
    backbone_fields: list[str] = []
    backbone_threshold = 0.5
    backbone_max_points = 40_000
    backbone_colorscale = "Viridis"
    available_fields: list[str] = []
    if debug.backbone_out is not None:
        available_fields = [
            name for name in ("occ_pr", "occ_input", "counts") if getattr(debug.backbone_out, name, None) is not None
        ]
    with st.expander("Backbone evidence overlay", expanded=False):
        show_backbone = st.checkbox(
            "Overlay backbone evidence",
            value=False,
            key="vin_geom_backbone_enable",
        )
        if not available_fields:
            st.info("Backbone evidence not available in debug outputs.")
        elif show_backbone:
            backbone_fields = st.multiselect(
                "Backbone fields",
                options=available_fields,
                default=available_fields,
                key="vin_geom_backbone_fields",
            )
            backbone_colorscale = st.selectbox(
                "Evidence colorscale",
                options=["Viridis", "Cividis", "Plasma", "Turbo", "Magma"],
                index=0,
                key="vin_geom_backbone_colorscale",
            )
            backbone_threshold = float(
                st.slider(
                    "Evidence threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.01,
                    key="vin_geom_backbone_threshold",
                ),
            )
            backbone_max_points = int(
                st.slider(
                    "Max evidence points",
                    min_value=1000,
                    max_value=200000,
                    value=40000,
                    step=1000,
                    key="vin_geom_backbone_max_points",
                ),
            )
    candidate_loss: torch.Tensor | None = None
    if candidate_color_mode == "loss":
        loss_error = None
        binner = getattr(state.module, "_binner", None)
        if binner is None:
            loss_error = "RRI binner unavailable; cannot compute loss hue."
        elif batch.rri is None:
            loss_error = "Loss hue requires oracle RRI labels."
        else:
            try:
                with torch.no_grad():
                    logits = pred.logits
                    rri = batch.rri.to(device=logits.device)
                    rri_flat = rri.reshape(-1)
                    mask = torch.isfinite(rri_flat)
                    if mask.any():
                        labels = binner.transform(rri_flat)
                        loss_per = coral_loss(
                            logits.reshape(-1, logits.shape[-1])[mask],
                            labels[mask],
                            num_classes=int(binner.num_classes),
                            reduction="none",
                        )
                        loss_flat = torch.full(
                            (rri_flat.numel(),),
                            float("nan"),
                            device=logits.device,
                            dtype=torch.float32,
                        )
                        loss_flat[mask] = loss_per
                        candidate_loss = loss_flat.reshape(rri.shape)
                    else:
                        loss_error = "Loss hue requires finite RRI labels."
            except Exception as exc:  # pragma: no cover - UI guard
                loss_error = f"{type(exc).__name__}: {exc}"
        if loss_error:
            st.info(loss_error)
    # apply_cw90_correction = False
    # if state.module is not None and hasattr(state.module, "vin"):
    #     apply_cw90_correction = bool(
    #         getattr(getattr(state.module.vin, "config", None), "apply_cw90_correction", False),
    #     )

    reference_pose_world_rig = batch.reference_pose_world_rig
    candidate_poses_world_cam = batch.candidate_poses_world_cam
    # if apply_cw90_correction:
    #     reference_pose_world_rig = rotate_yaw_cw90(
    #         reference_pose_world_rig,
    #     )
    #     candidate_poses_world_cam = rotate_yaw_cw90(
    #         candidate_poses_world_cam,
    #     )

    display_rotate_yaw_cw90 = True

    st.plotly_chart(
        build_geometry_overview_figure(
            debug,
            snippet=snippet_view,
            reference_pose_world_rig=reference_pose_world_rig,
            max_candidates=512,
            show_scene_bounds=plot_opts.show_scene_bounds,
            show_crop_bounds=plot_opts.show_crop_bounds,
            show_frustum=plot_opts.show_frustum,
            frustum_camera=cam_choice,
            frustum_frame_indices=frustum_indices,
            frustum_scale=plot_opts.frustum_scale,
            show_gt_obbs=plot_opts.show_gt_obbs,
            gt_timestamp=plot_opts.gt_timestamp,
            semidense_mode=plot_opts.semidense_mode,
            max_sem_points=plot_opts.max_sem_points,
            show_trajectory=plot_opts.mark_first_last,
            mark_first_last=plot_opts.mark_first_last,
            show_reference_axes=show_reference_axes,
            show_voxel_axes=show_voxel_axes,
            display_rotate_yaw_cw90=display_rotate_yaw_cw90,
            candidate_pose_mode="ref_rig"
            if candidate_pose_mode == "ref rig"
            else "world_cam"
            if candidate_pose_mode == "world cam"
            else "rig",
            candidate_poses_world_cam=candidate_poses_world_cam,
            candidate_color_mode="solid"
            if candidate_color_mode == "solid"
            else "loss"
            if candidate_color_mode == "loss"
            else "valid_fraction",
            candidate_color=candidate_color,
            candidate_colorscale=candidate_colorscale,
            candidate_loss=candidate_loss,
            candidate_frusta_indices=candidate_frusta_indices,
            candidate_frusta_camera=candidate_frusta_camera,
            candidate_frusta_frame_index=candidate_frusta_frame_index,
            candidate_frusta_scale=candidate_frusta_scale,
            candidate_frusta_color=candidate_frusta_color,
            candidate_frusta_show_axes=candidate_frusta_show_axes,
            candidate_frusta_show_center=candidate_frusta_show_center,
            backbone_fields=backbone_fields,
            backbone_occ_threshold=backbone_threshold,
            backbone_max_points=backbone_max_points,
            backbone_colorscale=backbone_colorscale,
        ),
        width="stretch",
        key="vin_geometry_overview",
    )

    if ctx.has_tokens:
        log1p_align_counts = st.checkbox(
            "Log1p alignment histogram counts",
            value=False,
            key="vin_geom_align_log1p",
        )
        alignment_figs = build_alignment_figures(
            debug,
            log1p_counts=log1p_align_counts,
        )
        for key, fig in alignment_figs.items():
            st.plotly_chart(fig, width="stretch", key=f"vin_align_{key}")


__all__ = ["render_geometry_tab"]
