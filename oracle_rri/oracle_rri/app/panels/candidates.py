"""Candidate pose panel rendering."""

from __future__ import annotations

import streamlit as st
import torch

from ...data import EfmSnippetView
from ...pose_generation import CandidateViewGeneratorConfig
from ...pose_generation.plotting import (
    CandidatePlotBuilder,
    _euler_histogram,
    plot_direction_marginals,
    plot_direction_polar,
    plot_direction_sphere,
    plot_candidate_centers_simple,
    plot_candidate_frusta_simple,
    plot_euler_reference,
    plot_euler_world,
    plot_min_distance_to_mesh,
    plot_path_collision_segments,
    plot_position_polar,
    plot_position_sphere,
    plot_radius_hist,
    plot_rule_masks,
    plot_rule_rejection_bar,
)
from ...pose_generation.types import CandidateSamplingResult
from ...pose_generation.utils import (
    rejected_pose_tensor,
    stats_to_markdown_table,
    summarise_dirs_ref,
    summarise_offsets_ref,
)
from ...utils.frames import world_up_tensor
from .common import _info_popover, _pretty_label


def render_candidates_page(
    sample: EfmSnippetView | None,
    candidates: CandidateSamplingResult,
    cand_cfg: CandidateViewGeneratorConfig | None,
    *,
    source_caption: str | None = None,
    source_note: str | None = None,
) -> None:
    st.header("Candidate Poses")

    if source_caption:
        st.caption(source_caption)
    if source_note:
        st.caption(source_note)

    shell_poses = candidates.shell_poses
    mask_valid = candidates.mask_valid

    cam_label = cand_cfg.camera_label if cand_cfg is not None else "cached"
    has_snippet = sample is not None

    tab_pos, tab_frusta = st.tabs(["Positions (3D)", "Frusta (3D)"])

    with tab_pos:
        _info_popover(
            "candidate positions",
            "Candidate centers are sampled around the reference pose according "
            "to the sampling shell (radius, azimuth, elevation). Only valid "
            "candidates (after rule filtering) are shown in blue. The reference "
            "axes show the sampling frame (gravity-aligned when enabled) to "
            "stay symmetric with the candidate cloud.",
        )
        if has_snippet:
            cand_fig = (
                CandidatePlotBuilder.from_candidates(
                    sample,
                    candidates,
                    title=_pretty_label(f"Candidate positions ({cam_label})"),
                )
                .add_mesh()
                .add_candidate_cloud(use_valid=True, color="royalblue", size=4, opacity=0.7)
                .add_reference_axes(display_rotate=False)
            ).finalize()
            st.plotly_chart(cand_fig, width="stretch")
        else:
            st.warning("EFM snippet not attached; rendering candidates without mesh.")
            st.plotly_chart(
                plot_candidate_centers_simple(
                    candidates,
                    title=_pretty_label("Candidate positions (cached)"),
                ),
                width="stretch",
            )

    offsets_ref, dirs_ref = candidates.get_offsets_and_dirs_ref(display_rotate=False)

    with tab_frusta:
        if mask_valid is None or mask_valid.sum() == 0:
            st.warning("All candidates were rejected; frustum plot omitted.")
        else:
            _info_popover(
                "candidate frusta",
                "Frusta visualize the candidate camera extrinsics with the "
                "selected camera intrinsics. Scale is a display-only factor "
                "and does not change rendering or scoring.",
            )
            opt_col1, opt_col2 = st.columns(2)
            with opt_col1:
                frustum_scale = st.slider(
                    "Frustum scale",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key="cand_frustum_scale",
                )
            with opt_col2:
                max_frustums = st.slider(
                    "Max frustums",
                    min_value=1,
                    max_value=24,
                    value=6,
                    step=1,
                    key="cand_max_frustums",
                )

            if has_snippet:
                frust_fig = (
                    CandidatePlotBuilder.from_candidates(
                        sample,
                        candidates,
                        title=_pretty_label(f"Candidate frusta ({cam_label})"),
                    )
                    .add_mesh()
                    .add_candidate_cloud(
                        use_valid=True,
                        color="royalblue",
                        size=3,
                        opacity=0.35,
                    )
                    .add_candidate_frusta(
                        scale=float(frustum_scale),
                        color="crimson",
                        name="Frustum",
                        max_frustums=int(max_frustums),
                        include_axes=False,
                        include_center=False,
                        display_rotate=False,
                    )
                    .add_reference_axes(display_rotate=False)
                ).finalize()
                st.plotly_chart(frust_fig, width="stretch")
            else:
                st.warning("EFM snippet not attached; rendering frusta without mesh.")
                st.plotly_chart(
                    plot_candidate_frusta_simple(
                        candidates,
                        scale=float(frustum_scale),
                        max_frustums=int(max_frustums),
                    ),
                    width="stretch",
                )

    with st.expander("Distributions & Diagnostics", expanded=False):
        _info_popover(
            "candidate diagnostics",
            "Offsets and directions are expressed in the reference rig frame. "
            "These plots diagnose sampling coverage, symmetry, and any bias "
            "introduced by constraints or collision rules.",
        )
        fixed_ranges = st.checkbox(
            "Clamp axes to standard ranges",
            value=True,
            key="cand_angles_fixed_ranges",
        )
        diag_offsets, diag_dirs, diag_rules, diag_rejected = st.tabs(
            ["Offsets", "Directions", "Rules", "Rejected"],
        )

        with diag_offsets:
            _info_popover(
                "offsets",
                "Offsets are candidate translations relative to the reference pose. "
                "Polar plots show azimuth/elevation of the offset direction; the "
                "radius histogram shows the sampled distance distribution.",
            )
            st.markdown(
                stats_to_markdown_table(
                    summarise_offsets_ref(offsets_ref),
                    header=None,
                ),
            )
            offsets_np = offsets_ref.cpu().numpy()
            colp1, colp2 = st.columns(2)
            with colp1:
                st.plotly_chart(
                    plot_position_polar(
                        offsets_np,
                        title=_pretty_label("Offsets from reference pose (az/elev)"),
                        fixed_ranges=fixed_ranges,
                    ),
                    width="stretch",
                )
            with colp2:
                st.plotly_chart(
                    plot_position_sphere(offsets_np, show_axes=True),
                    width="stretch",
                )
            st.plotly_chart(plot_radius_hist(offsets_np), width="stretch")

        with diag_dirs:
            _info_popover(
                "directions",
                "View directions are unit forward vectors in the reference frame. "
                "Marginals expose angular coverage; sphere plots reveal anisotropy. "
                "Euler plots are shown in both world and reference frames to "
                "highlight frame-dependent interpretations.",
            )
            st.markdown(
                stats_to_markdown_table(summarise_dirs_ref(dirs_ref), header=None),
            )
            dirs_np = dirs_ref.cpu().numpy()

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    plot_direction_polar(
                        dirs_np,
                        title=_pretty_label("View directions (reference frame)"),
                        fixed_ranges=fixed_ranges,
                    ),
                    width="stretch",
                )
            with col2:
                st.plotly_chart(
                    plot_direction_sphere(
                        dirs_np,
                        title=_pretty_label("View dirs on unit sphere"),
                        show_axes=True,
                    ),
                    width="stretch",
                )
            st.plotly_chart(
                plot_direction_marginals(
                    torch.as_tensor(dirs_np),
                    fixed_ranges=fixed_ranges,
                ),
                width="stretch",
            )

            r_wr = candidates.reference_pose.R
            if r_wr.ndim == 3:
                r_wr = r_wr[0]
            fwd_w = r_wr[:, 2].view(1, 3)
            up_w = r_wr[:, 1].view(1, 3)
            wup = world_up_tensor(device=fwd_w.device, dtype=fwd_w.dtype).view(1, 3)
            left0 = torch.cross(wup, fwd_w, dim=-1)
            left0_norm = left0.norm(dim=-1, keepdim=True)
            if float(left0_norm.item()) < 1e-6:
                alt = torch.tensor(
                    [1.0, 0.0, 0.0],
                    device=fwd_w.device,
                    dtype=fwd_w.dtype,
                ).view(1, 3)
                left0 = torch.cross(alt, fwd_w, dim=-1)
                left0_norm = left0.norm(dim=-1, keepdim=True)
                if float(left0_norm.item()) < 1e-6:
                    alt2 = torch.tensor(
                        [0.0, 1.0, 0.0],
                        device=fwd_w.device,
                        dtype=fwd_w.dtype,
                    ).view(1, 3)
                    left0 = torch.cross(alt2, fwd_w, dim=-1)
                    left0_norm = left0.norm(dim=-1, keepdim=True)
            left0 = left0 / left0_norm.clamp_min(1e-6)
            up0 = torch.cross(fwd_w, left0, dim=-1)
            sin_roll = (fwd_w * torch.cross(up0, up_w, dim=-1)).sum(dim=-1)
            cos_roll = (up0 * up_w).sum(dim=-1)
            roll = torch.rad2deg(torch.atan2(sin_roll, cos_roll)).item()
            yaw = torch.rad2deg(torch.atan2(fwd_w[:, 0], fwd_w[:, 1])).item()
            pitch = torch.rad2deg(torch.asin(fwd_w[:, 2].clamp(-1.0, 1.0))).item()
            st.markdown(
                f"Reference pose view angles (world): yaw={yaw:.2f}°, pitch={pitch:.2f}°, roll={roll:.2f}°",
            )
            st.plotly_chart(
                plot_euler_world(candidates, fixed_ranges=fixed_ranges),
                width="stretch",
            )
            st.plotly_chart(
                plot_euler_reference(candidates, fixed_ranges=fixed_ranges),
                width="stretch",
            )
            delta = candidates.extras.get("view_dirs_delta") if hasattr(candidates, "extras") else None
            if delta is not None:
                r_delta = delta.R
                fwd = r_delta[:, :, 2]
                yaw_d = torch.rad2deg(torch.atan2(fwd[:, 0], fwd[:, 2])).cpu()
                pitch_d = torch.rad2deg(torch.asin(fwd[:, 1].clamp(-1.0, 1.0))).cpu()
                roll_d = torch.rad2deg(
                    torch.atan2(r_delta[:, 1, 0], r_delta[:, 1, 1]),
                ).cpu()
                st.markdown(
                    stats_to_markdown_table(
                        {
                            "yaw_delta_deg": {
                                "min": float(yaw_d.min()),
                                "max": float(yaw_d.max()),
                                "mean": float(yaw_d.mean()),
                                "std": float(yaw_d.std(unbiased=False)),
                            },
                            "pitch_delta_deg": {
                                "min": float(pitch_d.min()),
                                "max": float(pitch_d.max()),
                                "mean": float(pitch_d.mean()),
                                "std": float(pitch_d.std(unbiased=False)),
                            },
                            "roll_delta_deg": {
                                "min": float(roll_d.min()),
                                "max": float(roll_d.max()),
                                "mean": float(roll_d.mean()),
                                "std": float(roll_d.std(unbiased=False)),
                            },
                        },
                        header="Orientation jitter stats (delta, LUF yaw/pitch/roll)",
                    ),
                )
                st.plotly_chart(
                    _euler_histogram(
                        yaw_d,
                        pitch_d,
                        roll_d,
                        bins=90,
                        title=_pretty_label("Orientation jitter (delta, deg)"),
                        fixed_ranges=fixed_ranges,
                    ),
                    width="stretch",
                )

        with diag_rules:
            _info_popover(
                "rules",
                "Rule masks mark which candidates were rejected by each constraint "
                "(collision, distance-to-mesh, visibility, etc.). The bar plot "
                "summarizes rejection counts per rule to reveal dominant filters.",
            )
            masks = candidates.masks
            if has_snippet and isinstance(masks, dict) and len(masks) > 0 and shell_poses is not None:
                masks_tensor = torch.stack(list(masks.values()))
                mask_fig = plot_rule_masks(
                    snippet=sample,
                    shell_poses=shell_poses.tensor() if hasattr(shell_poses, "tensor") else shell_poses,
                    masks=masks_tensor,
                    rule_names=list(masks.keys()),
                )
                st.plotly_chart(mask_fig, width="stretch")
            elif not has_snippet:
                st.info("Attach an EFM snippet to visualize rule masks in 3D.")

            extras = candidates.extras if hasattr(candidates, "extras") else {}
            dist_min = extras.get("min_distance_to_mesh")
            path_collide = extras.get("path_collision_mask")

            if has_snippet and dist_min is not None:
                st.plotly_chart(
                    plot_min_distance_to_mesh(
                        snippet=sample,
                        candidates=candidates,
                        distances=dist_min,
                    ),
                    width="stretch",
                )
            if has_snippet and path_collide is not None:
                st.plotly_chart(
                    plot_path_collision_segments(
                        snippet=sample,
                        candidates=candidates,
                        collision_mask=path_collide,
                    ),
                    width="stretch",
                )

            st.plotly_chart(plot_rule_rejection_bar(candidates), width="stretch")

        with diag_rejected:
            _info_popover(
                "rejected",
                "Rejected poses are candidates that failed at least one rule. "
                "Plotting them can reveal systematic failures (e.g., walls, "
                "occlusions, or sampling bias).",
            )
            plot_rejected_only = st.checkbox(
                "Plot rejected poses only (if any)",
                value=False,
                key="cand_plot_rejected_only",
            )
            if plot_rejected_only:
                rejected_poses = rejected_pose_tensor(candidates)
                if rejected_poses is None:
                    st.info(
                        "No rejected poses to plot; all sampled candidates survived rule filtering.",
                    )
                elif has_snippet:
                    rej_fig = (
                        CandidatePlotBuilder.from_candidates(
                            sample,
                            candidates,
                            title=_pretty_label(
                                f"Rejected candidate positions ({rejected_poses.shape[0]})",
                            ),
                        )
                        .add_mesh()
                        .add_rejected_cloud()
                        .add_reference_axes()
                    ).finalize()
                    st.plotly_chart(rej_fig, width="stretch")
                else:
                    st.info("Attach an EFM snippet to render rejected poses in 3D.")


__all__ = ["render_candidates_page"]
