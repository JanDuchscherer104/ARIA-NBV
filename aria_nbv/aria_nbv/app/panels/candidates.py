"""Candidate pose panel rendering."""

from __future__ import annotations

from types import SimpleNamespace

import streamlit as st
import torch
from efm3d.aria.pose import PoseTW

from ...data_handling import EfmSnippetView
from ...pose_generation import (
    CandidateViewGeneratorConfig,
    CounterfactualPoseGeneratorConfig,
    CounterfactualSelectionPolicy,
)
from ...pose_generation.counterfactuals import CounterfactualRolloutResult
from ...pose_generation.plotting import (
    CandidatePlotBuilder,
    CounterfactualPlotBuilder,
    _euler_histogram,
    plot_candidate_centers_simple,
    plot_candidate_frusta_simple,
    plot_counterfactual_paths_simple,
    plot_counterfactual_step_simple,
    plot_direction_marginals,
    plot_direction_polar,
    plot_direction_sphere,
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
from ...utils import Console
from ...utils.frames import world_up_tensor
from ..state_types import config_signature, sample_key
from .common import _info_popover, _pretty_label, _report_exception, _strip_ansi


def _shell_offsets_dirs_ref(
    candidates: CandidateSamplingResult,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Offsets and forward directions for the full sampling shell in reference frame."""

    shell = candidates.shell_poses
    if shell is None or shell._data is None or shell._data.numel() == 0:
        return None
    ref_inv = candidates.reference_pose.inverse().to(shell.t.device)
    poses_ref_cam = ref_inv.compose(shell)
    offsets = poses_ref_cam.t.view(-1, 3)
    z_cam = (
        torch.tensor([0.0, 0.0, 1.0], device=offsets.device, dtype=offsets.dtype).view(1, 3).expand(offsets.shape[0], 3)
    )
    dirs = poses_ref_cam.rotate(z_cam).view(-1, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return offsets, dirs


def _pose_orthonormality_stats(pose: PoseTW) -> dict[str, float]:
    """Summarize orthonormality statistics for a pose rotation matrix.

    Args:
        pose: PoseTW with rotation matrices in ``pose.R``.

    Returns:
        Aggregated orthonormality statistics across all poses in the batch.
    """

    r = pose.R.detach()
    if r.ndim == 2:
        r = r.unsqueeze(0)
    eye = torch.eye(3, device=r.device, dtype=r.dtype).unsqueeze(0)
    resid = r.transpose(-1, -2) @ r - eye
    resid_abs = resid.abs()
    ortho_max = resid_abs.amax(dim=(-1, -2))
    ortho_mean = resid_abs.mean(dim=(-1, -2))

    axis_norms = torch.linalg.norm(r, dim=-2)
    axis_norm_err = (axis_norms - 1.0).abs()
    axis_norm_max = axis_norm_err.amax(dim=-1)
    axis_norm_mean = axis_norm_err.mean(dim=-1)

    det = torch.linalg.det(r)

    return {
        "orth_max": float(ortho_max.max().item()),
        "orth_mean": float(ortho_mean.mean().item()),
        "axis_norm_max": float(axis_norm_max.max().item()),
        "axis_norm_mean": float(axis_norm_mean.mean().item()),
        "det_min": float(det.min().item()),
        "det_max": float(det.max().item()),
        "det_mean": float(det.mean().item()),
    }


def _render_pose_orthonormality(label: str, pose: PoseTW | None) -> None:
    """Render orthonormality metrics for a pose in Streamlit."""

    if pose is None:
        st.info(f"{label}: pose not available.")
        return

    stats = _pose_orthonormality_stats(pose)
    st.markdown(f"**{label}**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("max |R^T R - I|", f"{stats['orth_max']:.2e}")
    with col2:
        st.metric("mean |R^T R - I|", f"{stats['orth_mean']:.2e}")
    with col3:
        st.metric("max |axis_norm - 1|", f"{stats['axis_norm_max']:.2e}")
    with col4:
        st.metric("det(R) mean", f"{stats['det_mean']:.6f}")
    st.caption(
        "det(R) range: "
        f"[{stats['det_min']:.6f}, {stats['det_max']:.6f}] · "
        f"mean |axis_norm - 1|: {stats['axis_norm_mean']:.2e}",
    )


def _counterfactual_cache_key(
    sample: EfmSnippetView | SimpleNamespace | None,
    cand_cfg: CandidateViewGeneratorConfig | None,
    rollout_cfg: CounterfactualPoseGeneratorConfig,
) -> str:
    """Return a stable cache key for panel-scoped counterfactual rollouts."""

    sample_sig = "no-sample"
    if sample is not None and hasattr(sample, "scene_id") and hasattr(sample, "snippet_id"):
        sample_sig = f"{sample.scene_id}:{sample.snippet_id}"
        if isinstance(sample, EfmSnippetView):
            sample_sig = sample_key(sample)
    cand_sig = "no-candidate-config" if cand_cfg is None else config_signature(cand_cfg)
    rollout_sig = config_signature(rollout_cfg)
    return f"{sample_sig}|{cand_sig}|{rollout_sig}"


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


def _run_counterfactual_rollouts(
    sample: EfmSnippetView,
    rollout_cfg: CounterfactualPoseGeneratorConfig,
) -> tuple[CounterfactualRolloutResult, str]:
    """Execute rollouts while capturing Console output for the UI."""

    lines: list[str] = []

    def _sink(message: str) -> None:
        lines.append(_strip_ansi(message))

    Console.set_sink(_sink)
    try:
        rollouts = rollout_cfg.setup_target().generate_from_typed_sample(sample)
    finally:
        Console.set_sink(None)
    return rollouts, "\n".join(lines)


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

    with st.expander("Frame orthonormality", expanded=False):
        _info_popover(
            "frame orthonormality",
            "Reports orthonormality errors for the reference and sampling poses "
            "(R^T R should be close to I, det(R) near +1). These stats are "
            "computed before any display-only rotations.",
        )
        _render_pose_orthonormality("Reference pose (world <- rig)", candidates.reference_pose)
        sampling_pose = candidates.sampling_pose
        if sampling_pose is None:
            st.info("Sampling pose not available (cached data or gravity alignment disabled).")
        else:
            if sampling_pose is candidates.reference_pose:
                st.caption("Sampling pose matches reference pose.")
            _render_pose_orthonormality("Sampling pose (world <- sampling)", sampling_pose)

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
                .add_reference_axes(display_rotate=True)
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
    shell_data = _shell_offsets_dirs_ref(candidates)
    dirs_shell_ref = shell_data[1] if shell_data is not None else None

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
                    .add_reference_axes(display_rotate=True)
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
            show_view_dirs = st.checkbox(
                "Show view directions (rig frame)",
                value=False,
                key="cand_offsets_show_view_dirs",
            )
            st.markdown(
                stats_to_markdown_table(
                    summarise_offsets_ref(offsets_ref),
                    header=None,
                ),
            )
            offsets_np = offsets_ref.cpu().numpy()
            dirs_overlay = None
            if show_view_dirs:
                dirs_overlay = dirs_ref
                if dirs_shell_ref is not None and mask_valid.shape[0] == dirs_shell_ref.shape[0]:
                    dirs_overlay = dirs_shell_ref[mask_valid]
            dirs_overlay_np = dirs_overlay.cpu().numpy() if dirs_overlay is not None else None
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
                    plot_position_sphere(
                        offsets_np,
                        show_axes=True,
                        dirs=dirs_overlay_np,
                    ),
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
            dirs_plot = dirs_shell_ref if dirs_shell_ref is not None else dirs_ref
            st.markdown(
                stats_to_markdown_table(summarise_dirs_ref(dirs_plot), header=None),
            )
            dirs_np = dirs_plot.cpu().numpy()

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

    with st.expander("Counterfactual Rollouts", expanded=False):
        _info_popover(
            "counterfactual rollouts",
            "Generates multi-step candidate expansions by reusing the current one-step "
            "candidate generator at every step. This is an exploratory planning view: "
            "the rollout paths and per-step shell plots help diagnose branching behavior, "
            "beam pruning, and the selected continuation under simple geometric policies.",
        )

        if sample is None:
            st.info("Counterfactual rollouts require an attached EFM snippet.")
        elif cand_cfg is None:
            st.info("Counterfactual rollouts require a candidate generator config.")
        elif sample.mesh is None or sample.mesh_verts is None or sample.mesh_faces is None:
            st.info("Counterfactual rollouts require the sample's GT mesh, vertices, and faces.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                cf_horizon = st.slider("Horizon", min_value=1, max_value=5, value=3, step=1, key="cf_horizon")
                cf_branch = st.slider(
                    "Branch factor",
                    min_value=1,
                    max_value=4,
                    value=2,
                    step=1,
                    key="cf_branch_factor",
                )
            with col2:
                cf_beam_enabled = st.checkbox("Cap beam width", value=True, key="cf_beam_enabled")
                cf_beam = st.slider("Beam width", min_value=1, max_value=8, value=4, step=1, key="cf_beam_width")
                cf_policy = st.selectbox(
                    "Selection policy",
                    options=list(CounterfactualSelectionPolicy),
                    format_func=lambda policy: policy.value.replace("_", " "),
                    index=0,
                    key="cf_policy",
                )
            with col3:
                cf_hist_dist = st.slider(
                    "Min history distance (m)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    key="cf_min_history_distance",
                )
                cf_sibling_dist = st.slider(
                    "Min sibling distance (m)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.15,
                    step=0.05,
                    key="cf_min_sibling_distance",
                )

            cf_candidate_cfg = cand_cfg.model_copy(
                update={
                    "verbosity": int(cand_cfg.verbosity),
                    "is_debug": bool(cand_cfg.is_debug),
                }
            )
            cf_cfg = CounterfactualPoseGeneratorConfig(
                candidate_config=cf_candidate_cfg,
                horizon=int(cf_horizon),
                branch_factor=int(cf_branch),
                beam_width=int(cf_beam) if cf_beam_enabled else None,
                selection_policy=cf_policy,
                min_history_distance_m=float(cf_hist_dist),
                min_sibling_distance_m=float(cf_sibling_dist),
                verbosity=int(cand_cfg.verbosity),
                is_debug=bool(cand_cfg.is_debug),
            )

            run_key = _counterfactual_cache_key(sample, cand_cfg, cf_cfg)
            cache = st.session_state.setdefault("cand_counterfactual_cache", {})
            run_counterfactuals = st.button("Run / refresh counterfactual rollouts", key="cand_run_counterfactuals")

            if run_counterfactuals:
                try:
                    with st.spinner("Generating counterfactual rollouts..."):
                        rollouts, log_text = _run_counterfactual_rollouts(sample, cf_cfg)
                    cache[run_key] = {"rollouts": rollouts, "logs": log_text}
                except Exception as exc:  # pragma: no cover - UI guard
                    _report_exception(exc, context="Counterfactual rollout generation failed")

            cached_payload = cache.get(run_key)
            if cached_payload is None:
                st.caption("Configure the rollout settings above, then click run to materialize trajectories.")
            else:
                rollouts = cached_payload["rollouts"]
                log_text = cached_payload["logs"]
                rows = _counterfactual_trajectory_rows(rollouts)

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Trajectories", len(rollouts.trajectories))
                with metric_col2:
                    st.metric("Horizon", rollouts.horizon)
                with metric_col3:
                    best_score = max((traj.cumulative_score for traj in rollouts.trajectories), default=0.0)
                    st.metric("Best cumulative score", f"{best_score:.3f}")

                st.dataframe(rows, width="stretch", hide_index=True)

                plot_tab, step_tab, log_tab = st.tabs(["Paths", "Step Shell", "Logs"])

                with plot_tab:
                    show_selected_frusta = st.checkbox(
                        "Overlay selected frusta",
                        value=True,
                        key="cf_show_selected_frusta",
                    )
                    if sample is not None:
                        builder = (
                            CounterfactualPlotBuilder.from_rollouts(
                                sample,
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
                        st.plotly_chart(
                            plot_counterfactual_paths_simple(rollouts),
                            width="stretch",
                        )

                with step_tab:
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
                        max_step = len(trajectory.steps)
                        step_display_index = st.slider(
                            "Step",
                            min_value=1,
                            max_value=max_step,
                            value=min(2, max_step),
                            step=1,
                            key="cf_step_idx",
                        )
                        include_rejected = st.checkbox(
                            "Show rejected candidates",
                            value=False,
                            key="cf_step_include_rejected",
                        )
                        if sample is not None:
                            step_fig = (
                                CounterfactualPlotBuilder.from_rollouts(
                                    sample,
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
                        else:
                            step_fig = plot_counterfactual_step_simple(
                                trajectory,
                                step_index=int(step_display_index - 1),
                            )
                        st.plotly_chart(step_fig, width="stretch")

                with log_tab:
                    if log_text.strip():
                        st.code(log_text, language="text")
                    else:
                        st.caption("No Console output was emitted during this rollout run.")


__all__ = ["render_candidates_page"]
