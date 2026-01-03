"""Offline cache helpers for panels."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch

from ...configs import PathConfig
from ...data import AseEfmDatasetConfig, EfmSnippetView
from ...data.offline_cache import OracleRriCacheConfig, OracleRriCacheDataset, OracleRriCacheDatasetConfig
from ...lightning.aria_nbv_experiment import AriaNBVExperimentConfig
from ...utils import Stage
from ..state_types import config_signature


def _load_efm_snippet_for_cache(
    *,
    scene_id: str,
    snippet_id: str,
    dataset_payload: dict[str, Any] | None,
    device: str,
    paths: PathConfig,
    include_gt_mesh: bool,
) -> EfmSnippetView:
    payload = dict(dataset_payload or {})
    payload["paths"] = payload.get("paths", paths)
    payload["scene_ids"] = [scene_id]
    payload["snippet_ids"] = [snippet_id]
    payload["batch_size"] = 1
    payload["device"] = device
    payload["wds_shuffle"] = False
    payload["wds_repeat"] = False
    payload["load_meshes"] = bool(include_gt_mesh)
    payload.setdefault("require_mesh", False)
    cfg = AseEfmDatasetConfig(**payload)
    dataset = cfg.setup_target()
    return next(iter(dataset))


def _prepare_offline_cache_dataset(
    *,
    cache_dir: str | None,
    map_location: str,
    paths: PathConfig,
    state: Any,
    stage: Stage | None,
    include_efm_snippet: bool,
    include_gt_mesh: bool,
) -> OracleRriCacheDataset | None:
    if cache_dir is None:
        return None
    split = "all"
    if stage is Stage.TRAIN:
        split = "train"
    elif stage in (Stage.VAL, Stage.TEST):
        split = "val"
    cache_cfg = OracleRriCacheDatasetConfig(
        cache=OracleRriCacheConfig(cache_dir=Path(cache_dir), paths=paths),
        load_backbone=True,
        map_location=map_location,
        split=split,
        include_efm_snippet=include_efm_snippet,
        include_gt_mesh=include_gt_mesh,
    )
    cfg_sig = config_signature(cache_cfg)
    if state.offline_cache_sig != cfg_sig or state.offline_cache is None:
        cache_ds = cache_cfg.setup_target()
        state.offline_cache_sig = cfg_sig
        state.offline_cache = cache_ds
        state.offline_cache_len = len(cache_ds)
        state.offline_cache_idx = 0
        state.offline_snippet_key = None
        state.offline_snippet = None
        state.offline_snippet_error = None
    return state.offline_cache


def _collect_offline_cache_stats(
    *,
    toml_path: str | None,
    stage: Stage,
    cache_dir: str | None,
    map_location: str,
    max_samples: int | None,
    num_workers: int | None,
    train_val_split: float,
) -> dict[str, Any]:
    """Collect summary stats from an offline cache without keeping full samples in memory."""

    def _normalise(vec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return vec / vec.norm(dim=-1, keepdim=True).clamp_min(eps)

    def _roll_about_forward(
        *,
        forward: torch.Tensor,
        up_cam: torch.Tensor,
        up_ref: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        forward = _normalise(forward, eps=eps)
        up_cam = _normalise(up_cam, eps=eps)
        if up_ref.ndim == 1:
            up_ref = up_ref.view(1, 3).expand_as(forward)
        else:
            while up_ref.ndim < forward.ndim:
                up_ref = up_ref.unsqueeze(0)
            up_ref = up_ref.expand_as(forward)
        up_ref = _normalise(up_ref, eps=eps)

        left0 = torch.cross(up_ref, forward, dim=-1)
        left0_norm = left0.norm(dim=-1, keepdim=True)
        degenerate = left0_norm.squeeze(-1) < eps
        if degenerate.any():
            alt = torch.tensor(
                [1.0, 0.0, 0.0],
                device=forward.device,
                dtype=forward.dtype,
            )
            alt = alt.view(1, 3).expand_as(forward)
            alt = alt - (alt * forward).sum(dim=-1, keepdim=True) * forward
            alt_norm = alt.norm(dim=-1, keepdim=True)
            second = alt_norm.squeeze(-1) < eps
            if second.any():
                alt2 = torch.tensor(
                    [0.0, 1.0, 0.0],
                    device=forward.device,
                    dtype=forward.dtype,
                )
                alt2 = alt2.view(1, 3).expand_as(forward)
                alt2 = alt2 - (alt2 * forward).sum(dim=-1, keepdim=True) * forward
                alt[second] = alt2[second]
                alt_norm = alt.norm(dim=-1, keepdim=True)
            left0[degenerate] = alt[degenerate]
            left0_norm[degenerate] = alt_norm[degenerate]
        left0 = left0 / left0_norm.clamp_min(eps)

        up0 = torch.cross(forward, left0, dim=-1)
        up0 = _normalise(up0, eps=eps)
        up_cam = _normalise(up_cam, eps=eps)
        cosang = (up0 * up_cam).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.acos(cosang)
        sign = torch.sign((left0 * up_cam).sum(dim=-1))
        return angle * sign

    def _as_tensor(value: Any) -> torch.Tensor | None:
        if torch.is_tensor(value):
            return value
        if isinstance(value, np.ndarray):
            return torch.as_tensor(value)
        return None

    def _tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
        if tensor.numel() == 0:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "abs_mean": float("nan"),
                "nz_frac": 0.0,
            }
        vals = tensor.detach()
        vals = vals[torch.isfinite(vals)]
        if vals.numel() == 0:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "abs_mean": float("nan"),
                "nz_frac": 0.0,
            }
        vals = vals.to(dtype=torch.float32)
        return {
            "mean": float(vals.mean().item()),
            "std": float(vals.std(unbiased=False).item()),
            "abs_mean": float(vals.abs().mean().item()),
            "nz_frac": float((vals.abs() > 1e-6).float().mean().item()),
        }

    resolved_toml: Path | None = None
    if toml_path:
        try:
            resolved = PathConfig().resolve_config_toml_path(
                toml_path,
                must_exist=False,
            )
        except ValueError as exc:
            st.warning(f"Invalid config path ({toml_path}): {exc}")
        else:
            if not resolved.exists():
                st.warning(f"Config file not found: {resolved}")
            else:
                resolved_toml = resolved

    cfg = AriaNBVExperimentConfig.from_toml(resolved_toml) if resolved_toml is not None else AriaNBVExperimentConfig()
    cfg.run_mode = "summarize_vin"
    cfg.stage = stage
    cfg.trainer_config.use_wandb = False

    dm_cfg = cfg.datamodule_config
    dm_cfg.train_cache_new_samples_per_epoch = 0
    dm_cfg.val_cache_new_samples_per_epoch = 0

    paths = cfg.paths if isinstance(cfg.paths, PathConfig) else PathConfig()
    cache_root = cache_dir or str(
        paths.offline_cache_dir or (paths.data_root / "oracle_rri_cache"),
    )
    cache_cfg = OracleRriCacheDatasetConfig(
        cache=OracleRriCacheConfig(cache_dir=Path(cache_root), paths=paths),
        load_backbone=True,
        map_location=map_location,
        train_val_split=train_val_split,
    )
    dm_cfg.train_cache = cache_cfg.model_copy(deep=True, update={"split": "train"})
    dm_cfg.val_cache = cache_cfg.model_copy(deep=True, update={"split": "val"})
    dm_cfg.use_train_as_val = False

    if num_workers is not None and num_workers > 0:
        dm_cfg.num_workers = int(num_workers)

    datamodule = dm_cfg.setup_target()
    dataloader = datamodule.train_dataloader() if stage is Stage.TRAIN else datamodule.val_dataloader()

    total_batches = None
    try:
        total_batches = len(dataloader)
    except TypeError:
        total_batches = None

    sample_rows: list[dict[str, Any]] = []
    backbone_rows: list[dict[str, Any]] = []
    rri_values: list[float] = []
    pm_comp_after_values: list[float] = []
    pm_acc_after_values: list[float] = []
    num_valid_values: list[int] = []
    candidate_offsets: list[np.ndarray] = []
    candidate_yaw: list[np.ndarray] = []
    candidate_pitch: list[np.ndarray] = []
    candidate_roll: list[np.ndarray] = []
    candidate_rot_deg: list[np.ndarray] = []

    max_batches = None if max_samples in (None, 0) else int(max_samples)
    progress = st.progress(0.0)
    for idx, batch in enumerate(dataloader):
        if max_batches is not None and idx >= max_batches:
            break

        rri = batch.rri.detach().flatten()
        rri_mask = torch.isfinite(rri)
        rri_valid = rri[rri_mask]
        num_valid = int(rri_valid.numel())
        num_valid_values.append(num_valid)
        if num_valid:
            rri_values.extend(rri_valid.cpu().tolist())

        pm_comp_after = batch.pm_comp_after.detach().flatten()
        pm_acc_after = batch.pm_acc_after.detach().flatten()
        pm_comp_valid = pm_comp_after[torch.isfinite(pm_comp_after)]
        pm_acc_valid = pm_acc_after[torch.isfinite(pm_acc_after)]
        if pm_comp_valid.numel():
            pm_comp_after_values.extend(pm_comp_valid.cpu().tolist())
        if pm_acc_valid.numel():
            pm_acc_after_values.extend(pm_acc_valid.cpu().tolist())

        def _finite_stats(vals: torch.Tensor) -> tuple[float, float, float]:
            vals = vals[torch.isfinite(vals)]
            if vals.numel() == 0:
                return float("nan"), float("nan"), float("nan")
            return (
                float(vals.mean().item()),
                float(vals.min().item()),
                float(vals.max().item()),
            )

        rri_mean, rri_min, rri_max = _finite_stats(rri)
        pm_comp_mean, _, _ = _finite_stats(pm_comp_after)
        pm_acc_mean, _, _ = _finite_stats(pm_acc_after)
        sample_rows.append(
            {
                "scene_id": batch.scene_id,
                "snippet_id": batch.snippet_id,
                "num_valid": num_valid,
                "rri_mean": rri_mean,
                "rri_min": rri_min,
                "rri_max": rri_max,
                "pm_comp_after_mean": pm_comp_mean,
                "pm_acc_after_mean": pm_acc_mean,
            },
        )

        backbone_out = batch.backbone_out
        if backbone_out is not None:
            if is_dataclass(backbone_out):
                items = [(field.name, getattr(backbone_out, field.name)) for field in fields(backbone_out)]
            else:
                items = list(vars(backbone_out).items())
            for name, value in items:
                tensor = _as_tensor(value)
                if tensor is None:
                    continue
                stats = _tensor_stats(tensor)
                backbone_rows.append(
                    {
                        "scene_id": batch.scene_id,
                        "snippet_id": batch.snippet_id,
                        "field": name,
                        "shape": str(tuple(tensor.shape)),
                        "numel": int(tensor.numel()),
                        **stats,
                    },
                )

        poses_world_cam = batch.candidate_poses_world_cam
        ref_pose = batch.reference_pose_world_rig
        if poses_world_cam is not None and ref_pose is not None:
            r_wc = poses_world_cam.R
            t_wc = poses_world_cam.t
            r_wr = ref_pose.R
            t_wr = ref_pose.t
            if r_wr.ndim == 2:
                r_wr = r_wr.unsqueeze(0)
                t_wr = t_wr.unsqueeze(0)
            if r_wr.shape[0] == 1 and r_wc.shape[0] > 1:
                r_wr = r_wr.expand(r_wc.shape[0], -1, -1)
                t_wr = t_wr.expand(r_wc.shape[0], -1)

            r_rw = r_wr.transpose(-1, -2)
            t_rw = -(r_rw @ t_wr.unsqueeze(-1)).squeeze(-1)
            r_rc = r_rw @ r_wc
            t_rc = t_rw + (r_rw @ t_wc.unsqueeze(-1)).squeeze(-1)

            candidate_offsets.append(t_rc.detach().cpu().numpy())

            fwd = r_rc[:, :, 2]
            up = r_rc[:, :, 1]
            yaw = torch.atan2(fwd[:, 0], fwd[:, 2])
            pitch = torch.asin(_normalise(fwd)[:, 1].clamp(-1.0, 1.0))
            up_ref = torch.tensor(
                [0.0, 1.0, 0.0],
                device=fwd.device,
                dtype=fwd.dtype,
            )
            roll = _roll_about_forward(
                forward=fwd,
                up_cam=up,
                up_ref=up_ref,
            )
            yaw_deg = torch.rad2deg(yaw).detach().cpu().numpy()
            pitch_deg = torch.rad2deg(pitch).detach().cpu().numpy()
            roll_deg = torch.rad2deg(roll).detach().cpu().numpy()
            candidate_yaw.append(yaw_deg)
            candidate_pitch.append(pitch_deg)
            candidate_roll.append(roll_deg)

            trace = r_rc[:, 0, 0] + r_rc[:, 1, 1] + r_rc[:, 2, 2]
            cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
            rot_angle = torch.acos(cos_angle)
            candidate_rot_deg.append(
                torch.rad2deg(rot_angle).detach().cpu().numpy(),
            )

        if total_batches:
            progress.progress(min(1.0, float(idx + 1) / float(total_batches)))
        else:
            progress.progress(0.0)

    progress.empty()

    sample_df = pd.DataFrame(sample_rows)
    backbone_df = pd.DataFrame(backbone_rows)
    summary = {
        "samples": len(sample_rows),
        "total_candidates": int(sum(num_valid_values)),
        "rri_mean": float(np.mean(rri_values)) if rri_values else float("nan"),
        "rri_median": float(np.median(rri_values)) if rri_values else float("nan"),
        "pm_comp_after_mean": float(np.mean(pm_comp_after_values)) if pm_comp_after_values else float("nan"),
        "pm_acc_after_mean": float(np.mean(pm_acc_after_values)) if pm_acc_after_values else float("nan"),
    }
    return {
        "summary": summary,
        "sample_df": sample_df,
        "backbone_df": backbone_df,
        "rri_values": rri_values,
        "pm_comp_after_values": pm_comp_after_values,
        "pm_acc_after_values": pm_acc_after_values,
        "num_valid_values": num_valid_values,
        "candidate_offsets": np.concatenate(candidate_offsets, axis=0)
        if candidate_offsets
        else np.zeros((0, 3), dtype=np.float32),
        "candidate_yaw": np.concatenate(candidate_yaw, axis=0) if candidate_yaw else np.zeros((0,), dtype=np.float32),
        "candidate_pitch": np.concatenate(candidate_pitch, axis=0)
        if candidate_pitch
        else np.zeros((0,), dtype=np.float32),
        "candidate_roll": np.concatenate(candidate_roll, axis=0)
        if candidate_roll
        else np.zeros((0,), dtype=np.float32),
        "candidate_rot_deg": np.concatenate(candidate_rot_deg, axis=0)
        if candidate_rot_deg
        else np.zeros((0,), dtype=np.float32),
    }


__all__ = [
    "_collect_offline_cache_stats",
    "_load_efm_snippet_for_cache",
    "_prepare_offline_cache_dataset",
]
