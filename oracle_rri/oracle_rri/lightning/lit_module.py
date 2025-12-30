"""LightningModule for training VIN (View Introspection Network).

This module implements the same core logic as `oracle_rri/scripts/train_vin.py`,
but with PyTorch Lightning training loops and optional W&B logging via the
trainer factory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import pytorch_lightning as pl
import torch
from efm3d.aria.pose import PoseTW
from pydantic import Field, model_validator
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Metric
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.regression import SpearmanCorrCoef

from ..configs import PathConfig
from ..utils import BaseConfig, Console, Stage
from ..vin import RriOrdinalBinner, VinModelConfig, coral_loss
from ..vin.plotting import plot_vin_encodings_from_debug
from .lit_datamodule import VinOracleBatch


def _to_jsonable(value: Any) -> Any:
    """Convert nested config dumps to logger-friendly primitives."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_jsonable(v) for v in value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, torch.device):
        return str(value)
    return value


def _shape_str(value: Any) -> str:
    if isinstance(value, PoseTW):
        return f"PoseTW{tuple(value.tensor().shape)}"
    if isinstance(value, torch.Tensor):
        return f"Tensor{tuple(value.shape)}[{str(value.dtype).replace('torch.', '')}]"
    return type(value).__name__


class LabelHistogram(Metric):
    """Accumulate label counts for ordinal classes."""

    full_state_update = False

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.add_state(
            "counts",
            default=torch.zeros(self.num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, target: Tensor) -> None:
        if target.numel() == 0:
            return
        labels = target.to(dtype=torch.int64).reshape(-1)
        counts = torch.bincount(labels, minlength=self.num_classes)
        self.counts = self.counts + counts.to(device=self.counts.device)

    def compute(self) -> Tensor:
        return self.counts


class AdamWConfig(BaseConfig[Optimizer]):
    """AdamW optimizer configuration for VIN."""

    target: type[Optimizer] = Field(default_factory=lambda: AdamW, exclude=True)
    """Factory target for :meth:`~oracle_rri.utils.base_config.BaseConfig.setup_target`."""

    learning_rate: float = 8e-5
    """Learning rate for AdamW."""

    weight_decay: float = 1e-3
    """Weight decay for AdamW."""

    def setup_target(self, params: list[Tensor]) -> Optimizer:  # type: ignore[override]
        return AdamW(params=params, lr=float(self.learning_rate), weight_decay=float(self.weight_decay))


class VinLightningModuleConfig(BaseConfig["VinLightningModule"]):
    """Configuration for :class:`VinLightningModule`."""

    target: type["VinLightningModule"] = Field(default_factory=lambda: VinLightningModule, exclude=True)

    vin: VinModelConfig = Field(default_factory=VinModelConfig)
    """Underlying VIN model configuration (frozen EVL backbone + CORAL head)."""

    optimizer: AdamWConfig = Field(default_factory=AdamWConfig)
    """Optimizer configuration."""

    num_classes: int = 15
    """Number of ordinal classes (must match `vin.head.num_classes`)."""

    binner_fit_snippets: int = 2
    """Number of oracle-labelled snippets used to fit the ordinal binner."""

    binner_max_attempts: int = 64
    """Maximum number of skipped oracle batches while fitting the binner (guards against bad oracle settings)."""

    save_binner: bool = True
    """Persist `rri_binner.json` into the run directory on fit start."""

    binner_path: Path | None = None
    """Optional explicit path to save `rri_binner.json` (defaults to trainer root dir)."""

    use_valid_frac_weight: bool = True
    """Whether to weight loss by per-candidate frustum coverage."""

    valid_frac_weight_floor: float = Field(default=0.2, ge=0.0, le=1.0)
    """Minimum loss weight for candidates with low voxel coverage."""

    @model_validator(mode="after")
    def _validate_num_classes(self) -> Self:
        model_classes = int(self.vin.head.num_classes)
        if int(self.num_classes) != model_classes:
            raise ValueError(f"num_classes={self.num_classes} must match vin.head.num_classes={model_classes}.")
        return self


class VinLightningModule(pl.LightningModule):
    """PyTorch Lightning module for VIN training with CORAL ordinal regression."""

    def __init__(self, config: VinLightningModuleConfig) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters(_to_jsonable(config.model_dump()))

        self.console = Console.with_prefix(self.__class__.__name__)

        self.vin = self.config.vin.setup_target()
        self._binner: RriOrdinalBinner | None = None
        num_classes = int(self.config.num_classes)

        self._spearman = {
            Stage.TRAIN: SpearmanCorrCoef(),
            Stage.VAL: SpearmanCorrCoef(),
            Stage.TEST: SpearmanCorrCoef(),
        }
        self._confusion = {
            Stage.TRAIN: MulticlassConfusionMatrix(num_classes=num_classes),
            Stage.VAL: MulticlassConfusionMatrix(num_classes=num_classes),
            Stage.TEST: MulticlassConfusionMatrix(num_classes=num_classes),
        }
        self._label_hist = {
            Stage.TRAIN: LabelHistogram(num_classes=num_classes),
            Stage.VAL: LabelHistogram(num_classes=num_classes),
            Stage.TEST: LabelHistogram(num_classes=num_classes),
        }

    # --------------------------------------------------------------------- lifecycle
    def setup(self, stage: str) -> None:  # noqa: A003
        super().setup(stage)
        self._integrate_console()
        if self._binner is None:
            self._binner = self._load_binner_from_config()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self._binner is not None:
            checkpoint["rri_binner"] = self._binner.to_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        data = checkpoint.get("rri_binner")
        if data is not None:
            self._binner = RriOrdinalBinner.from_dict(data)

    # ------------------------------------------------------------------ training/val/test
    def training_step(self, batch: VinOracleBatch, batch_idx: int) -> Tensor | None:
        return self._step(batch, batch_idx, stage=Stage.TRAIN)

    def validation_step(self, batch: VinOracleBatch, batch_idx: int) -> Tensor | None:
        return self._step(batch, batch_idx, stage=Stage.VAL)

    def test_step(self, batch: VinOracleBatch, batch_idx: int) -> Tensor | None:
        return self._step(batch, batch_idx, stage=Stage.TEST)

    # ------------------------------------------------------------------ epoch-end metrics
    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics(Stage.TRAIN)

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics(Stage.VAL)

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics(Stage.TEST)

    # ------------------------------------------------------------------ optim
    def configure_optimizers(self) -> dict[str, Any]:
        params = [p for p in self.vin.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters found (did you freeze everything?).")
        optimizer = self.config.optimizer.setup_target(params=params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, patience=20, factor=0.2),
                "monitor": "train/loss",
                "interval": "step",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------ internals
    def _step(self, batch: VinOracleBatch, batch_idx: int, *, stage: Stage) -> Tensor | None:
        # TODO: should we pass the number of candidates as the batch size instead of 1?
        if self._binner is None:
            raise RuntimeError(
                "RRI binner not initialized. Provide `VinLightningModuleConfig.binner_path` (a fitted .json), "
                "or resume from a checkpoint that contains `rri_binner`."
            )

        pred = self.vin.forward(
            batch.efm,
            candidate_poses_world_cam=batch.candidate_poses_world_cam,
            reference_pose_world_rig=batch.reference_pose_world_rig,
            p3d_cameras=batch.p3d_cameras,
        )
        logits = pred.logits.squeeze(0)  # N x (K-1)
        valid_frac = pred.valid_frac.squeeze(0)  # N

        rri = batch.rri.to(device=logits.device)
        labels = self._binner.transform(rri.reshape(-1))

        mask = torch.isfinite(rri)
        if not mask.any():
            self.log(f"{stage}/skip_no_valid", 1.0, on_step=True, prog_bar=False, batch_size=1)
            return None

        probs = pred.prob.squeeze(0)
        midpoints = self._binner.class_midpoints().to(device=probs.device, dtype=probs.dtype)
        pred_rri_proxy = (probs * midpoints.view(1, -1)).sum(dim=-1)

        loss_per = coral_loss(
            logits[mask],
            labels[mask],
            num_classes=int(self._binner.num_classes),
            reduction="none",
        )
        if self.config.use_valid_frac_weight:
            weights = (
                self.config.valid_frac_weight_floor + (1.0 - self.config.valid_frac_weight_floor) * valid_frac[mask]
            )
            loss = (loss_per * weights).sum() / weights.sum().clamp_min(1e-6)
        else:
            loss = loss_per.mean()

        prefix = f"{stage}"
        self.log(
            f"{prefix}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=(stage is Stage.TRAIN),
            batch_size=1,
        )
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=1)
        self.log(
            f"{prefix}/voxel_valid_fraction",
            float(valid_frac.mean().item()),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        self.log(
            f"{prefix}/candidate_valid_fraction",
            float(pred.candidate_valid.float().mean().item()),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        self.log(
            f"{prefix}/rri_mean",
            rri[mask].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        self.log(
            f"{prefix}/pred_rri_mean",
            pred_rri_proxy[mask].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        self.log(
            f"{prefix}/pred_score_mean",
            pred.expected_normalized.squeeze(0)[mask].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )

        pred_class = probs.argmax(dim=-1).to(dtype=torch.int64)
        self._spearman[stage].update(
            pred.expected_normalized.squeeze(0)[mask].to(dtype=torch.float32),
            rri[mask].to(dtype=torch.float32),
        )
        self._confusion[stage].update(pred_class[mask], labels[mask])
        self._label_hist[stage].update(labels[mask])

        pm_dist_before = batch.pm_dist_before.to(device=logits.device)
        pm_dist_after = batch.pm_dist_after.to(device=logits.device)
        pm_acc_before = batch.pm_acc_before.to(device=logits.device)
        pm_comp_before = batch.pm_comp_before.to(device=logits.device)
        pm_acc_after = batch.pm_acc_after.to(device=logits.device)
        pm_comp_after = batch.pm_comp_after.to(device=logits.device)

        self.log(
            f"{prefix}/pm_dist_before_mean",
            pm_dist_before[mask].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        self.log(
            f"{prefix}/pm_dist_after_mean",
            pm_dist_after[mask].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        self.log(
            f"{prefix}/pm_acc_before_mean",
            pm_acc_before[mask].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        self.log(
            f"{prefix}/pm_comp_before_mean",
            pm_comp_before[mask].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        self.log(
            f"{prefix}/pm_acc_after_mean",
            pm_acc_after[mask].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{prefix}/pm_comp_after_mean",
            pm_comp_after[mask].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        return loss

    def _log_epoch_metrics(self, stage: Stage) -> None:
        prefix = f"{stage}"

        spearman = self._spearman[stage].compute()
        if torch.isfinite(spearman):
            self.log(
                f"{prefix}/spearman",
                spearman,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=1,
            )
        self._spearman[stage].reset()

        confusion = self._confusion[stage].compute().detach().cpu()
        if confusion.numel() > 0:
            conf_dict = {
                f"{prefix}/confusion/{row}_{col}": float(confusion[row, col].item())
                for row in range(confusion.shape[0])
                for col in range(confusion.shape[1])
            }
            self.log_dict(conf_dict, on_step=False, on_epoch=True)
        self._confusion[stage].reset()

        label_hist = self._label_hist[stage].compute().detach().cpu()
        if label_hist.numel() > 0:
            hist_dict = {
                f"{prefix}/label_hist/{idx}": float(label_hist[idx].item()) for idx in range(label_hist.shape[0])
            }
            self.log_dict(hist_dict, on_step=False, on_epoch=True)
            self.log(
                f"{prefix}/label_hist_total",
                float(label_hist.sum().item()),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=1,
            )
        self._label_hist[stage].reset()

    def _load_binner_from_config(self) -> RriOrdinalBinner:
        if self.config.binner_path is None:
            raise RuntimeError(
                "Missing `VinLightningModuleConfig.binner_path`. Fit a binner first (e.g. via `nbv-fit-binner`) "
                "and point this config field to the resulting `rri_binner.json`, or resume from a checkpoint."
            )

        resolved = PathConfig().resolve_artifact_path(
            self.config.binner_path, expected_suffix=".json", create_parent=False
        )
        if not resolved.exists():
            raise FileNotFoundError(
                f"RRI binner not found at {resolved}. Run `nbv-fit-binner --out-dir <run_dir>` to create it "
                "or set `VinLightningModuleConfig.binner_path` to an existing fitted binner JSON."
            )
        return RriOrdinalBinner.load(resolved)

    def _integrate_console(self) -> None:
        logger = getattr(self, "logger", None)
        if logger is not None:
            Console.integrate_with_logger(logger, global_step=int(self.global_step))

    def summarize_vin(
        self,
        batch: VinOracleBatch,
        *,
        include_torchsummary: bool = True,
        torchsummary_depth: int = 3,
    ) -> str:
        """Summarize VIN inputs/outputs for a single oracle-labeled batch.

        Args:
            batch: Oracle-labeled VIN batch from :class:`VinDataModule`.
            include_torchsummary: Whether to append torchsummary module summaries.
            torchsummary_depth: Max depth for torchsummary module traversal.

        Returns:
            Multiline string with VIN summary information.
        """

        from efm3d.aria.aria_constants import ARIA_CALIB, ARIA_IMG, ARIA_POSE_T_WORLD_RIG

        was_training = self.vin.training
        self.vin.eval()
        with torch.no_grad():
            pred, debug = self.vin.forward_with_debug(
                batch.efm,
                candidate_poses_world_cam=batch.candidate_poses_world_cam,
                reference_pose_world_rig=batch.reference_pose_world_rig,
                p3d_cameras=batch.p3d_cameras,
            )
        if was_training:
            self.vin.train()

        lines: list[str] = []
        lines.append("VIN summary (oracle batch)")
        lines.append(f"  scene_id: {batch.scene_id}")
        lines.append(f"  snippet_id: {batch.snippet_id}")
        lines.append(f"  device: {debug.candidate_center_rig_m.device}")
        lines.append(f"  candidates: {_shape_str(batch.candidate_poses_world_cam)}")
        lines.append("")

        efm = batch.efm
        lines.append("EFM snippet inputs (key shapes)")
        for key in ARIA_IMG:
            lines.append(f"  {key}: {_shape_str(efm.get(key))}")
        for key in ARIA_CALIB:
            lines.append(f"  {key}: {_shape_str(efm.get(key))}")
        lines.append(f"  {ARIA_POSE_T_WORLD_RIG}: {_shape_str(efm.get(ARIA_POSE_T_WORLD_RIG))}")
        for key in ["points/p3s_world", "points/dist_std", "pose/gravity_in_world"]:
            if key in efm:
                lines.append(f"  {key}: {_shape_str(efm.get(key))}")
        lines.append("")

        backbone_out = debug.backbone_out
        lines.append("EVL feature contract (VIN inputs)")
        if isinstance(backbone_out.occ_pr, torch.Tensor):
            lines.append(f"  occ_pr: {tuple(backbone_out.occ_pr.shape)}")
        if isinstance(backbone_out.occ_input, torch.Tensor):
            lines.append(f"  occ_input: {tuple(backbone_out.occ_input.shape)}")
        if isinstance(backbone_out.counts, torch.Tensor):
            lines.append(f"  counts: {tuple(backbone_out.counts.shape)}")
        lines.append(f"  T_world_voxel: PoseTW{tuple(backbone_out.t_world_voxel.tensor().shape)}")
        lines.append(f"  voxel_extent: {tuple(backbone_out.voxel_extent.shape)}")
        lines.append("")

        lines.append("Candidate pose descriptor (shapes)")
        lines.append(f"  t (center in ref): {tuple(debug.candidate_center_rig_m.shape)}")
        lines.append(f"  r = ||t||: {tuple(debug.candidate_radius_m.shape)}")
        lines.append(f"  u = t/r: {tuple(debug.candidate_center_dir_rig.shape)}")
        lines.append(f"  f = R z_cam: {tuple(debug.candidate_forward_dir_rig.shape)}")
        lines.append(f"  s = <f,-u>: {tuple(debug.view_alignment.shape)}")
        lines.append("")

        if debug.voxel_pose_enc is not None and debug.voxel_center_rig_m is not None:
            lines.append("Voxel pose descriptor (shapes)")
            lines.append(f"  t_vox (center in ref): {tuple(debug.voxel_center_rig_m.shape)}")
            if debug.voxel_radius_m is not None:
                lines.append(f"  r_vox = ||t||: {tuple(debug.voxel_radius_m.shape)}")
            if debug.voxel_center_dir_rig is not None:
                lines.append(f"  u_vox = t/r: {tuple(debug.voxel_center_dir_rig.shape)}")
            if debug.voxel_forward_dir_rig is not None:
                lines.append(f"  f_vox = R z_vox: {tuple(debug.voxel_forward_dir_rig.shape)}")
            if debug.voxel_view_alignment is not None:
                lines.append(f"  s_vox = <f,-u>: {tuple(debug.voxel_view_alignment.shape)}")
            lines.append(f"  voxel_pose_enc: {tuple(debug.voxel_pose_enc.shape)}")
            lines.append("")

        valid_frac = debug.token_valid.float().mean(dim=-1)
        num_candidates = int(debug.candidate_valid.shape[1])
        lines.append("VIN head input features (shapes)")
        lines.append(f"  pose_enc: {tuple(debug.pose_enc.shape)}")
        if debug.global_feat is not None:
            lines.append(f"  global_feat: {tuple(debug.global_feat.shape)}")
        lines.append(f"  local_feat: {tuple(debug.local_feat.shape)}")
        lines.append(
            f"  token_valid: {tuple(debug.token_valid.shape)} | valid_frac mean={float(valid_frac.mean().item()):.3f}"
        )
        lines.append(
            f"  candidate_valid: {tuple(debug.candidate_valid.shape)} | "
            f"keep={int(debug.candidate_valid.sum().item())}/{num_candidates}"
        )
        lines.append(f"  concat feats: {tuple(debug.feats.shape)}  (in_dim={int(debug.feats.shape[-1])})")
        lines.append("")

        lines.append("VIN outputs (shapes)")
        lines.append(f"  logits: {tuple(pred.logits.shape)}")
        lines.append(f"  prob: {tuple(pred.prob.shape)}")
        lines.append(f"  expected: {tuple(pred.expected.shape)}")
        lines.append(f"  expected_normalized: {tuple(pred.expected_normalized.shape)}")
        lines.append("")

        trainable_params = sum(p.numel() for p in self.vin.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.vin.parameters())
        lines.append(
            f"Trainable VIN params: {trainable_params:,} (vin total params: {total_params:,}; EVL frozen not counted)"
        )
        lines.append("")

        if include_torchsummary:
            from torchsummary import summary as torch_summary

            feats_2d = debug.feats.reshape(debug.feats.shape[0] * debug.feats.shape[1], -1)

            lines.append("torchsummary: Vin (trainable modules)")
            lines.append(
                str(
                    torch_summary(
                        self.vin,
                        input_data=None,
                        verbose=0,
                        depth=torchsummary_depth,
                        device=debug.candidate_center_rig_m.device,
                    )
                )
            )
            lines.append("")

            lines.append("torchsummary: ShellShPoseEncoder (trainable)")
            lines.append(
                str(
                    torch_summary(
                        self.vin.pose_encoder_lff,
                        input_data=(debug.candidate_center_dir_rig, debug.candidate_forward_dir_rig),
                        r=debug.candidate_radius_m,
                        scalars=debug.view_alignment,
                        verbose=0,
                        depth=torchsummary_depth,
                        device=debug.candidate_center_rig_m.device,
                    )
                )
            )
            lines.append("")

            lines.append("torchsummary: field_proj (trainable)")
            lines.append(
                str(
                    torch_summary(
                        self.vin.field_proj,
                        input_data=debug.field_in,
                        verbose=0,
                        depth=torchsummary_depth,
                        device=debug.candidate_center_rig_m.device,
                    )
                )
            )
            lines.append("")

            lines.append("torchsummary: VinScorerHead (trainable)")
            lines.append(
                str(
                    torch_summary(
                        self.vin.head,
                        input_data=feats_2d,
                        verbose=0,
                        depth=torchsummary_depth,
                        device=debug.candidate_center_rig_m.device,
                    )
                )
            )

        return "\n".join(lines)

    def plot_vin_encodings_batch(
        self,
        batch: VinOracleBatch,
        *,
        out_dir: Path,
        lmax: int,
        sh_normalization: str,
        radius_freqs: list[float],
        file_stem_prefix: str,
    ) -> dict[str, Path]:
        """Generate VIN encoding plots for a single oracle-labeled batch.

        Args:
            batch: Oracle-labeled VIN batch from :class:`VinDataModule`.
            out_dir: Output directory for plots.
            lmax: Max SH degree for visualization.
            sh_normalization: Spherical harmonics normalization mode.
            radius_freqs: Fourier frequencies for radius plot.
            file_stem_prefix: Filename prefix for the plots.

        Returns:
            Mapping of plot names to saved paths.
        """

        was_training = self.vin.training
        self.vin.eval()
        with torch.no_grad():
            _, debug = self.vin.forward_with_debug(
                batch.efm,
                candidate_poses_world_cam=batch.candidate_poses_world_cam,
                reference_pose_world_rig=batch.reference_pose_world_rig,
                p3d_cameras=batch.p3d_cameras,
            )
        if was_training:
            self.vin.train()

        return plot_vin_encodings_from_debug(
            debug,
            out_dir=out_dir,
            lmax=int(lmax),
            sh_normalization=str(sh_normalization),
            radius_freqs=radius_freqs,
            file_stem_prefix=file_stem_prefix,
            pose_encoder_lff=self.vin.pose_encoder_lff,
        )


__all__ = ["AdamWConfig", "VinLightningModule", "VinLightningModuleConfig"]
