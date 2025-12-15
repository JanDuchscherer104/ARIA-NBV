"""VIN model on top of a frozen EVL backbone."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from efm3d.aria.aria_constants import ARIA_POSE_T_WORLD_RIG
from efm3d.aria.pose import PoseTW, so3log_map
from efm3d.utils.voxel_sampling import pc_to_vox, sample_voxels
from pydantic import Field
from torch import nn

from ..utils import BaseConfig
from .backbone_evl import EvlBackboneConfig
from .coral import CoralLayer, coral_expected_from_logits, coral_logits_to_prob
from .pose_encoding import LearnableFourierFeaturesConfig
from .types import VinPrediction

Tensor = torch.Tensor


class VinScorerHead(nn.Module):
    """Candidate scoring head producing CORAL logits."""

    def __init__(
        self,
        *,
        in_dim: int | None,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0.")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2.")

        act: nn.Module
        match activation.lower():
            case "relu":
                act = nn.ReLU()
            case "gelu":
                act = nn.GELU()
            case other:
                raise ValueError(f"Unsupported activation: {other}")

        layers: list[nn.Module] = []
        if in_dim is None:
            layers.append(nn.LazyLinear(hidden_dim))
        else:
            layers.append(nn.Linear(int(in_dim), hidden_dim))
        layers.append(act)
        if dropout > 0:
            layers.append(nn.Dropout(p=float(dropout)))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))

        self.mlp = nn.Sequential(*layers)
        self.coral = CoralLayer(in_dim=hidden_dim, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.coral(self.mlp(x))


class VinScorerHeadConfig(BaseConfig[VinScorerHead]):
    """Configuration for :class:`VinScorerHead`."""

    target: type[VinScorerHead] = Field(default_factory=lambda: VinScorerHead, exclude=True)
    """Factory target for :meth:`BaseConfig.setup_target`."""

    hidden_dim: int = 256
    """Hidden dimension for MLP layers."""

    num_layers: int = 2
    """Number of MLP layers before the CORAL layer."""

    dropout: float = 0.0
    """Dropout probability in the MLP."""

    num_classes: int = 15
    """Number of ordinal bins (VIN-NBV uses 15)."""

    activation: str = "gelu"
    """Activation function ('gelu' or 'relu')."""

    def setup_target(self, *, in_dim: int | None = None) -> VinScorerHead:  # type: ignore[override]
        return self.target(
            in_dim=in_dim,
            hidden_dim=int(self.hidden_dim),
            num_layers=int(self.num_layers),
            dropout=float(self.dropout),
            num_classes=int(self.num_classes),
            activation=str(self.activation),
        )


def _vin_target() -> type["VinModel"]:
    return VinModel


class VinModelConfig(BaseConfig["VinModel"]):
    """Configuration for :class:`VinModel`."""

    target: type["VinModel"] = Field(default_factory=_vin_target, exclude=True)
    """Factory target for :meth:`BaseConfig.setup_target`."""

    backbone: EvlBackboneConfig = Field(default_factory=EvlBackboneConfig)
    """Frozen EVL backbone configuration."""

    pose_encoder: LearnableFourierFeaturesConfig = Field(default_factory=LearnableFourierFeaturesConfig)
    """Learnable Fourier pose encoding configuration (candidate poses relative to reference)."""

    head: VinScorerHeadConfig = Field(default_factory=VinScorerHeadConfig)
    """Scoring head configuration."""

    use_occ_feat: bool = True
    """Use EVL occupancy neck features."""

    use_obb_feat: bool = True
    """Use EVL OBB neck features."""

    use_global_pool: bool = True
    """Concatenate global pooled voxel features."""

    use_local_sample: bool = True
    """Concatenate voxel features sampled at the candidate camera center."""


class VinModel(nn.Module):
    """View Introspection Network (VIN) predicting RRI from EVL voxel features + candidate pose."""

    def __init__(self, config: VinModelConfig) -> None:
        super().__init__()
        self.config = config

        self.backbone = self.config.backbone.setup_target()
        self.pose_encoder = self.config.pose_encoder.setup_target()

        # Head input dim is data-dependent (feature channel count depends on EVL cfg).
        self.head = self.config.head.setup_target(in_dim=None)
        # Keep the trainable head modules on the same device as the frozen backbone.
        # (EvlBackbone is not an nn.Module, so nn.Module.to() won't affect it.)
        self.to(self.backbone.device)

    def _get_reference_pose_world_rig(self, efm: Mapping[str, Any]) -> PoseTW:
        pose_tw = efm.get(ARIA_POSE_T_WORLD_RIG)
        if not isinstance(pose_tw, PoseTW):
            raise KeyError(f"Missing {ARIA_POSE_T_WORLD_RIG} PoseTW in efm snippet.")
        return PoseTW.from_matrix3x4(pose_tw.matrix3x4[..., -1, :, :])

    @staticmethod
    def _ensure_candidate_batch(candidate_poses_world_cam: PoseTW) -> PoseTW:
        if candidate_poses_world_cam.ndim == 2:  # N x 12
            return PoseTW(candidate_poses_world_cam._data.unsqueeze(0))
        return candidate_poses_world_cam

    def forward(
        self,
        efm: Mapping[str, Any],
        candidate_poses_world_cam: PoseTW,
        *,
        reference_pose_world_rig: PoseTW | None = None,
    ) -> VinPrediction:
        """Score candidate poses for one snippet.

        Args:
            efm: Raw EFM snippet dict.
            candidate_poses_world_cam: Candidate camera poses as world←camera.
                Shape can be ``(N,12)`` or ``(B,N,12)``.
            reference_pose_world_rig: Optional override for reference rig pose (world←rig).
                If omitted, uses the last pose in ``pose/t_world_rig`` from the snippet.

        Returns:
            :class:`VinPrediction` with CORAL logits and expected scores.
        """

        backbone_out = self.backbone.forward(efm)
        device = backbone_out.occ_feat.device
        dtype = backbone_out.occ_feat.dtype

        cand_w_c = self._ensure_candidate_batch(candidate_poses_world_cam).to(device=device)  # type: ignore[arg-type]
        b, n = cand_w_c.shape[0], cand_w_c.shape[1]

        if reference_pose_world_rig is None:
            ref_w_r = self._get_reference_pose_world_rig(efm).to(device=device)  # type: ignore[arg-type]
        else:
            ref_w_r = reference_pose_world_rig.to(device=device)  # type: ignore[arg-type]
        if ref_w_r.ndim == 1:
            ref_w_r = PoseTW(ref_w_r._data.unsqueeze(0))
        if ref_w_r.shape[0] != b:
            ref_w_r = PoseTW(ref_w_r._data.expand(b, 12))

        # ------------------------------------------------------------------ pose encoding (rig <- camera)
        t_r_w = ref_w_r.inverse()[:, None]  # B x 1 x 12
        t_r_c = t_r_w @ cand_w_c  # B x N x 12

        pose_vec = torch.cat(
            [
                t_r_c.t.to(dtype=torch.float32),
                so3log_map(t_r_c.R).to(dtype=torch.float32),
            ],
            dim=-1,
        )  # B N 6
        pose_enc = self.pose_encoder(pose_vec)  # B N E

        # ------------------------------------------------------------------ voxel feature queries
        parts: list[Tensor] = [pose_enc.to(device=device, dtype=dtype)]

        occ_feat = backbone_out.occ_feat if self.config.use_occ_feat else None
        obb_feat = backbone_out.obb_feat if self.config.use_obb_feat else None

        if not self.config.use_occ_feat and not self.config.use_obb_feat:
            raise ValueError("At least one of use_occ_feat/use_obb_feat must be True.")

        if self.config.use_global_pool:
            pooled: list[Tensor] = []
            if occ_feat is not None:
                pooled.append(occ_feat.mean(dim=(-3, -2, -1)))
            if obb_feat is not None:
                pooled.append(obb_feat.mean(dim=(-3, -2, -1)))
            global_feat = torch.cat(pooled, dim=-1).unsqueeze(1).expand(b, n, -1)
            parts.append(global_feat)

        if self.config.use_local_sample:
            if occ_feat is None and obb_feat is None:
                raise AssertionError("Unexpected: local sample enabled but no features selected.")

            d, h, w = backbone_out.occ_feat.shape[-3:]
            t_v_w = backbone_out.t_world_voxel.inverse()
            cand_centers_world = cand_w_c.t.to(dtype=dtype)
            cand_centers_voxel = t_v_w * cand_centers_world  # B N 3 in voxel frame (meters)
            voxel_extent = backbone_out.voxel_extent.to(device=device, dtype=torch.float32)
            if voxel_extent.ndim == 1:
                voxel_extent = voxel_extent.view(1, 6).expand(b, 6)
            cand_voxel_ids, valid = pc_to_vox(
                cand_centers_voxel,
                vW=int(w),
                vH=int(h),
                vD=int(d),
                voxel_extent=voxel_extent,
            )
            valid = valid & torch.isfinite(cand_voxel_ids).all(dim=-1)
            local_parts: list[Tensor] = []
            if occ_feat is not None:
                occ_samp, _ = sample_voxels(occ_feat, cand_voxel_ids, differentiable=False)
                local_parts.append(occ_samp.transpose(1, 2))
            if obb_feat is not None:
                obb_samp, _ = sample_voxels(obb_feat, cand_voxel_ids, differentiable=False)
                local_parts.append(obb_samp.transpose(1, 2))
            local_feat = torch.cat(local_parts, dim=-1)
            parts.append(local_feat)
        else:
            valid = torch.ones((b, n), device=device, dtype=torch.bool)

        feats = torch.cat(parts, dim=-1)
        logits = self.head(feats.reshape(b * n, -1)).reshape(b, n, -1)

        prob = coral_logits_to_prob(logits)
        expected, expected_norm = coral_expected_from_logits(logits)

        return VinPrediction(
            logits=logits,
            prob=prob,
            expected=expected,
            expected_normalized=expected_norm,
            candidate_valid=valid,
        )
