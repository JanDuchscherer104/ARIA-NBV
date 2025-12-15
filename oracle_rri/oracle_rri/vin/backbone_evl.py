"""EVL backbone adapter for VIN.

VIN consumes raw EFM snippet dicts as input and queries EVL's 3D neck features
to score candidate poses.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
import omegaconf
import torch
from efm3d.aria.aria_constants import ARIA_IMG
from efm3d.aria.tensor_wrapper import TensorWrapper
from efm3d.dataset.wds_dataset import batchify
from pydantic import Field, ValidationInfo, field_validator

from ..configs import PathConfig
from ..utils import BaseConfig, Console
from .types import EvlBackboneOutput

Tensor = torch.Tensor


def _target_cls() -> type["EvlBackbone"]:
    return EvlBackbone


class EvlBackboneConfig(BaseConfig["EvlBackbone"]):
    """Configuration for :class:`EvlBackbone`."""

    target: type["EvlBackbone"] = Field(default_factory=_target_cls, exclude=True)
    """Factory target for :meth:`BaseConfig.setup_target`."""

    paths: PathConfig = Field(default_factory=PathConfig)
    """Project path resolver."""

    model_cfg: Path = Field(default_factory=lambda: Path(".configs") / "evl_inf_desktop.yaml")
    """Hydra YAML used to instantiate :class:`efm3d.model.evl.EVL`."""

    model_ckpt: Path = Field(default_factory=lambda: Path(".logs") / "ckpts" / "model_lite.pth")
    """Checkpoint containing an ``EVL`` state dict under ``['state_dict']``."""

    device: torch.device = Field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    """Torch device to run EVL on (auto-selects CUDA if available)."""

    freeze: bool = True
    """Disable gradients for all EVL parameters when True."""

    @field_validator("model_cfg", "model_ckpt", mode="before")
    @classmethod
    def _resolve_paths(cls, value: str | Path, info: ValidationInfo) -> Path:
        paths: PathConfig = info.data.get("paths") or PathConfig()
        path = Path(value)
        if not path.is_absolute():
            path = (paths.root / path).resolve()
        return path.expanduser().resolve()

    @field_validator("device", mode="before")
    @classmethod
    def _resolve_device(cls, value: str | torch.device) -> torch.device:
        if isinstance(value, torch.device):
            return value
        if value is None or str(value).lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(value)


class EvlBackbone:
    """Frozen EVL backbone wrapper that exposes neck features and voxel grid pose."""

    def __init__(self, config: EvlBackboneConfig) -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

        if not self.config.model_cfg.exists():
            raise FileNotFoundError(f"Missing EVL model cfg: {self.config.model_cfg}")
        if not self.config.model_ckpt.exists():
            raise FileNotFoundError(f"Missing EVL checkpoint: {self.config.model_ckpt}")

        self.device = torch.device(self.config.device)

        checkpoint = torch.load(self.config.model_ckpt, weights_only=True, map_location=self.device)
        model_config = omegaconf.OmegaConf.load(self.config.model_cfg)
        model = hydra.utils.instantiate(model_config)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(self.device)
        model.eval()

        if self.config.freeze:
            for param in model.parameters():
                param.requires_grad_(False)

        self.model = model
        self.voxel_extent = torch.tensor(getattr(model, "ve", [-2.0, 2.0, 0.0, 4.0, -2.0, 2.0]), dtype=torch.float32)

    def _prepare_batch(self, efm: Mapping[str, Any]) -> dict[str, Any]:
        batch: dict[str, Any] = dict(efm)

        img = batch.get(ARIA_IMG[0])
        needs_batchify = isinstance(img, torch.Tensor) and img.ndim == 4  # T C H W
        if needs_batchify:
            batchify(batch, device=self.device)
            return batch

        # Already batched (or missing rgb/img); just move tensor-like values.
        for key, value in batch.items():
            if isinstance(value, (torch.Tensor, TensorWrapper)):
                batch[key] = value.to(self.device)
        return batch

    def forward(self, efm: Mapping[str, Any]) -> EvlBackboneOutput:
        """Run EVL and return the feature volumes needed by VIN.

        Args:
            efm: Raw EFM snippet dict (unbatched ``T×...`` or batched ``B×T×...``).

        Returns:
            :class:`EvlBackboneOutput` with neck features and voxel grid pose.
        """

        batch = self._prepare_batch(efm)
        with torch.no_grad():
            out = self.model(batch)

        occ_feat = out["neck/occ_feat"]
        obb_feat = out["neck/obb_feat"]
        t_world_voxel = out["voxel/T_world_voxel"]
        voxel_extent = out.get("voxel_extent")
        if voxel_extent is None:
            voxel_extent = self.voxel_extent.to(device=occ_feat.device, dtype=occ_feat.dtype)
        if not isinstance(voxel_extent, torch.Tensor):
            raise TypeError(f"Expected voxel_extent Tensor, got {type(voxel_extent)}")

        return EvlBackboneOutput(
            occ_feat=occ_feat,
            obb_feat=obb_feat,
            t_world_voxel=t_world_voxel,
            voxel_extent=voxel_extent,
        )
