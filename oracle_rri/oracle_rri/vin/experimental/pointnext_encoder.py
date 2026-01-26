"""PointNeXt-S adapter for semidense point cloud features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from pydantic import Field, field_validator
from torch import Tensor, nn

from oracle_rri.configs.path_config import PathConfig

from ...utils import BaseConfig, Optimizable, optimizable_field


def _extract_tensor(output: Any) -> Tensor:
    """Best-effort extractor for tensors returned by external point encoders."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("feat", "features", "logits", "pred", "out"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError("Point encoder output did not contain a tensor.")


def _load_pointnext_cfg(cfg_path: Path) -> Any:
    """Load a PointNeXt/OpenPoints YAML config with EasyConfig."""
    from openpoints.utils import EasyConfig

    cfg = EasyConfig()
    cfg.load(str(cfg_path), recursive=True)
    return cfg


def _load_checkpoint_strict(model: nn.Module, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("model", "net", "network", "state_dict", "base_model"):
            if key in state:
                state = state[key]
                break
    if not isinstance(state, dict):
        raise RuntimeError("PointNeXt checkpoint did not contain a state dict.")
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


class PointNeXtSEncoderConfig(BaseConfig["PointNeXtSEncoder"]):
    """Configuration for the optional PointNeXt-S semidense encoder."""

    target: type["PointNeXtSEncoder"] = Field(default_factory=lambda: PointNeXtSEncoder, exclude=True)
    """Factory target for :meth:`~oracle_rri.utils.base_config.BaseConfig.setup_target`."""

    cfg_path: Path = Field(default_factory=lambda: Path("PointNeXt/cfgs/s3dis/pointnext-s.yaml"))  #
    """Path to the PointNeXt YAML config (relative to relative to PathConfig().external)."""

    checkpoint_path: Path = Field(
        default_factory=lambda: Path(
            "s3dis-train-pointnext-s-ngpus1-seed1742-20220525-162639-Gz4ViiL95b6KP9MMH2ytG3_ckpt_best.pth"
        )
    )
    """Optional pretrained checkpoint (PointNeXt/OpenPoints format)."""

    out_dim: int = optimizable_field(
        default=128,
        optimizable=Optimizable.discrete(
            low=64,
            high=256,
            step=64,
            description="Output embedding dimension for the semidense point encoder.",
            relies_on={"module_config.vin.use_point_encoder": (True,)},
        ),
        gt=0,
    )
    """Output embedding dimension produced for the semidense point cloud."""

    max_points: int = optimizable_field(
        default=3000,
        optimizable=Optimizable.discrete(
            low=1000,
            high=6000,
            step=500,
            description="Maximum semidense points to encode per snippet.",
            relies_on={"module_config.vin.use_point_encoder": (True,)},
        ),
        gt=0,
    )
    """Subsample semidense points to this count before encoding."""

    freeze: bool = True
    """Whether to freeze PointNeXt-S weights during training."""

    strict_load: bool = False
    """Whether to enforce strict checkpoint loading."""

    @field_validator("checkpoint_path", mode="before")
    @classmethod
    def _resolve_checkpoint_path(cls, v: Path | str) -> Path:
        return PathConfig().resolve_external_checkpoint_path(v)

    @field_validator("cfg_path", mode="before")
    @classmethod
    def _resolve_cfg_path(cls, v: Path | str) -> Path:
        pth = PathConfig().external_dir / v
        assert pth.exists(), f"PointNeXt cfg path does not exist: {pth}"

        return pth


class PointNeXtSEncoder(nn.Module):
    """Optional PointNeXt-S adapter for semidense point cloud features."""

    def __init__(self, config: PointNeXtSEncoderConfig) -> None:
        super().__init__()
        self.config = config
        try:
            from openpoints.models.build import build_model_from_cfg
            from openpoints.utils import load_checkpoint
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "PointNeXt-S requires the openpoints package. Install it or disable the point encoder.",
            ) from exc

        cfg_path = Path(self.config.cfg_path)

        cfg = _load_pointnext_cfg(cfg_path)
        model_cfg = cfg.model if hasattr(cfg, "model") else cfg["model"]
        encoder_args = getattr(model_cfg, "encoder_args", None)
        if encoder_args is None and isinstance(model_cfg, dict):
            encoder_args = model_cfg.get("encoder_args")
        if encoder_args is not None and not isinstance(encoder_args, dict):
            encoder_args = getattr(encoder_args, "__dict__", encoder_args)
        in_channels = None
        if isinstance(encoder_args, dict):
            in_channels = encoder_args.get("in_channels")
        self.input_channels = int(in_channels or 3)
        self.model = build_model_from_cfg(model_cfg)
        self.model.eval()

        if self.config.checkpoint_path is not None:
            ckpt_path = Path(self.config.checkpoint_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"PointNeXt-S checkpoint_path does not exist: {ckpt_path}")
            if self.config.strict_load:
                _load_checkpoint_strict(self.model, ckpt_path)
            else:
                load_checkpoint(self.model, str(ckpt_path))

        if self.config.freeze:
            for param in self.model.parameters():
                param.requires_grad_(False)

        raw_dim = None
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "out_channels"):
            raw_dim = int(self.model.encoder.out_channels)
        elif hasattr(self.model, "out_channels"):
            raw_dim = int(self.model.out_channels)

        if raw_dim is None:
            with torch.no_grad():
                dummy = torch.zeros((1, 32, 3), dtype=torch.float32, device="cuda")
                self.model.to(dummy.device)
                raw = _extract_tensor(self._forward_features(dummy))
                raw_dim = int(raw.shape[-1])

        self.out_dim = int(self.config.out_dim)
        self.proj = nn.Identity() if raw_dim == self.out_dim else nn.Linear(raw_dim, self.out_dim)

    def train(self, mode: bool = True) -> "PointNeXtSEncoder":
        super().train(mode)
        if self.config.freeze:
            self.model.eval()
        return self

    def _forward_features(self, points: Tensor, features: Tensor | None = None) -> Tensor:
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "forward_cls_feat"):
            if features is None:
                return self.model.encoder.forward_cls_feat(points)
            return self.model.encoder.forward_cls_feat(points, features)
        if hasattr(self.model, "forward_cls_feat"):
            if features is None:
                return self.model.forward_cls_feat(points)
            return self.model.forward_cls_feat(points, features)
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features(points)
        return self.model(points)

    def forward(self, points: Tensor) -> Tensor:
        """Encode point clouds into a compact semidense embedding.

        Args:
            points: ``Tensor["B N 3", float32]`` or ``Tensor["B N C", float32]`` semidense points
                in the rig frame. When ``C > 3``, the first three channels are XYZ and the
                remaining channels are treated as per-point features.

        Returns:
            ``Tensor["B out_dim", float32]`` semidense embeddings.
        """
        if points.ndim != 3:
            raise ValueError(f"Expected points shape (B,N,3), got {tuple(points.shape)}.")
        if points.shape[-1] != 3 and points.shape[1] == 3:
            points = points.transpose(1, 2)
        if points.device.type != "cuda":
            raise RuntimeError(
                "PointNeXt-S uses CUDA-only ops; move points to CUDA or disable the point encoder.",
            )
        if next(self.model.parameters()).device != points.device:
            self.model.to(points.device)
        xyz = points[..., :3].contiguous()
        features: Tensor | None = None
        if points.shape[-1] > 3:
            features = points

        if self.input_channels > 3 and features is None:
            pad = torch.zeros(
                (*xyz.shape[:-1], self.input_channels - 3),
                device=xyz.device,
                dtype=xyz.dtype,
            )
            features = torch.cat([xyz, pad], dim=-1)

        if features is not None:
            if features.shape[-1] > self.input_channels:
                features = features[..., : self.input_channels]
            elif features.shape[-1] < self.input_channels:
                pad = torch.zeros(
                    (*features.shape[:-1], self.input_channels - features.shape[-1]),
                    device=features.device,
                    dtype=features.dtype,
                )
                features = torch.cat([features, pad], dim=-1)
            raw = _extract_tensor(self._forward_features(xyz, features.transpose(1, 2).contiguous()))
        else:
            raw = _extract_tensor(self._forward_features(xyz))
        if raw.ndim > 2:
            reduce_dims = tuple(range(2, raw.ndim))
            raw = raw.mean(dim=reduce_dims)
        raw = raw.to(dtype=self.proj.weight.dtype if isinstance(self.proj, nn.Linear) else raw.dtype)
        if isinstance(self.proj, nn.Linear) and self.proj.weight.device != raw.device:
            self.proj.to(raw.device)
        return self.proj(raw)


__all__ = ["PointNeXtSEncoder", "PointNeXtSEncoderConfig"]
