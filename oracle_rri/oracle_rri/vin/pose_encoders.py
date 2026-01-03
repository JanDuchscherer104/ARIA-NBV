"""Pose encoder variants for VIN candidates.

This module centralizes pose-encoding logic that was previously embedded in
`VinModelV2`, allowing different encoders to be selected via configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias

import torch
from efm3d.aria.pose import PoseTW
from pydantic import Field, field_validator
from pytorch3d.transforms import matrix_to_rotation_6d
from torch import Tensor, nn

from ..utils import BaseConfig
from .pose_encoding import LearnableFourierFeatures, LearnableFourierFeaturesConfig
from .spherical_encoding import ShellShPoseEncoderConfig


@dataclass(slots=True)
class PoseEncodingOutput:
    """Pose-encoding outputs for a pose expressed in a reference frame.

    Attributes:
        center_m: ``Tensor["... 3", float32]`` translation in the reference frame.
        pose_vec: ``Tensor["... D", float32]`` pose vector fed into the encoder.
        pose_enc: ``Tensor["... E", float32]`` encoded pose embedding.
        center_dir: Optional ``Tensor["... 3", float32]`` unit direction to center.
        forward_dir: Optional ``Tensor["... 3", float32]`` forward direction in ref frame.
        radius_m: Optional ``Tensor["... 1", float32]`` radius ``||t||``.
        view_alignment: Optional ``Tensor["... 1", float32]`` dot ``<f, -u>``.
    """

    center_m: Tensor
    """``Tensor["... 3", float32]`` translation in the reference frame."""

    pose_vec: Tensor
    """``Tensor["... D", float32]`` pose vector fed into the encoder."""

    pose_enc: Tensor
    """``Tensor["... E", float32]`` encoded pose embedding."""

    center_dir: Tensor | None = None
    """``Tensor["... 3", float32]`` unit center direction (optional)."""

    forward_dir: Tensor | None = None
    """``Tensor["... 3", float32]`` forward direction (optional)."""

    radius_m: Tensor | None = None
    """``Tensor["... 1", float32]`` radius ``||t||`` (optional)."""

    view_alignment: Tensor | None = None
    """``Tensor["... 1", float32]`` dot ``<f, -u>`` (optional)."""


class PoseEncoder(nn.Module):
    """Base interface for VIN pose encoders."""

    pose_encoder_lff: LearnableFourierFeatures | None
    """Optional LFF submodule for diagnostics (None for SH-only encoders)."""

    @property
    def out_dim(self) -> int:  # pragma: no cover - interface only
        """Return output embedding dimension."""
        raise NotImplementedError

    def encode(self, pose_rig: PoseTW) -> PoseEncodingOutput:  # pragma: no cover - interface only
        """Encode a pose expressed in the reference frame."""
        raise NotImplementedError


class R6dLffPoseEncoder(PoseEncoder):
    """Encode poses as translation + rotation-6D passed through LFF."""

    def __init__(self, config: "R6dLffPoseEncoderConfig") -> None:
        super().__init__()
        self.config = config
        self.pose_encoder_lff = self.config.pose_encoder_lff.setup_target()

        scale_init = torch.tensor(self.config.pose_scale_init, dtype=torch.float32)
        if self.config.pose_scale_learnable:
            self.pose_scale_log = nn.Parameter(torch.log(scale_init))
        else:
            self.register_buffer("pose_scale_log", torch.log(scale_init), persistent=False)
        self.pose_scale_eps = float(self.config.pose_scale_eps)

    @property
    def out_dim(self) -> int:
        return int(self.pose_encoder_lff.out_dim)

    def _pose_scales(self) -> tuple[Tensor, Tensor]:
        scales = self.pose_scale_log.exp() + self.pose_scale_eps
        return scales[0], scales[1]

    def encode(self, pose_rig: PoseTW) -> PoseEncodingOutput:
        """Encode poses in the reference rig frame.

        Args:
            pose_rig: ``PoseTW["... 12"]`` pose in reference rig frame.

        Returns:
            PoseEncodingOutput containing translation, pose vector, and embedding.
        """
        center_m = pose_rig.t.to(dtype=torch.float32)
        r6d = matrix_to_rotation_6d(pose_rig.R.to(dtype=torch.float32))
        scale_t, scale_r = self._pose_scales()
        pose_vec = torch.cat([center_m * scale_t, r6d * scale_r], dim=-1)
        pose_enc = self.pose_encoder_lff(pose_vec)
        return PoseEncodingOutput(center_m=center_m, pose_vec=pose_vec, pose_enc=pose_enc)


class ShellLffPoseEncoder(PoseEncoder):
    """Encode shell descriptors with LFF.

    Note: The shell descriptor uses only the forward direction. Roll about the
    forward axis is not represented; this is acceptable when roll jitter is
    small. Use the R6D encoder if roll needs to be captured.
    """

    def __init__(self, config: "ShellLffPoseEncoderConfig") -> None:
        super().__init__()
        self.config = config
        self.pose_encoder_lff = self.config.pose_encoder_lff.setup_target()

    @property
    def out_dim(self) -> int:
        return int(self.pose_encoder_lff.out_dim)

    def encode(self, pose_rig: PoseTW) -> PoseEncodingOutput:
        """Encode shell pose descriptors in the reference rig frame."""
        center_m = pose_rig.t.to(dtype=torch.float32)
        radius_m = torch.linalg.vector_norm(center_m, dim=-1, keepdim=True)
        center_dir = center_m / (radius_m + 1e-8)

        cam_forward_axis = torch.tensor(
            [0.0, 0.0, 1.0],
            device=center_m.device,
            dtype=torch.float32,
        )
        forward_dir = torch.einsum(
            "...ij,j->...i",
            pose_rig.R.to(dtype=torch.float32),
            cam_forward_axis,
        )
        forward_dir = forward_dir / (torch.linalg.vector_norm(forward_dir, dim=-1, keepdim=True) + 1e-8)
        view_alignment = (forward_dir * (-center_dir)).sum(dim=-1, keepdim=True)

        pose_vec = torch.cat([center_dir, forward_dir, radius_m, view_alignment], dim=-1)
        pose_enc = self.pose_encoder_lff(pose_vec)
        return PoseEncodingOutput(
            center_m=center_m,
            pose_vec=pose_vec,
            pose_enc=pose_enc,
            center_dir=center_dir,
            forward_dir=forward_dir,
            radius_m=radius_m,
            view_alignment=view_alignment,
        )


class ShellShPoseEncoderAdapter(PoseEncoder):
    """Encode shell descriptors with SH-based encoder.

    Note: The SH descriptor uses only the forward direction and therefore does
    not encode roll about the forward axis. This is acceptable when roll jitter
    is small; use R6D LFF if roll sensitivity is needed.
    """

    def __init__(self, config: "ShellShPoseEncoderAdapterConfig") -> None:
        super().__init__()
        self.config = config
        self.sh_encoder = self.config.sh_encoder.setup_target()
        self.pose_encoder_lff = None

    @property
    def out_dim(self) -> int:
        return int(self.sh_encoder.out_dim)

    def encode(self, pose_rig: PoseTW) -> PoseEncodingOutput:
        """Encode shell pose descriptors with spherical harmonics."""
        center_m = pose_rig.t.to(dtype=torch.float32)
        radius_m = torch.linalg.vector_norm(center_m, dim=-1, keepdim=True)
        center_dir = center_m / (radius_m + 1e-8)

        cam_forward_axis = torch.tensor(
            [0.0, 0.0, 1.0],
            device=center_m.device,
            dtype=torch.float32,
        )
        forward_dir = torch.einsum(
            "...ij,j->...i",
            pose_rig.R.to(dtype=torch.float32),
            cam_forward_axis,
        )
        forward_dir = forward_dir / (torch.linalg.vector_norm(forward_dir, dim=-1, keepdim=True) + 1e-8)
        view_alignment = (forward_dir * (-center_dir)).sum(dim=-1, keepdim=True)

        pose_vec = torch.cat([center_dir, forward_dir, radius_m, view_alignment], dim=-1)
        pose_enc = self.sh_encoder(center_dir, forward_dir, r=radius_m, scalars=view_alignment)
        return PoseEncodingOutput(
            center_m=center_m,
            pose_vec=pose_vec,
            pose_enc=pose_enc,
            center_dir=center_dir,
            forward_dir=forward_dir,
            radius_m=radius_m,
            view_alignment=view_alignment,
        )


class R6dLffPoseEncoderConfig(BaseConfig[R6dLffPoseEncoder]):
    """Config for :class:`R6dLffPoseEncoder`."""

    target: type[R6dLffPoseEncoder] = Field(default=R6dLffPoseEncoder, exclude=True)

    kind: Literal["r6d_lff"] = "r6d_lff"
    """Discriminator for pose-encoder selection."""

    pose_encoder_lff: LearnableFourierFeaturesConfig = Field(
        default_factory=lambda: LearnableFourierFeaturesConfig(
            input_dim=9,
            fourier_dim=64,
            hidden_dim=128,
            output_dim=32,
        ),
    )
    """LFF encoder for ``[t, r6d]`` pose vectors (input_dim=9)."""

    pose_scale_init: tuple[float, float] = (1.0, 1.0)
    """Initial per-group scale for translation and rotation (t, r6d)."""

    pose_scale_learnable: bool = True
    """Whether pose scaling parameters are learned."""

    pose_scale_eps: float = Field(default=1e-6, gt=0.0)
    """Numerical epsilon for pose scaling."""

    @field_validator("pose_encoder_lff")
    @classmethod
    def _validate_pose_encoder_lff(
        cls,
        value: LearnableFourierFeaturesConfig,
    ) -> LearnableFourierFeaturesConfig:
        if value.input_dim != 9:
            raise ValueError(
                "pose_encoder_lff.input_dim must be 9 for [t, r6d] pose vectors.",
            )
        return value

    @field_validator("pose_scale_init")
    @classmethod
    def _validate_pose_scale_init(
        cls,
        value: tuple[float, float],
    ) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError(
                "pose_scale_init must have two entries (t_scale, r6d_scale).",
            )
        if value[0] <= 0 or value[1] <= 0:
            raise ValueError("pose_scale_init values must be > 0.")
        return value


class ShellLffPoseEncoderConfig(BaseConfig[ShellLffPoseEncoder]):
    """Config for :class:`ShellLffPoseEncoder`."""

    target: type[ShellLffPoseEncoder] = Field(default=ShellLffPoseEncoder, exclude=True)

    kind: Literal["shell_lff"] = "shell_lff"
    """Discriminator for pose-encoder selection."""

    pose_encoder_lff: LearnableFourierFeaturesConfig = Field(
        default_factory=lambda: LearnableFourierFeaturesConfig(input_dim=8),
    )
    """LFF encoder for shell pose vectors (input_dim=8)."""

    @field_validator("pose_encoder_lff")
    @classmethod
    def _validate_pose_encoder_lff(
        cls,
        value: LearnableFourierFeaturesConfig,
    ) -> LearnableFourierFeaturesConfig:
        if value.input_dim != 8:
            raise ValueError(
                "pose_encoder_lff.input_dim must be 8 for [u, f, r, s] pose vectors.",
            )
        return value


class ShellShPoseEncoderAdapterConfig(BaseConfig[ShellShPoseEncoderAdapter]):
    """Config for :class:`ShellShPoseEncoderAdapter`."""

    target: type[ShellShPoseEncoderAdapter] = Field(
        default=ShellShPoseEncoderAdapter,
        exclude=True,
    )

    kind: Literal["shell_sh"] = "shell_sh"
    """Discriminator for pose-encoder selection."""

    sh_encoder: ShellShPoseEncoderConfig = Field(
        default_factory=ShellShPoseEncoderConfig,
    )
    """Spherical-harmonics encoder configuration."""


def infer_pose_vec_groups(pose_vec_dim: int) -> list[tuple[str, slice]]:
    """Infer semantic groups for a pose-encoder input vector.

    Args:
        pose_vec_dim: Size of the pose vector ``D``.

    Returns:
        Ordered list of ``(name, slice)`` entries describing how to split the
        pose vector into semantic groups. Falls back to a single ``pose_vec``
        group when the dimensionality is unknown.
    """
    dim = int(pose_vec_dim)
    if dim == 9:
        return [
            ("translation", slice(0, 3)),
            ("rotation6d", slice(3, 9)),
        ]
    if dim == 8:
        return [
            ("center_dir", slice(0, 3)),
            ("forward_dir", slice(3, 6)),
            ("radius", slice(6, 7)),
            ("view_alignment", slice(7, 8)),
        ]
    return [("pose_vec", slice(0, dim))]


PoseEncoderConfig: TypeAlias = Annotated[
    R6dLffPoseEncoderConfig | ShellLffPoseEncoderConfig | ShellShPoseEncoderAdapterConfig,
    Field(discriminator="kind"),
]


__all__ = [
    "PoseEncoder",
    "PoseEncoderConfig",
    "PoseEncodingOutput",
    "R6dLffPoseEncoder",
    "R6dLffPoseEncoderConfig",
    "ShellLffPoseEncoder",
    "ShellLffPoseEncoderConfig",
    "ShellShPoseEncoderAdapter",
    "ShellShPoseEncoderAdapterConfig",
    "infer_pose_vec_groups",
]
