"""Backend profile resolution for the oracle RRI pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

import torch

from ..pose_generation.types import CollisionBackend
from ..rendering import DepthRendererBackend
from ..rendering.candidate_pointclouds import PointCloudBackend
from ..rri_metrics.oracle_rri import OracleDistanceBackend
from ..utils.devices import TorchAccelerator, is_mps_available

if TYPE_CHECKING:
    from .oracle_rri_labeler import OracleRriLabelerConfig


class OracleBackendProfile(StrEnum):
    """Mutually exclusive production backend profiles for oracle RRI."""

    PYTORCH3D_CUDA = "pytorch3d_cuda"
    APPLE_MPS_MOJO = "apple_mps_mojo"


class OracleBackendProfileError(RuntimeError):
    """Raised when a selected oracle backend profile cannot be resolved."""


def accelerator_options_for_profile(profile: OracleBackendProfile | str) -> tuple[TorchAccelerator, ...]:
    """Return UI-safe accelerator choices for a production profile."""

    profile = OracleBackendProfile(profile)
    if profile == OracleBackendProfile.PYTORCH3D_CUDA:
        return (TorchAccelerator.AUTO, TorchAccelerator.CUDA)
    if profile == OracleBackendProfile.APPLE_MPS_MOJO:
        return (TorchAccelerator.AUTO, TorchAccelerator.MPS)
    raise OracleBackendProfileError(f"Unsupported oracle backend profile: {profile}")


def resolve_oracle_backend_profile(
    config: "OracleRriLabelerConfig",
    *,
    require_available: bool = True,
) -> "OracleRriLabelerConfig":
    """Return a deep-copied config with the selected backend profile applied.

    The input config is never mutated. Production profiles apply all stage
    backends atomically; explicit mixed stage backends require
    ``allow_backend_overrides=True`` and are intended for tests/diagnostics.
    """

    profile = OracleBackendProfile(config.backend_profile)
    accelerator = TorchAccelerator(config.torch_accelerator)
    allow_overrides = bool(config.allow_backend_overrides)
    profile_device = _resolve_profile_device(
        profile=profile,
        accelerator=accelerator,
        allow_overrides=allow_overrides,
    )
    stage_devices = _stage_devices(profile=profile, profile_device=profile_device)
    effective_device = profile_device
    if allow_overrides and _field_was_set(config, "device"):
        effective_device = torch.device(config.device)
        stage_devices = {"generator": effective_device, "depth": effective_device}

    expected = _expected_backends(profile)
    if not allow_overrides:
        _reject_explicit_mixed_backends(config, expected)
        _reject_explicit_mixed_devices(config, expected_device=profile_device, stage_devices=stage_devices)

    resolved = config.model_copy(deep=True)
    _apply_profile_to_copy(
        source=config,
        target=resolved,
        expected=expected,
        device=effective_device,
        stage_devices=stage_devices,
        allow_overrides=allow_overrides,
    )

    if require_available:
        _raise_if_unavailable(resolved, production_profile=not allow_overrides)

    return resolved


def _expected_backends(profile: OracleBackendProfile) -> dict[str, StrEnum]:
    if profile == OracleBackendProfile.PYTORCH3D_CUDA:
        return {
            "collision_backend": CollisionBackend.P3D,
            "depth_backend": DepthRendererBackend.PYTORCH3D,
            "pointcloud_backend": PointCloudBackend.PYTORCH3D,
            "oracle_backend": OracleDistanceBackend.PYTORCH3D,
        }
    if profile == OracleBackendProfile.APPLE_MPS_MOJO:
        return {
            "collision_backend": CollisionBackend.MOJO,
            "depth_backend": DepthRendererBackend.MOJO,
            "pointcloud_backend": PointCloudBackend.MOJO,
            "oracle_backend": OracleDistanceBackend.MOJO,
        }
    raise OracleBackendProfileError(f"Unsupported oracle backend profile: {profile}")


def _resolve_profile_device(
    *,
    profile: OracleBackendProfile,
    accelerator: TorchAccelerator,
    allow_overrides: bool,
) -> torch.device:
    if profile == OracleBackendProfile.PYTORCH3D_CUDA:
        if accelerator in (TorchAccelerator.AUTO, TorchAccelerator.CUDA):
            return torch.device("cuda")
        if accelerator == TorchAccelerator.CPU and allow_overrides:
            return torch.device("cpu")
        raise OracleBackendProfileError(
            "`pytorch3d_cuda` supports only the CUDA accelerator in production.",
        )
    if profile == OracleBackendProfile.APPLE_MPS_MOJO:
        if accelerator in (TorchAccelerator.AUTO, TorchAccelerator.MPS):
            return torch.device("mps")
        if accelerator == TorchAccelerator.CPU and allow_overrides:
            return torch.device("cpu")
        raise OracleBackendProfileError(
            "`apple_mps_mojo` supports only the MPS accelerator in production.",
        )
    raise OracleBackendProfileError(f"Unsupported oracle backend profile: {profile}")


def _stage_devices(*, profile: OracleBackendProfile, profile_device: torch.device) -> dict[str, torch.device]:
    if profile == OracleBackendProfile.APPLE_MPS_MOJO:
        # Candidate sampling still uses EFM/PoseTW-heavy tensor ops that are not
        # stable on MPS in the Streamlit path. Keep this fallback explicit and
        # profile-owned instead of letting app code mutate devices.
        return {"generator": torch.device("cpu"), "depth": profile_device}
    return {"generator": profile_device, "depth": profile_device}


def _field_was_set(model: Any, field_name: str) -> bool:
    return field_name in getattr(model, "model_fields_set", set())


def _device_matches(value: torch.device | str, expected_device: torch.device) -> bool:
    return torch.device(value) == expected_device


def _reject_explicit_mixed_backends(config: "OracleRriLabelerConfig", expected: dict[str, StrEnum]) -> None:
    checks = (
        (config.generator, "collision_backend", expected["collision_backend"], "generator.collision_backend"),
        (config.depth, "backend", expected["depth_backend"], "depth.backend"),
        (config.pointcloud, "backend", expected["pointcloud_backend"], "pointcloud.backend"),
        (config.oracle, "backend", expected["oracle_backend"], "oracle.backend"),
    )
    for model, field_name, expected_value, label in checks:
        if _field_was_set(model, field_name) and getattr(model, field_name) != expected_value:
            raise OracleBackendProfileError(
                f"{label}={getattr(model, field_name)!s} conflicts with "
                f"backend_profile={config.backend_profile!s}; expected {expected_value!s}. "
                "Set allow_backend_overrides=True only for tests or diagnostics.",
            )


def _reject_explicit_mixed_devices(
    config: "OracleRriLabelerConfig",
    *,
    expected_device: torch.device,
    stage_devices: dict[str, torch.device],
) -> None:
    checks = (
        (config, "device", "device"),
        (config.generator, "device", "generator.device"),
        (config.depth, "device", "depth.device"),
    )
    for model, field_name, label in checks:
        field_expected_device = stage_devices["generator"] if label == "generator.device" else expected_device
        if _field_was_set(model, field_name) and not _device_matches(getattr(model, field_name), field_expected_device):
            raise OracleBackendProfileError(
                f"{label}={getattr(model, field_name)!s} conflicts with "
                f"backend_profile={config.backend_profile!s}; expected {field_expected_device!s}. "
                "Set allow_backend_overrides=True only for tests or diagnostics.",
            )

    nested_renderer = (
        config.depth.pytorch3d if config.backend_profile == OracleBackendProfile.PYTORCH3D_CUDA else config.depth.mojo
    )
    nested_label = (
        "depth.pytorch3d.device"
        if config.backend_profile == OracleBackendProfile.PYTORCH3D_CUDA
        else "depth.mojo.device"
    )
    if _field_was_set(nested_renderer, "device") and not _device_matches(nested_renderer.device, expected_device):
        raise OracleBackendProfileError(
            f"{nested_label}={nested_renderer.device!s} conflicts with "
            f"backend_profile={config.backend_profile!s}; expected {expected_device!s}. "
            "Set allow_backend_overrides=True only for tests or diagnostics.",
        )


def _set_profile_field(
    *,
    source: Any,
    target: Any,
    field_name: str,
    value: Any,
    allow_overrides: bool,
) -> None:
    if allow_overrides and _field_was_set(source, field_name):
        return
    object.__setattr__(target, field_name, value)


def _apply_profile_to_copy(
    *,
    source: "OracleRriLabelerConfig",
    target: "OracleRriLabelerConfig",
    expected: dict[str, StrEnum],
    device: torch.device,
    stage_devices: dict[str, torch.device],
    allow_overrides: bool,
) -> None:
    _set_profile_field(source=source, target=target, field_name="device", value=device, allow_overrides=allow_overrides)
    _set_profile_field(
        source=source.generator,
        target=target.generator,
        field_name="collision_backend",
        value=expected["collision_backend"],
        allow_overrides=allow_overrides,
    )
    _set_profile_field(
        source=source.generator,
        target=target.generator,
        field_name="device",
        value=stage_devices["generator"],
        allow_overrides=allow_overrides,
    )
    _set_profile_field(
        source=source.depth,
        target=target.depth,
        field_name="backend",
        value=expected["depth_backend"],
        allow_overrides=allow_overrides,
    )
    _set_profile_field(
        source=source.depth,
        target=target.depth,
        field_name="device",
        value=stage_devices["depth"],
        allow_overrides=allow_overrides,
    )
    _set_profile_field(
        source=source.depth.pytorch3d,
        target=target.depth.pytorch3d,
        field_name="device",
        value=stage_devices["depth"],
        allow_overrides=allow_overrides,
    )
    _set_profile_field(
        source=source.depth.mojo,
        target=target.depth.mojo,
        field_name="device",
        value=stage_devices["depth"],
        allow_overrides=allow_overrides,
    )
    _set_profile_field(
        source=source.pointcloud,
        target=target.pointcloud,
        field_name="backend",
        value=expected["pointcloud_backend"],
        allow_overrides=allow_overrides,
    )
    _set_profile_field(
        source=source.oracle,
        target=target.oracle,
        field_name="backend",
        value=expected["oracle_backend"],
        allow_overrides=allow_overrides,
    )


def _raise_if_unavailable(config: "OracleRriLabelerConfig", *, production_profile: bool) -> None:
    errors: list[str] = []
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        errors.append("Torch CUDA is not available.")
    if device.type == "mps" and not is_mps_available():
        errors.append("Torch MPS is not available.")

    if production_profile:
        if config.backend_profile == OracleBackendProfile.PYTORCH3D_CUDA:
            _append_pytorch3d_errors(errors)
            if device.type != "cuda":
                errors.append("The `pytorch3d_cuda` profile requires device=cuda.")
        elif config.backend_profile == OracleBackendProfile.APPLE_MPS_MOJO:
            _append_mojo_errors(errors)
            if device.type != "mps":
                errors.append("The `apple_mps_mojo` profile requires device=mps.")
    else:
        if _uses_pytorch3d_backend(config):
            _append_pytorch3d_errors(errors)
        if _uses_mojo_backend(config):
            _append_mojo_errors(errors)

    if errors:
        profile = (
            config.backend_profile.value
            if isinstance(config.backend_profile, OracleBackendProfile)
            else str(config.backend_profile)
        )
        raise OracleBackendProfileError(
            f"Oracle backend profile `{profile}` is unavailable:\n- " + "\n- ".join(errors),
        )


def _append_pytorch3d_errors(errors: list[str]) -> None:
    from ..utils.pytorch3d_compat import pytorch3d_import_error

    import_error = pytorch3d_import_error()
    if import_error is not None:
        errors.append(f"PyTorch3D import failed: {import_error}")


def _append_mojo_errors(errors: list[str]) -> None:
    from ..pose_generation.mojo_backend import is_mojo_available as pose_mojo_available
    from ..rendering.mojo_backend import is_mojo_available as rendering_mojo_available
    from ..rri_metrics.mojo_backend import is_mojo_available as rri_mojo_available

    if not pose_mojo_available():
        errors.append("Mojo pose-generation kernels are not importable.")
    if not rendering_mojo_available():
        errors.append("Mojo rendering kernels are not importable.")
    if not rri_mojo_available():
        errors.append("Mojo RRI distance kernels are not importable.")


def _uses_pytorch3d_backend(config: "OracleRriLabelerConfig") -> bool:
    return (
        config.generator.collision_backend == CollisionBackend.P3D
        or config.depth.backend == DepthRendererBackend.PYTORCH3D
        or config.pointcloud.backend == PointCloudBackend.PYTORCH3D
        or config.oracle.backend == OracleDistanceBackend.PYTORCH3D
    )


def _uses_mojo_backend(config: "OracleRriLabelerConfig") -> bool:
    return (
        config.generator.collision_backend == CollisionBackend.MOJO
        or config.depth.backend == DepthRendererBackend.MOJO
        or config.pointcloud.backend == PointCloudBackend.MOJO
        or config.oracle.backend == OracleDistanceBackend.MOJO
    )


__all__ = [
    "OracleBackendProfile",
    "OracleBackendProfileError",
    "accelerator_options_for_profile",
    "resolve_oracle_backend_profile",
]
