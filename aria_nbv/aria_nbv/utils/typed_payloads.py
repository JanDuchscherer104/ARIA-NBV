"""Shared typed-payload helpers for cache and offline-store serialization.

This module centralizes the project-wide logic for converting dataclass-based
runtime objects to CPU-only, msgspec-compatible payloads and reconstructing
them later. It supports the tensor-wrapper types commonly used across Aria-NBV,
including ``PoseTW``, ``CameraTW``, ``ObbTW``, generic ``TensorWrapper``
subclasses, and PyTorch3D ``PerspectiveCameras``.

The helpers here are intentionally shared across both ``aria_nbv.data`` and
``aria_nbv.data_handling`` so the project does not maintain duplicate custom
serialization code in multiple data packages.
"""

from __future__ import annotations

import sys
import types
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import msgspec
import numpy as np
import torch
from efm3d.aria.camera import CameraTW
from efm3d.aria.obb import ObbTW
from efm3d.aria.pose import PoseTW
from efm3d.aria.tensor_wrapper import TensorWrapper
from pytorch3d.renderer.cameras import PerspectiveCameras  # type: ignore[import-untyped]

Tensor = torch.Tensor
SerializableType = TypeVar("SerializableType")


class TensorPayload(msgspec.Struct, frozen=True):
    """Msgspec payload for a CPU tensor.

    Attributes:
        kind: Stable type tag used when decoding untyped payloads.
        dtype: NumPy dtype string for the stored array.
        shape: Full tensor shape.
        data: Contiguous raw bytes for the CPU tensor.
    """

    kind: str = "tensor"
    """Stable type tag used during untyped restoration."""

    dtype: str = ""
    """NumPy dtype string for the stored array."""

    shape: tuple[int, ...] = ()
    """Full tensor shape."""

    data: bytes = b""
    """Contiguous raw bytes for the CPU tensor."""

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> "TensorPayload":
        """Build a payload from a tensor.

        Args:
            tensor: Tensor to serialize.

        Returns:
            CPU payload describing the tensor.
        """

        array = tensor.detach().cpu().contiguous().numpy()
        return cls(
            dtype=str(array.dtype),
            shape=tuple(int(dim) for dim in array.shape),
            data=array.tobytes(order="C"),
        )

    def to_tensor(self, *, device: torch.device | None = None) -> Tensor:
        """Reconstruct the tensor represented by this payload.

        Args:
            device: Optional destination device.

        Returns:
            Reconstructed tensor.
        """

        array = np.frombuffer(self.data, dtype=np.dtype(self.dtype)).reshape(self.shape).copy()
        tensor = torch.from_numpy(array)
        return tensor.to(device=device) if device is not None else tensor


class PerspectiveCameraPayload(msgspec.Struct, frozen=True):
    """Msgspec payload for a PyTorch3D ``PerspectiveCameras`` batch."""

    kind: str = "p3d_camera"
    """Stable type tag used during untyped restoration."""

    R: TensorPayload = msgspec.field(default_factory=TensorPayload)
    """Rotation matrices."""

    T: TensorPayload = msgspec.field(default_factory=TensorPayload)
    """Translation vectors."""

    focal_length: TensorPayload = msgspec.field(default_factory=TensorPayload)
    """Focal lengths."""

    principal_point: TensorPayload = msgspec.field(default_factory=TensorPayload)
    """Principal points."""

    image_size: TensorPayload = msgspec.field(default_factory=TensorPayload)
    """Image sizes."""

    in_ndc: bool = False
    """Whether the batch is in NDC coordinates."""

    znear: TensorPayload | float | int | None = None
    """Optional near-plane value."""

    zfar: TensorPayload | float | int | None = None
    """Optional far-plane value."""

    @classmethod
    def from_cameras(cls, cameras: PerspectiveCameras) -> "PerspectiveCameraPayload":
        """Build a payload from a PyTorch3D camera batch.

        Args:
            cameras: Camera batch to serialize.

        Returns:
            CPU payload describing the camera batch.
        """

        in_ndc = getattr(cameras, "in_ndc", False)
        if callable(in_ndc):
            in_ndc = in_ndc()
        znear = getattr(cameras, "znear", None)
        zfar = getattr(cameras, "zfar", None)
        return cls(
            R=TensorPayload.from_tensor(cameras.R),
            T=TensorPayload.from_tensor(cameras.T),
            focal_length=TensorPayload.from_tensor(cameras.focal_length),
            principal_point=TensorPayload.from_tensor(cameras.principal_point),
            image_size=TensorPayload.from_tensor(cameras.image_size),
            in_ndc=bool(in_ndc),
            znear=TensorPayload.from_tensor(znear) if isinstance(znear, torch.Tensor) else znear,
            zfar=TensorPayload.from_tensor(zfar) if isinstance(zfar, torch.Tensor) else zfar,
        )

    def to_cameras(self, *, device: torch.device) -> PerspectiveCameras:
        """Reconstruct the camera batch represented by this payload.

        Args:
            device: Destination device.

        Returns:
            Reconstructed ``PerspectiveCameras`` batch.
        """

        kwargs: dict[str, Any] = {
            "device": device,
            "R": self.R.to_tensor(device=device),
            "T": self.T.to_tensor(device=device),
            "focal_length": self.focal_length.to_tensor(device=device),
            "principal_point": self.principal_point.to_tensor(device=device),
            "image_size": self.image_size.to_tensor(device=device),
            "in_ndc": bool(self.in_ndc),
        }
        znear = self.znear.to_tensor(device=device) if isinstance(self.znear, TensorPayload) else self.znear
        zfar = self.zfar.to_tensor(device=device) if isinstance(self.zfar, TensorPayload) else self.zfar
        if znear is not None:
            kwargs["znear"] = znear
        if zfar is not None:
            kwargs["zfar"] = zfar

        try:
            return PerspectiveCameras(**kwargs)
        except TypeError:
            kwargs.pop("znear", None)
            kwargs.pop("zfar", None)
            cameras = PerspectiveCameras(**kwargs)
            if znear is not None:
                cameras.znear = znear
            if zfar is not None:
                cameras.zfar = zfar
            return cameras


def to_serializable(value: Any, *, exclude: set[str] | None = None) -> Any:
    """Convert a runtime value into a msgspec-compatible CPU payload.

    Args:
        value: Runtime value to serialize.
        exclude: Optional dataclass field names to omit.

    Returns:
        Serializable builtins composed of scalars, lists, dicts, bytes, and
        tagged msgspec payloads.
    """

    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    if isinstance(value, torch.Tensor):
        return msgspec.to_builtins(TensorPayload.from_tensor(value))
    if isinstance(value, PerspectiveCameras):
        return msgspec.to_builtins(PerspectiveCameraPayload.from_cameras(value))
    if isinstance(value, (PoseTW, CameraTW, ObbTW)):
        return msgspec.to_builtins(TensorPayload.from_tensor(value.tensor()))
    if isinstance(value, TensorWrapper):
        return msgspec.to_builtins(TensorPayload.from_tensor(value.tensor()))
    if is_dataclass(value):
        omitted = exclude or set()
        return {
            field.name: to_serializable(getattr(value, field.name))
            for field in fields(value)
            if field.name not in omitted
        }
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def from_serializable(
    cls: type[SerializableType],
    payload: Any,
    *,
    device: torch.device | None,
    include_fields: set[str] | None = None,
) -> SerializableType:
    """Reconstruct a typed runtime object from a serialized payload.

    Args:
        cls: Target runtime type.
        payload: Serialized payload.
        device: Optional destination device for tensors and wrappers.
        include_fields: Optional subset of dataclass fields to decode.

    Returns:
        Reconstructed runtime object.
    """

    return _decode_value(payload, cls, device=device, include_fields=include_fields)


def _strip_optional(expected_type: Any) -> Any:
    """Return the inner type for ``Optional[T]`` values."""

    origin = get_origin(expected_type)
    if origin is None:
        return expected_type

    union_types = {Union}
    union_type = getattr(types, "UnionType", None)
    if union_type is not None:
        union_types.add(union_type)

    if origin in union_types:
        args = get_args(expected_type)
        if args and type(None) in args:
            non_none = [arg for arg in args if arg is not type(None)]
            if len(non_none) == 1:
                return non_none[0]
    return expected_type


def _restore_untyped(value: Any, *, device: torch.device | None) -> Any:
    """Restore tagged payloads nested inside untyped containers."""

    if isinstance(value, dict):
        normalized = _normalize_payload_dict(value)
        kind = normalized.get("kind")
        if kind == "tensor":
            return msgspec.convert(normalized, type=TensorPayload).to_tensor(device=device)
        if kind == "p3d_camera":
            target_device = device or torch.device("cpu")
            return msgspec.convert(normalized, type=PerspectiveCameraPayload).to_cameras(device=target_device)
        return {key: _restore_untyped(item, device=device) for key, item in normalized.items()}
    if isinstance(value, list):
        return [_restore_untyped(item, device=device) for item in value]
    return value


def _normalize_payload_dict(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy payload dictionaries into msgspec-compatible form.

    Args:
        value: Potentially legacy payload dictionary.

    Returns:
        Dictionary normalized for ``msgspec.convert``.
    """

    normalized: dict[str, Any] = {key: _normalize_payload_value(item) for key, item in value.items()}
    if {"dtype", "shape", "data"}.issubset(normalized):
        dtype = normalized["dtype"]
        if isinstance(dtype, torch.dtype):
            normalized["dtype"] = str(dtype).removeprefix("torch.")
        elif not isinstance(dtype, str):
            normalized["dtype"] = str(dtype)
        shape = normalized["shape"]
        if isinstance(shape, list):
            normalized["shape"] = tuple(int(dim) for dim in shape)
        normalized.setdefault("kind", "tensor")
    elif {
        "R",
        "T",
        "focal_length",
        "principal_point",
        "image_size",
    }.issubset(normalized):
        normalized.setdefault("kind", "p3d_camera")
    return normalized


def _normalize_payload_value(value: Any) -> Any:
    """Normalize nested legacy payload values recursively."""

    if isinstance(value, dict):
        return _normalize_payload_dict(value)
    if isinstance(value, list):
        return [_normalize_payload_value(item) for item in value]
    return value


def _move_tensor(value: Tensor, *, device: torch.device | None) -> Tensor:
    """Move a tensor to the requested device when needed."""

    return value.to(device=device) if device is not None else value


def _decode_legacy_perspective_cameras(value: dict[str, Any], *, device: torch.device) -> PerspectiveCameras:
    """Decode a legacy PyTorch3D camera payload stored as raw tensors.

    Args:
        value: Legacy camera payload dictionary.
        device: Destination device.

    Returns:
        Reconstructed ``PerspectiveCameras`` batch.
    """

    kwargs: dict[str, Any] = {
        "device": device,
        "R": _move_tensor(value["R"], device=device),
        "T": _move_tensor(value["T"], device=device),
        "focal_length": _move_tensor(value["focal_length"], device=device),
        "principal_point": _move_tensor(value["principal_point"], device=device),
        "image_size": _move_tensor(value["image_size"], device=device),
        "in_ndc": bool(value.get("in_ndc", False)),
    }
    znear = value.get("znear")
    zfar = value.get("zfar")
    if isinstance(znear, torch.Tensor):
        znear = _move_tensor(znear, device=device)
    if isinstance(zfar, torch.Tensor):
        zfar = _move_tensor(zfar, device=device)
    if znear is not None:
        kwargs["znear"] = znear
    if zfar is not None:
        kwargs["zfar"] = zfar
    try:
        return PerspectiveCameras(**kwargs)
    except TypeError:
        kwargs.pop("znear", None)
        kwargs.pop("zfar", None)
        cameras = PerspectiveCameras(**kwargs)
        if znear is not None:
            cameras.znear = znear
        if zfar is not None:
            cameras.zfar = zfar
        return cameras


def _resolve_type_hints(cls: type[Any]) -> dict[str, Any]:
    """Resolve runtime type hints for a dataclass, tolerating forward refs."""

    try:
        module = sys.modules.get(cls.__module__)
        globalns = dict(vars(module)) if module is not None else {}
        globalns.setdefault("PoseTW", PoseTW)
        globalns.setdefault("CameraTW", CameraTW)
        globalns.setdefault("ObbTW", ObbTW)
        globalns.setdefault("PerspectiveCameras", PerspectiveCameras)
        globalns.setdefault("Tensor", Tensor)
        return get_type_hints(cls, globalns=globalns, localns=globalns)
    except NameError:
        return {}


def _decode_dataclass(
    cls: type[SerializableType],
    payload: dict[str, Any],
    *,
    device: torch.device | None,
    include_fields: set[str] | None = None,
) -> SerializableType:
    """Decode a dataclass payload using field-level type hints."""

    type_hints = _resolve_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for field in fields(cls):
        if include_fields is not None and field.name not in include_fields:
            if field.default is not MISSING or field.default_factory is not MISSING:
                continue
            raise KeyError(f"Missing required field '{field.name}' for {cls.__name__}")
        if field.name not in payload:
            if field.default is not MISSING or field.default_factory is not MISSING:
                continue
            raise KeyError(f"Missing field '{field.name}' for {cls.__name__}")
        expected_type = type_hints.get(field.name, field.type)
        kwargs[field.name] = _decode_value(payload[field.name], expected_type, device=device)
    return cls(**kwargs)


def _decode_value(
    value: Any,
    expected_type: Any,
    *,
    device: torch.device | None,
    include_fields: set[str] | None = None,
) -> Any:
    """Decode one serialized value into the requested target type."""

    if value is None:
        return None

    expected_type = _strip_optional(expected_type)
    origin = get_origin(expected_type)

    if expected_type in (torch.Tensor, Tensor):
        if isinstance(value, torch.Tensor):
            return _move_tensor(value, device=device)
        return msgspec.convert(_normalize_payload_value(value), type=TensorPayload).to_tensor(device=device)
    if expected_type in (PoseTW, CameraTW, ObbTW):
        if isinstance(value, torch.Tensor):
            return expected_type(_move_tensor(value, device=device))
        tensor = msgspec.convert(_normalize_payload_value(value), type=TensorPayload).to_tensor(device=device)
        return expected_type(tensor)
    if isinstance(expected_type, type) and issubclass(expected_type, TensorWrapper):
        if isinstance(value, torch.Tensor):
            return expected_type(_move_tensor(value, device=device))
        tensor = msgspec.convert(_normalize_payload_value(value), type=TensorPayload).to_tensor(device=device)
        return expected_type(tensor)
    if expected_type is PerspectiveCameras:
        target_device = device or torch.device("cpu")
        if isinstance(value, PerspectiveCameras):
            return value.to(device=target_device)
        if isinstance(value, dict) and {
            "R",
            "T",
            "focal_length",
            "principal_point",
            "image_size",
        }.issubset(value):
            if all(
                isinstance(value[key], torch.Tensor)
                for key in ("R", "T", "focal_length", "principal_point", "image_size")
            ):
                return _decode_legacy_perspective_cameras(value, device=target_device)
        return msgspec.convert(_normalize_payload_value(value), type=PerspectiveCameraPayload).to_cameras(
            device=target_device
        )
    if expected_type is Any:
        return _restore_untyped(value, device=device)
    if is_dataclass(expected_type):
        return _decode_dataclass(expected_type, value, device=device, include_fields=include_fields)
    if origin in (list, tuple):
        args = get_args(expected_type)
        elem_type = args[0] if args else Any
        decoded = [_decode_value(item, elem_type, device=device) for item in value]
        return decoded if origin is list else tuple(decoded)
    if origin is dict:
        args = get_args(expected_type)
        val_type = args[1] if len(args) == 2 else Any
        return {key: _decode_value(item, val_type, device=device) for key, item in value.items()}
    return value


__all__ = [
    "PerspectiveCameraPayload",
    "TensorPayload",
    "from_serializable",
    "to_serializable",
]
