"""Serialization helpers for offline cache payloads."""

from __future__ import annotations

import sys
import types
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, Union, get_args, get_origin, get_type_hints

import torch
from efm3d.aria.camera import CameraTW
from efm3d.aria.obb import ObbTW
from efm3d.aria.pose import PoseTW
from efm3d.aria.tensor_wrapper import TensorWrapper
from pytorch3d.renderer.cameras import PerspectiveCameras  # type: ignore[import-untyped]

from ..pose_generation.types import CandidateSamplingResult
from ..rendering.candidate_depth_renderer import CandidateDepths
from ..rendering.candidate_pointclouds import CandidatePointClouds
from ..rri_metrics.types import RriResult
from ..vin.types import EvlBackboneOutput

Tensor = torch.Tensor


# ----------------------------------------------------------------------------- Tensor helpers
def _cpu_tensor(value: Tensor | None) -> Tensor | None:
    """Detach a tensor and move it onto CPU for serialization."""
    if value is None:
        return None
    return value.detach().cpu()


def _encode_p3d_cameras(cameras: PerspectiveCameras) -> dict[str, Any]:
    """Serialize a PyTorch3D camera batch into CPU tensors and scalars."""
    in_ndc = getattr(cameras, "in_ndc", False)
    if callable(in_ndc):
        in_ndc = in_ndc()
    znear = getattr(cameras, "znear", None)
    zfar = getattr(cameras, "zfar", None)
    return {
        "R": _cpu_tensor(cameras.R),
        "T": _cpu_tensor(cameras.T),
        "focal_length": _cpu_tensor(cameras.focal_length),
        "principal_point": _cpu_tensor(cameras.principal_point),
        "image_size": _cpu_tensor(cameras.image_size),
        "in_ndc": bool(in_ndc),
        "znear": _cpu_tensor(znear) if isinstance(znear, torch.Tensor) else znear,
        "zfar": _cpu_tensor(zfar) if isinstance(zfar, torch.Tensor) else zfar,
    }


def _decode_p3d_cameras(data: dict[str, Any], *, device: torch.device) -> PerspectiveCameras:
    """Rebuild a PyTorch3D camera batch from serialized payload data."""
    kwargs: dict[str, Any] = {
        "device": device,
        "R": data["R"].to(device=device),
        "T": data["T"].to(device=device),
        "focal_length": data["focal_length"].to(device=device),
        "principal_point": data["principal_point"].to(device=device),
        "image_size": data["image_size"].to(device=device),
        "in_ndc": bool(data["in_ndc"]),
    }
    znear = data.get("znear")
    zfar = data.get("zfar")
    if isinstance(znear, torch.Tensor):
        znear = znear.to(device=device)
    if isinstance(zfar, torch.Tensor):
        zfar = zfar.to(device=device)
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


# ----------------------------------------------------------------------------- Dataclass serialization
def _encode_value(value: Any) -> Any:
    """Recursively serialize tensors, wrappers, dataclasses, and containers."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.Tensor):
        return _cpu_tensor(value)
    if isinstance(value, (PoseTW, CameraTW)):
        return value.tensor().detach().cpu()
    if isinstance(value, TensorWrapper):
        return value.tensor().detach().cpu()
    if isinstance(value, PerspectiveCameras):
        return _encode_p3d_cameras(value)
    if is_dataclass(value):
        return encode_dataclass(value)
    if isinstance(value, dict):
        return {str(k): _encode_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_value(v) for v in value]
    return value


def _strip_optional(tp: Any) -> tuple[Any, bool]:
    """Return the non-optional inner type together with an optional flag."""
    origin = get_origin(tp)
    if origin is None:
        return tp, False

    union_types = {Union}
    union_type = getattr(types, "UnionType", None)
    if union_type is not None:
        union_types.add(union_type)

    if origin in union_types:
        args = get_args(tp)
        if args and type(None) in args:
            non_none = [arg for arg in args if arg is not type(None)]
            if len(non_none) == 1:
                return non_none[0], True
    return tp, False


def _decode_value(value: Any, expected_type: Any, *, device: torch.device | None) -> Any:
    """Recursively deserialize a payload value using the requested target type."""
    if value is None:
        return None

    expected_type, _ = _strip_optional(expected_type)
    origin = get_origin(expected_type)

    if expected_type in (PoseTW, CameraTW):
        tensor = value.to(device=device) if isinstance(value, torch.Tensor) and device else value
        return expected_type(tensor)
    if isinstance(expected_type, type) and issubclass(expected_type, TensorWrapper):
        tensor = value.to(device=device) if isinstance(value, torch.Tensor) and device else value
        return expected_type(tensor)
    if expected_type is PerspectiveCameras:
        if device is None:
            device = value["R"].device if isinstance(value, dict) and isinstance(value.get("R"), torch.Tensor) else None
        return _decode_p3d_cameras(value, device=device or torch.device("cpu"))
    if expected_type in (torch.Tensor, Tensor):
        return value.to(device=device) if isinstance(value, torch.Tensor) and device else value
    if expected_type is Any:
        return value.to(device=device) if isinstance(value, torch.Tensor) and device else value
    if is_dataclass(expected_type):
        return decode_dataclass(expected_type, value, device=device)
    if origin in (list, tuple):
        elem_type = get_args(expected_type)[0] if get_args(expected_type) else Any
        decoded = [_decode_value(v, elem_type, device=device) for v in value]
        return decoded if origin is list else tuple(decoded)
    if origin is dict:
        args = get_args(expected_type)
        val_type = args[1] if len(args) == 2 else Any
        return {k: _decode_value(v, val_type, device=device) for k, v in value.items()}

    if isinstance(value, torch.Tensor) and device is not None:
        return value.to(device=device)
    return value


def encode_dataclass(obj: Any, *, exclude: set[str] | None = None) -> dict[str, Any]:
    """Serialize a dataclass by encoding each field."""
    if not is_dataclass(obj):
        raise TypeError(f"Expected dataclass instance, got {type(obj)}")
    excluded = exclude or set()
    payload: dict[str, Any] = {}
    for field in fields(obj):
        if field.name in excluded:
            continue
        payload[field.name] = _encode_value(getattr(obj, field.name))
    return payload


def decode_dataclass(
    cls: type[Any],
    payload: dict[str, Any],
    *,
    device: torch.device | None,
    include_fields: set[str] | None = None,
) -> Any:
    """Deserialize a dataclass from a payload dict."""
    if not is_dataclass(cls):
        raise TypeError(f"Expected dataclass type, got {cls}")
    type_hints: dict[str, Any] = {}
    try:
        module = sys.modules.get(cls.__module__)
        globalns = dict(vars(module)) if module is not None else {}
        globalns.setdefault("PoseTW", PoseTW)
        globalns.setdefault("CameraTW", CameraTW)
        globalns.setdefault("ObbTW", ObbTW)
        globalns.setdefault("PerspectiveCameras", PerspectiveCameras)
        globalns.setdefault("Tensor", Tensor)
        type_hints = get_type_hints(cls, globalns=globalns, localns=globalns)
    except NameError:
        type_hints = {}
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


def encode_candidates(candidates: CandidateSamplingResult) -> dict[str, Any]:
    """Serialize CandidateSamplingResult to CPU tensors."""
    return encode_dataclass(candidates)


def decode_candidates(payload: dict[str, Any]) -> CandidateSamplingResult:
    """Deserialize CandidateSamplingResult from tensors (device preserved)."""
    return decode_dataclass(CandidateSamplingResult, payload, device=None)


def encode_depths(depths: CandidateDepths) -> dict[str, Any]:
    """Serialize CandidateDepths to CPU tensors."""
    return encode_dataclass(depths)


def decode_depths(payload: dict[str, Any], *, device: torch.device) -> CandidateDepths:
    """Deserialize CandidateDepths onto the requested device."""
    return decode_dataclass(CandidateDepths, payload, device=device)


def encode_candidate_pcs(pcs: CandidatePointClouds) -> dict[str, Any]:
    """Serialize CandidatePointClouds to CPU tensors."""
    return encode_dataclass(pcs)


def decode_candidate_pcs(payload: dict[str, Any], *, device: torch.device) -> CandidatePointClouds:
    """Deserialize CandidatePointClouds onto the requested device."""
    return decode_dataclass(CandidatePointClouds, payload, device=device)


def encode_rri(rri: RriResult) -> dict[str, Any]:
    """Serialize RriResult to CPU tensors."""
    return encode_dataclass(rri)


def decode_rri(payload: dict[str, Any], *, device: torch.device) -> RriResult:
    """Deserialize RriResult onto the requested device."""
    return decode_dataclass(RriResult, payload, device=device)


def encode_backbone(backbone: EvlBackboneOutput) -> dict[str, Any]:
    """Serialize EvlBackboneOutput to CPU tensors."""
    return encode_dataclass(backbone)


def decode_backbone(
    payload: dict[str, Any],
    *,
    device: torch.device,
    include_fields: set[str] | None = None,
) -> EvlBackboneOutput:
    """Deserialize EvlBackboneOutput onto the requested device."""
    return decode_dataclass(
        EvlBackboneOutput,
        payload,
        device=device,
        include_fields=include_fields,
    )


__all__ = [
    "decode_backbone",
    "decode_candidate_pcs",
    "decode_candidates",
    "decode_dataclass",
    "decode_depths",
    "decode_rri",
    "encode_backbone",
    "encode_candidate_pcs",
    "encode_candidates",
    "encode_dataclass",
    "encode_depths",
    "encode_rri",
]
