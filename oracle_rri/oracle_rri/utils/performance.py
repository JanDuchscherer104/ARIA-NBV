"""Global performance / accelerator preferences.

Provides a single place to express whether the NBV stack should prefer GPU
execution. Components can call :func:`select_device` to resolve torch devices
consistently and :func:`pick_fast_depth_renderer` to upgrade depth-renderer
configs to the fastest available backend.

The mode is controlled by :data:`NBV_PERFORMANCE_MODE` environment variable
(`auto`|`gpu`|`cpu`) or programmatically via :func:`set_performance_mode`.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from oracle_rri.rendering.candidate_depth_renderer import RendererConfig


ENV_VAR = "NBV_PERFORMANCE_MODE"


class PerformanceMode(str, Enum):
    """Global accelerator preference."""

    AUTO = "auto"
    GPU = "gpu"
    CPU = "cpu"


def _initial_mode() -> PerformanceMode:
    env = os.getenv(ENV_VAR, "").strip().lower()
    if env in {"gpu", "cuda", "on"}:
        return PerformanceMode.GPU
    if env in {"cpu", "off"}:
        return PerformanceMode.CPU
    return PerformanceMode.AUTO


_MODE: PerformanceMode = _initial_mode()


def get_performance_mode() -> PerformanceMode:
    """Return the current global performance mode."""

    return _MODE


def set_performance_mode(mode: PerformanceMode | str) -> PerformanceMode:
    """Set the global performance mode.

    Args:
        mode: One of ``PerformanceMode`` or ``{"auto", "gpu", "cpu"}``.

    Returns:
        The normalised mode.
    """

    global _MODE
    _MODE = PerformanceMode(mode)
    return _MODE


def prefer_gpu() -> bool:
    """Whether GPU/CUDA should be preferred for new components."""

    if _MODE == PerformanceMode.GPU:
        return True
    if _MODE == PerformanceMode.AUTO:
        return torch.cuda.is_available()
    return False


def prefer_cpu() -> bool:
    """Whether CPU should be forced regardless of CUDA availability."""

    return _MODE == PerformanceMode.CPU


def _warn(component: str | None, message: str) -> None:
    if component is None:
        return
    try:  # Lazy import to avoid circular deps.
        from .console import Console

        Console.with_prefix(component, "performance").warn(message)
    except Exception:
        # Logging failures should never break device resolution.
        pass


def select_device(
    preferred: str | torch.device | None = None,
    *,
    allow_cpu_fallback: bool = True,
    component: str | None = None,
) -> torch.device:
    """Resolve a torch device given global performance mode.

    Args:
        preferred: Optional device string/instance (``"cuda"``, ``"cpu"``,
            ``"auto"``). When ``None`` the global performance mode is used.
        allow_cpu_fallback: When ``True``, a missing CUDA runtime falls back to
            CPU instead of raising.
        component: Optional component name for contextual warnings.

    Returns:
        A concrete :class:`torch.device`.
    """

    requested: torch.device | None
    if isinstance(preferred, torch.device):
        requested = preferred
    elif isinstance(preferred, str):
        pref = preferred.strip().lower()
        if pref in {"auto", ""}:
            requested = None
        elif pref in {"gpu", "cuda"}:
            requested = torch.device("cuda")
        elif pref == "cpu":
            requested = torch.device("cpu")
        else:
            requested = torch.device(pref)
    else:
        requested = None

    if requested is not None:
        if requested.type == "cuda" and not torch.cuda.is_available():
            if allow_cpu_fallback:
                _warn(component, "CUDA requested but unavailable; falling back to CPU.")
                return torch.device("cpu")
            msg = f"CUDA requested for {component or 'component'} but unavailable and fallback disabled."
            raise RuntimeError(msg)
        return requested

    # No explicit request; use global mode.
    if prefer_gpu():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if allow_cpu_fallback:
            _warn(component, "Global GPU mode set but CUDA unavailable; using CPU.")
            return torch.device("cpu")
        msg = f"Global GPU mode set for {component or 'component'} but CUDA unavailable."
        raise RuntimeError(msg)

    if prefer_cpu():
        return torch.device("cpu")

    # AUTO default
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_fast_depth_renderer(renderer: "RendererConfig") -> "RendererConfig":
    """Upgrade a depth-renderer config to the fastest available backend.

    - When GPU is preferred *or available* and CPU is not forced, ensure we use
      the PyTorch3D renderer on CUDA.
    - When CPU is forced, leave the config unchanged.
    """

    try:
        from oracle_rri.rendering.efm3d_depth_renderer import Efm3dDepthRendererConfig
        from oracle_rri.rendering.pytorch3d_depth_renderer import Pytorch3DDepthRendererConfig
    except Exception:
        return renderer

    # Respect explicit CPU ray-tracer requests.
    if isinstance(renderer, Efm3dDepthRendererConfig):
        if getattr(renderer, "device", "").lower() == "cpu":
            return renderer

    # If CPU is forced, keep as is.
    if prefer_cpu():
        return renderer

    gpu_ok = torch.cuda.is_available() and (prefer_gpu() or get_performance_mode() in {PerformanceMode.AUTO})

    if gpu_ok:
        if isinstance(renderer, Pytorch3DDepthRendererConfig):
            dev = select_device(getattr(renderer, "device", "cuda"), component="Pytorch3DDepthRenderer")
            return renderer.model_copy(update={"device": str(dev)})
        if isinstance(renderer, Efm3dDepthRendererConfig):
            dev = select_device("cuda", component="CandidateDepthRenderer")
            return Pytorch3DDepthRendererConfig(zfar=renderer.zfar, device=str(dev))

    return renderer


__all__ = [
    "PerformanceMode",
    "get_performance_mode",
    "set_performance_mode",
    "prefer_gpu",
    "prefer_cpu",
    "select_device",
    "pick_fast_depth_renderer",
    "ENV_VAR",
]
