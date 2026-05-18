"""Shared Rerun recording startup helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

from ._entities import ENTITY_WORLD

if TYPE_CHECKING:
    from ._config import RerunInspectorOutputConfig

RerunEntityFactory: TypeAlias = Callable[..., object]


class RerunModule(Protocol):
    """Subset of the Rerun module used by the offline inspector."""

    Points3D: RerunEntityFactory
    LineStrips3D: RerunEntityFactory
    Boxes3D: RerunEntityFactory
    AnyValues: RerunEntityFactory
    TextDocument: RerunEntityFactory
    Scalar: RerunEntityFactory
    Scalars: RerunEntityFactory
    SeriesLines: RerunEntityFactory
    SeriesPoints: RerunEntityFactory
    Transform3D: RerunEntityFactory
    Mesh3D: RerunEntityFactory
    Image: RerunEntityFactory
    DepthImage: RerunEntityFactory
    Pinhole: RerunEntityFactory
    ViewCoordinates: Any
    TransformRelation: Any

    def init(self, *args: object, **kwargs: object) -> None:
        """Initialize a recording."""

    def save(self, *args: object, **kwargs: object) -> None:
        """Open a save sink."""

    def spawn(self, *args: object, **kwargs: object) -> None:
        """Open a viewer sink."""

    def connect_grpc(self, *args: object, **kwargs: object) -> None:
        """Connect to a Rerun server."""

    def log(self, entity_path: str, entity: object, *args: object, **kwargs: object) -> None:
        """Log one entity."""

    def set_time(self, timeline: str, *, sequence: int | None = None, **kwargs: object) -> None:
        """Set a timeline for subsequent logs."""

    def set_time_sequence(self, timeline: str, sequence: int, **kwargs: object) -> None:
        """Set an integer timeline for subsequent logs."""


def start_rerun_recording(rr_module: RerunModule, output: RerunInspectorOutputConfig) -> None:
    """Initialize Rerun and open the configured sink before any entity logs."""

    rr_module.init(output.application_id, recording_id=output.recording_id)
    if output.mode == "save":
        output.save_path.parent.mkdir(parents=True, exist_ok=True)
        rr_module.save(output.save_path)
    elif output.mode == "spawn":
        rr_module.spawn(
            port=output.spawn_port,
            connect=True,
            memory_limit=output.spawn_memory_limit,
            hide_welcome_screen=output.hide_welcome_screen,
        )
    elif output.mode == "connect":
        rr_module.connect_grpc(output.connect_addr)
    else:  # pragma: no cover - pydantic constrains this.
        raise ValueError(f"Unsupported Rerun output mode: {output.mode}")


def log_world_coordinates(rr_module: RerunModule) -> None:
    """Declare the Rerun scene root as ARIA's right-handed Z-up world."""

    rr_module.log(ENTITY_WORLD, rr_module.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


__all__ = ["RerunModule", "log_world_coordinates", "start_rerun_recording"]
