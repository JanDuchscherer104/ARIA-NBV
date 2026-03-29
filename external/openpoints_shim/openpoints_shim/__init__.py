"""OpenPoints shim bootstrap helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _find_pointnext_root() -> Path | None:
    env_root = os.environ.get("OPENPOINTS_POINTNEXT_ROOT") or os.environ.get(
        "OPENPOINTS_ROOT",
    )
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root))

    exe = Path(sys.executable).resolve()
    candidates.extend(exe.parents)

    module_path = Path(__file__).resolve()
    candidates.extend(module_path.parents)

    cwd = Path.cwd().resolve()
    candidates.extend(cwd.parents)

    for root in _unique_paths(candidates):
        pointnext_root = root / "external" / "PointNeXt"
        if (pointnext_root / "openpoints").is_dir():
            return pointnext_root
        if (root / "PointNeXt" / "openpoints").is_dir():
            return root / "PointNeXt"
    return None


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def bootstrap() -> Path | None:
    """Ensure the PointNeXt/OpenPoints source tree is importable.

    Returns:
        The resolved PointNeXt root if found, else ``None``.
    """
    pointnext_root = _find_pointnext_root()
    if pointnext_root is None:
        return None

    _prepend_sys_path(pointnext_root)

    cpp_dirs = [
        pointnext_root / "openpoints" / "cpp" / "pointnet2_batch",
        pointnext_root / "openpoints" / "cpp" / "pointops",
        pointnext_root / "openpoints" / "cpp" / "subsampling",
        pointnext_root / "openpoints" / "cpp" / "chamfer_dist",
        pointnext_root / "openpoints" / "cpp" / "emd",
    ]
    for cpp_dir in cpp_dirs:
        if cpp_dir.is_dir():
            _prepend_sys_path(cpp_dir)

    return pointnext_root


__all__ = ["bootstrap"]
