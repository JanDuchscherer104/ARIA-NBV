"""Build helpers for openpoints-shim."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _pointnext_root() -> Path:
    return _repo_root() / "external" / "PointNeXt"


def _glob_any(directory: Path, patterns: Iterable[str]) -> bool:
    return any(any(directory.glob(pattern)) for pattern in patterns)


def _pointnet2_built(pointnext_root: Path) -> bool:
    build_dir = pointnext_root / "openpoints" / "cpp" / "pointnet2_batch"
    return _glob_any(
        build_dir,
        ("pointnet2_batch_cuda*.so", "pointnet2_batch_cuda*.pyd"),
    )


def _pointops_built(pointnext_root: Path) -> bool:
    build_dir = pointnext_root / "openpoints" / "cpp" / "pointops"
    return _glob_any(
        build_dir,
        ("pointops_cuda*.so", "pointops_cuda*.pyd"),
    )


def _subsampling_built(pointnext_root: Path) -> bool:
    build_dir = pointnext_root / "openpoints" / "cpp" / "subsampling"
    return _glob_any(
        build_dir,
        ("grid_subsampling*.so", "grid_subsampling*.pyd"),
    )


def _chamfer_built(pointnext_root: Path) -> bool:
    build_dir = pointnext_root / "openpoints" / "cpp" / "chamfer_dist"
    return _glob_any(
        build_dir,
        ("chamfer*.so", "chamfer*.pyd"),
    )


def _emd_built(pointnext_root: Path) -> bool:
    build_dir = pointnext_root / "openpoints" / "cpp" / "emd"
    return _glob_any(
        build_dir,
        ("emd_cuda*.so", "emd_cuda*.pyd"),
    )


def _ensure_openpoints_submodule(pointnext_root: Path) -> None:
    openpoints_dir = pointnext_root / "openpoints"
    if openpoints_dir.is_dir() and any(openpoints_dir.iterdir()):
        return

    git_path = shutil.which("git")
    if git_path is None:
        raise FileNotFoundError("git not found on PATH")

    repo_root = _repo_root()
    subprocess.run(  # noqa: S603
        [
            git_path,
            "-c",
            "url.https://github.com/.insteadOf=git@github.com/",
            "submodule",
            "update",
            "--init",
            "--recursive",
            "external/PointNeXt",
        ],
        cwd=repo_root,
        check=True,
    )


def _run_setup_py(path: Path, env: dict[str, str]) -> None:
    subprocess.run(  # noqa: S603
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=path,
        check=True,
        env=env,
    )


def _infer_cuda_home_from_venv(repo_root: Path) -> Path | None:
    venv_cfg = repo_root / "oracle_rri" / ".venv" / "pyvenv.cfg"
    if not venv_cfg.is_file():
        return None

    home_line = None
    for line in venv_cfg.read_text(encoding="utf-8").splitlines():
        if line.startswith("home = "):
            home_line = line.split("=", 1)[1].strip()
            break
    if not home_line:
        return None

    home_path = Path(home_line)
    cuda_home = home_path.parent
    nvcc = cuda_home / "bin" / "nvcc"
    if nvcc.exists():
        return cuda_home
    return None


def _ensure_cuda_env(env: dict[str, str]) -> dict[str, str]:
    if "CUDA_HOME" in env and Path(env["CUDA_HOME"]).exists():
        return env

    nvcc = shutil.which("nvcc", path=env.get("PATH"))
    if nvcc:
        cuda_home = Path(nvcc).resolve().parents[1]
        env["CUDA_HOME"] = str(cuda_home)
        env["PATH"] = f"{cuda_home / 'bin'}:{env.get('PATH', '')}"
        return env

    cuda_home = _infer_cuda_home_from_venv(_repo_root())
    if cuda_home is None:
        return env

    env["CUDA_HOME"] = str(cuda_home)
    env["PATH"] = f"{cuda_home / 'bin'}:{env.get('PATH', '')}"
    return env


def ensure_openpoints_built() -> None:
    """Build OpenPoints CUDA extensions when missing."""
    pointnext_root = _pointnext_root()
    if not pointnext_root.is_dir():
        message = f"Missing PointNeXt checkout at {pointnext_root}"
        raise FileNotFoundError(message)

    _ensure_openpoints_submodule(pointnext_root)

    env = os.environ.copy()
    env = _ensure_cuda_env(env)
    env.setdefault("OPENPOINTS_INSTALL_DEPS", "0")
    env.setdefault("OPENPOINTS_BUILD_POINTNET2", "1")
    env.setdefault("OPENPOINTS_BUILD_POINTOPS", "0")
    env.setdefault("OPENPOINTS_BUILD_SUBSAMPLING", "0")
    env.setdefault("OPENPOINTS_BUILD_CHAMFER_DIST", "1")
    env.setdefault("OPENPOINTS_BUILD_EMD", "0")
    env.setdefault("PYTHON_BIN", sys.executable)

    build_pointnet2 = env.get("OPENPOINTS_BUILD_POINTNET2") == "1"
    build_pointops = env.get("OPENPOINTS_BUILD_POINTOPS") == "1"
    build_subsampling = env.get("OPENPOINTS_BUILD_SUBSAMPLING") == "1"
    build_chamfer = env.get("OPENPOINTS_BUILD_CHAMFER_DIST") == "1"
    build_emd = env.get("OPENPOINTS_BUILD_EMD") == "1"

    needs_pointnet2 = build_pointnet2 and not _pointnet2_built(pointnext_root)
    needs_pointops = build_pointops and not _pointops_built(pointnext_root)
    needs_subsampling = build_subsampling and not _subsampling_built(pointnext_root)
    needs_chamfer = build_chamfer and not _chamfer_built(pointnext_root)
    needs_emd = build_emd and not _emd_built(pointnext_root)

    if not any(
        [needs_pointnet2, needs_pointops, needs_subsampling, needs_chamfer, needs_emd],
    ):
        return

    if env.get("OPENPOINTS_INSTALL_DEPS") == "1":
        requirements = pointnext_root / "requirements.txt"
        if requirements.is_file():
            subprocess.run(  # noqa: S603
                [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
                cwd=pointnext_root,
                check=True,
                env=env,
            )

    if needs_pointnet2:
        _run_setup_py(pointnext_root / "openpoints" / "cpp" / "pointnet2_batch", env)

    if needs_pointops:
        _run_setup_py(pointnext_root / "openpoints" / "cpp" / "pointops", env)

    if needs_subsampling:
        _run_setup_py(pointnext_root / "openpoints" / "cpp" / "subsampling", env)

    if needs_chamfer:
        _run_setup_py(pointnext_root / "openpoints" / "cpp" / "chamfer_dist", env)

    if needs_emd:
        _run_setup_py(pointnext_root / "openpoints" / "cpp" / "emd", env)
