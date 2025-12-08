"""Shared helpers for processing and caching GT meshes.

This module centralises mesh cropping/simplification and persistence so that
both candidate generation (collision checks) and depth rendering can reuse the
same processed mesh without duplicating logic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import trimesh  # type: ignore[import-untyped]

from ..configs.path_config import PathConfig
from ..utils import Console

if False:  # pragma: no cover - import guard for type checking without runtime dep
    from .efm_views import EfmSnippetView


@dataclass(slots=True)
class MeshProcessSpec:
    """Specification that uniquely defines a processed mesh artifact."""

    scene_id: str
    snippet_id: str | None
    bounds_min: list[float]
    bounds_max: list[float]
    margin_m: float
    simplify_ratio: float | None
    max_faces: int | None
    crop_min_keep_ratio: float

    def hash(self) -> str:
        """Stable short hash for filenames and cache keys."""

        payload: dict[str, Any] = {
            "scene_id": self.scene_id,
            "snippet_id": self.snippet_id,
            "bounds_min": [round(x, 4) for x in self.bounds_min],
            "bounds_max": [round(x, 4) for x in self.bounds_max],
            "margin_m": round(self.margin_m, 4),
            "simplify_ratio": None if self.simplify_ratio is None else round(self.simplify_ratio, 4),
            "max_faces": self.max_faces,
            "crop_min_keep_ratio": round(self.crop_min_keep_ratio, 4),
        }
        spec_json = json.dumps(payload, sort_keys=True)
        return hashlib.sha1(spec_json.encode("utf-8")).hexdigest()[:12]


@dataclass(slots=True)
class ProcessedMesh:
    """Container for processed mesh and cached tensors."""

    mesh: trimesh.Trimesh
    bounds: tuple[torch.Tensor, torch.Tensor]
    verts: torch.Tensor
    faces: torch.Tensor
    cache_hit: bool
    path: Path
    spec_hash: str


def _crop_mesh(
    mesh: trimesh.Trimesh,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    margin_m: float,
    *,
    crop_min_keep_ratio: float,
    console: Console | None,
) -> trimesh.Trimesh:
    lo = (bounds_min - margin_m).numpy()
    hi = (bounds_max + margin_m).numpy()

    verts = mesh.vertices
    in_bounds = ((verts >= lo) & (verts <= hi)).all(axis=1)
    if not in_bounds.any():
        if console:
            console.warn("Mesh cropping skipped: no vertices within bounds; returning original mesh.")
        return mesh

    face_mask = in_bounds[mesh.faces].any(axis=1)
    if not face_mask.any():
        if console:
            console.warn("Mesh cropping skipped: no faces intersect bounds; returning original mesh.")
        return mesh

    cropped = mesh.submesh([face_mask], append=True)
    assert isinstance(cropped, trimesh.Trimesh)

    keep_ratio = cropped.faces.shape[0] / float(mesh.faces.shape[0])
    if keep_ratio < crop_min_keep_ratio:
        if console:
            console.warn(f"Cropping dropped too many faces (keep_ratio={keep_ratio:.3f} < {crop_min_keep_ratio}).")

    return cropped


def load_or_process_mesh(
    mesh: trimesh.Trimesh,
    spec: MeshProcessSpec,
    paths: PathConfig,
    *,
    console: Console | None = None,
) -> ProcessedMesh:
    """Crop/simplify a mesh once and persist the result on disk.

    Args:
        mesh: Raw scene mesh.
        spec: Processing specification (bounds, simplification, etc.).
        paths: PathConfig providing the processed-mesh directory.
        console: Optional logger.

    Returns:
        ProcessedMesh with trimesh object, bounds, torch tensors, and cache flag.
    """

    spec_hash = spec.hash()
    out_path = paths.resolve_processed_mesh_path(spec.scene_id, spec.snippet_id, spec_hash)

    if out_path.exists():
        mesh_proc = trimesh.load(out_path, process=False)
        assert isinstance(mesh_proc, trimesh.Trimesh)
        if console:
            console.log(f"Loaded processed mesh from cache: {out_path}")

        return ProcessedMesh(
            mesh=mesh_proc,
            bounds=(
                torch.as_tensor(spec.bounds_min, dtype=torch.float32),
                torch.as_tensor(spec.bounds_max, dtype=torch.float32),
            ),
            verts=torch.as_tensor(mesh_proc.vertices, dtype=torch.float32),
            faces=torch.as_tensor(mesh_proc.faces, dtype=torch.int64),
            cache_hit=True,
            path=out_path,
            spec_hash=spec_hash,
        )

    mesh_work = mesh.copy()

    bounds_min = torch.as_tensor(spec.bounds_min, dtype=torch.float32)
    bounds_max = torch.as_tensor(spec.bounds_max, dtype=torch.float32)
    mesh_work = _crop_mesh(
        mesh_work,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        margin_m=spec.margin_m,
        crop_min_keep_ratio=spec.crop_min_keep_ratio,
        console=console,
    )

    faces_before = mesh_work.faces.shape[0]
    target_faces = faces_before
    if spec.simplify_ratio not in (None, 0):
        target_faces = min(target_faces, int(faces_before * spec.simplify_ratio))
    if spec.max_faces is not None:
        target_faces = min(target_faces, int(spec.max_faces))
    if target_faces < mesh_work.faces.shape[0]:
        mesh_work = mesh_work.simplify_quadric_decimation(face_count=target_faces)
        if console:
            console.dbg(
                f"Simplified mesh from {faces_before} to {mesh_work.faces.shape[0]} faces (target={target_faces})."
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_work.export(out_path)
    if console:
        console.log(f"Saved processed mesh to {out_path}")

    return ProcessedMesh(
        mesh=mesh_work,
        bounds=(bounds_min, bounds_max),
        verts=torch.as_tensor(mesh_work.vertices, dtype=torch.float32),
        faces=torch.as_tensor(mesh_work.faces, dtype=torch.int64),
        cache_hit=False,
        path=out_path,
        spec_hash=spec_hash,
    )


# ---------------------------------------------------------------------------
# Shared in-memory cache for PyTorch3D Meshes
# ---------------------------------------------------------------------------

_P3D_STRUCT_CACHE: dict[str, Any] = {}


def get_pytorch3d_mesh(
    verts: torch.Tensor,
    faces: torch.Tensor,
    *,
    cache_key: str,
    device: torch.device | None = None,
    meshes_cls: Any | None = None,
) -> Any:
    """Return a cached PyTorch3D Meshes built from tensors.

    Args:
        verts: ``[V,3]`` float32 tensor.
        faces: ``[F,3]`` int64 tensor.
        cache_key: Stable key (e.g., processed mesh spec hash).
        device: Optional device to place tensors on.
        Meshes_cls: Optional override for testing; defaults to import-time Meshes.
    """

    from pytorch3d.structures import Meshes  # type: ignore[import-untyped]

    meshes_cls = Meshes if meshes_cls is None else meshes_cls

    cached = _P3D_STRUCT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    verts_t = verts if device is None else verts.to(device=device)
    faces_t = faces if device is None else faces.to(device=device)

    mesh_struct = meshes_cls(verts=[verts_t], faces=[faces_t])
    _P3D_STRUCT_CACHE[cache_key] = mesh_struct
    return mesh_struct


# ---------------------------------------------------------------------------
# Snippet-centric helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MeshArtifact:
    """Bundle linking spec, processed mesh, and optional PyTorch3D struct."""

    spec: MeshProcessSpec
    processed: ProcessedMesh
    p3d: Any | None = None


def mesh_from_snippet(
    sample: "EfmSnippetView",
    *,
    paths: PathConfig | None = None,
    device: torch.device | None = None,
    want_p3d: bool = False,
    meshes_cls: Any | None = None,
    console: Console | None = None,
) -> MeshArtifact:
    """Materialise mesh assets for a snippet using its cached spec.

    - Prefers existing ``sample.mesh_verts/faces`` tensors.
    - Falls back to loading the processed mesh from disk using the snippet's
      ``mesh_cache_key``.
    - As a last resort, re-processes ``sample.mesh`` with the stored spec.

    Args:
        sample: Snippet view containing mesh metadata.
        paths: Optional :class:`PathConfig` override; defaults to singleton.
        device: Optional target device for returned tensors / PyTorch3D struct.
        want_p3d: Build and return a PyTorch3D ``Meshes`` instance.
        meshes_cls: Optional override for the ``Meshes`` class (testing).
        console: Optional logger.

    Returns:
        MeshArtifact with processed mesh, spec, and optional PyTorch3D struct.
    """

    if sample.mesh_specs is None:
        raise ValueError("EfmSnippetView.mesh_specs is required to load meshes.")

    paths = paths or PathConfig()
    spec = sample.mesh_specs
    cache_key = sample.mesh_cache_key or spec.hash()

    verts: torch.Tensor | None = sample.mesh_verts
    faces: torch.Tensor | None = sample.mesh_faces
    mesh_obj = sample.mesh

    if verts is None or faces is None:
        proc_path = paths.resolve_processed_mesh_path(spec.scene_id, spec.snippet_id, cache_key)
        if proc_path.exists():
            mesh_obj = trimesh.load(proc_path, process=False)
            assert isinstance(mesh_obj, trimesh.Trimesh)
            verts = torch.as_tensor(mesh_obj.vertices, dtype=torch.float32)
            faces = torch.as_tensor(mesh_obj.faces, dtype=torch.int64)
            processed = ProcessedMesh(
                mesh=mesh_obj,
                bounds=(
                    torch.as_tensor(spec.bounds_min, dtype=torch.float32),
                    torch.as_tensor(spec.bounds_max, dtype=torch.float32),
                ),
                verts=verts,
                faces=faces,
                cache_hit=True,
                path=proc_path,
                spec_hash=cache_key,
            )
        else:
            if mesh_obj is None:
                raise ValueError("Snippet has no mesh attached and no cached processed mesh on disk.")
            processed = load_or_process_mesh(mesh_obj, spec=spec, paths=paths, console=console)
            mesh_obj = processed.mesh
            verts = processed.verts
            faces = processed.faces
    else:
        processed = ProcessedMesh(
            mesh=mesh_obj if mesh_obj is not None else trimesh.Trimesh(vertices=verts, faces=faces),
            bounds=(
                torch.as_tensor(spec.bounds_min, dtype=torch.float32),
                torch.as_tensor(spec.bounds_max, dtype=torch.float32),
            ),
            verts=verts,
            faces=faces,
            cache_hit=True,
            path=paths.resolve_processed_mesh_path(spec.scene_id, spec.snippet_id, cache_key),
            spec_hash=cache_key,
        )

    if device is not None:
        verts = verts.to(device=device)
        faces = faces.to(device=device)
        processed = ProcessedMesh(
            mesh=processed.mesh,
            bounds=processed.bounds,
            verts=verts,
            faces=faces,
            cache_hit=processed.cache_hit,
            path=processed.path,
            spec_hash=processed.spec_hash,
        )

    p3d_struct = None
    if want_p3d:
        p3d_struct = get_pytorch3d_mesh(
            verts=verts, faces=faces, cache_key=cache_key, device=device, meshes_cls=meshes_cls
        )

    return MeshArtifact(spec=spec, processed=processed, p3d=p3d_struct)


__all__ = [
    "MeshProcessSpec",
    "ProcessedMesh",
    "MeshArtifact",
    "load_or_process_mesh",
    "get_pytorch3d_mesh",
    "mesh_from_snippet",
]
