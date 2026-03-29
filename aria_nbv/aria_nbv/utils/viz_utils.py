"""Shared visualization-friendly helpers.

This module centralises lightweight utilities that are needed by multiple
plotting modules. Functions previously defined in
``oracle_rri.data.utils`` now live here and remain re-exported from their old
location for compatibility.
"""

from __future__ import annotations

from pathlib import Path


def extract_scene_id_from_sequence_name(sequence_name: str) -> str:
    """Extract scene ID from an ATEK/ASE sequence name.

    Args:
        sequence_name: Sequence name (e.g., ``"82832_seq_000"``).

    Returns:
        Scene ID (e.g., ``"82832"``).
    """

    return sequence_name.split("_")[0]


def validate_scene_data(
    scene_id: str,
    data_dir: Path,
    require_mesh: bool = False,
    require_atek: bool = False,
) -> dict[str, bool | Path | None]:
    """Check if a scene has the expected mesh and ATEK shards.

    Args:
        scene_id: Scene identifier.
        data_dir: Root data directory containing ``ase_meshes`` and
            ``ase_atek``.
        require_mesh: Raise ``FileNotFoundError`` when the mesh is missing.
        require_atek: Raise ``FileNotFoundError`` when ATEK shards are missing.

    Returns:
        Dictionary with existence flags and resolved paths.
    """

    data_dir = Path(data_dir)

    mesh_path = data_dir / "ase_meshes" / f"scene_ply_{scene_id}.ply"
    mesh_exists = mesh_path.exists()
    if require_mesh and not mesh_exists:
        raise FileNotFoundError(f"GT mesh not found: {mesh_path}")

    atek_dir = data_dir / "ase_atek" / scene_id
    atek_exists = atek_dir.exists() and len(list(atek_dir.glob("*.tar"))) > 0
    if require_atek and not atek_exists:
        raise FileNotFoundError(f"ATEK data not found in: {atek_dir}")

    return {
        "mesh_exists": mesh_exists,
        "mesh_path": mesh_path if mesh_exists else None,
        "atek_exists": atek_exists,
        "atek_path": atek_dir if atek_exists else None,
    }


__all__ = ["extract_scene_id_from_sequence_name", "validate_scene_data"]
