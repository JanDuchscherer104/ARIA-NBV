"""Helper utilities for ASE data handling."""

from pathlib import Path


def extract_scene_id_from_sequence_name(sequence_name: str) -> str:
    """Extract scene ID from ATEK sequence name.

    Args:
        sequence_name: Sequence name (e.g., "82832_seq_000")

    Returns:
        Scene ID (e.g., "82832")

    Example:
        >>> extract_scene_id_from_sequence_name("82832_seq_000")
        '82832'
    """
    return sequence_name.split("_")[0]


def validate_scene_data(
    scene_id: str,
    data_dir: Path,
    require_mesh: bool = False,
    require_atek: bool = False,
) -> dict[str, bool | Path | None]:
    """Check if scene has required data files.

    Args:
        scene_id: Scene identifier
        data_dir: Root data directory
        require_mesh: Raise error if mesh missing
        require_atek: Raise error if ATEK data missing

    Returns:
        Dictionary with validation results:
            - mesh_exists: bool
            - mesh_path: Path | None
            - atek_exists: bool
            - atek_path: Path | None

    Example:
        >>> info = validate_scene_data("82832", Path(".data"))
        >>> print(f"Has mesh: {info['mesh_exists']}")
    """
    data_dir = Path(data_dir)

    # Check mesh
    mesh_path = data_dir / "ase_meshes" / f"scene_ply_{scene_id}.ply"
    mesh_exists = mesh_path.exists()

    if require_mesh and not mesh_exists:
        raise FileNotFoundError(f"GT mesh not found: {mesh_path}")

    # Check ATEK data
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
