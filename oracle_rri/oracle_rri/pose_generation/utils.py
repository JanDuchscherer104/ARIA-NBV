import torch


def _axis_stats(x: torch.Tensor) -> dict[str, float]:
    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
    }


def summarise_offsets_ref(offsets_ref: torch.Tensor) -> dict[str, dict[str, float]]:
    # offsets_ref: (N,3) in LUF reference frame
    r = offsets_ref.norm(dim=-1)
    az = torch.rad2deg(torch.atan2(offsets_ref[:, 0], offsets_ref[:, 2]))
    el = torch.rad2deg(torch.atan2(offsets_ref[:, 1], torch.linalg.norm(offsets_ref[:, (0, 2)], dim=-1) + 1e-8))
    return {"radius_m": _axis_stats(r), "az_deg": _axis_stats(az), "el_deg": _axis_stats(el)}


def summarise_dirs_ref(dirs_ref: torch.Tensor) -> dict[str, dict[str, float]]:
    # dirs_ref: (N,3) unit forward vectors in reference frame
    dirs = dirs_ref / dirs_ref.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    az = torch.rad2deg(torch.atan2(dirs[:, 0], dirs[:, 2]))
    el = torch.rad2deg(torch.asin(dirs[:, 1].clamp(-1.0, 1.0)))
    return {"az_deg": _axis_stats(az), "el_deg": _axis_stats(el)}


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------


def stats_to_markdown_table(stats: dict[str, dict[str, float]], *, header: str | None = None) -> str:
    """Convert nested stats dict into a GitHub-flavoured Markdown table.

    Args:
        stats: Mapping like ``{"radius_m": {"min": .., "max": .., ...}}``.
        header: Optional table title inserted as a preceding bold line.

    Returns:
        Markdown string containing a table with columns ``metric | min | max | mean | std``.
    """

    lines: list[str] = []
    if header:
        lines.append(f"**{header}**")
    lines.append("metric | min | max | mean | std")
    lines.append(":--|--:|--:|--:|--:")
    for key, vals in stats.items():
        lines.append(f"{key} | {vals['min']:.3f} | {vals['max']:.3f} | {vals['mean']:.3f} | {vals['std']:.3f}")
    return "\n".join(lines)
