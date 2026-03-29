"""Export Streamlit-style diagnostic figures used in the Typst paper.

This script runs the oracle candidate→render→RRI pipeline on a single ASE
snippet and exports a small set of Plotly figures that match the paper's
expected file layout under `docs/figures/app/`.

Notes:
    - PNG/PDF export for Plotly requires `kaleido`. If it is not available, the
      script still exports `.html` files and prints a warning.
    - For reproducibility, prefer running with `--config-path` pointing to a
      pinned dataset snippet (scene + sample key filter).

Example:
    `uv run python aria_nbv/scripts/export_paper_figures.py \\
      --config-path .configs/paper_figures_oracle_labeler.toml \\
      --output-dir docs/figures/app \\
      --overwrite`
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import plotly.graph_objects as go  # type: ignore[import]
from plotly.subplots import make_subplots  # type: ignore[import]
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from aria_nbv.data import AseEfmDatasetConfig
from aria_nbv.pipelines import OracleRriLabelerConfig
from aria_nbv.pose_generation.plotting import CandidatePlotBuilder
from aria_nbv.rendering.plotting import depth_grid
from aria_nbv.utils import BaseConfig, Console


def _extract_config_path(argv: list[str]) -> Path | None:
    for idx, arg in enumerate(argv):
        if arg in ("--config_path", "--config-path") and idx + 1 < len(argv):
            return Path(argv[idx + 1])
        if arg.startswith("--config_path=") or arg.startswith("--config-path="):
            return Path(arg.split("=", 1)[1])
    return None


def _normalize_cli_args(argv: list[str]) -> list[str]:
    out: list[str] = []
    for arg in argv:
        if arg.startswith("--") and "_" in arg:
            out.append(arg.replace("_", "-"))
        else:
            out.append(arg)
    return out


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


class PaperFigureExportConfig(BaseConfig["PaperFigureExporter"]):
    """Configuration for exporting paper figures from the oracle label pipeline."""

    target: type["PaperFigureExporter"] = Field(
        default_factory=lambda: PaperFigureExporter,
        exclude=True,
    )

    dataset: AseEfmDatasetConfig = Field(default_factory=AseEfmDatasetConfig)
    """Dataset configuration used to select the snippet."""

    labeler: OracleRriLabelerConfig = Field(default_factory=OracleRriLabelerConfig)
    """Oracle label pipeline configuration (candidates → depth → RRI)."""

    output_dir: Path = Path("docs/figures/app")
    """Directory to write figures into (defaults to the paper's figure folder)."""

    overwrite: bool = False
    """Overwrite existing files when True."""

    export_html: bool = True
    """Export `.html` alongside image files for debugging."""

    frustum_scale: float = 0.5
    """Display-only scale factor for frusta."""

    max_frustums: int = 6
    """Maximum number of frusta to draw in the 3D plot."""

    rri_baseline_label: str = "-1"
    """Label used for the baseline (semi-dense only) bars."""


class CLIPaperFigureExportConfig(BaseSettings, PaperFigureExportConfig):
    """CLI-enabled wrapper with optional TOML config path."""

    config_path: Path | None = Field(default=None)
    """Path to a TOML configuration file."""

    model_config = SettingsConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=True,
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_implicit_flags=True,
        env_prefix="ARIA_NBV_",
    )


class PaperFigureExporter:
    """Export a compact set of Plotly figures for the paper."""

    def __init__(self, config: PaperFigureExportConfig) -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

    def run(self) -> None:
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        dataset = self.config.dataset.setup_target()
        sample = next(iter(dataset))
        self.console.log(f"Using sample scene={sample.scene_id} snippet={sample.snippet_id}")

        labeler = self.config.labeler.setup_target()
        batch = labeler.run(sample)

        name_tag = self._candidate_name_tag()

        fig_frusta = (
            CandidatePlotBuilder.from_candidates(
                sample,
                batch.candidates,
                title=f"Candidate frusta ({name_tag})",
                height=900,
            )
            .add_mesh()
            .add_candidate_cloud(use_valid=True, color="royalblue", size=3, opacity=0.35)
            .add_candidate_frusta(
                scale=float(self.config.frustum_scale),
                color="crimson",
                max_frustums=int(self.config.max_frustums),
                include_axes=False,
                include_center=False,
                display_rotate=False,
            )
            .add_reference_axes(display_rotate=False)
        ).finalize()

        self._write_plotly(
            fig_frusta,
            out_dir / f"cand_frusta_{name_tag}.png",
        )

        depths = batch.depths.depths
        candidate_ids = batch.depths.candidate_indices.cpu().tolist()
        titles = [f"cand {i} (id {int(cid)})" for i, cid in enumerate(candidate_ids)]
        fig_depth_grid = depth_grid(
            depths,
            titles=titles,
            zmax=float(depths.max().item()),
        )
        self._write_plotly(fig_depth_grid, out_dir / "candidate_renders.png")

        fig_rri = self._plot_rri_forward(
            candidate_ids=[int(cid) for cid in candidate_ids],
            rri=batch.rri,
        )
        self._write_plotly(fig_rri, out_dir / "rri_forward.png")

    def _candidate_name_tag(self) -> str:
        cfg = self.config.labeler.generator
        kappa = int(round(float(cfg.kappa)))
        rmin = int(round(float(cfg.min_radius) * 10))
        rmax = int(round(float(cfg.max_radius) * 10))
        return f"kappa{kappa}_r{rmin:02d}-{rmax:02d}"

    def _write_plotly(self, fig: go.Figure, path: Path) -> None:
        if path.exists() and not self.config.overwrite:
            self.console.warn(f"Skip existing file (overwrite=false): {path}")
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        if self.config.export_html:
            html_path = path.with_suffix(".html")
            fig.write_html(html_path.as_posix(), include_plotlyjs="cdn")
            self.console.log(f"Wrote {html_path}")

        try:
            fig.write_image(path.as_posix())
            self.console.log(f"Wrote {path}")
        except Exception as exc:  # noqa: BLE001
            self.console.warn(
                f"Failed to export Plotly image. Install `kaleido` for PNG/PDF export. Error: {exc}",
            )

    def _plot_rri_forward(self, *, candidate_ids: list[int], rri) -> go.Figure:
        labels = [str(cid) for cid in candidate_ids]

        rri_vals = rri.rri.detach().cpu().tolist()
        pm_dist_before = float(rri.pm_dist_before[0].detach().cpu().item())
        pm_acc_before = float(rri.pm_acc_before[0].detach().cpu().item())
        pm_comp_before = float(rri.pm_comp_before[0].detach().cpu().item())

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Oracle RRI",
                "Chamfer-like (bidirectional)",
                "Point→Mesh (accuracy)",
                "Mesh→Point (completeness)",
            ),
        )

        fig.add_trace(go.Bar(x=labels, y=rri_vals, name="RRI"), row=1, col=1)

        fig.add_trace(
            go.Bar(
                x=[self.config.rri_baseline_label],
                y=[pm_dist_before],
                marker_color="lightgray",
                name="before",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(x=labels, y=rri.pm_dist_after.detach().cpu().tolist(), name="after", showlegend=False),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Bar(
                x=[self.config.rri_baseline_label],
                y=[pm_acc_before],
                marker_color="lightgray",
                name="before",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(x=labels, y=rri.pm_acc_after.detach().cpu().tolist(), name="after", showlegend=False),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=[self.config.rri_baseline_label],
                y=[pm_comp_before],
                marker_color="lightgray",
                name="before",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Bar(x=labels, y=rri.pm_comp_after.detach().cpu().tolist(), name="after", showlegend=False),
            row=2,
            col=2,
        )

        fig.update_layout(height=720, title_text="Oracle RRI diagnostics")
        return fig


def main() -> None:
    argv = _normalize_cli_args(sys.argv[1:])
    config_path = _extract_config_path(argv)

    if config_path is None:
        cfg = CLIPaperFigureExportConfig(_cli_parse_args=argv)
        cfg.setup_target().run()
        return

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_cfg = PaperFigureExportConfig.from_toml(config_path)
    cli_cfg = CLIPaperFigureExportConfig(_cli_parse_args=argv)
    overrides = cli_cfg.model_dump(exclude_unset=True)
    overrides.pop("config_path", None)

    merged = _deep_update(base_cfg.model_dump(), overrides)
    cfg = PaperFigureExportConfig.model_validate(merged)
    cfg.setup_target().run()


if __name__ == "__main__":  # pragma: no cover
    main()
