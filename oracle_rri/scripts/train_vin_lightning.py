"""Lightning training/eval entrypoint for VIN (View Introspection Network).

This is a Lightning-based alternative to `scripts/train_vin.py`.

It keeps the same core data-flow (online oracle labels) but runs it through:

- `oracle_rri.lightning.VinDataModule` (oracle label generation),
- `oracle_rri.lightning.VinLightningModule` (VIN forward + CORAL loss),
- `oracle_rri.lightning.TrainerFactoryConfig` (Trainer + optional W&B logging).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch


def _ensure_oracle_rri_importable() -> None:
    """Add the `oracle_rri/` project root to `sys.path` if needed."""
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_from_arg(value: str) -> torch.device:
    if value.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _resolve_root_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _find_first_scene_with_tars(tar_root: Path, *, require_mesh: bool, mesh_root: Path) -> str | None:
    if not tar_root.exists():
        return None
    for scene_dir in sorted(tar_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        if any(scene_dir.glob("*.tar")):
            if require_mesh and not (mesh_root / f"scene_ply_{scene_dir.name}.ply").exists():
                continue
            return scene_dir.name
    return None


def _collect_scenes_with_tars(tar_root: Path, *, require_mesh: bool, mesh_root: Path) -> list[str]:
    if not tar_root.exists():
        return []
    scenes: list[str] = []
    for scene_dir in sorted(tar_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        if not any(scene_dir.glob("*.tar")):
            continue
        if require_mesh and not (mesh_root / f"scene_ply_{scene_dir.name}.ply").exists():
            continue
        scenes.append(scene_dir.name)
    return scenes


def main() -> None:
    _ensure_oracle_rri_importable()

    from oracle_rri.configs import PathConfig
    from oracle_rri.configs.wandb_config import WandbConfig
    from oracle_rri.data import AseEfmDatasetConfig
    from oracle_rri.lightning import (
        AdamWConfig,
        TrainerCallbacksConfig,
        TrainerFactoryConfig,
        VinDataModuleConfig,
        VinLightningModuleConfig,
    )
    from oracle_rri.pipelines.oracle_rri_labeler import OracleRriLabelerConfig
    from oracle_rri.pose_generation import CandidateViewGeneratorConfig
    from oracle_rri.rendering import CandidateDepthRendererConfig, Pytorch3DDepthRendererConfig
    from oracle_rri.utils import Stage, Verbosity
    from oracle_rri.vin import EvlBackboneConfig, RriOrdinalBinner, VinModelConfig
    from oracle_rri.vin.model import VinScorerHeadConfig

    parser = argparse.ArgumentParser(description="VIN training via PyTorch Lightning (online oracle labels).")
    parser.add_argument("--stage", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--ckpt-path", type=str, default=None, help="Optional Lightning checkpoint path for val/test.")

    parser.add_argument("--scene-id", type=str, default=None, help="ASE scene id (defaults to first local scene).")
    parser.add_argument(
        "--atek-variant",
        type=str,
        default="efm",
        choices=["efm", "efm_eval", "cubercnn", "cubercnn_eval"],
        help="ATEK dataset variant (maps to `.data/ase_<variant>`).",
    )
    parser.add_argument(
        "--wds-shuffle",
        action="store_true",
        help="Enable WebDataset shuffle inside AseEfmDataset (non-deterministic unless WebDataset RNG is controlled).",
    )
    parser.add_argument("--device", type=str, default="auto", help="Torch device for EVL/VIN/oracle (e.g. cuda:0).")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max-steps", type=int, default=10, help="Optimizer steps (mapped to limit_train_batches).")
    parser.add_argument("--fit-snippets", type=int, default=2, help="Snippets used to fit ordinal bin edges.")
    parser.add_argument(
        "--max-binner-attempts",
        type=int,
        default=25,
        help="Maximum number of skipped/invalid oracle batches while fitting the ordinal binner.",
    )
    parser.add_argument("--max-step-attempts", type=int, default=50)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)

    parser.add_argument("--num-classes", type=int, default=15)
    parser.add_argument("--tanh-scale", type=float, default=1.0)

    parser.add_argument("--max-candidates", type=int, default=8, help="Rendered candidates per snippet.")
    parser.add_argument("--backprojection-stride", type=int, default=8)
    parser.add_argument("--min-distance-to-mesh", type=float, default=0.2)
    parser.add_argument("--labeler-render-oversample", type=float, default=2.0)

    parser.add_argument("--out-dir", type=str, default=".logs/vin_train_lightning")
    parser.add_argument(
        "--fit-binner-only",
        action="store_true",
        help="Fit the ordinal binner on oracle labels, save it, and exit (no training).",
    )
    parser.add_argument(
        "--binner-load-path",
        type=str,
        default=None,
        help="Optional path to an existing `rri_binner.json` to load (skips fitting; required for val/test without ckpt).",
    )
    parser.add_argument(
        "--binner-save-path",
        type=str,
        default=None,
        help="Optional output path for `rri_binner.json` (defaults to `<out-dir>/rri_binner.json`).",
    )
    parser.add_argument("--use-wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--wandb-project", type=str, default="oracle-rri")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-offline", action="store_true", help="Enable offline W&B logging.")

    args = parser.parse_args()

    _set_seed(int(args.seed))
    device = _device_from_arg(str(args.device))

    paths = PathConfig()
    tar_root = paths.resolve_atek_data_dir(str(args.atek_variant))
    mesh_root = paths.ase_meshes
    available_scenes = _collect_scenes_with_tars(tar_root, require_mesh=True, mesh_root=mesh_root)
    if not available_scenes:
        raise FileNotFoundError(f"No ASE scene with both tars and meshes found under {tar_root} and {mesh_root}.")

    if args.scene_id is not None:
        scene_ids = [str(args.scene_id)]
    else:
        scene_ids = available_scenes if args.fit_binner_only else [available_scenes[0]]

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (paths.root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    binner_save_path = (
        _resolve_root_path(paths.root, args.binner_save_path)
        if args.binner_save_path
        else (out_dir / "rri_binner.json")
    )
    binner_load_path = _resolve_root_path(paths.root, args.binner_load_path) if args.binner_load_path else None

    meta = {
        "stage": str(args.stage),
        "scene_id": str(scene_ids[0]) if scene_ids else None,
        "scene_ids": [str(s) for s in scene_ids],
        "atek_variant": str(args.atek_variant),
        "wds_shuffle": bool(args.wds_shuffle),
        "device": str(device),
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "fit_snippets": int(args.fit_snippets),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "num_classes": int(args.num_classes),
        "tanh_scale": float(args.tanh_scale),
        "max_candidates": int(args.max_candidates),
        "backprojection_stride": int(args.backprojection_stride),
        "min_distance_to_mesh": float(args.min_distance_to_mesh),
        "labeler_render_oversample": float(args.labeler_render_oversample),
        "fit_binner_only": bool(args.fit_binner_only),
        "binner_load_path": binner_load_path.as_posix() if binner_load_path is not None else None,
        "binner_save_path": binner_save_path.as_posix(),
        "use_wandb": bool(args.use_wandb),
        "wandb_project": str(args.wandb_project),
        "wandb_name": args.wandb_name,
        "wandb_offline": bool(args.wandb_offline),
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    # --------------------------------------------------------------------- dataset + labeler configs
    ds_cfg = AseEfmDatasetConfig(
        atek_variant=str(args.atek_variant),
        scene_ids=[str(s) for s in scene_ids],
        load_meshes=True,
        require_mesh=True,
        wds_shuffle=bool(args.wds_shuffle),
        batch_size=None,
        verbosity=Verbosity.QUIET,
        is_debug=False,
    )

    label_cfg = OracleRriLabelerConfig(
        generator=CandidateViewGeneratorConfig(
            num_samples=max(4 * int(args.max_candidates), 32),
            device=device,
            min_distance_to_mesh=float(args.min_distance_to_mesh),
            verbosity=Verbosity.QUIET,
            is_debug=False,
        ),
        depth=CandidateDepthRendererConfig(
            max_candidates_final=int(args.max_candidates),
            oversample_factor=float(args.labeler_render_oversample),
            renderer=Pytorch3DDepthRendererConfig(device=str(device), verbosity=Verbosity.QUIET),
            verbosity=Verbosity.QUIET,
            is_debug=False,
        ),
        backprojection_stride=int(args.backprojection_stride),
        output_device=None,
    )

    dm_cfg = VinDataModuleConfig(
        train_dataset=ds_cfg,
        val_dataset=ds_cfg,
        labeler=label_cfg,
        stage_id=0,
        max_attempts_per_batch=int(args.max_step_attempts),
        num_workers=0,
        persistent_workers=False,
        verbosity=Verbosity.QUIET,
        is_debug=False,
    )

    datamodule = dm_cfg.setup_target()

    def _fit_binner_from_oracle(*, fit_snippets: int, max_attempts: int) -> RriOrdinalBinner:
        it = datamodule.iter_oracle_batches(stage=Stage.TRAIN)
        rri_all: list[torch.Tensor] = []
        stage_all: list[torch.Tensor] = []

        successes = 0
        failed = 0
        while successes < fit_snippets:
            if failed >= max_attempts:
                raise RuntimeError(
                    f"Unable to fit binner: only {successes}/{fit_snippets} snippets after {failed} skipped batches."
                )

            try:
                batch = next(it)
            except StopIteration as exc:  # pragma: no cover
                raise RuntimeError(
                    f"Dataset exhausted while fitting binner ({successes}/{fit_snippets} snippets collected)."
                ) from exc

            rri_full = batch.rri.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
            stage_full = batch.stage.detach().reshape(-1).to(device="cpu", dtype=torch.int64)
            mask = torch.isfinite(rri_full)
            if rri_full.numel() == 0 or not mask.any():
                failed += 1
                continue

            rri = rri_full[mask]
            stage_ids = stage_full[mask]

            rri_all.append(rri)
            stage_all.append(stage_ids)
            successes += 1
            print(
                f"[binner-fit] {successes:02d}/{fit_snippets:02d} scene={batch.scene_id} snip={batch.snippet_id} "
                f"C={int(rri.numel())} rri_mean={float(rri.mean().item()):.4f} rri_std={float(rri.std().item()):.4f}"
            )

        binner = RriOrdinalBinner.fit(
            torch.cat(rri_all, dim=0),
            torch.cat(stage_all, dim=0),
            num_classes=int(args.num_classes),
            tanh_scale=float(args.tanh_scale),
        )
        binner.edges = binner.edges.detach().cpu()
        return binner

    if args.fit_binner_only:
        if binner_load_path is not None:
            raise ValueError("--fit-binner-only cannot be combined with --binner-load-path.")
        if int(args.fit_snippets) <= 0:
            raise ValueError("--fit-snippets must be > 0 when fitting the binner.")

        binner = _fit_binner_from_oracle(
            fit_snippets=int(args.fit_snippets), max_attempts=int(args.max_binner_attempts)
        )
        binner_save_path.parent.mkdir(parents=True, exist_ok=True)
        saved_path = binner.save(binner_save_path)
        print(f"[binner-fit] saved: {saved_path}")
        return

    # --------------------------------------------------------------------- VIN module config
    vin_cfg = VinModelConfig(
        backbone=EvlBackboneConfig(device=device),
        head=VinScorerHeadConfig(num_classes=int(args.num_classes)),
    )

    module_cfg = VinLightningModuleConfig(
        vin=vin_cfg,
        optimizer=AdamWConfig(learning_rate=float(args.lr), weight_decay=float(args.weight_decay)),
        num_classes=int(args.num_classes),
        binner_fit_snippets=int(args.fit_snippets),
        binner_tanh_scale=float(args.tanh_scale),
        binner_max_attempts=int(args.max_binner_attempts),
        save_binner=True,
        binner_path=binner_save_path,
    )

    # --------------------------------------------------------------------- trainer config
    cb_cfg = TrainerCallbacksConfig(
        checkpoint_dir=out_dir / "checkpoints",
        checkpoint_monitor="val_loss",
        checkpoint_filename="epoch={epoch}-step={step}-val_loss={val_loss:.4f}",
        use_rich_progress_bar=False,
        use_tqdm_progress_bar=True,
        use_rich_model_summary=True,
        rich_summary_max_depth=-1,
    )
    trainer_cfg = TrainerFactoryConfig(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        max_epochs=10,
        limit_train_batches=int(args.max_steps) if args.stage == "train" else 1,
        limit_val_batches=1,
        check_val_every_n_epoch=1,
        enable_model_summary=False,
        callbacks=cb_cfg,
        use_wandb=bool(args.use_wandb),
        wandb_config=WandbConfig(
            project=str(args.wandb_project),
            name=args.wandb_name,
            offline=bool(args.wandb_offline),
        ),
        is_debug=False,
    )

    trainer = trainer_cfg.setup_target()
    module = module_cfg.setup_target()

    if binner_load_path is not None:
        binner = RriOrdinalBinner.load(binner_load_path)
        binner.edges = binner.edges.detach().cpu()
        module._binner = binner
        if binner_save_path != binner_load_path:
            binner_save_path.parent.mkdir(parents=True, exist_ok=True)
            binner.save(binner_save_path)
        print(f"[binner] loaded: {binner_load_path}")

    resolved_stage = Stage.from_str(str(args.stage))
    ckpt_path = str(args.ckpt_path) if args.ckpt_path else None
    if resolved_stage is Stage.TRAIN:
        trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
    elif resolved_stage is Stage.VAL:
        trainer.validate(module, datamodule=datamodule, ckpt_path=ckpt_path)
    elif resolved_stage is Stage.TEST:
        trainer.test(module, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":  # pragma: no cover
    main()
