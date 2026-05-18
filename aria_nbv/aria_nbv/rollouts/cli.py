"""CLI for building standalone target-RRI rollout stores."""

from __future__ import annotations

# TODO: Use typer for nicer CLI!
import argparse
import sys
from pathlib import Path

from ..utils import Console
from ..utils.config_paths import resolve_config_toml_path
from .dataset_writer import RolloutDatasetWriterConfig
from .manifest import RolloutStoreInvocation
from .shards import (
    load_rollout_shard_entry_for_cli,
    run_rollout_shard,
    summarize_rollout_shard_campaign,
    write_rollout_shard_manifest_from_config,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nbv-build-rollouts",
        description="Build a standalone target-RRI rollout Zarr store from VIN offline rows.",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        type=Path,
        help="Path to a RolloutDatasetWriterConfig TOML file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the TOML and print resolved paths without loading data or writing Zarr.",
    )
    parser.add_argument(
        "--shard-manifest",
        type=Path,
        help="JSONL rollout shard manifest emitted by nbv-plan-rollout-shards.",
    )
    parser.add_argument(
        "--shard-id",
        help="Rollout shard id to build, for example shard-000123 or a Slurm array task id.",
    )
    parser.add_argument(
        "--output-tmp",
        type=Path,
        help="Temporary shard output directory used before atomic promotion.",
    )
    parser.add_argument(
        "--output-final",
        type=Path,
        help="Final shard output directory that receives _SUCCESS.json last.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    raw_argv = sys.argv[1:] if argv is None else argv
    parser = _build_parser()
    args = parser.parse_args(raw_argv)
    console = Console.with_prefix("nbv-build-rollouts")
    config_path = resolve_config_toml_path(args.config_path)
    cfg = RolloutDatasetWriterConfig.from_toml(config_path)
    console.log(f"Loaded rollout writer config: {config_path}")
    console.log(f"Resolved source store: {cfg.source.store.store_dir}")
    console.log(f"Resolved rollout store: {cfg.store.store_dir}")
    console.log(f"Target top-k: {cfg.target_selector.k}")
    console.log(f"Candidate mixture budget: {cfg.candidate_mixture.total_count}")
    shard_args = (args.shard_manifest, args.shard_id, args.output_tmp, args.output_final)
    if any(value is not None for value in shard_args) and not all(value is not None for value in shard_args):
        parser.error("--shard-manifest, --shard-id, --output-tmp, and --output-final must be supplied together.")
    if all(value is not None for value in shard_args):
        shard_entry = load_rollout_shard_entry_for_cli(args.shard_manifest, args.shard_id)
        console.log(f"Resolved rollout shard: {shard_entry.shard_id} rows={len(shard_entry.rows)}")
        console.log(f"Shard tmp output: {args.output_tmp}")
        console.log(f"Shard final output: {args.output_final}")
        if args.dry_run:
            console.log("Dry run complete; shard manifest was loaded but no rollout writer was instantiated.")
            return
        shard_result = run_rollout_shard(
            cfg,
            shard_entry=shard_entry,
            output_tmp=args.output_tmp,
            output_final=args.output_final,
            invocation=RolloutStoreInvocation.from_cli(argv=["nbv-build-rollouts", *raw_argv], config_path=config_path),
        )
        if shard_result.skipped:
            console.log(f"Skipped completed rollout shard: {shard_result.final_dir}")
            return
        assert shard_result.store_result is not None
        result = shard_result.store_result
        console.log(
            "Wrote rollout shard: "
            f"rollouts={result.num_rollouts} steps={result.num_steps} candidates={result.num_candidates} "
            f"path={shard_result.final_dir} success={shard_result.success_path}",
        )
        return
    if args.dry_run:
        console.log("Dry run complete; no VIN offline dataset or rollout writer was instantiated.")
        return
    result = cfg.setup_target().run(
        invocation=RolloutStoreInvocation.from_cli(argv=["nbv-build-rollouts", *raw_argv], config_path=config_path)
    )
    console.log(
        "Wrote rollout Zarr store: "
        f"rollouts={result.num_rollouts} steps={result.num_steps} candidates={result.num_candidates} "
        f"path={result.store_dir} manifest={result.manifest_path}",
    )


def _build_plan_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nbv-plan-rollout-shards",
        description="Plan deterministic source-row rollout shard manifests from a writer TOML.",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        type=Path,
        help="Path to a RolloutDatasetWriterConfig TOML file.",
    )
    parser.add_argument(
        "--output-manifest",
        required=True,
        type=Path,
        help="Destination JSONL shard manifest path.",
    )
    parser.add_argument(
        "--rows-per-shard",
        required=True,
        type=int,
        help="Maximum number of VIN source rows owned by one rollout shard.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan shards and print counts without writing the JSONL manifest.",
    )
    return parser


def plan_main(argv: list[str] | None = None) -> None:
    """CLI entry point for planning rollout source-row shards."""

    raw_argv = sys.argv[1:] if argv is None else argv
    args = _build_plan_parser().parse_args(raw_argv)
    console = Console.with_prefix("nbv-plan-rollout-shards")
    config_path = resolve_config_toml_path(args.config_path)
    cfg = RolloutDatasetWriterConfig.from_toml(config_path)
    if args.dry_run:
        from .shards import plan_rollout_shards

        entries = plan_rollout_shards(cfg, rows_per_shard=args.rows_per_shard)
    else:
        entries = write_rollout_shard_manifest_from_config(
            cfg,
            manifest_path=args.output_manifest,
            rows_per_shard=args.rows_per_shard,
        )
    console.log(
        f"Planned rollout shards: count={len(entries)} rows={sum(len(entry.rows) for entry in entries)} "
        f"rows_per_shard={args.rows_per_shard}",
    )
    if args.dry_run:
        console.log(f"Dry run complete; manifest was not written: {args.output_manifest}")
    else:
        console.log(f"Wrote rollout shard manifest: {args.output_manifest}")


def _build_status_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nbv-status-rollout-shards",
        description="Summarize succeeded, failed, incomplete, and missing rollout shards for one campaign.",
    )
    parser.add_argument(
        "--shard-manifest",
        required=True,
        type=Path,
        help="JSONL rollout shard manifest emitted by nbv-plan-rollout-shards.",
    )
    parser.add_argument(
        "--final-root",
        required=True,
        type=Path,
        help="Directory containing final shard directories named shard-000000, shard-000001, ...",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for a machine-readable campaign status JSON file.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit with status 2 when any planned shard is not succeeded.",
    )
    return parser


def status_main(argv: list[str] | None = None) -> None:
    """CLI entry point for rollout shard campaign status reporting."""

    args = _build_status_parser().parse_args(sys.argv[1:] if argv is None else argv)
    console = Console.with_prefix("nbv-status-rollout-shards")
    status = summarize_rollout_shard_campaign(
        args.shard_manifest,
        final_root=args.final_root,
    )
    counts = status.counts
    console.log(
        "Rollout shard campaign status: "
        f"total={len(status.shards)} succeeded={counts['succeeded']} failed={counts['failed']} "
        f"incomplete={counts['incomplete']} missing={counts['missing']}",
    )
    problems = [shard for shard in status.shards if shard.status != "succeeded"]
    if problems:
        problem_ids = ", ".join(f"{shard.shard_id}:{shard.status}" for shard in problems[:20])
        suffix = "" if len(problems) <= 20 else f", ... +{len(problems) - 20} more"
        console.log(f"Problem shards: {problem_ids}{suffix}")
    if args.output_json is not None:
        from .manifest import manifest_json_bytes

        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(manifest_json_bytes(status.to_jsonable()))
        console.log(f"Wrote rollout shard campaign status JSON: {output_path}")
    if args.require_complete and problems:
        raise SystemExit(2)


__all__ = ["main", "plan_main", "status_main"]
