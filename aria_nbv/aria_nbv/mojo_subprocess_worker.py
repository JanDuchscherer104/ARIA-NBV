"""Subprocess entrypoint for thread-incompatible Mojo-backed stages."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from efm3d.aria import CameraTW, PoseTW

from .pose_generation.mojo_backend import (
    clearance_mask_mojo,
    path_collision_mask_mojo,
    point_mesh_distance_mojo,
)
from .rendering.camera_batches import NativeCameraBatch
from .rendering.candidate_pointclouds import _backproject_depths_mojo_batch
from .rendering.mojo_depth_renderer import MojoDepthRenderer, MojoDepthRendererConfig
from .rri_metrics.oracle_rri import OracleDistanceBackend, OracleRRIConfig


def _render_payload(payload: dict[str, Any]) -> dict[str, Any]:
    renderer = MojoDepthRenderer(
        MojoDepthRendererConfig(
            device="cpu",
            znear=float(payload["znear"]),
            zfar=float(payload["zfar"]),
            workers=payload.get("workers"),
        ),
    )
    poses = PoseTW(payload["poses"])
    camera = CameraTW(payload["camera"])
    depths, mask = renderer.render_depths(
        poses=poses,
        mesh=(payload["verts"], payload["faces"]),
        camera=camera,
    )
    return {"depths": depths.cpu(), "depths_valid_mask": mask.cpu()}


def _unproject_payload(payload: dict[str, Any]) -> dict[str, Any]:
    padded, lengths = _backproject_depths_mojo_batch(
        depths=payload["depths"],
        mask_valid=payload["mask_valid"],
        poses=PoseTW(payload["poses"]),
        camera_batch=NativeCameraBatch(camera_tw=CameraTW(payload["camera"]), pose_world_cam=PoseTW(payload["poses"])),
        stride=int(payload["stride"]),
    )
    return {"points": padded.cpu(), "lengths": lengths.cpu()}


def _oracle_payload(payload: dict[str, Any]) -> dict[str, Any]:
    oracle = OracleRRIConfig(
        backend=OracleDistanceBackend.MOJO,
        mojo={"workers": payload.get("workers")},
    ).setup_target()
    result = oracle.score(
        points_t=payload["points_t"],
        points_q=payload["points_q"],
        lengths_q=payload["lengths_q"],
        gt_verts=payload["gt_verts"],
        gt_faces=payload["gt_faces"],
        extend=payload["extend"],
    )
    return {
        "rri": result.rri.cpu(),
        "pm_dist_before": result.pm_dist_before.cpu(),
        "pm_dist_after": result.pm_dist_after.cpu(),
        "pm_acc_before": result.pm_acc_before.cpu(),
        "pm_comp_before": result.pm_comp_before.cpu(),
        "pm_acc_after": result.pm_acc_after.cpu(),
        "pm_comp_after": result.pm_comp_after.cpu(),
        "fscore_tau": result.fscore_tau.cpu() if result.fscore_tau is not None else None,
    }


def _collision_clearance_payload(payload: dict[str, Any]) -> dict[str, Any]:
    keep = clearance_mask_mojo(
        payload["points"],
        payload["triangles"],
        min_distance=float(payload["min_distance"]),
        workers=payload.get("workers"),
    )
    return {"keep": keep.cpu()}


def _collision_path_payload(payload: dict[str, Any]) -> dict[str, Any]:
    collide = path_collision_mask_mojo(
        payload["origin"],
        payload["targets"],
        payload["triangles"],
        workers=payload.get("workers"),
    )
    return {"collide": collide.cpu()}


def _collision_distance_payload(payload: dict[str, Any]) -> dict[str, Any]:
    distance = point_mesh_distance_mojo(
        payload["points"],
        payload["triangles"],
        workers=payload.get("workers"),
    )
    return {"distance": distance.cpu()}


def _dispatch(mode: str, payload: dict[str, Any]) -> dict[str, Any]:
    if mode == "render_depths":
        return _render_payload(payload)
    if mode == "unproject_points":
        return _unproject_payload(payload)
    if mode == "oracle_score":
        return _oracle_payload(payload)
    if mode == "collision_clearance":
        return _collision_clearance_payload(payload)
    if mode == "collision_path":
        return _collision_path_payload(payload)
    if mode == "collision_distance":
        return _collision_distance_payload(payload)
    raise ValueError(f"Unsupported mojo subprocess mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = torch.load(Path(args.input), map_location="cpu", weights_only=False)
    result = _dispatch(args.mode, payload)
    torch.save(result, Path(args.output))


if __name__ == "__main__":
    main()
