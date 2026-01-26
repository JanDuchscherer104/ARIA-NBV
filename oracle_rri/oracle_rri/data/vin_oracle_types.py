"""Shared VIN oracle batch types and collation utilities."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from efm3d.aria.pose import PoseTW
from pytorch3d.renderer.cameras import PerspectiveCameras  # type: ignore[import-untyped]

from ..vin.types import EvlBackboneOutput
from .efm_views import EfmSnippetView, VinSnippetView

if TYPE_CHECKING:
    from ..pipelines.oracle_rri_labeler import OracleRriLabelBatch

Tensor = torch.Tensor


@dataclass(slots=True)
class VinOracleBatch:
    """Single-snippet VIN training batch produced from an oracle label run.

    Attributes:
        efm_snippet_view: EFM snippet view or minimal VIN snippet (None when loading from cache).
        candidate_poses_world_cam: ``PoseTW["N 12"]`` or ``PoseTW["B N 12"]`` candidate poses as world←camera.
        reference_pose_world_rig: ``PoseTW["12"]`` or ``PoseTW["B 12"]`` reference pose as world←rig_reference.
        rri: ``Tensor["N", float32]`` or ``Tensor["B N", float32]`` oracle RRI per candidate.
        pm_dist_before: ``Tensor["N", float32]`` or ``Tensor["B N", float32]`` Chamfer distance before (broadcasted).
        pm_dist_after: ``Tensor["N", float32]`` or ``Tensor["B N", float32]`` Chamfer distance after (per-candidate).
        pm_acc_before: ``Tensor["N", float32]`` or ``Tensor["B N", float32]`` accuracy distance before.
        pm_comp_before: ``Tensor["N", float32]`` or ``Tensor["B N", float32]`` completeness distance before.
        pm_acc_after: ``Tensor["N", float32]`` or ``Tensor["B N", float32]`` accuracy distance after.
        pm_comp_after: ``Tensor["N", float32]`` or ``Tensor["B N", float32]`` completeness distance after.
        p3d_cameras: PyTorch3D cameras used for depth rendering/unprojection (same ordering as candidates).
        scene_id: ASE scene id for diagnostics (string or list when batched).
        snippet_id: Snippet id (tar key/url stem) for diagnostics (string or list when batched).
        backbone_out: Optional cached EVL backbone outputs.
    """

    efm_snippet_view: EfmSnippetView | VinSnippetView | None
    """Optional snippet view (None when loading from cache)."""
    candidate_poses_world_cam: PoseTW
    reference_pose_world_rig: PoseTW
    rri: Tensor
    pm_dist_before: Tensor
    pm_dist_after: Tensor
    pm_acc_before: Tensor
    pm_comp_before: Tensor
    pm_acc_after: Tensor
    pm_comp_after: Tensor
    p3d_cameras: PerspectiveCameras
    scene_id: str | list[str]
    snippet_id: str | list[str]
    backbone_out: EvlBackboneOutput | None = None
    """Optional cached EVL backbone outputs (used to skip backbone inference)."""

    def shape_summary(self) -> dict[str, str]:
        """Summarize tensor shapes for diagnostics/logging."""

        def _shape(value: object) -> str:
            if torch.is_tensor(value):
                return str(tuple(value.shape))  # type: ignore[attr-defined]
            return str(type(value).__name__)

        out: dict[str, str] = {
            "candidate_poses_world_cam": str(tuple(self.candidate_poses_world_cam.tensor().shape)),
            "reference_pose_world_rig": str(tuple(self.reference_pose_world_rig.tensor().shape)),
            "rri": _shape(self.rri),
            "pm_dist_before": _shape(self.pm_dist_before),
            "pm_dist_after": _shape(self.pm_dist_after),
            "pm_acc_before": _shape(self.pm_acc_before),
            "pm_comp_before": _shape(self.pm_comp_before),
            "pm_acc_after": _shape(self.pm_acc_after),
            "pm_comp_after": _shape(self.pm_comp_after),
            "p3d_cameras.R": _shape(self.p3d_cameras.R),
            "p3d_cameras.T": _shape(self.p3d_cameras.T),
            "p3d_cameras.focal_length": _shape(self.p3d_cameras.focal_length),
            "p3d_cameras.principal_point": _shape(self.p3d_cameras.principal_point),
            "p3d_cameras.image_size": _shape(self.p3d_cameras.image_size),
        }

        poses = self.candidate_poses_world_cam.tensor()
        batch_size = None
        num_candidates = None
        if poses.ndim == 2:
            batch_size = 1
            num_candidates = int(poses.shape[0])
        elif poses.ndim == 3:
            batch_size = int(poses.shape[0])
            num_candidates = int(poses.shape[1])
        if batch_size is not None and num_candidates is not None:
            cam_count = int(self.p3d_cameras.R.shape[0])
            if cam_count == batch_size * num_candidates:
                out["p3d_cameras.batch_mode"] = "flat (B*N)"
                out["p3d_cameras.R_grouped"] = str((batch_size, num_candidates, 3, 3))
                out["p3d_cameras.T_grouped"] = str((batch_size, num_candidates, 3))
                out["p3d_cameras.focal_length_grouped"] = str((batch_size, num_candidates, 2))
                out["p3d_cameras.principal_point_grouped"] = str((batch_size, num_candidates, 2))
                out["p3d_cameras.image_size_grouped"] = str((batch_size, num_candidates, 2))

        if isinstance(self.efm_snippet_view, VinSnippetView):
            out["vin_snippet.points_world"] = _shape(self.efm_snippet_view.points_world)
            out["vin_snippet.lengths"] = _shape(self.efm_snippet_view.lengths)
            out["vin_snippet.t_world_rig"] = _shape(self.efm_snippet_view.t_world_rig.tensor())
        elif isinstance(self.efm_snippet_view, EfmSnippetView):
            out["efm_snippet_view"] = "EfmSnippetView"
        elif self.efm_snippet_view is None:
            out["efm_snippet_view"] = "None"

        if self.backbone_out is not None:
            if is_dataclass(self.backbone_out):
                items = [(field.name, getattr(self.backbone_out, field.name)) for field in fields(self.backbone_out)]
            else:
                items = list(vars(self.backbone_out).items())
            for name, value in items:
                if torch.is_tensor(value):
                    out[f"backbone.{name}"] = _shape(value)
        else:
            out["backbone"] = "None"

        return out

    @classmethod
    def from_label(
        cls,
        label_batch: "OracleRriLabelBatch",
        *,
        efm_keep_keys: set[str] | None,
    ) -> "VinOracleBatch":
        """Build a VIN oracle batch from an online label batch."""
        rri = label_batch.rri
        sample = label_batch.sample
        if efm_keep_keys is not None:
            sample = sample.prune_efm(efm_keep_keys)

        return cls(
            efm_snippet_view=sample,
            candidate_poses_world_cam=label_batch.depths.poses,
            reference_pose_world_rig=label_batch.depths.reference_pose,
            rri=rri.rri,
            pm_dist_before=rri.pm_dist_before,
            pm_dist_after=rri.pm_dist_after,
            pm_acc_before=rri.pm_acc_before,
            pm_comp_before=rri.pm_comp_before,
            pm_acc_after=rri.pm_acc_after,
            pm_comp_after=rri.pm_comp_after,
            p3d_cameras=label_batch.depths.p3d_cameras,
            scene_id=sample.scene_id,
            snippet_id=sample.snippet_id,
            backbone_out=None,
        )

    @classmethod
    def collate(cls, samples: list["VinOracleBatch"]) -> "VinOracleBatch":
        """Collate cached VIN batches by padding candidate sets to a shared length."""
        if not samples:
            raise ValueError("Empty batch passed to VinOracleBatch.collate.")

        snippet_views = [sample.efm_snippet_view for sample in samples]
        has_snippet = any(view is not None for view in snippet_views)
        if has_snippet:
            if any(view is None for view in snippet_views):
                raise ValueError(
                    "Mixed snippet views in batch; ensure include_efm_snippet is consistent.",
                )
            if not all(isinstance(view, VinSnippetView) for view in snippet_views):
                raise NotImplementedError(
                    "Batching with full EfmSnippetView is not supported. Use VinSnippetView from the offline cache.",
                )

        candidate_counts = [int(sample.candidate_poses_world_cam.shape[-2]) for sample in samples]
        max_candidates = max(candidate_counts) if candidate_counts else 0
        if max_candidates <= 0:
            raise ValueError("Cannot batch empty candidate sets.")

        poses = torch.stack(
            [
                cls._pad_candidate_poses(sample.candidate_poses_world_cam, target_len=max_candidates)
                for sample in samples
            ],
            dim=0,
        )
        candidate_poses_world_cam = PoseTW(poses)

        reference_pose_world_rig = cls._stack_reference_poses(
            [sample.reference_pose_world_rig for sample in samples],
        )

        rri = torch.stack(
            [cls._pad_1d(sample.rri, target_len=max_candidates, pad_value=float("nan")) for sample in samples],
            dim=0,
        )
        pm_dist_before = torch.stack(
            [
                cls._pad_1d(sample.pm_dist_before, target_len=max_candidates, pad_value=float("nan"))
                for sample in samples
            ],
            dim=0,
        )
        pm_dist_after = torch.stack(
            [
                cls._pad_1d(sample.pm_dist_after, target_len=max_candidates, pad_value=float("nan"))
                for sample in samples
            ],
            dim=0,
        )
        pm_acc_before = torch.stack(
            [
                cls._pad_1d(sample.pm_acc_before, target_len=max_candidates, pad_value=float("nan"))
                for sample in samples
            ],
            dim=0,
        )
        pm_comp_before = torch.stack(
            [
                cls._pad_1d(sample.pm_comp_before, target_len=max_candidates, pad_value=float("nan"))
                for sample in samples
            ],
            dim=0,
        )
        pm_acc_after = torch.stack(
            [cls._pad_1d(sample.pm_acc_after, target_len=max_candidates, pad_value=float("nan")) for sample in samples],
            dim=0,
        )
        pm_comp_after = torch.stack(
            [
                cls._pad_1d(sample.pm_comp_after, target_len=max_candidates, pad_value=float("nan"))
                for sample in samples
            ],
            dim=0,
        )

        p3d_cameras = cls._stack_p3d_cameras(
            [sample.p3d_cameras for sample in samples],
            target_len=max_candidates,
        )

        scene_id = [sample.scene_id for sample in samples]
        snippet_id = [sample.snippet_id for sample in samples]

        backbone_out = None
        if any(sample.backbone_out is not None for sample in samples):
            backbone_out = cls._stack_backbone_outputs([sample.backbone_out for sample in samples])  # type: ignore
        vin_snippet = None
        if has_snippet:
            points_list = [view.points_world for view in snippet_views if view is not None]
            lengths_list = [view.lengths for view in snippet_views if view is not None]
            traj_list = [view.t_world_rig for view in snippet_views if view is not None]
            max_points = max(int(points.shape[0]) for points in points_list)
            max_frames = max(int(traj.shape[0]) for traj in traj_list)
            points_world = torch.stack(
                [cls._pad_points(points, target_len=max_points) for points in points_list],
                dim=0,
            )
            lengths = torch.stack(
                [
                    length.reshape(-1)[0]
                    if length.numel() > 0
                    else torch.tensor(0, device=points_world.device, dtype=torch.int64)
                    for length in lengths_list
                ],
                dim=0,
            ).to(device=points_world.device, dtype=torch.int64)
            t_world_rig = PoseTW(
                torch.stack(
                    [cls._pad_trajectory(traj, target_len=max_frames) for traj in traj_list],
                    dim=0,
                ),
            )
            vin_snippet = VinSnippetView(points_world=points_world, lengths=lengths, t_world_rig=t_world_rig)

        return cls(
            efm_snippet_view=vin_snippet,
            candidate_poses_world_cam=candidate_poses_world_cam,
            reference_pose_world_rig=reference_pose_world_rig,
            rri=rri,
            pm_dist_before=pm_dist_before,
            pm_dist_after=pm_dist_after,
            pm_acc_before=pm_acc_before,
            pm_comp_before=pm_comp_before,
            pm_acc_after=pm_acc_after,
            pm_comp_after=pm_comp_after,
            p3d_cameras=p3d_cameras,
            scene_id=scene_id,
            snippet_id=snippet_id,
            backbone_out=backbone_out,
        )

    @staticmethod
    def _pad_candidate_poses(poses: PoseTW, *, target_len: int) -> Tensor:
        data = poses.tensor()
        if data.ndim == 1:
            data = data.unsqueeze(0)
        if data.shape[-1] != 12:
            raise ValueError("candidate_poses_world_cam must have shape (N,12).")
        num = int(data.shape[0])
        if num == target_len:
            return data
        if num == 0:
            pad = PoseTW().tensor().expand(target_len, -1)
            return pad
        if num > target_len:
            return data[:target_len]
        pad = data[-1:].expand(target_len - num, -1).clone()
        return torch.cat([data, pad], dim=0)

    @staticmethod
    def _pad_points(points: Tensor, *, target_len: int) -> Tensor:
        if points.ndim == 3 and points.shape[0] == 1:
            points = points.squeeze(0)
        if points.ndim != 2:
            raise ValueError("points_world must have shape (K, D).")
        num = int(points.shape[0])
        if num == target_len:
            return points
        if num > target_len:
            return points[:target_len]
        pad = torch.full(
            (target_len - num, points.shape[1]),
            float("nan"),
            dtype=points.dtype,
            device=points.device,
        )
        return torch.cat([points, pad], dim=0)

    @staticmethod
    def _pad_trajectory(poses: PoseTW, *, target_len: int) -> Tensor:
        data = poses.tensor()
        if data.ndim == 3 and data.shape[0] == 1:
            data = data.squeeze(0)
        if data.ndim != 2 or data.shape[-1] != 12:
            raise ValueError("t_world_rig must have shape (F,12).")
        num = int(data.shape[0])
        if num == target_len:
            return data
        if num == 0:
            pad = PoseTW().tensor().expand(target_len, -1)
            return pad
        if num > target_len:
            return data[:target_len]
        pad = data[-1:].expand(target_len - num, -1).clone()
        return torch.cat([data, pad], dim=0)

    @staticmethod
    def _pad_1d(values: Tensor, *, target_len: int, pad_value: float) -> Tensor:
        flat = values.reshape(-1)
        num = int(flat.shape[0])
        if num == target_len:
            return flat
        if num > target_len:
            return flat[:target_len]
        pad = torch.full(
            (target_len - num,),
            pad_value,
            dtype=flat.dtype,
            device=flat.device,
        )
        return torch.cat([flat, pad], dim=0)

    @staticmethod
    def _stack_reference_poses(poses: list[PoseTW]) -> PoseTW:
        tensors = []
        for pose in poses:
            data = pose.tensor()
            if data.ndim == 2:
                if data.shape[0] != 1:
                    raise ValueError("reference_pose_world_rig must have shape (12,) or (1,12).")
                data = data.squeeze(0)
            if data.ndim != 1:
                raise ValueError("reference_pose_world_rig must have shape (12,) or (1,12).")
            tensors.append(data)
        return PoseTW(torch.stack(tensors, dim=0))

    @staticmethod
    def _expand_camera_param(param: Tensor, *, target_len: int, name: str) -> Tensor:
        if param.shape[0] == target_len:
            return param
        if param.shape[0] == 1:
            return param.expand(target_len, *param.shape[1:])
        raise ValueError(f"{name} batch size {param.shape[0]} does not match target_len={target_len}.")

    @staticmethod
    def _pad_camera_param(param: Tensor, *, target_len: int) -> Tensor:
        if param.shape[0] == target_len:
            return param
        if param.shape[0] > target_len:
            return param[:target_len]
        pad = param[-1:].expand(target_len - param.shape[0], *param.shape[1:]).clone()
        return torch.cat([param, pad], dim=0)

    @classmethod
    def _stack_p3d_cameras(
        cls,
        cameras: list[PerspectiveCameras],
        *,
        target_len: int,
    ) -> PerspectiveCameras:
        if not cameras:
            raise ValueError("No cameras provided for collation.")

        in_ndc = getattr(cameras[0], "in_ndc", False)
        if callable(in_ndc):
            in_ndc = in_ndc()

        znear = getattr(cameras[0], "znear", None)
        zfar = getattr(cameras[0], "zfar", None)
        for cam in cameras[1:]:
            cam_in_ndc = getattr(cam, "in_ndc", False)
            if callable(cam_in_ndc):
                cam_in_ndc = cam_in_ndc()
            if bool(cam_in_ndc) != bool(in_ndc):
                raise ValueError("All PerspectiveCameras must agree on in_ndc for batching.")

        r_list: list[Tensor] = []
        t_list: list[Tensor] = []
        f_list: list[Tensor] = []
        pp_list: list[Tensor] = []
        im_list: list[Tensor] = []
        for cam in cameras:
            num = int(cam.R.shape[0])
            rot = cam.R
            trans = cam.T
            focal = cls._expand_camera_param(cam.focal_length, target_len=num, name="focal_length")
            principal = cls._expand_camera_param(cam.principal_point, target_len=num, name="principal_point")
            image_size = cls._expand_camera_param(cam.image_size, target_len=num, name="image_size")

            rot = cls._pad_camera_param(rot, target_len=target_len)
            trans = cls._pad_camera_param(trans, target_len=target_len)
            focal = cls._pad_camera_param(focal, target_len=target_len)
            principal = cls._pad_camera_param(principal, target_len=target_len)
            image_size = cls._pad_camera_param(image_size, target_len=target_len)

            r_list.append(rot)
            t_list.append(trans)
            f_list.append(focal)
            pp_list.append(principal)
            im_list.append(image_size)

        rot_all = torch.cat(r_list, dim=0)
        trans_all = torch.cat(t_list, dim=0)
        focal_all = torch.cat(f_list, dim=0)
        principal_all = torch.cat(pp_list, dim=0)
        image_all = torch.cat(im_list, dim=0)

        kwargs = {
            "device": rot_all.device,
            "R": rot_all,
            "T": trans_all,
            "focal_length": focal_all,
            "principal_point": principal_all,
            "image_size": image_all,
            "in_ndc": bool(in_ndc),
        }
        if znear is not None:
            kwargs["znear"] = znear
        if zfar is not None:
            kwargs["zfar"] = zfar
        try:
            return PerspectiveCameras(**kwargs)
        except TypeError:
            kwargs.pop("znear", None)
            kwargs.pop("zfar", None)
            cameras_out = PerspectiveCameras(**kwargs)
            if znear is not None:
                cameras_out.znear = znear
            if zfar is not None:
                cameras_out.zfar = zfar
            return cameras_out

    @classmethod
    def _stack_tensor_field(cls, values: list[Tensor | None], *, name: str) -> Tensor | None:
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"Cannot batch backbone field '{name}': missing values.")
        tensors = [value for value in values if value is not None]
        first = tensors[0]
        for tensor in tensors[1:]:
            if tensor.shape != first.shape:
                raise ValueError(f"Cannot batch backbone field '{name}': mismatched shapes.")
        if first.ndim > 1 and first.shape[0] == 1:
            return torch.cat(tensors, dim=0)
        return torch.stack(tensors, dim=0)

    @classmethod
    def _stack_tensor_dict(
        cls,
        values: list[dict[str, Tensor]],
        *,
        name: str,
    ) -> dict[str, Tensor]:
        if all(len(value) == 0 for value in values):
            return {}
        keys = set(values[0].keys())
        for value in values[1:]:
            if set(value.keys()) != keys:
                raise ValueError(f"Cannot batch backbone field '{name}': mismatched dict keys.")
        return {key: cls._stack_tensor_field([value[key] for value in values], name=f"{name}.{key}") for key in keys}

    @classmethod
    def _stack_backbone_outputs(cls, outputs: list[EvlBackboneOutput]) -> EvlBackboneOutput:
        if any(output is None for output in outputs):
            raise ValueError("Cannot batch backbone outputs when some entries are missing.")

        t_world_voxel = cls._stack_reference_poses([output.t_world_voxel for output in outputs])
        voxel_extent = cls._stack_tensor_field([output.voxel_extent for output in outputs], name="voxel_extent")
        if voxel_extent is None:
            raise ValueError("voxel_extent is required for VIN batching.")

        obb_fields = [
            ("obbs_pr_nms", [output.obbs_pr_nms for output in outputs]),
            ("obb_pred", [output.obb_pred for output in outputs]),
            ("obb_pred_viz", [output.obb_pred_viz for output in outputs]),
        ]
        for field_name, values in obb_fields:
            if any(value is not None for value in values):
                raise NotImplementedError(
                    f"Batching backbone field '{field_name}' is not supported yet. "
                    "Disable OBB outputs or set batch_size=None.",
                )

        list_fields = [
            ("obb_pred_probs_full", [output.obb_pred_probs_full for output in outputs]),
            ("obb_pred_probs_full_viz", [output.obb_pred_probs_full_viz for output in outputs]),
        ]
        for field_name, values in list_fields:
            if any(value is not None for value in values):
                raise NotImplementedError(
                    f"Batching backbone field '{field_name}' is not supported yet. "
                    "Disable OBB outputs or set batch_size=None.",
                )

        name_lists = [output.obb_pred_sem_id_to_name for output in outputs]
        if any(name is not None for name in name_lists):
            if any(name != name_lists[0] for name in name_lists[1:]):
                raise ValueError("obb_pred_sem_id_to_name must match across batch.")
            obb_pred_sem_id_to_name = name_lists[0]
        else:
            obb_pred_sem_id_to_name = None

        feat2d = cls._stack_tensor_dict([output.feat2d_upsampled for output in outputs], name="feat2d_upsampled")
        token2d = cls._stack_tensor_dict([output.token2d for output in outputs], name="token2d")

        return EvlBackboneOutput(
            t_world_voxel=t_world_voxel,
            voxel_extent=voxel_extent,
            voxel_feat=cls._stack_tensor_field([output.voxel_feat for output in outputs], name="voxel_feat"),
            occ_feat=cls._stack_tensor_field([output.occ_feat for output in outputs], name="occ_feat"),
            obb_feat=cls._stack_tensor_field([output.obb_feat for output in outputs], name="obb_feat"),
            occ_pr=cls._stack_tensor_field([output.occ_pr for output in outputs], name="occ_pr"),
            occ_input=cls._stack_tensor_field([output.occ_input for output in outputs], name="occ_input"),
            free_input=cls._stack_tensor_field([output.free_input for output in outputs], name="free_input"),
            counts=cls._stack_tensor_field([output.counts for output in outputs], name="counts"),
            counts_m=cls._stack_tensor_field([output.counts_m for output in outputs], name="counts_m"),
            voxel_select_t=cls._stack_tensor_field(
                [output.voxel_select_t for output in outputs], name="voxel_select_t"
            ),
            cent_pr=cls._stack_tensor_field([output.cent_pr for output in outputs], name="cent_pr"),
            bbox_pr=cls._stack_tensor_field([output.bbox_pr for output in outputs], name="bbox_pr"),
            clas_pr=cls._stack_tensor_field([output.clas_pr for output in outputs], name="clas_pr"),
            cent_pr_nms=cls._stack_tensor_field([output.cent_pr_nms for output in outputs], name="cent_pr_nms"),
            obbs_pr_nms=None,
            obb_pred=None,
            obb_pred_viz=None,
            obb_pred_sem_id_to_name=obb_pred_sem_id_to_name,
            obb_pred_probs_full=None,
            obb_pred_probs_full_viz=None,
            pts_world=cls._stack_tensor_field([output.pts_world for output in outputs], name="pts_world"),
            feat2d_upsampled=feat2d,
            token2d=token2d,
        )


@runtime_checkable
class VinOracleDatasetBase(Protocol):
    """Shared interface for datasets that yield :class:`VinOracleBatch`."""

    is_map_style: bool
    """Whether the dataset is map-style (supports random access + batching)."""

    def __iter__(self) -> Iterator[VinOracleBatch]:
        """Iterate over VIN oracle batches."""


__all__ = ["VinOracleBatch", "VinOracleDatasetBase"]
