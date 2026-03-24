= Coordinate Conventions and Geometry

#import "../../shared/macros.typ": *

// TODO(paper-cleanup): Cross-check all frame/pose conventions against Appendix
// @sec:appendix-pose-frames and slides_4.typ; avoid drifting “CW90” explanations across sections.

We follow the EFM3D/ATEK coordinate conventions throughout the pipeline. The
world frame is gravity-aligned, the rig frame moves with the headset, and each
camera frame is expressed in left-up-forward (LUF) coordinates. All poses are
represented as SE(3) transforms, and cameras are represented by calibrated
camera objects that bundle intrinsics and extrinsics.

LUF is defined as $+x$ left, $+y$ up, and $+z$ forward (along the optical axis).
Image-plane pixel coordinates follow the standard convention: origin at the
top-left corner, $u$ increasing to the right, and $v$ increasing downward.

For the fixed CW90 rig-basis correction (`rotate_yaw_cw90`) used in candidate
generation and VIN inputs, see Appendix @sec:appendix-pose-frames.

== Transformation notation

We write #T("A", "B") for the transform from frame $B$ to frame $A$. Such
transforms lie in $"SE"(3)$; composing poses follows standard multiplication,
and inversion yields the reverse transform. For example, a world point $bold(x)$
can be expressed in a (candidate) camera frame via

#block[
  #align(center)[
    $ bold(x)^(#symb.frame.cq) = #T(symb.frame.cq, symb.frame.w) bold(x)^(#symb.frame.w) $
  ]
]


These conventions are critical when projecting semi-dense points into candidate
views and when mapping EVL voxel coordinates back to the rig frame.

== EVL voxel grid contract

EVL represents the scene in a local, fixed-size voxel grid aligned to a
gravity-aware frame. The grid is specified by a rigid transform between voxel
and world coordinates and a metric extent (grid bounds in metres). Candidates
can fall partially or fully outside this extent; we therefore track per-candidate
validity signals so the NBV head can down-weight unreliable voxel context.

== Camera model

EFM3D represents Aria cameras as batched `CameraTW` calibration objects (fisheye
model parameters plus per-frame extrinsics) and provides utilities such as
valid-radius masking for diagnostics @EFM3D-straub2024. For oracle rendering,
backprojection, and view-conditioned projections, we convert these calibrations
to a PyTorch3D `PerspectiveCameras` instance so that rasterization and
unprojection share a single camera convention @PyTorch3D-Cameras-2025.

We use PyTorch3D's `PerspectiveCameras` for the oracle renderer because it
enables fast batched mesh rasterization on GPU and provides a consistent
projection/unprojection interface that we reuse for view-conditioned features
@PyTorch3D-Cameras-2025. Concretely, we convert Aria calibrations to a batched
PyTorch3D camera instance and render metric depth maps with `in_ndc=false`.
Using the same camera model for (i) depth rendering, (ii) backprojection, and
(iii) screen-space projection of semi-dense points prevents convention bugs
(pixel vs NDC, principal point ordering, and axis directions) and keeps the
oracle labels and VIN features geometrically consistent.
// TODO(paper-cleanup): Keep the “why PyTorch3D” rationale here, but move pixel↔NDC specifics
// to Section @sec:oracle-rri to avoid redundancy.
