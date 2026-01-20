= Coordinate Conventions and Geometry

#import "/typst/shared/macros.typ": *

We follow the EFM3D/ATEK coordinate conventions throughout the pipeline. The
world frame is gravity-aligned, the rig frame moves with the headset, and each
camera frame is expressed in left-up-forward (LUF) coordinates. All poses are
represented as SE(3) transforms, and cameras are represented by calibrated
camera objects that bundle intrinsics and extrinsics.

LUF is defined as $+x$ left, $+y$ up, and $+z$ forward (along the optical axis).
Image-plane pixel coordinates follow the standard convention: origin at the
top-left corner, $u$ increasing to the right, and $v$ increasing downward.

For visualization only, our dashboard applies a fixed 90#sym.degree yaw
rotation to match its display convention. All geometric reasoning and learning
components operate in the canonical coordinate frames.

== Transformation notation

We write #T("A", "B") for the transform from frame $B$ to frame $A$. Such
transforms lie in $"SE"(3)$; composing poses follows standard multiplication,
and inversion yields the reverse transform. For example, a world point $x$ can
be expressed in the camera frame via

#block[
  #align(center)[
    $ bold(x)^"cam" = #T(fr_cam, fr_world) bold(x)^"w" $
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

EFM3D provides a batched fisheye camera model that supports projection,
unprojection, and valid-radius masking. We rely on these utilities when
rendering depth maps, unprojecting valid pixels, and computing candidate
frustum diagnostics. For view conditioning, we project semi-dense points into
candidate views using a consistent screen-space camera model aligned with the
renderer. Using calibrated camera objects avoids mismatches between
camera-specific parameters and the global frame.
