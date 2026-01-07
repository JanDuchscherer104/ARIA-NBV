= Coordinate Conventions and Geometry

#import "/typst/shared/macros.typ": *

We follow the EFM3D/ATEK coordinate conventions throughout the pipeline. The
world frame is gravity-aligned, the rig frame moves with the headset, and each
camera frame is expressed in left-up-forward (LUF) coordinates. All poses are
represented as SE(3) transforms, and cameras are represented by calibrated
camera objects that bundle intrinsics and extrinsics.

Implementation note: our visualization stack uses a fixed 90#sym.degree yaw
rotation to match the dashboard's display convention. All geometric reasoning
and learning components operate in the canonical EFM3D coordinate conventions.

== Transformation notation

We write $T_(A<-B)$ for the transform from frame $B$ to frame $A$. Composing
poses follows standard SE(3) multiplication, and inversion yields the reverse
transform. For example, a world point $x$ can be expressed in the camera frame
via

#block[
  #align(center)[
    $ bold(x)^"cam" = #sym_T _(#fr_cam <- #fr_world) bold(x)^"w" $
  ]
]

These conventions are critical when projecting semidense points into candidate
views and when mapping EVL voxel coordinates back to the rig frame.

== Camera model

EFM3D provides a batched fisheye camera model that supports projection,
unprojection, and valid-radius masking. We rely on these utilities when
rendering depth maps, unprojecting valid pixels, and computing candidate
frustum diagnostics. For view conditioning, we project semidense points into
candidate views using a consistent screen-space camera model aligned with the
renderer. Using calibrated camera objects avoids mismatches between
camera-specific parameters and the global frame.
