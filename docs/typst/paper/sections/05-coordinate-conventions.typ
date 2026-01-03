= Coordinate Conventions and Geometry

#import "/typst/shared/macros.typ": *

We follow the EFM3D/ATEK coordinate conventions throughout the pipeline. The
world frame is gravity-aligned, the rig frame moves with the headset, and each
camera frame is expressed in left-up-forward (LUF) coordinates. All poses are
represented as SE(3) transforms using `PoseTW`, and cameras are represented
with `CameraTW` to preserve intrinsics and extrinsics.

Implementation note: for UI alignment in our diagnostics dashboard, the
candidate generator applies a fixed 90° rotation about the local +Z axis
(`rotate_yaw_cw90`) to the reference and candidate poses. Since EVL backbone
outputs follow the canonical EFM3D conventions, `VinModelV2` undoes this
rotation (`apply_cw90_correction=true`) before computing pose encodings and
view-conditioned features.

== Transformation notation

We write $T_{A<-B}$ for the transform from frame $B$ to frame $A$. Composing
poses follows standard SE(3) multiplication, and inversion yields the reverse
transform. For example, a world point $x$ can be expressed in the camera frame
via

#block[
  #align(center)[
    $ bold(x)_{"cam"} = (#sym_T)_{#fr_cam <- #fr_world} bold(x)_{"world"} $
  ]
]

These conventions are critical when projecting semidense points into candidate
views and when mapping EVL voxel coordinates back to the rig frame.

== Camera model

EFM3D provides a batched fisheye camera model that supports projection,
unprojection, and valid-radius masking. We rely on these utilities when
rendering depth maps, unprojecting valid pixels, and computing candidate
frustum diagnostics. For VIN view conditioning, we project semidense points
into candidate views using PyTorch3D `PerspectiveCameras` (screen-space
projection) built from the same intrinsics and extrinsics. Using typed camera
objects avoids mismatches between camera-specific parameters and the global
frame.
