Pose generation tests update (2025-11-30)
---------------------------------------

What changed
- Rebuilt pose_generation tests to target individual components: PositionSampler shape/radius bounds, OrientationBuilder forward-rig orientation, MinDistance/PathCollision/FreeSpace rules masking, and a light-weight generator integration on a synthetic cube mesh.
- Removed legacy ASE-dependent test to keep the suite self-contained and fast.

Notes
- PathCollisionRule test uses P3D backend; relies on PyTorch3D availability in the env.
- Camera template is passed as None; generator handles default template via helper.
