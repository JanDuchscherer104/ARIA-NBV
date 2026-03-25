Task: Verify GPT5-Pro review about candidate rendering pose conventions (2025-11-30).

Findings:
- CandidateSamplingResult currently stores `T_camera_rig` as camera<-reference (`candidate_generation.py:587-590`, docstring in `pose_generation/types.py:84-86`).
- CandidateDepthRenderer assumes `T_camera_rig` is reference<-camera and composes `poses_world_cam = reference_pose @ T_camera_rig` without inversion (`rendering/candidate_depth_renderer.py:182-191`).
- This mismatch yields incorrect world<-camera poses in the renderer and explains the misaligned frusta/depth noted in the review.
- Meshes stay in world coordinates and PyTorch3D extrinsics are built via `poses.inverse()` in `rendering/pytorch3d_depth_renderer.py`, so no global axis flip is needed; the bug is the pose-direction swap above.

Suggested fix:
- Pick one convention (recommend aligning to the docstring: camera<-reference) and update the other side. Minimal change: restore inversion in `_select_candidates` or, alternatively, emit reference<-camera in `_finalise` and adjust docs to match.

Tests: Not run (analysis only).
