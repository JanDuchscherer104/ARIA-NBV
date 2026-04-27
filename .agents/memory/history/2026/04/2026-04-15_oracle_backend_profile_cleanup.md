---
id: 2026-04-15_oracle_backend_profile_cleanup
date: 2026-04-15
title: "Oracle Backend Profile Cleanup"
status: done
topics: [oracle-rri, backend-profile, mojo, pytorch3d, streamlit]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/memory/state/GOTCHAS.md
---

## Task

Implemented explicit Oracle RRI backend profiles so the default CUDA/PyTorch3D
path stays default-preserving and the Apple MPS/Mojo path is selected only by
profile.

## Method

- Kept the existing backend enum and Mojo kernel groundwork.
- Removed Streamlit app-level PyTorch3D-availability fallback mutation.
- Added a profile resolver that deep-copies configs and applies stage backends
  atomically.
- Kept candidate sampling on a resolver-owned CPU fallback for the Apple
  profile after the Streamlit smoke exposed an MPS crash in that path.
- Added a worker-thread guard for Python-imported Mojo collision kernels after
  Streamlit candidate generation reproduced a segfault in `clearance_mask_mojo`.
- Added subprocess execution for Mojo collision, depth rendering,
  point-cloud backprojection, and oracle-distance scoring when those stages are
  invoked from Streamlit's worker thread.
- Preserved `PerspectiveCameras` compatibility for CUDA/VIN/cache surfaces.
- Kept finite-segment path-collision behavior.

## Verification

Targeted resolver, renderer-alias, and RRI chunking tests were run during the
cleanup. PyTorch3D-specific tests skip on this Apple host because the local
PyTorch3D binary has an ABI import failure.

## Canonical State Impact

`PROJECT_STATE.md` now records that Oracle RRI backend selection is profile
driven, and `GOTCHAS.md` records that local PyTorch3D ABI failures must not
cause silent Mojo default switching.
