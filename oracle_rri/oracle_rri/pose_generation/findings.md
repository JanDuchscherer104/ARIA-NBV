
## 1. What we learned / what we now want

From the voice conversation and the current code, the key implications are:

1. **Forward-bias bug is on the *analysis* side, not in the generator.**

   * The generator already samples directions in the **rig (camera) frame**, maps them to world via `dirs_world = dirs_rig @ Rᵀ`, and filters by elevation/azimuth in the *world* frame.
   * The misalignment you saw (forward vs world‑Y) came from plotting code that:

     * Mixed rotated and unrotated frames.
     * Used the wrong transform (rig→world again instead of world→rig).
     * Fed `N×12` flattened poses into functions expecting `N×3` offsets.

2. **We want a clean separation: “where do I stand?” vs “which way do I look?”**

   Conceptually:

   * **Stage 1 (position shell):** sample directions + radii around `last_pose` in a controlled band:

     * `min_radius`, `max_radius`, `min_elev_deg`, `max_elev_deg`, `delta_azimuth_deg`, `sampling_strategy` (uniform vs forward PowerSpherical) are all about **positions**.
   * **Stage 2 (orientation):** given candidate centres, decide **camera rotation** separately.

3. **Orientation requirements (from the voice discussion):**

   * We want **view orientation to be configurable**, with multiple modes:

     * Keep orientation roughly like the last pose (forward in rig frame).
     * Optionally radial “look away / look towards last pose”.
     * Possibly look at a fixed target point.
   * We want **roll/pitch/yaw bounds** as *relative deltas*:

     * **Zero angles (Δroll=Δpitch=Δyaw=0) must yield *exactly* the same rotation matrix as the last pose** (at least for the “forward‑rig” mode).
     * We can bound roll (and maybe pitch) tightly so the x‑axis stays “horizontal”, but still allow small random jitter as data augmentation.
   * Existing **azimuth / elevation bounds remain for *position***, not for orientation, but we may introduce a similar parametrisation or simply use yaw/pitch/roll bounds for the view cone.

4. **Diagnostics / plotting expectations:**

   * Position shell is controlled by world‑frame az/elev; we want to *see* that via:

     * The mesh + sampling band.
     * Position distributions in the **rig frame** (offsets).
   * View directions should be visualized in a way that’s clearly **rig‑frame directions relative to the last pose**, not world‑Y.

That’s the spec.

---

## 2. Revised design for candidate view generation

### 2.1 Config changes: make orientation explicit

Extend your types with a **view‑direction mode** enum and jitter parameters.

**`types.py`** – add:

```py
class ViewDirectionMode(StrEnum):
    """How to derive the *base* camera orientation for candidates."""

    FORWARD_RIG = "forward_rig"      # same orientation as last pose (plus jitter)
    RADIAL_AWAY = "radial_away"      # look away from last pose along centre-last vector
    RADIAL_TOWARDS = "radial_towards"  # look at last pose
    TARGET_POINT = "target_point"    # look at a given world-space point
```

You can add this next to `SamplingStrategy`.

**`candidate_generation.py` – `CandidateViewGeneratorConfig`**: extend with orientation fields:

```py
from .types import SamplingStrategy, CollisionBackend, ViewDirectionMode

class CandidateViewGeneratorConfig(BaseConfig["CandidateViewGenerator"]):
    ...

    # --- Orientation / view-direction ---
    view_mode: ViewDirectionMode = ViewDirectionMode.FORWARD_RIG
    """Strategy for constructing camera orientations."""

    view_yaw_jitter_deg: float = 0.0
    """Symmetric yaw jitter around base orientation (deg)."""

    view_pitch_jitter_deg: float = 0.0
    """Symmetric pitch jitter around base orientation (deg)."""

    view_roll_jitter_deg: float = 0.0
    """Symmetric roll jitter around base orientation (deg)."""

    view_target_point_world: torch.Tensor | None = None
    """Optional world-space target point for TARGET_POINT mode (3,)."""
```

Interpretation:

* `view_mode = FORWARD_RIG` → **baseline orientation** is exactly `last_pose.R` (for every candidate).
* `view_*_jitter_deg` define **local perturbations** around that baseline.
* Zero jitter → orientations of all candidates == last pose rotation (**your key requirement**).

You can later still use `RADIAL_AWAY` / `RADIAL_TOWARDS` for “look along shell radius” behaviour.

Optional but recommended: enrich `CandidateSamplingResult` so downstream code can see positional info directly:

```py
@dataclass
class CandidateSamplingResult:
    poses: PoseTW
    mask_valid: torch.Tensor
    masks: dict[str, torch.Tensor]
    shell_poses: PoseTW          # pre-pruning shell orientations
    centers_world: torch.Tensor  # (N,3)
    shell_offsets_rig: torch.Tensor  # (N,3) in rig frame
    extras: dict[str, Any] = field(default_factory=dict)
```

And pass those from `_finalise`.

---

### 2.2 Stage 1: position shell (unchanged in spirit)

Refactor `_sample_shell_poses` into a position-only helper, e.g. `_sample_shell`:

```py
def _sample_shell(
    last_pose: PoseTW,
    cfg: CandidateViewGeneratorConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample directions+radii around last_pose.

    Returns:
        dirs_world_all: (N,3) world-frame directions from last_pose
        centers_world: (N,3) candidate centres
        offsets_rig_all: (N,3) sampled directions in rig frame
    """
    device = torch.device(cfg.device)
    sampler = _DIRECTION_SAMPLERS[cfg.sampling_strategy]

    last_pose_dev = last_pose.to(device)
    r_wc = last_pose_dev.R
    if r_wc.ndim == 3:
        r_wc = r_wc[0]
    t_last = last_pose_dev.t
    fwd_world = _forward_world(last_pose_dev)
    if fwd_world.ndim > 1:
        fwd_world = fwd_world[0]

    dirs_world_list: list[torch.Tensor] = []
    offsets_rig_list: list[torch.Tensor] = []
    remaining = cfg.num_samples
    rounds = 0

    while remaining > 0 and rounds < cfg.max_resamples:
        rounds += 1
        n_draw = ceil(cfg.oversample_factor * remaining)

        # Sample in rig frame
        dirs_rig = sampler.sample(cfg, n_draw, device=device)          # (n_draw,3)
        # Map to world
        dirs_world = dirs_rig @ r_wc.transpose(-1, -2)                 # rig→world

        # Filter by world-frame elev/az bounds
        mask = _filter_directions_world(dirs_world, fwd_world, cfg)

        if mask.any():
            dirs_world_list.append(dirs_world[mask])
            offsets_rig_list.append(dirs_rig[mask])
            remaining = cfg.num_samples - sum(d.shape[0] for d in dirs_world_list)

    if not dirs_world_list:
        raise RuntimeError("Directional sampling failed; relax elevation/azimuth constraints.")

    dirs_world_all = torch.cat(dirs_world_list, dim=0)[: cfg.num_samples]
    offsets_rig_all = torch.cat(offsets_rig_list, dim=0)[: cfg.num_samples]

    # Radii: uniform shell
    radii = torch.empty(
        dirs_world_all.shape[0], device=device, dtype=dirs_world_all.dtype
    ).uniform_(cfg.min_radius, cfg.max_radius)
    centers_world = t_last.view(1, 3) + radii[:, None] * dirs_world_all

    return dirs_world_all, centers_world, offsets_rig_all
```

This keeps the **positional** behaviour identical to your current implementation: world‑frame directional band, forward‑biased `PowerSpherical` if requested, uniform radii in `[min_radius, max_radius]`.

---

### 2.3 Stage 2: orientation builder with view modes + yaw/pitch/roll jitter

Add a helper that turns centres + config into `PoseTW`:

```py
def _build_view_poses(
    last_pose: PoseTW,
    centers_world: torch.Tensor,
    cfg: CandidateViewGeneratorConfig,
) -> PoseTW:
    """Construct candidate camera poses given centres + orientation config."""

    device = centers_world.device
    last_pose_dev = last_pose.to(device)
    n = centers_world.shape[0]

    # --- Base rotation matrix for each candidate ---
    if cfg.view_mode is ViewDirectionMode.FORWARD_RIG:
        # All candidates start with the same orientation as the last pose:
        r_base = last_pose_dev.R
        if r_base.ndim == 2:
            r_base = r_base.unsqueeze(0)
        r_base = r_base.expand(n, 3, 3).contiguous()

    elif cfg.view_mode in (ViewDirectionMode.RADIAL_AWAY, ViewDirectionMode.RADIAL_TOWARDS):
        # Reuse your existing view_axes_from_poses logic for radial modes.
        t_centers = torch.zeros(n, 3, 4, device=device, dtype=centers_world.dtype)
        t_centers[..., :3, :3] = torch.eye(3, device=device, dtype=centers_world.dtype)
        t_centers[..., :3, 3] = centers_world
        centers_pose = PoseTW.from_matrix3x4(t_centers)

        shell_poses = view_axes_from_poses(
            from_pose=last_pose_dev,
            to_pose=centers_pose,
            look_away=(cfg.view_mode is ViewDirectionMode.RADIAL_AWAY),
        )
        r_base = shell_poses.R

    elif cfg.view_mode is ViewDirectionMode.TARGET_POINT and cfg.view_target_point_world is not None:
        # Look-at a fixed point from each centre, zero roll using world up.
        target = cfg.view_target_point_world.to(device=device, dtype=centers_world.dtype).view(1, 3)
        wup = world_up_tensor(device=device, dtype=centers_world.dtype)  # (3,)
        v = (target - centers_world)  # candidate->target
        fwd = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

        # Build camera axes: z=fwd, x=normalized(left), y=up
        # Ensure roll ~0 by projecting world-up into plane orthogonal to fwd.
        dot_up = (fwd * wup).sum(dim=-1, keepdim=True)
        up_proj = wup.view(1, 3) - dot_up * fwd
        up = up_proj / (up_proj.norm(dim=-1, keepdim=True) + 1e-8)
        left = torch.cross(up, fwd, dim=-1)

        r_base = torch.stack([left, up, fwd], dim=-1)  # (N,3,3) world←cam

    else:
        raise ValueError(f"Unsupported view_mode: {cfg.view_mode}")

    # --- Local yaw/pitch/roll jitter around r_base ---
    yaw_range = torch.deg2rad(torch.tensor(cfg.view_yaw_jitter_deg, device=device, dtype=centers_world.dtype))
    pitch_range = torch.deg2rad(torch.tensor(cfg.view_pitch_jitter_deg, device=device, dtype=centers_world.dtype))
    roll_range = torch.deg2rad(torch.tensor(cfg.view_roll_jitter_deg, device=device, dtype=centers_world.dtype))

    if (yaw_range == 0) and (pitch_range == 0) and (roll_range == 0):
        r_final = r_base
    else:
        # Sample symmetric jitters in [-range, +range]
        yaw = (torch.rand(n, device=device) * 2.0 - 1.0) * yaw_range
        pitch = (torch.rand(n, device=device) * 2.0 - 1.0) * pitch_range
        roll = (torch.rand(n, device=device) * 2.0 - 1.0) * roll_range

        cy, sy = torch.cos(yaw), torch.sin(yaw)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cr, sr = torch.cos(roll), torch.sin(roll)

        # Define local rotations in camera (LUF) frame:
        # yaw around +Y, pitch around +X, roll around +Z.
        # R_local maps *new* camera coords to old camera coords (intrinsic).
        R_yaw = torch.zeros(n, 3, 3, device=device, dtype=centers_world.dtype)
        R_pitch = torch.zeros_like(R_yaw)
        R_roll = torch.zeros_like(R_yaw)

        # Yaw (around Y)
        R_yaw[:, 0, 0] = cy
        R_yaw[:, 0, 2] = sy
        R_yaw[:, 1, 1] = 1.0
        R_yaw[:, 2, 0] = -sy
        R_yaw[:, 2, 2] = cy

        # Pitch (around X)
        R_pitch[:, 0, 0] = 1.0
        R_pitch[:, 1, 1] = cp
        R_pitch[:, 1, 2] = -sp
        R_pitch[:, 2, 1] = sp
        R_pitch[:, 2, 2] = cp

        # Roll (around Z)
        R_roll[:, 0, 0] = cr
        R_roll[:, 0, 1] = -sr
        R_roll[:, 1, 0] = sr
        R_roll[:, 1, 1] = cr
        R_roll[:, 2, 2] = 1.0

        # Intrinsic rotations: first yaw, then pitch, then roll in camera frame.
        R_local = R_roll @ R_pitch @ R_yaw     # (N,3,3)

        # For PoseTW.R as world←cam, applying local rotation R_local
        # gives: world = R_base * (R_local⁻¹ * v_cam_new)
        # so world←cam_new = R_base @ R_local.T
        r_final = r_base @ R_local.transpose(-1, -2)

    # Assemble PoseTW with centres_world + r_final
    t = centers_world
    mat = torch.zeros(n, 3, 4, device=device, dtype=centers_world.dtype)
    mat[..., :3, :3] = r_final
    mat[..., :3, 3] = t
    return PoseTW.from_matrix3x4(mat)
```

Key properties:

* **FORWARD_RIG + zero jitter** → `r_final == last_pose.R` for all candidates, exactly what you asked for.
* You can keep roll very small (`view_roll_jitter_deg ≈ 0–3°`) to ensure the x‑axis is practically horizontal; or set it to 0 for strict no‑roll.

You can use `view_mode="radial_away"` or `"radial_towards"` to recover the original “look away/toward” behaviour, now with optional jitter on top, because we reuse `view_axes_from_poses`.

---

### 2.4 Stage 3: generator wiring

Update `CandidateViewGenerator.generate` to use the two‑stage geometry:

```py
class CandidateViewGenerator:
    ...

    def generate(
        self,
        *,
        last_pose: PoseTW,
        gt_mesh: trimesh.Trimesh | None = None,
        mesh_verts: torch.Tensor | None = None,
        mesh_faces: torch.Tensor | None = None,
        occupancy_extent: torch.Tensor | None = None,
        camera_fov: torch.Tensor | None = None,
    ) -> CandidateSamplingResult:
        cfg = self.config
        device = torch.device(cfg.device)

        occ_extent = occupancy_extent if occupancy_extent is not None else cfg.occupancy_extent

        # --- Stage 1: position shell ---
        dirs_world, centers_world, offsets_rig = _sample_shell(last_pose, cfg)

        # --- Stage 2: orientations (view directions) ---
        shell_poses = _build_view_poses(last_pose.to(device), centers_world, cfg)

        # --- Stage 3: pruning rules ---
        ctx = CandidateContext(
            cfg=cfg,
            last_pose=last_pose.to(device),
            gt_mesh=gt_mesh,
            mesh_verts=mesh_verts.to(device) if mesh_verts is not None else None,
            mesh_faces=mesh_faces.to(device) if mesh_faces is not None else None,
            occupancy_extent=occ_extent.to(device) if occ_extent is not None else None,
            camera_fov=camera_fov,
            shell_poses=shell_poses,
            centers_world=centers_world,
            shell_offsets_rig=offsets_rig,
            mask_valid=torch.ones(centers_world.shape[0], dtype=torch.bool, device=device),
        )

        self._apply_rules(ctx)
        return self._finalise(ctx)
```

And `_finalise` can optionally propagate `centers_world` and `shell_offsets_rig` into the result (as suggested earlier).

---

### 2.5 Plotting / diagnostics alignment (quick recap)

To keep everything consistent with the generator and avoid the original bug:

* **Direction distributions** in the Streamlit page should be based on **rig‑frame offsets**:

  ```py
  # from CandidateSamplingResult
  offsets_rig = candidates.shell_offsets_rig  # (N,3) in rig frame

  # For positions:
  plot_position_polar(offsets_rig)
  plot_position_sphere(offsets_rig, show_axes=True)
  ```

* **View direction distributions** relative to the last rig frame can be derived from centres:

  ```py
  centers_world = candidates.centers_world  # (N,3)
  last_center = last_pose_rig.t.view(1, 3)
  offsets_world = centers_world - last_center

  r_wr = last_pose_rig.R
  if r_wr.ndim == 3:
      r_wr = r_wr[0]
  offsets_rig = offsets_world @ r_wr         # world→rig

  dirs_rig = offsets_rig / (offsets_rig.norm(dim=1, keepdim=True) + 1e-8)
  plot_direction_polar(dirs_rig)
  plot_direction_sphere(dirs_rig, show_axes=True)
  ```

This ensures:

* No mixing of rotated/unrotated frames.

* No incorrect use of `Rᵀ` for world→rig (you use `dirs_world @ R`, not `@ Rᵀ`).

* The **forward direction bias is correctly expressed in the last camera frame**, not world‑Y.

---

### 2.6 What you gain from this design

* **Exact baseline:** `Δroll = Δpitch = Δyaw = 0` → candidate orientation == last pose orientation (for `FORWARD_RIG`), as requested.
* **Decoupled control:**

  * Position shell: radius + world‑frame az/elev band + forward bias (already implemented).
  * Orientation: view mode + yaw/pitch/roll jitter around a well-defined base.
* **Backwards compatibility:**

  * `RADIAL_AWAY` with zero jitter reproduces current “look away from last pose along the shell” behaviour.
* **Diagnostics become meaningful:** rig‑frame position/view distributions you plot correspond exactly to what the generator is doing.

From here, you can layer in further extras (stats, adaptive oversampling, rule scores), but the geometry and semantics are now clean and satisfy everything we hammered out in the voice discussion.
