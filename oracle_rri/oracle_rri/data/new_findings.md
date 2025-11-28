# Cleanly separate *position* and *orientation* pipelines

The current design mixes “where do I stand?” and “which way do I look?” in
`_sample_shell_poses` by calling `view_axes_from_poses` at the end. :contentReference[oaicite:1]{index=1}

We refactor the generator into **two explicit geometry stages** plus **rule-based pruning**:

1. `_sample_candidate_positions(last_pose, cfg)` → `centers_world, offsets_rig`
2. `_build_candidate_orientations(last_pose, centers_world, cfg)` → `shell_poses`
3. Rule application (unchanged): operates on `centers_world` + `shell_poses` to produce final `poses`.

Each geometry stage has its own config knobs. This gives:

- Ability to reuse the same *positional* candidates with different orientation modes
  (e.g. radial vs forward‑rig vs target‑point) without resampling.
- Easier testing: you can verify elevation/azimuth constraints independently of
  orientation logic.
- A clean extension path to surface‑aware or semantics‑aware position sampling while
  keeping orientation logic modular.

On top of that, we:

- Use the existing rig‑frame `HypersphericalUniform` / `PowerSpherical` for both
  **position directions** *and* **view directions**. :contentReference[oaicite:2]{index=2}
- Use `PoseTW` for coordinate transforms (`rotate`, `compose`, `from_Rt`) rather than raw
  matrix multiplications. :contentReference[oaicite:3]{index=3}

---

## 1. Spec recap with the new requirements

We want:

### 1.1 Positions (Stage 1)

Exactly as before, just factored out:

- Directions sampled in rig frame using `HypersphericalUniform` or
  `PowerSpherical(mu=[0,0,1], kappa)` via the existing `SamplingStrategy`. :contentReference[oaicite:4]{index=4}
- Directions are mapped to world with `last_pose.rotate(dirs_rig)` (using `PoseTW.rotate` rather than
  manual `@ R.T`) and filtered by world‑frame:

  - `min_elev_deg`, `max_elev_deg` relative to world horizontal.
  - `delta_azimuth_deg` around last forward, w.r.t. world‑up.
- Radii drawn uniformly in `[min_radius, max_radius]` to yield `centers_world`.

Output of Stage 1:

- `centers_world: (N,3)` – candidate camera centres in world.
- `offsets_rig: (N,3)` – sampled directions in the **rig** (last camera) frame.

### 1.2 Orientations (Stage 2)

We now have two concepts:

- A **base orientation** per candidate controlled by `ViewDirectionMode`.
- Optional **view‑direction jitter** in the base camera frame using `PowerSpherical`.

Requirements:

- If view jitter is “off”, candidate rotations are **purely determined by**
  `last_pose.R` and `view_direction_mode`:

  - `FORWARD_RIG`: all candidates share the last camera orientation.
  - `RADIAL_AWAY` / `RADIAL_TOWARDS`: the existing roll‑free radial behaviour
    (via `view_axes_from_poses`) is preserved. :contentReference[oaicite:5]{index=5}
  - `TARGET_POINT`: candidates look at a fixed target in world space.
- If view jitter is “on”, for each candidate we:
  - Sample a direction on S² in the **base camera frame** using
    `HypersphericalUniform` or `PowerSpherical(mu=[0,0,1], scale=view_kappa)`.
  - Use that direction as the new forward axis in base‑cam coords.
  - Build an orthonormal camera basis in base‑cam coords, then compose it with
    the base orientation using `PoseTW.compose`.
  - Optionally apply a random roll around the new forward axis
    (`view_roll_jitter_deg`) for augmentation.

That means:

- `PowerSpherical` is used for **position directions** (existing code) and for
  **view directions** (new Stage 2).
- The special case “no jitter” (no view sampling and no roll jitter) yields
  `shell_poses.R ==` the base orientation (e.g. `last_pose.R` for `FORWARD_RIG`).

---

## 2. Config changes

We extend `types.py` and `CandidateViewGeneratorConfig` to support view‑direction modes and view‑direction sampling.

### 2.1 `ViewDirectionMode` (in `types.py`)

```py
class ViewDirectionMode(StrEnum):
    """How to derive the *base* camera orientation for candidates."""

    FORWARD_RIG = "forward_rig"
    # same orientation as last pose (all candidates share last_pose.R)

    RADIAL_AWAY = "radial_away"
    # look away from last pose along center-last vector (equivalent to existing
    # view_axes_from_poses(..., look_away=True) behaviour, optionally with jitter)

    RADIAL_TOWARDS = "radial_towards"
    # look towards last pose along last-center vector (look_away=False), optionally with jitter

    TARGET_POINT = "target_point"
    # look at a given world-space point, keeping roll minimal, optionally with jitter
````

Add import in `candidate_generation.py`:

```py
from .types import SamplingStrategy, CollisionBackend, ViewDirectionMode
```

### 2.2 New fields in `CandidateViewGeneratorConfig`

```py
class CandidateViewGeneratorConfig(BaseConfig["CandidateViewGenerator"]):
    # ... existing fields ...

    # --- View direction base mode ---
    view_direction_mode: ViewDirectionMode = ViewDirectionMode.FORWARD_RIG
    """
    Strategy for constructing the *base* camera orientation before any jitter.
    """

    # --- View direction sampling (in base camera frame) ---
    view_sampling_strategy: SamplingStrategy | None = None
    """
    Distribution for view directions in the base camera frame.

    None → no view-direction sampling: view_dir_cam = [0,0,1] for all candidates
            (purely deterministic base orientation, see `view_direction_mode`).
    Non-None → sample view directions on S² using HypersphericalUniform /
               PowerSpherical, analogous to position direction sampling.
    """

    view_kappa: float | None = None
    """
    Concentration for PowerSpherical view-direction sampler.
    If None, fall back to the position sampler's `kappa`.
    """

    view_max_angle_deg: float = 0.0
    """
    Optional hard cap on the angular deviation (deg) between the
    base forward [0,0,1] in camera frame and the sampled view direction.
    0 → no explicit cap (distribution alone controls spread).
    """

    view_roll_jitter_deg: float = 0.0
    """
    Symmetric roll jitter (deg) around the *new* forward axis in camera frame.

    0 → no roll jitter; x-axis remains aligned as defined by the base orientation.
    """

    # Optional target point for TARGET_POINT mode
    view_target_point_world: torch.Tensor | None = None
    """
    Optional world-space target point (3,) for TARGET_POINT mode.
    """

    @model_validator(mode="after")
    def _propagate_view_defaults(self) -> "CandidateViewGeneratorConfig":
        # Default view_kappa to the position sampler kappa
        if self.view_kappa is None:
            object.__setattr__(self, "view_kappa", self.kappa)

        # Preserve existing behaviour for azimuth_full_circle
        object.__setattr__(self, "azimuth_full_circle", bool(self.delta_azimuth_deg >= 360.0))
        return self
```

Semantics:

* `view_direction_mode` selects the **deterministic base orientation** for each candidate.
* `view_sampling_strategy is None` ⇒ no view‑direction sampling:
  `shell_poses` are entirely determined by `view_direction_mode`.
* If `view_sampling_strategy` is set, view directions are sampled on S² in the **base camera frame**
  using `HypersphericalUniform` / `PowerSpherical(mu=[0,0,1], scale=view_kappa)` and used
  to perturb the base orientation.
* `view_roll_jitter_deg` always applies in the camera frame around the new forward axis (if > 0).

---

## 3. Stage 1: position sampling (`_sample_candidate_positions`)

We refactor the geometry part of `_sample_shell_poses` into a position‑only helper that returns
centres and rig‑frame offsets. The logic (direction sampling, world‑frame filter, radii) stays
the same as current code, but we use `PoseTW.rotate` instead of manual `@ R.T`.

```py
from math import ceil

def _sample_candidate_positions(
    last_pose: PoseTW,
    cfg: CandidateViewGeneratorConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample candidate centres around the last pose.

    Returns:
        centers_world: (N, 3) candidate camera centres in world frame.
        offsets_rig:   (N, 3) sampled direction vectors in rig (last cam) frame.
    """
    device = torch.device(cfg.device)
    sampler = _DIRECTION_SAMPLERS[cfg.sampling_strategy]

    last_pose_dev = last_pose.to(device)
    t_last = last_pose_dev.t               # (3,)
    fwd_world = _forward_world(last_pose_dev)  # uses PoseTW.rotate internally

    dirs_world_list: list[torch.Tensor] = []
    offsets_rig_list: list[torch.Tensor] = []
    remaining = cfg.num_samples
    rounds = 0

    while remaining > 0 and rounds < cfg.max_resamples:
        rounds += 1
        n_draw = ceil(cfg.oversample_factor * remaining)

        # rig-frame directions (HypersphericalUniform or PowerSpherical)
        dirs_rig = sampler.sample(cfg, n_draw, device=device)  # (n_draw, 3)

        # map to world using PoseTW.rotate instead of manual R^T
        dirs_world = last_pose_dev.rotate(dirs_rig)            # (n_draw, 3)

        # world-frame elevation/azimuth filter
        mask = _filter_directions_world(dirs_world, fwd_world, cfg)

        if mask.any():
            dirs_world_list.append(dirs_world[mask])
            offsets_rig_list.append(dirs_rig[mask])
            remaining = cfg.num_samples - sum(d.shape[0] for d in dirs_world_list)

    if not dirs_world_list:
        raise RuntimeError("Directional sampling failed; relax elevation/azimuth constraints.")

    dirs_world_all = torch.cat(dirs_world_list, dim=0)[: cfg.num_samples]
    offsets_rig_all = torch.cat(offsets_rig_list, dim=0)[: cfg.num_samples]

    # radii on [min_radius, max_radius]
    radii = torch.empty(
        dirs_world_all.shape[0],
        device=device,
        dtype=dirs_world_all.dtype,
    ).uniform_(cfg.min_radius, cfg.max_radius)

    centers_world = t_last.view(1, 3) + radii[:, None] * dirs_world_all
    return centers_world, offsets_rig_all
```

---

## 4. Stage 2: orientations using `ViewDirectionMode` + `PowerSpherical`

### 4.1 Sampling view directions in the base camera frame

We reuse `HypersphericalUniform` / `PowerSpherical`, now in the **base camera frame**:

```py
from power_spherical import HypersphericalUniform, PowerSpherical

def _sample_view_dirs_cam(
    cfg: CandidateViewGeneratorConfig,
    num: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample view directions on S² in the base camera frame."""

    strat = cfg.view_sampling_strategy
    if strat is None:
        # deterministic: forward only (no view-direction jitter)
        v = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        return v.view(1, 3).expand(num, 3)

    if strat == SamplingStrategy.SHELL_UNIFORM:
        dist = HypersphericalUniform(dim=3, device=device, dtype=dtype)
    elif strat == SamplingStrategy.FORWARD_GAUSSIAN:
        mu = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        scale = torch.tensor(cfg.view_kappa, device=device, dtype=dtype)
        dist = PowerSpherical(mu, scale)
    else:
        raise ValueError(f"Unsupported view_sampling_strategy: {strat}")

    dirs = dist.rsample((num,))
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    # Optional hard cone: clamp max angular deviation from [0,0,1]
    if cfg.view_max_angle_deg > 0.0:
        mu = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        cos_max = math.cos(math.radians(cfg.view_max_angle_deg))

        mask = (dirs * mu).sum(dim=-1) < cos_max
        max_tries = 10
        tries = 0
        while mask.any() and tries < max_tries:
            tries += 1
            resample_n = int(mask.sum().item())
            new_dirs = dist.rsample((resample_n,))
            new_dirs = new_dirs / new_dirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            dirs[mask] = new_dirs
            mask = (dirs * mu).sum(dim=-1) < cos_max

    return dirs
```

### 4.2 Constructing orientations via `ViewDirectionMode` and `PoseTW`

We now build `shell_poses` (pre‑pruning candidates) from `centers_world` using:

* `ViewDirectionMode` to define a base orientation.
* `_sample_view_dirs_cam` to jitter view directions in base‑cam frame.
* `PoseTW.compose` to apply the rotational jitter without manually composing `R` matrices.

```py
from ..utils.frames import view_axes_from_poses, world_up_tensor

def _build_candidate_orientations(
    last_pose: PoseTW,
    centers_world: torch.Tensor,
    cfg: CandidateViewGeneratorConfig,
) -> PoseTW:
    """Construct candidate orientations using ViewDirectionMode + PowerSpherical view directions."""

    device = centers_world.device
    dtype = centers_world.dtype
    N = centers_world.shape[0]

    last_pose_dev = last_pose.to(device)

    # --- 1) Base orientations per ViewDirectionMode ---

    if cfg.view_direction_mode is ViewDirectionMode.FORWARD_RIG:
        # Same orientation as last pose, just translated to the candidate centre.
        R_last = last_pose_dev.R
        if R_last.ndim == 3:
            R_last = R_last[0]
        R_base = R_last.unsqueeze(0).expand(N, 3, 3)
        base_poses = PoseTW.from_Rt(R_base, centers_world)

    elif cfg.view_direction_mode in (ViewDirectionMode.RADIAL_AWAY, ViewDirectionMode.RADIAL_TOWARDS):
        # Reuse existing roll-free radial behaviour.
        eye = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)
        centers_pose = PoseTW.from_Rt(eye, centers_world)
        shell_like = view_axes_from_poses(
            from_pose=last_pose_dev,
            to_pose=centers_pose,
            look_away=(cfg.view_direction_mode is ViewDirectionMode.RADIAL_AWAY),
        )
        base_poses = shell_like

    elif cfg.view_direction_mode is ViewDirectionMode.TARGET_POINT:
        if cfg.view_target_point_world is None:
            raise ValueError("TARGET_POINT mode requires `view_target_point_world` to be set.")
        target = cfg.view_target_point_world.to(device=device, dtype=dtype).view(1, 3)
        wup = world_up_tensor(device=device, dtype=dtype)  # (3,)

        # World-frame forward = from centre to target
        v = (target - centers_world)  # (N, 3)
        z_world = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        # Build camera basis with minimal roll using world up.
        dot_up = (z_world * wup.view(1, 3)).sum(dim=-1, keepdim=True)
        y_world = wup.view(1, 3) - dot_up * z_world
        y_world = y_world / y_world.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        x_world = torch.cross(y_world, z_world, dim=-1)

        R_base = torch.stack([x_world, y_world, z_world], dim=-1)
        base_poses = PoseTW.from_Rt(R_base, centers_world)

    else:
        raise ValueError(f"Unsupported view_direction_mode: {cfg.view_direction_mode}")

    # Short-circuit: pure deterministic base orientation with no jitter
    if cfg.view_sampling_strategy is None and cfg.view_roll_jitter_deg == 0.0:
        return base_poses

    # --- 2) View-direction jitter in base camera frame ---

    dirs_cam = _sample_view_dirs_cam(cfg, N, device=device, dtype=dtype)  # (N,3)
    z_new = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    # old camera "up" in its own coords is (0,1,0)
    up_cam = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).view(1, 3).expand_as(z_new)

    # x_new = normalize(up_cam × z_new), y_new = normalize(z_new × x_new)
    x_new = torch.cross(up_cam, z_new, dim=-1)
    x_new = x_new / x_new.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    y_new = torch.cross(z_new, x_new, dim=-1)
    y_new = y_new / y_new.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    # base_cam ← new_cam: columns are new basis vectors expressed in base-cam coords
    R_delta = torch.stack([x_new, y_new, z_new], dim=-1)  # (N, 3, 3)

    # --- 3) Optional roll jitter around new forward axis ---

    if cfg.view_roll_jitter_deg > 0.0:
        roll = (
            2.0 * torch.rand(N, device=device, dtype=dtype) - 1.0
        ) * math.radians(cfg.view_roll_jitter_deg)
        cr, sr = torch.cos(roll), torch.sin(roll)

        R_roll = torch.zeros(N, 3, 3, device=device, dtype=dtype)
        R_roll[:, 0, 0] = cr
        R_roll[:, 0, 1] = -sr
        R_roll[:, 1, 0] = sr
        R_roll[:, 1, 1] = cr
        R_roll[:, 2, 2] = 1.0

        # apply roll in new camera frame: base←new_roll = base←new @ (new←new_roll)
        R_delta = torch.matmul(R_delta, R_roll)

    # --- 4) Compose base poses with rotational deltas using PoseTW.compose ---

    delta_poses = PoseTW.from_Rt(R_delta, torch.zeros_like(centers_world))
    shell_poses = base_poses.compose(delta_poses)
    # base_poses.t + base_poses.R @ 0 = centers_world → translations preserved

    return shell_poses
```

Key invariants:

* `view_sampling_strategy is None` and `view_roll_jitter_deg == 0` ⇒ `shell_poses == base_poses` (exact).
* For `ViewDirectionMode.FORWARD_RIG`, `base_poses.R == last_pose.R`; for others, behaviour matches existing radial `view_axes_from_poses` or a TARGET_POINT look‑at.
* View jitter always lives in the base camera frame; `PowerSpherical` is camera‑centric.

---

## 5. Stage 3: generator wiring

Finally, `CandidateViewGenerator.generate` uses the two geometry stages plus the existing pruning rules. We keep `CandidateContext.shell_poses` as the pre‑pruning pose set for compatibility.

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
        """Sample candidate poses around `last_pose` and apply pruning rules."""

        cfg = self.config
        device = torch.device(cfg.device)

        occ_extent = occupancy_extent if occupancy_extent is not None else cfg.occupancy_extent

        # Stage 1: positions
        centers_world, offsets_rig = _sample_candidate_positions(last_pose, cfg)

        # Stage 2: orientations
        shell_poses = _build_candidate_orientations(last_pose, centers_world, cfg)

        # Stage 3: pruning (unchanged)
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

`_apply_rules` and `_finalise` remain as in the current implementation: rules see world‑space centres and pre‑pruning `shell_poses`, and `CandidateSamplingResult.poses` contains only those candidates that survive rule filtering.
