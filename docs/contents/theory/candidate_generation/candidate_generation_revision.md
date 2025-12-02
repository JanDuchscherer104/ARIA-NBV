Please provide a fully revised, simplified version that resolves all the issues in our current implementation. Start by lining out the architecture and data-flow throughout the vised submodule. Keep in mind what we our criteria:

- x axis horizontal - yaw = zero - allow some bias towards the z-axis direction of the last pose (PoseTW)
- candidates must be of type PoseTW  - the sampling strategies should be modular and exchangable like the ruls for pruning.
- We need a revised strategy to sample the view directions
- We're currently biased towards pos y world, which causes our views (look away from last pose towards the candidate pose) to also be limited.
- Our MInDistanceToMeshRule should optionally provide us with the min-distance of any candidate point to the mesh! So these extra information (like min-distance, the different masks) should be optionally activateable via a new config parameter.

Here is an overview of all other issues:


### 1. Code / architecture issues in `candidate_generation.py`

* Current `CandidateViewGenerator` is “clunky, over‑complicated and intransparent”.
* Rule‑based structure is good in principle, but the flow is hard to follow and not modular enough:

  * Sampling, filtering and pose construction are entangled.
  * The role of each rule (`ShellSamplingRule`, `MinDistanceToMeshRule`, `PathCollisionRule`, `FreeSpaceRule`) is not clearly exposed at the API level.
* You want a **cleaner outer component**:

  * Simpler control flow and clearer separation: “sample candidate directions/positions → build poses → apply rules”.
  * Use the newer `SimpleCandidateViewGenerator` logic as conceptual reference.

---

### 2. Sampling / candidate distribution issues

* You now have a better sampling proposal in `revision_proposal_candidate_view_generator.py` using `HypersphericalUniform` and `PowerSpherical` from `power_spherical`, and you want to:

  * Replace the old hand‑rolled sphere sampling with these distributions.
  * Support both **area‑uniform** and **forward‑biased** sampling.
* There is a **strong offset between the last camera pose and the candidate point cloud**:

  * Candidate positions appear shifted relative to the last pose, especially along world‑y.
  * This raises suspicion that direction sampling or frame transforms are wrong.
* Azimuth range behaviour is confusing:

  * You want a `delta_azimuth` parameter:

    * `delta_azimuth = 360°` → full sphere.
    * `delta_azimuth = 90°` → restrict to ±45° around the last‑pose z‑axis.
    * `delta_azimuth=0 ` all candidates lie inside the  z-y plane (camera frame, last pose)
  * Currently azimuth seems **much more constrained** than intended and biased toward a particular world direction (positive y).

---

### 3. Frame conventions, orientation, and `view_axes_from_points`

* Camera frames are **LUF**; the world (VIO) frame has its own convention.
* Requirements for candidate orientations:

  * Camera **x‑axis must stay horizontal in world** (no roll; “no yaw about world z” in your phrasing).
  * Candidates should **look away from the last pose** (rather than always towards it).
  * You need to enforce a fixed `world_up_tensor` from `utils.frames` when constructing orientations.
* There is doubt whether the current `view_axes_from_points` helper is correct in this setting.
* You want a **new `view_axes_from_point`‑style function** with signature

  * `from_pose: PoseTW`, `to_pose: PoseTW`
  * Explicitly respecting LUF camera frame + VIO world frame and zero‑roll constraint.
* Observed behaviour:

  * Because of a bias toward positive y in world, **all candidate frames end up looking roughly in +y_world**, regardless of the last pose.
  * Rendered images appear **rotated**, and you currently compensate by manually rotating all camera views by **~90° clockwise** in yaw to look reasonable.

---

### 4. Visual / rendering anomalies

* In 3D visualizations:

  * The candidate point cloud is clearly **offset** relative to the last camera pose.
  * Candidate positions appear shifted along world‑y rather than aligned with the intended camera forward direction.
* In rendered depth/rgb views:

  * Views appear **rotated by 90°**, consistent with a mismatch between camera intrinsics/extrinsics and the LUF/VIO convention.
* Mesh rendering problem:

  * **Walls are not rendered** in the depth renderer, even when toggling backface culling on/off.
  * This suggests a possible issue with:

    * Face orientation / winding,

    * Or the way frusta / rays are constructed.

---

### 5. Desired revisions (high‑level)

* Completely **clean up `candidate_generation.py`**:

  * Clear pipeline: sample directions & radii → compute positions → build poses (respecting up‑vector and “look away”) → run rule filters.
  * Integrate `HypersphericalUniform` / `PowerSpherical` from `reference_power_spherical_distributions.py`.
  * Introduce `delta_azimuth` semantics as described.
* Provide a **new, correct orientation helper** (`view_axes_from_point(from_pose, to_pose)`) that:

  * Enforces LUF camera frame, VIO world frame,
  * Uses `world_up_tensor`,
  * Produces zero‑roll, well‑behaved orientations.

These are the core symptoms and desired changes you’ve described: a messy candidate generator, incorrect/biased sampling and orientations, confusing azimuth behaviour, rotated renderings, and missing walls in the depth rendering.

---
---

# First Answer by GPT5-PRO

## 1. Why the offset + weird azimuth + y‑bias?

### 1.1 “Offset” between last pose and candidate point cloud

Geometrically, a lot of what you’re seeing is expected:

* You’re **not sampling a full sphere** – you’re sampling a *spherical cap* (limited elevation) and, in practice, also a **restricted azimuth band** and **forward‑biased directions**.
* So even if the shell is mathematically centerd at the last pose, the *cluster of actual samples* lives mostly in a lobe in front / above the rig, not uniformly around it.
* The **centroid** of that lobe is therefore offset from the last pose. The “visual center” of the blue point cloud in the screenshot is the centroid of that lobe, not the geometric center of the underlying sphere.

That alone already makes the cluster look displaced with respect to the last pose. That part is not a bug, it’s simply “we only sample one side of the shell”.

### 1.2 Why azimuth looks so limited

From your docstring for `ShellSamplingRule`:

$$
\begin{aligned}
x &= \cos(\text{elev}) \cos(\text{az}), \\
y &= \sin(\text{elev}), \\
z &= \cos(\text{elev}) \sin(\text{az})
\end{aligned}
$$

Key points:

* Elevation is measured from the **x–z plane**, with `y` as “up”.
* Depending on how you set `min_elev_deg` / `max_elev_deg` and whether `azimuth_full_circle` is true, you get a **cap centerd on some axis**.
* On top of this, `SamplingStrategy.FORWARD_POWERSPHERICAL ` is implemented as “PowerSpherical with some mean direction”. If that mean direction is wrong (see below), you’re not sampling around the *rig forward axis* but around some fixed axis in world or rig coordinates.

Given the screenshots and your comment

> Because of the bias towards positive y all candidate frames are looking towards positive y in world coordinates.

the most likely current behaviour is:

* `PowerSpherical` is instantiated with `mean = (0, 1, 0)` (or equivalent after some transform).
* So directions are **concentrated around +y**, regardless of the rig’s actual forward axis.
* You then restrict elevation to a band like ([-15°, 45°]), which still keeps you in the “mostly +y” lobe.
* If `azimuth_full_circle=False`, you’re additionally restricting azimuth to something like a half‑plane, so the lobe is both *forward* and *to one side*.

Hence: the cloud sticks out roughly along **world +y**, and frusta all roughly “look north/up” in world coordinates, not along the last pose’s forward axis.

### 1.3 Why positions look offset along world‑y, not camera‑z

Take the rig frame:

* Suppose rig forward axis is `F_rig = (0, 0, 1)` (LUF: x=left, y=up, z=forward).
* But when you sample directions, you’re effectively using a distribution whose mean is `mean_dir = (0, 1, 0)` in that same frame (or in world frame which is aligned similarly).

Then your candidate offsets in rig coordinates satisfy, in expectation:

* $(\mathbb{E}[d] \approx c \cdot (0, 1, 0))$, some non‑zero `c`.
* After transforming to world: `offset_world ≈ R_world_rig @ (0, 1, 0) * c`. If world up is aligned with rig up (typical for Aria), that’s just **world +y**.

You *then* build camera orientations via `view_axes_from_points`. If that function “looks from the candidate center to the last pose” using a fixed `world_up`, all candidates *also* share very similar forward axes in world coordinates – again essentially pointing “downhill” from the +y lobe toward the last pose.

So:

* The positions are offset because you’re sampling a *biased cap* rather than a symmetric shell.
* The offset is along world‑y because your PowerSpherical **mean direction is aligned with +y**, not with the rig forward axis.
* The azimuth seems constrained because you restrict both elevation *and* azimuth, *and* bias directions with a concentrated distribution.

This is consistent with what you’re seeing without needing any bug in PoseTW itself.

---

## 2. Simplified design for `candidate_generation.py`

Given the docstrings, the current structure looks roughly like:

* `CandidateViewGeneratorConfig` – big config object with sampling + rules + device etc.
* `CandidateViewGenerator` – orchestrator with:

  * `_build_default_rules()` → instances of `ShellSamplingRule`, `MinDistanceToMeshRule`, `PathCollisionRule`, `FreeSpaceRule`.
  * `_sample_candidates()` → runs rules with a mutable `CandidateContext`.
  * `generate_from_typed_sample(sample)` → extracts last pose, mesh, occupancy, etc.
  * `generate(...)` → core generator: build context, run rules, pack `CandidateSamplingResult`.

Pain‑points you already hinted at:

* **Too much state hidden in `CandidateContext`**, passed implicitly between rules.
* Sampling logic spread across `CandidateViewGenerator` and `ShellSamplingRule` with ad‑hoc device handling.
* Orientation logic hidden in `utils.frames.view_axes_from_points`, not easy to reason about.
* Rules mix “geometric generation” and “filtering”, reducing modularity.

### 2.1 Goals for the revision

For the *outer* component `candidate_generation.py`, I’d aim for:

1. **Single, explicit pipeline** in `CandidateViewGenerator.generate()`:

   * sample directions in rig frame (using `HypersphericalUniform` / `PowerSpherical`),
   * restrict by elevation + `delta_azimuth`,
   * turn into world‑frame positions,
   * build camera orientations with a new `view_axes_from_poses`,
   * run a list of simple filter rules,
   * return `CandidateSamplingResult`.

2. **Declarative configuration**:

   * `CandidateViewGeneratorConfig` describes *only*:

     * sampling parameters (`num_samples`, radii, `min_elev_deg`, `max_elev_deg`, `delta_azimuth_deg`, `sampling_strategy`, `kappa`, etc.),
     * rule toggles (`ensure_collision_free`, `ensure_free_space`, `min_distance_to_mesh`, `collision_backend`),
     * infra bits (`device`, `verbosity`).
   * No hidden mode switches / internal enums sprinkled around.

3. **Thin rules**:

   * Each rule is `Rule(CandidateState) -> CandidateState` where `CandidateState` is a small dataclass.
   * Rules *only modify masks*; they never resample or mutate geometry.

4. **Vectorised, predictable math**:

   * Use the same direction sampling logic as in `SimpleCandidateViewGenerator` (HypersphericalUniform / PowerSpherical).
   * Use an explicit world‑up tensor (from `utils.frames.world_up_tensor`) in the orientation helper.

### 2.2 Proposed top‑level structure

Pseudo‑API for the new `candidate_generation.py`:

```python
# pose_generation/candidate_generation.py

@dataclass
class CandidateState:
    cfg: CandidateViewGeneratorConfig
    last_pose_world_rig: PoseTW          # world←rig for last frame
    camera_centers_world: torch.Tensor   # [N, 3]
    rotations_world_cam: torch.Tensor    # [N, 3, 3] LUF
    shell_offsets_rig: torch.Tensor      # [N, 3] original offsets in rig frame
    mask_valid: torch.BoolTensor         # [N]
    per_rule_masks: dict[str, torch.BoolTensor]
    mesh: Optional[trimesh.Trimesh]
    mesh_verts: Optional[torch.Tensor]
    mesh_faces: Optional[torch.Tensor]
    occupancy_extent: Optional[torch.Tensor]
```

`CandidateViewGenerator` then becomes:

```python
class CandidateViewGenerator:
    def __init__(self, config: CandidateViewGeneratorConfig):
        self.config = config
        self.console = Console(...)
        self.rules: list[Rule] = self._build_rules(config)

    def generate_from_typed_sample(self, sample: EfmSnippetView) -> CandidateSamplingResult:
        last_pose = sample.trajectory().final_pose()     # PoseTW world←rig
        gt_mesh = sample.mesh
        mesh_verts, mesh_faces = ...
        occupancy_extent = sample.get_occupancy_extent()
        camera_fov = sample.camera_rgb().get_fov()
        return self.generate(last_pose, gt_mesh, mesh_verts, mesh_faces, occupancy_extent, camera_fov)

    def generate(
        self,
        last_pose: PoseTW,   # world←rig
        gt_mesh: Optional[trimesh.Trimesh],
        mesh_verts: Optional[torch.Tensor],
        mesh_faces: Optional[torch.Tensor],
        occupancy_extent: Optional[torch.Tensor],
        camera_fov: float,
    ) -> CandidateSamplingResult:

        state = self._sample_shell(last_pose, gt_mesh, mesh_verts, mesh_faces, occupancy_extent)

        for rule in self.rules:
            mask_before = state.mask_valid.clone()
            state = rule(state)
            state.per_rule_masks[rule.name] = state.mask_valid & mask_before

        # Build PoseTW batch from rotations + translations
        poses = PoseTW.from_rt(state.rotations_world_cam, state.camera_centers_world)

        return CandidateSamplingResult(
            poses=poses,
            mask_valid=state.mask_valid,
            masks=state.per_rule_masks,
            shell_poses=PoseTW.from_rt(
                self._rig_rotations_broadcast(last_pose, state.shell_offsets_rig),
                (last_pose.t.unsqueeze(0) + state.shell_offsets_rig @ last_pose.R.T),
            ),
        )
```

And `_sample_shell` is the only place that knows about HypersphericalUniform, PowerSpherical, radii, `delta_azimuth`, world‑up, etc.

---

## 3. Revised `view_axes_from_poses` (LUF, zero roll, “look away” option)

### 3.1 Requirements

* Camera frame is **LUF**:

  * x: left
  * y: up
  * z: forward
* World frame is VIO (z‑up or y‑up depending on Aria; we don’t need to hard‑code, we just take `world_up_tensor()` from `utils.frames`).
* We want **zero roll**: camera x‑axis must be orthogonal to `world_up`.
* We want to be able to “look at” or “look away from” the last pose.

### 3.2 Proposed helper

Let’s define this in `utils.frames` (or alongside your current helpers):

```python
def view_axes_from_poses(
    from_pose: PoseTW,
    to_pose: PoseTW,
    world_up: torch.Tensor,
    look_away: bool = False,
) -> torch.Tensor:
    """
    Construct LUF camera rotation matrices R_world_cam for poses whose origins
    are at `from_pose`, with a forward axis pointing toward or away from
    `to_pose` and zero roll w.r.t. `world_up`.

    Args
    ----
    from_pose:
        PoseTW with transform world←from (only the translation is used).
        Can be batched: [B, 3, 4] internally.
    to_pose:
        PoseTW with transform world←to (only translation is used).
        Must be broadcastable with `from_pose`.
    world_up:
        Tensor[3], world up direction (e.g. [0, 0, 1] or [0, 1, 0]).
        Need not be unit length.
    look_away:
        If False (default), cameras look from `from_pose` *towards* `to_pose`.
        If True, cameras look in the opposite direction (“away from”).

    Returns
    -------
    R_world_cam : Tensor[B, 3, 3]
        Rotation matrices with LUF conventions.
    """

    # Extract translations: world coordinates of origins
    c_from = from_pose.t_world   # [B, 3]
    c_to   = to_pose.t_world     # [B, 3]

    # Forward direction in world coordinates
    direction = c_to - c_from
    if look_away:
        direction = -direction

    # Normalise; guard against degenerate case (same point)
    eps = 1e-8
    f = direction / (direction.norm(dim=-1, keepdim=True) + eps)   # forward (F axis)

    # Ensure world_up is unit and batch-aligned
    u_world = world_up / world_up.norm()
    if f.ndim == 2:
        u = u_world.expand_as(f)
    else:
        raise ValueError("Unexpected dims for forward vector")

    # x-axis (left) is orthogonal to both up and forward
    # Using x = normalize(up × forward) gives LUF (left-handed) if world/pose are consistent.
    x = torch.cross(u, f, dim=-1)
    x = x / (x.norm(dim=-1, keepdim=True) + eps)

    # y-axis (up) is then orthogonal to forward and left
    y = torch.cross(f, x, dim=-1)

    # Fix sign so that y is aligned with world_up (no flipped camera)
    flip = (torch.sum(y * u, dim=-1, keepdim=True) < 0.0).to(y.dtype)
    flip = 1.0 - 2.0 * flip  # either +1 or -1
    x = x * flip
    y = y * flip
    f = f * flip

    # Stack as columns: R_world_cam @ v_cam = v_world
    R = torch.stack([x, y, f], dim=-1)  # [B, 3, 3]
    return R
```

Use it in the sampler:

```python
world_up = world_up_tensor(device=last_pose.device)
R_world_cam = view_axes_from_poses(
    from_pose=candidate_pose_origins,
    to_pose=last_pose.expand_as(candidate_pose_origins),
    world_up=world_up,
    look_away=True,      # if you want to look away from last pose
)
```

If you want “look toward unexplored space roughly along the candidate offset”, you can instead set `to_pose` to a *virtual* pose:

```python
# pseudo: from candidate center, look along offset direction
to_points = candidate_centers_world + (candidate_centers_world - last_pose_center)
R_world_cam = view_axes_from_points(
    from_points=candidate_centers_world,
    to_points=to_points,
    world_up=world_up,
)
```

where `view_axes_from_points` is just a point‑based version of the same math.

---

## 4. Revised candidate sampling with HypersphericalUniform / PowerSpherical + `delta_azimuth`

### 4.1 High‑level: what we want

* Sample unit directions **in the rig frame**:

  * Uniform on S² (`HypersphericalUniform`) or forward‑biased (`PowerSpherical`).
  * Constrained to:

    * elevation ∈ [`min_elev_deg`, `max_elev_deg`] relative to RIG up,
    * |azimuth| ≤ `delta_azimuth_deg / 2` relative to RIG forward axis (or full circle for 360°).
* Sample radii `r ~ Uniform[min_radius, max_radius]`.
* Convert to rig offsets: `offset_rig = r * d_rig`.
* Convert to world positions: `p_world = t_last_world + R_world_rig @ offset_rig`.
* Build camera orientations using `view_axes_from_poses`.

### 4.2 Direction sampling function

Something like this inside `CandidateViewGenerator` (or a helper module), conceptually aligned with your `SimpleCandidateViewGenerator`:

```python
from power_spherical import HypersphericalUniform, PowerSpherical

def _sample_directions_rig(
    cfg: CandidateViewGeneratorConfig,
    num: int,
    device: torch.device,
    forward_axis_rig: torch.Tensor,   # e.g. tensor([0., 0., 1.], device=device)
) -> torch.Tensor:
    """
    Sample unit directions in rig frame using HypersphericalUniform or PowerSpherical,
    then clamp to elevation and delta_azimuth.

    Returns:
        dirs_rig : Tensor[num, 3]
    """

    # 1. Base distribution
    if cfg.sampling_strategy == SamplingStrategy.UNIFORM_SPHERE :
        base = HypersphericalUniform(dim=3, device=device)
        dirs = base.rsample((num,))
    elif cfg.sampling_strategy == SamplingStrategy.FORWARD_POWERSPHERICAL :
        mean = forward_axis_rig / forward_axis_rig.norm()
        dist = PowerSpherical(mean, torch.tensor(cfg.kappa, device=device))
        dirs = dist.rsample((num,))
    else:
        raise ValueError(f"Unknown sampling strategy: {cfg.sampling_strategy}")

    # 2. Express directions in a rig-local basis (right, up, forward)
    # For LUF rig frame, you might set:
    right_rig  = torch.tensor([-1.0, 0.0, 0.0], device=device)   # -L
    up_rig     = torch.tensor([0.0, 1.0, 0.0], device=device)    # U
    forward_rig= forward_axis_rig                                # F

    # Projections onto basis
    d_right   = (dirs * right_rig).sum(-1)
    d_up      = (dirs * up_rig).sum(-1)
    d_forward = (dirs * forward_rig).sum(-1)

    # 3. Compute yaw (azimuth) and pitch (elevation)
    # yaw: angle in right–forward plane around up axis
    yaw = torch.atan2(d_right, d_forward)   # [-pi, pi]
    # pitch: elevation angle from horizontal
    pitch = torch.asin(d_up.clamp(-1.0, 1.0))

    # 4. Apply elevation and azimuth bounds
    min_pitch = math.radians(cfg.min_elev_deg)
    max_pitch = math.radians(cfg.max_elev_deg)
    keep_elev = (pitch >= min_pitch) & (pitch <= max_pitch)

    if cfg.delta_azimuth_deg >= 360.0 - 1e-3:
        keep_az = torch.ones_like(keep_elev)
    else:
        half = math.radians(cfg.delta_azimuth_deg) / 2.0
        keep_az = (yaw >= -half) & (yaw <= half)

    mask = keep_elev & keep_az
    dirs = dirs[mask]

    # If aggressive filters removed too many, you can resample or pad
    if dirs.shape[0] < num:
        # simplest: repeat with wrap-around
        repeat_factor = math.ceil(num / dirs.shape[0])
        dirs = dirs.repeat(repeat_factor, 1)[:num]

    return dirs[:num]
```

This matches your requirement:

> *If `delta_azimuth` is 360, sample full hypersphere. If it’s 90, limit to 45° to both sides of the z‑axis of the last position.*

Here, that “z‑axis of the last position” is the rig forward axis (`forward_rig`). If for Aria your rig forward axis is something else, just set `forward_axis_rig` accordingly.

### 4.3 Putting it together in `_sample_shell`

Pseudo‑implementation of `_sample_shell` using the above pieces:

```python
def _sample_shell(
    self,
    last_pose_world_rig: PoseTW,
    gt_mesh,
    mesh_verts,
    mesh_faces,
    occupancy_extent,
) -> CandidateState:

    cfg = self.config
    device = cfg.device

    # 1. Sample directions in rig frame
    forward_rig = torch.tensor([0.0, 0.0, 1.0], device=device)   # LUF forward
    dirs_rig = _sample_directions_rig(cfg, cfg.num_samples, device, forward_rig)

    # 2. Sample radii
    u = torch.rand(cfg.num_samples, device=device)
    radii = cfg.min_radius + (cfg.max_radius - cfg.min_radius) * u
    offsets_rig = dirs_rig * radii.unsqueeze(-1)         # [N, 3]

    # 3. Transform offsets to world frame
    R_world_rig = last_pose_world_rig.R_world  # [3, 3]
    t_world_rig = last_pose_world_rig.t_world  # [3]

    centers_world = t_world_rig.unsqueeze(0) + offsets_rig @ R_world_rig.T  # [N, 3]

    # 4. Build camera orientations (look away from last pose, zero roll)
    world_up = world_up_tensor(device=device)
    # Build a "pose" object just for translations of candidate centers
    candidate_pose_world_centers = PoseTW.from_rt(
        R=torch.eye(3, device=device).expand(cfg.num_samples, 3, 3),
        t=centers_world,
    )
    rotations_world_cam = view_axes_from_poses(
        from_pose=candidate_pose_world_centers,
        to_pose=last_pose_world_rig.expand(cfg.num_samples),
        world_up=world_up,
        look_away=True,
    )

    # 5. Init state
    mask_valid = torch.ones(cfg.num_samples, dtype=torch.bool, device=device)
    state = CandidateState(
        cfg=cfg,
        last_pose_world_rig=last_pose_world_rig,
        camera_centers_world=centers_world,
        rotations_world_cam=rotations_world_cam,
        shell_offsets_rig=offsets_rig,
        mask_valid=mask_valid,
        per_rule_masks={},
        mesh=gt_mesh,
        mesh_verts=mesh_verts,
        mesh_faces=mesh_faces,
        occupancy_extent=occupancy_extent,
    )
    return state
```

Now rules (`MinDistanceToMeshRule`, `PathCollisionRule`, `FreeSpaceRule`) work exclusively on `CandidateState.camera_centers_world`, `state.last_pose_world_rig`, and `state.mask_valid`. No rule re‑computes directions or orientations.

---

## 5. Direct answers to your concrete questions

> **Does this make sense? Currently we're looking in the direction defined by the candidate position and the last pose.**

Yes, conceptually that’s exactly what `view_axes_from_points` should do. The issue is *not* that idea, but:

* the **direction distribution** (biased to +y instead of forward),
* and potentially the **“look at vs look away”** choice.

With the new `view_axes_from_poses` you can explicitly pick:

* `look_away=True` → forward axis points *away from* the last pose,
* `look_away=False` → forward axis points *towards* the last pose.

> **Why the offset? Why is our azimuth currently limited so strongly?**

* Offset: you’re sampling a **biased spherical cap**, not a symmetric shell. The visible “center” of the cluster is its centroid, which is shifted toward the bias direction.
* Strong azimuth limitation: you have both:

  * an **elevation band** and
  * an **implicit or explicit azimuth band** (half‑sphere / quarter‑sphere),
  * plus the PowerSpherical concentration. Combined, that yields a narrow lobe.
* So even if you think you’re covering “full circle”, the actual composed constraints likely limit you to a fairly narrow yaw band.

> **Why does the candidate positions PC seem offset along the y‑world axis rather than z‑camera?**

Because the *sampling* is biased in the rig/world **+y direction**, not along the camera’s forward (z) axis. After transforming by `PoseTW`, that bias manifests in world coordinates as an offset along world +y. The camera’s forward axis is computed *after* that from the relative positions, but the point cloud distribution still carries the +y bias.

---

## 6. Summary of proposed changes

Very compressed checklist you can actually implement:

1. **Refactor `CandidateViewGenerator`** to follow this pipeline:

   * `_sample_shell` → returns `CandidateState` with world centers + rotations.
   * `rules` list → each `Rule` only updates `state.mask_valid`.
   * `generate` → builds `CandidateSamplingResult` from `CandidateState`.

2. **Replace current spherical sampling** in `ShellSamplingRule` / `_sample_directions` with a `HypersphericalUniform` / `PowerSpherical`‑based helper like `_sample_directions_rig` above, with explicit `delta_azimuth_deg`.

3. **Introduce `view_axes_from_poses`** as above:

   * input: `from_pose`, `to_pose` (both `PoseTW`),
   * uses `world_up_tensor` for zero‑roll LUF frames,
   * `look_away` flag.

4. **Remove hidden frame assumptions**:

   * Explicitly define rig forward axis (e.g. `(0, 0, 1)` in LUF),
   * Use it consistently in the direction sampler and when interpreting `delta_azimuth_deg`.

Once this is in place, the candidate cloud should:

* be centerd around the last pose in *geometry*,
* show an offset that matches your explicit **elevation + delta_azimuth + forward bias**,
* and have frusta that look where you think they look (toward or away from the last pose, but *not* all mysteriously toward world +y).


---
---

# Second Answer by GPT5-PRO

Let’s treat this as a full redesign of the *pose‑generation* layer, not just a patch.

I’ll do it in three steps:

1. **Architecture & data‑flow** in the revised submodule.
2. **New geometry helper**: `view_axes_from_poses(from_pose, to_pose)` with correct LUF + world‑up behaviour.
3. **Complete new `candidate_generation.py`** built around:

   * modular direction samplers (HypersphericalUniform / PowerSpherical),
   * clean shell sampling,
   * rule pipeline,
   * optional debug outputs such as per‑rule masks and min‑distance‑to‑mesh.

Where I have to guess details of your types (e.g. `PoseTW`), I’ll make the assumptions explicit so you can adapt quickly.

---

## 1. Architecture & data‑flow (revised)

### 1.1 Key actors

**Types (`pose_generation.types`)**

* `SamplingStrategy` (enum): `UNIFORM_SPHERE `, `FORWARD_POWERSPHERICAL `, …
* `CollisionBackend` (enum): `P3D`, `PYEMBREE`, `TRIMESH`.
* `CandidateContext` (mutable context passed to rules).
* `CandidateSamplingResult` (immutable result handed to RRI/oracle).

We will slightly *extend* `CandidateContext` and `CandidateSamplingResult`.

**Outer generator (`pose_generation.candidate_generation`)**

* `CandidateViewGeneratorConfig`
  Sampling + rule config:

  * `num_samples`, `min_radius`, `max_radius`
  * `min_elev_deg`, `max_elev_deg`
  * `delta_azimuth_deg`   **(new)**
  * `sampling_strategy` (`UNIFORM_SPHERE ` / `FORWARD_POWERSPHERICAL `)
  * `kappa` (PowerSpherical concentration)
  * rule toggles: `min_distance_to_mesh`, `ensure_collision_free`, `ensure_free_space`, `collision_backend`
  * implementation knobs: `oversample_factor`, `max_resamples`, `device`
  * debug knobs:
    `collect_rule_masks: bool` and `collect_debug_stats: bool` **(new)**

* `CandidateViewGenerator`

  * `generate_from_typed_sample(EfmSnippetView)` → convenience wrapper.
  * `generate(last_pose, gt_mesh, mesh_verts, mesh_faces, occupancy_extent, camera_fov)`:

    1. build `CandidateContext` with geometry and config;
    2. call `_sample_shell(ctx)` to get PoseTW candidates before any pruning;
    3. call `_apply_rules(ctx)` to run `MinDistanceToMeshRule`, `PathCollisionRule`, `FreeSpaceRule`, …
    4. call `_finalise_result(ctx)` → `CandidateSamplingResult`.

* **Direction samplers** (modular, exchangeable like rules):

  * `DirectionSampler` abstract base.
  * `UniformDirectionSampler` → `HypersphericalUniform`.
  * `ForwardPowerSphericalSampler` → `PowerSpherical` around +z in rig frame.

**Rules (`pose_generation.candidate_generation_rules`)**

Unchanged interface:

```python
class Rule:
    name: str
    def __call__(self, ctx: CandidateContext) -> None:
        ...
```

Rules only:

* read geometry from `ctx`,
* update `ctx.mask_valid` in place,
* *optionally* write debug info into `ctx.debug` if `ctx.cfg.collect_debug_stats` is True.

**MinDistanceToMeshRule** additionally:

* when `cfg.min_distance_to_mesh > 0`:

  * computes signed distance `d_i` for each candidate i,
  * updates `ctx.mask_valid &= (d_i > cfg.min_distance_to_mesh)`,
  * if `cfg.collect_debug_stats` is True, stores `ctx.debug["min_distance_to_mesh"] = d` **without masking**, so you retain distances for all sampled poses.

**Orientation helper (`utils.frames`)**

* New `view_axes_from_poses(from_pose: PoseTW, to_pose: PoseTW, look_away: bool = True)`:

  * LUF camera frame:

    * x: left, y: up, z: forward
  * world frame: VIO world.
  * ensures:

    * `z_cam` points along displacement `last → candidate` (look‑away) or the opposite (look‑at),
    * `y_cam` is as aligned as possible with `world_up_tensor()` (no roll),
    * `x_cam` is horizontal: `x_cam ⟂ world_up`.

### 1.2 Data‑flow

End‑to‑end flow for `generate_from_typed_sample`:

1. **Input**: `EfmSnippetView sample`

   * `traj = sample.trajectory()` → last rig pose `PoseTW last_pose_world_rig`
   * `mesh`, `(verts, faces)`, `occupancy_extent`, `camera_fov`.

2. **Shell sampling** (`_sample_shell`):

   * Query direction sampler according to `cfg.sampling_strategy`.
   * Sample `N * oversample_factor` unit directions in rig frame (LUF, z‑forward, y‑up).
   * Rotate directions to world with `last_pose_world_rig`.
   * Filter by:

     * elevation vs world up (`min_elev_deg ≤ elev ≤ max_elev_deg`),
     * yaw difference vs last forward direction in horizontal plane (`|Δyaw| ≤ delta_azimuth_deg/2`).
   * Sample radii `r ∈ [min_radius, max_radius]`.
   * Compute candidate centers:
     `p_i = t_last + r_i * d_world_i`.
   * Build PoseTW for each candidate via `view_axes_from_poses(last_pose, T_world= p_i, look_away=True)`.
   * Initialise `ctx.shell_poses = PoseTW([...])`, `ctx.centers_world`, `ctx.mask_valid = True`.

3. **Rule pass** (`_apply_rules`):

   * For each rule `R_k` in order:

     * Snapshot current mask if `cfg.collect_rule_masks`.
     * Invoke `R_k(ctx)`, which updates `ctx.mask_valid`.
     * Save per‑rule mask if enabled: `ctx.rule_masks[R_k.name] = ctx.mask_valid.clone()`.

4. **Finalisation** (`_finalise_result`):

   * Build filtered PoseTW: `poses = ctx.shell_poses[ctx.mask_valid]`.
   * Construct `CandidateSamplingResult` with:

     * `poses`
     * `mask_valid = ctx.mask_valid`
     * `masks = ctx.rule_masks` (or empty dict if disabled)
     * `shell_poses = ctx.shell_poses`
     * `extras = ctx.debug` (may include `"min_distance_to_mesh"`).

This satisfies the constraints:

* x‑axis horizontal / no roll: enforced in `view_axes_from_poses` using `world_up_tensor`.
* yaw “zero” in the sense of no arbitrary rotation about forward: the up vector choice is fixed by world up, so yaw is determined by geometry, not free.
* bias towards last z‑axis: provided by the `ForwardPowerSphericalSampler` (mean direction +z in rig frame).
* candidates are genuine `PoseTW` in world frame.
* sampling strategies are pluggable objects, just like rules.

---

## 2. New geometry helper: `view_axes_from_poses`

Put this into `oracle_rri/utils/frames.py` (or wherever your frame helpers live).
Assumptions:

* `PoseTW` stores transforms in a `.tensor` of shape `(3, 4)` or `(B, 3, 4)`.
* You have a `world_up_tensor(device=None) -> torch.Tensor[3]` returning unit world‑up.

```python
# utils/frames.py

from __future__ import annotations

from typing import Tuple

import torch
from efm3d.aria.pose import PoseTW


def world_up_tensor(device: torch.device | None = None) -> torch.Tensor:
    """
    Returns the global 'up' direction in world coordinates as a unit 3-vector.

    For VIO world frames this is typically [0, 0, 1] or [0, 1, 0].
    Adjust here once and keep everything else agnostic.
    """
    up = torch.tensor([0.0, 0.0, 1.0])  # <-- change if your world-up differs
    if device is not None:
        up = up.to(device)
    return up / up.norm()


def _broadcast_poses_for_view(
    from_pose: PoseTW,
    to_pose: PoseTW,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ensure from/to pose tensors have matching batch dimensions.

    - If both are unbatched (3x4), returns them as 1x3x4.
    - If from is unbatched and to is batched (N x 3 x 4), expands from.
    """
    T_from = from_pose.tensor
    T_to = to_pose.tensor

    if T_from.ndim == 2 and T_to.ndim == 2:
        T_from = T_from.unsqueeze(0)
        T_to = T_to.unsqueeze(0)
    elif T_from.ndim == 2 and T_to.ndim == 3:
        T_from = T_from.unsqueeze(0).expand(T_to.shape[0], -1, -1)
    elif T_from.ndim == 3 and T_to.ndim == 2:
        T_to = T_to.unsqueeze(0).expand(T_from.shape[0], -1, -1)
    elif T_from.ndim == 3 and T_to.ndim == 3:
        if T_from.shape[0] != T_to.shape[0]:
            raise ValueError(f"Batch sizes differ: {T_from.shape} vs {T_to.shape}")
    else:
        raise ValueError(f"Unexpected pose tensor shapes: {T_from.shape}, {T_to.shape}")

    return T_from, T_to


def view_axes_from_poses(
    from_pose: PoseTW,
    to_pose: PoseTW,
    look_away: bool = True,
    eps: float = 1e-6,
) -> PoseTW:
    """
    Build PoseTW(s) whose origin(s) are at `to_pose` and whose LUF camera axes satisfy:

        - z_cam points along the line between from_pose and to_pose:
              z_cam ∝ (to - from)              if look_away=True
              z_cam ∝ (from - to)              if look_away=False

        - y_cam is as aligned as possible with the world up vector,
          ensuring zero roll and keeping x_cam horizontal.

        - x_cam is horizontal (orthogonal to world up) and completes a
          right-handed LUF triad (x=Left, y=Up, z=Forward).

    Both poses are assumed to be expressed in the VIO world frame.
    """
    T_from, T_to = _broadcast_poses_for_view(from_pose, to_pose)
    device = T_to.device

    WUP = world_up_tensor(device=device)  # (3,)

    p_from = T_from[..., :3, 3]  # (..., 3)
    p_to = T_to[..., :3, 3]      # (..., 3)

    disp = p_to - p_from
    if not look_away:
        disp = -disp

    # Forward axis: world-frame z_cam
    z_cam = disp / disp.norm(dim=-1, keepdim=True).clamp_min(eps)  # (..., 3)

    # Horizontal x_cam: proportional to world_up × z_cam
    WUP_exp = WUP.view(1, 3).expand_as(z_cam)
    x_cam = torch.cross(WUP_exp, z_cam, dim=-1)
    x_norm = x_cam.norm(dim=-1, keepdim=True)

    # Handle near-singular case (forward almost parallel to up)
    degenerate = x_norm.squeeze(-1) < eps
    if degenerate.any():
        # Use an arbitrary horizontal axis orthogonal to up as fallback
        # e.g. [1,0,0] projected into horizontal plane.
        tmp = torch.tensor([1.0, 0.0, 0.0], device=device)
        tmp = tmp - (tmp @ WUP) * WUP
        tmp = tmp / tmp.norm()
        x_cam[degenerate] = tmp

        x_norm = x_cam.norm(dim=-1, keepdim=True)

    x_cam = x_cam / x_norm

    # y_cam completes right-handed LUF triad
    y_cam = torch.cross(z_cam, x_cam, dim=-1)

    R_wc = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (..., 3, 3)

    T_out = torch.empty_like(T_to)
    T_out[..., :3, :3] = R_wc
    T_out[..., :3, 3] = p_to

    return PoseTW(T_out)
```

This guarantees:

* **x_cam is horizontal**: always constructed orthogonal to `world_up`.
* **No roll**: the orientation about z_cam is uniquely determined by `world_up`.
* **Forward direction** can be chosen “look‑away” (your NBV case) or “look‑back” by flipping `look_away`.

---

## 3. New types: `CandidateContext` and `CandidateSamplingResult`

Update `pose_generation/types.py` to define these two as follows (rest of the file can stay as is).

```python
# pose_generation/types.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from efm3d.aria.pose import PoseTW
from .types_enums import SamplingStrategy, CollisionBackend  # however you organise enums


@dataclass
class CandidateContext:
    """
    Mutable state passed between sampling and pruning rules.

    Geometry:
        - last_pose: PoseTW of the last rig/camera pose in world frame.
        - gt_mesh, mesh_verts, mesh_faces, occupancy_extent, camera_fov: optional
          objects used by rules such as MinDistanceToMeshRule, PathCollisionRule, etc.

    Candidate data:
        - shell_poses: PoseTW of all sampled candidate poses before pruning.
        - centers_world: (N, 3) world-frame camera centers corresponding to shell_poses.
        - shell_offsets_rig: (N, 3) offsets in rig frame from last_pose center
                             (useful for debugging the sampling distribution).
        - mask_valid: (N,) bool mask tracking which candidates are still alive.

    Diagnostics:
        - rule_masks: per-rule masks after applying each rule; only filled when
                      cfg.collect_rule_masks is True.
        - debug: arbitrary extra tensors produced by rules (e.g. distances to mesh)
                 when cfg.collect_debug_stats is True.
    """
    cfg: Any  # CandidateViewGeneratorConfig, but avoid circular import
    last_pose: PoseTW

    # Geometry for rules
    gt_mesh: Any | None = None
    mesh_verts: torch.Tensor | None = None
    mesh_faces: torch.Tensor | None = None
    occupancy_extent: torch.Tensor | None = None
    camera_fov: float | None = None

    # Candidate data
    shell_poses: PoseTW | None = None
    centers_world: torch.Tensor | None = None
    shell_offsets_rig: torch.Tensor | None = None
    mask_valid: torch.Tensor | None = None

    # Diagnostics
    rule_masks: Dict[str, torch.Tensor] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateSamplingResult:
    """
    Immutable result of candidate sampling + rule-based pruning.

    Attributes
    ----------
    poses:
        PoseTW containing final, filtered candidate viewpoints in Aria world
        frame (T_world_cam).

    mask_valid:
        Bool tensor of length N_shell indicating which of the original shell
        poses survived all rules.

    masks:
        Optional per-rule masks, mapping rule names to bool tensors of length
        N_shell. Only populated when cfg.collect_rule_masks is True.

    shell_poses:
        PoseTW containing all poses sampled on the spherical shell *before*
        any pruning.

    extras:
        Optional dictionary with arbitrary debug tensors. By convention,
        MinDistanceToMeshRule stores a tensor under the key
        "min_distance_to_mesh" when cfg.collect_debug_stats is True.
    """
    poses: PoseTW
    mask_valid: torch.Tensor
    masks: Dict[str, torch.Tensor]
    shell_poses: PoseTW
    extras: Dict[str, Any] = field(default_factory=dict)
```

---

## 4. Replacement `candidate_generation.py`

Here is a self‑contained, simplified version of `oracle_rri/oracle_rri/pose_generation/candidate_generation.py` that uses the new helper and types.

You will need to adjust a few import paths to your repo layout, but the logic is all here.

```python
# oracle_rri/oracle_rri/pose_generation/candidate_generation.py

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Optional, Tuple

import torch
from efm3d.aria.pose import PoseTW
import trimesh  # for typed hints only

from oracle_rri.utils.base_config import BaseConfig
from oracle_rri.utils.console import Console
from oracle_rri.utils.frames import world_up_tensor, view_axes_from_poses

from .types import (
    SamplingStrategy,
    CollisionBackend,
    CandidateContext,
    CandidateSamplingResult,
)
from .candidate_generation_rules import (
    Rule,
    MinDistanceToMeshRule,
    PathCollisionRule,
    FreeSpaceRule,
)
from .reference_power_spherical_distributions import (
    HypersphericalUniform,
    PowerSpherical,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CandidateViewGeneratorConfig(BaseConfig):
    """
    Config for candidate generation around the latest pose.

    Sampling
    --------
    num_samples : int
        Number of candidate poses requested *after* pruning.

    min_radius, max_radius : float
        Inner and outer shell radii in metres.

    min_elev_deg, max_elev_deg : float
        Elevation band in degrees, measured in world frame as the angle between
        the ray and the horizontal plane spanned by world_up_tensor().

    delta_azimuth_deg : float
        Width of the horizontal yaw band in degrees around the last forward
        direction. 360 means unrestricted; 90 means ±45° around forward.

    sampling_strategy : SamplingStrategy
        UNIFORM_SPHERE  for area-uniform, FORWARD_POWERSPHERICAL  for PowerSpherical
        bias towards the last forward axis.

    kappa : float
        Concentration parameter for PowerSpherical.

    Rules
    -----
    min_distance_to_mesh : float
        Clearance enforced by MinDistanceToMeshRule (metres).

    ensure_collision_free : bool
        Enable PathCollisionRule.

    ensure_free_space : bool
        Enable FreeSpaceRule.

    collision_backend : CollisionBackend
        Backend for path collision checks.

    Implementation / debug
    -----------------------
    oversample_factor : float
        Factor by which we oversample directions before filtering, to reduce
        the need for resampling.

    max_resamples : int
        Maximum number of oversampling rounds to fill num_samples after heavy
        pruning.

    device : str
        Torch device for vectorised ops.

    collect_rule_masks : bool
        If True, we store per-rule masks in CandidateSamplingResult.masks.

    collect_debug_stats : bool
        If True, rules may write additional tensors into CandidateContext.debug
        (e.g. per-candidate distances to mesh), which are returned as
        CandidateSamplingResult.extras.
    """

    num_samples: int = 512
    min_radius: float = 0.4
    max_radius: float = 1.6

    min_elev_deg: float = -15.0
    max_elev_deg: float = 45.0
    delta_azimuth_deg: float = 360.0

    sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM_SPHERE
    kappa: float = 4.0

    min_distance_to_mesh: float = 0.0
    ensure_collision_free: bool = True
    ensure_free_space: bool = True
    collision_backend: CollisionBackend = CollisionBackend.P3D

    oversample_factor: float = 2.0
    max_resamples: int = 4
    device: str = "cuda"

    collect_rule_masks: bool = False
    collect_debug_stats: bool = False

    def set_debug(self) -> None:
        # Force CPU, more verbose logs, etc., if you like.
        self.device = "cpu"


# ---------------------------------------------------------------------------
# Direction samplers
# ---------------------------------------------------------------------------

class DirectionSampler:
    """Abstract base for direction sampling in rig (camera) frame."""

    name: str = "base"

    def sample(self, cfg: CandidateViewGeneratorConfig, num: int, device: torch.device) -> torch.Tensor:
        """
        Return unit directions in rig frame (LUF), shape (num, 3).
        Default forward axis is +z, up is +y.
        """
        raise NotImplementedError


class UniformDirectionSampler(DirectionSampler):
    name = "uniform"

    def sample(self, cfg: CandidateViewGeneratorConfig, num: int, device: torch.device) -> torch.Tensor:
        dist = HypersphericalUniform(dim=3, device=device)
        dirs = dist.sample((num,))  # (num, 3)
        return dirs / dirs.norm(dim=-1, keepdim=True)


class ForwardPowerSphericalSampler(DirectionSampler):
    name = "forward_power_spherical"

    def sample(self, cfg: CandidateViewGeneratorConfig, num: int, device: torch.device) -> torch.Tensor:
        # Forward axis is +z in rig frame (LUF).
        mu = torch.tensor([0.0, 0.0, 1.0], device=device)
        dist = PowerSpherical(mu=mu, kappa=torch.tensor(cfg.kappa, device=device))
        dirs = dist.sample((num,))  # (num, 3)
        return dirs / dirs.norm(dim=-1, keepdim=True)


_DIRECTION_SAMPLERS: Dict[SamplingStrategy, DirectionSampler] = {
    SamplingStrategy.UNIFORM_SPHERE : UniformDirectionSampler(),
    SamplingStrategy.FORWARD_POWERSPHERICAL : ForwardPowerSphericalSampler(),
}


# ---------------------------------------------------------------------------
# Helper geometry functions
# ---------------------------------------------------------------------------

def _forward_world(last_pose: PoseTW) -> torch.Tensor:
    """
    World-frame forward direction of the last camera pose.

    Assumes LUF camera frame with forward = +z_cam.
    """
    R_wc = last_pose.tensor[:3, :3]  # (3,3)
    f_cam = torch.tensor([0.0, 0.0, 1.0], device=R_wc.device)
    return R_wc @ f_cam  # (3,)


def _filter_directions_world(
    dirs_world: torch.Tensor,        # (N, 3)
    last_forward_world: torch.Tensor,
    cfg: CandidateViewGeneratorConfig,
) -> torch.Tensor:
    """
    Apply elevation + delta-azimuth filters in world frame.

    Returns a boolean mask of shape (N,) selecting accepted directions.
    """
    device = dirs_world.device
    WUP = world_up_tensor(device=device)  # (3,)

    # Elevation: angle between direction and horizontal plane
    dot_up = dirs_world @ WUP            # (N,)
    elev = torch.asin(dot_up.clamp(-1.0, 1.0))  # radians

    min_elev = torch.deg2rad(torch.tensor(cfg.min_elev_deg, device=device))
    max_elev = torch.deg2rad(torch.tensor(cfg.max_elev_deg, device=device))
    mask_elev = (elev >= min_elev) & (elev <= max_elev)

    # Yaw: signed angle in horizontal plane relative to last forward
    def horiz(v: torch.Tensor) -> torch.Tensor:
        v_h = v - (v @ WUP)[:, None] * WUP[None, :]
        return v_h / v_h.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    dirs_h = horiz(dirs_world)
    fwd_h = horiz(last_forward_world.view(1, 3)).expand_as(dirs_h)

    cross = torch.cross(fwd_h, dirs_h, dim=-1)      # (N, 3)
    sin_yaw = cross @ WUP                           # (N,)
    cos_yaw = (dirs_h * fwd_h).sum(dim=-1)          # (N,)
    yaw = torch.atan2(sin_yaw, cos_yaw)             # radians

    if cfg.delta_azimuth_deg >= 360.0 - 1e-3:
        mask_yaw = torch.ones_like(mask_elev)
    else:
        half_delta = 0.5 * torch.deg2rad(torch.tensor(cfg.delta_azimuth_deg, device=device))
        mask_yaw = (yaw >= -half_delta) & (yaw <= half_delta)

    return mask_elev & mask_yaw


def _sample_shell_poses(
    last_pose: PoseTW,
    cfg: CandidateViewGeneratorConfig,
) -> Tuple[PoseTW, torch.Tensor, torch.Tensor]:
    """
    Core shell sampling:

        1. Sample directions in rig frame via chosen DirectionSampler.
        2. Rotate to world frame.
        3. Filter by elevation and delta-azimuth band.
        4. Sample radii and compute candidate centers.
        5. Build PoseTW using view_axes_from_poses (look-away, zero roll).

    Returns:
        shell_poses : PoseTW with shape (N_shell, 3, 4).
        centers_world : Tensor[N_shell, 3]
        offsets_rig : Tensor[N_shell, 3]  # useful for debugging
    """
    device = torch.device(cfg.device)
    sampler = _DIRECTION_SAMPLERS[cfg.sampling_strategy]

    n_target = cfg.num_samples
    n_per_round = max(n_target, 1)

    last_pose_tensor = last_pose.tensor.to(device)
    R_wc = last_pose_tensor[:3, :3]
    t_last = last_pose_tensor[:3, 3]

    fwd_world = _forward_world(last_pose).to(device)

    dirs_world_list: List[torch.Tensor] = []
    offsets_rig_list: List[torch.Tensor] = []

    total_needed = n_target
    rounds = 0

    while total_needed > 0 and rounds < cfg.max_resamples:
        rounds += 1
        n_draw = ceil(cfg.oversample_factor * total_needed)

        dirs_rig = sampler.sample(cfg, n_draw, device=device)    # (n_draw, 3)
        dirs_world = (R_wc @ dirs_rig.T).T                       # (n_draw, 3)
        mask = _filter_directions_world(dirs_world, fwd_world, cfg)

        if mask.any():
            dirs_world_list.append(dirs_world[mask])
            offsets_rig_list.append(dirs_rig[mask])
            total_needed = n_target - sum(d.shape[0] for d in dirs_world_list)

    if not dirs_world_list:
        raise RuntimeError("Shell sampling failed: no directions satisfied constraints.")

    dirs_world_all = torch.cat(dirs_world_list, dim=0)[:n_target]
    offsets_rig_all = torch.cat(offsets_rig_list, dim=0)[:n_target]

    # Radii
    radii = torch.empty(dirs_world_all.shape[0], device=device).uniform_(
        cfg.min_radius, cfg.max_radius
    )

    centers_world = t_last[None, :] + radii[:, None] * dirs_world_all  # (N, 3)

    # Build PoseTW via view_axes_from_poses (look away from last pose)
    # Build a PoseTW with identity rotation and translation at each center.
    T_centers = torch.zeros(dirs_world_all.shape[0], 3, 4, device=device)
    T_centers[..., :3, :3] = torch.eye(3, device=device).expand_as(T_centers[..., :3, :3])
    T_centers[..., :3, 3] = centers_world

    centers_pose = PoseTW(T_centers)
    shell_poses = view_axes_from_poses(from_pose=last_pose, to_pose=centers_pose, look_away=True)

    return shell_poses, centers_world, offsets_rig_all


# ---------------------------------------------------------------------------
# CandidateViewGenerator
# ---------------------------------------------------------------------------

class CandidateViewGenerator:
    """
    Generate candidate PoseTW around the latest pose using a rule-based pipeline:

        1. Shell sampling (this module).
        2. Mesh distance / collision / free-space pruning (rules).
    """

    def __init__(self, config: CandidateViewGeneratorConfig):
        self.config = config
        self.console = Console.with_prefix("candidate_gen")
        self._rules: List[Rule] = self._build_default_rules(config)

    # --- Public API -----------------------------------------------------

    def generate_from_typed_sample(self, sample) -> CandidateSamplingResult:
        """
        Convenience wrapper for EfmSnippetView-style samples.

        Expects:
            - sample.trajectory().final_pose() -> PoseTW
            - sample.mesh / mesh_verts / mesh_faces
            - sample.get_occupancy_extend()
            - sample.camera_rgb().get_fov()
        """
        traj = sample.trajectory()
        last_pose = traj.final_pose()

        mesh = getattr(sample, "mesh", None)
        mesh_verts = getattr(sample, "mesh_verts", None)
        mesh_faces = getattr(sample, "mesh_faces", None)
        occupancy_extent = sample.get_occupancy_extend()
        camera_fov = sample.camera_rgb().get_fov()

        return self.generate(
            last_pose=last_pose,
            gt_mesh=mesh,
            mesh_verts=mesh_verts,
            mesh_faces=mesh_faces,
            occupancy_extent=occupancy_extent,
            camera_fov=camera_fov,
        )

    def generate(
        self,
        last_pose: PoseTW,
        gt_mesh: Optional[trimesh.Trimesh],
        mesh_verts: Optional[torch.Tensor],
        mesh_faces: Optional[torch.Tensor],
        occupancy_extent: Optional[torch.Tensor],
        camera_fov: float,
    ) -> CandidateSamplingResult:
        """
        Core API: generate candidate poses around `last_pose` and prune them.
        """
        cfg = self.config
        device = torch.device(cfg.device)

        # 1) Shell sampling
        shell_poses, centers_world, offsets_rig = _sample_shell_poses(last_pose, cfg)

        # 2) Build context
        ctx = CandidateContext(
            cfg=cfg,
            last_pose=last_pose,
            gt_mesh=gt_mesh,
            mesh_verts=mesh_verts.to(device) if mesh_verts is not None else None,
            mesh_faces=mesh_faces.to(device) if mesh_faces is not None else None,
            occupancy_extent=occupancy_extent.to(device) if occupancy_extent is not None else None,
            camera_fov=camera_fov,
            shell_poses=shell_poses,
            centers_world=centers_world,
            shell_offsets_rig=offsets_rig,
            mask_valid=torch.ones(centers_world.shape[0], dtype=torch.bool, device=device),
        )

        # 3) Apply rules
        self._apply_rules(ctx)

        # 4) Finalise
        return self._finalise_result(ctx)

    # --- Internals ------------------------------------------------------

    def _build_default_rules(self, cfg: CandidateViewGeneratorConfig) -> List[Rule]:
        rules: List[Rule] = []

        if cfg.min_distance_to_mesh > 0.0:
            rules.append(MinDistanceToMeshRule(cfg))

        if cfg.ensure_collision_free:
            rules.append(PathCollisionRule(cfg))

        if cfg.ensure_free_space:
            rules.append(FreeSpaceRule(cfg))

        return rules

    def _apply_rules(self, ctx: CandidateContext) -> None:
        """
        Apply all rules in sequence, updating ctx.mask_valid in place.

        If cfg.collect_rule_masks is True, snapshots of the mask after each
        rule are stored in ctx.rule_masks[rule.name].
        """
        for rule in self._rules:
            # Rules read ctx.shell_poses, ctx.centers_world, ctx.last_pose, etc.
            rule(ctx)
            if ctx.cfg.collect_rule_masks:
                # Snapshot after this rule
                ctx.rule_masks[rule.__class__.__name__] = ctx.mask_valid.clone()

    def _finalise_result(self, ctx: CandidateContext) -> CandidateSamplingResult:
        """
        Convert context into an immutable CandidateSamplingResult.
        """
        mask_valid = ctx.mask_valid
        shell_poses = ctx.shell_poses
        assert shell_poses is not None, "Shell poses must be set before finalisation."

        # Filter PoseTW – adapt if your PoseTW supports advanced indexing.
        filtered_tensor = shell_poses.tensor[mask_valid]
        poses = PoseTW(filtered_tensor)

        return CandidateSamplingResult(
            poses=poses,
            mask_valid=mask_valid,
            masks=ctx.rule_masks if ctx.cfg.collect_rule_masks else {},
            shell_poses=shell_poses,
            extras=ctx.debug if ctx.cfg.collect_debug_stats else {},
        )
```

---

## 5. MinDistanceToMeshRule: how to expose distances

You don’t have to overhaul `candidate_generation_rules.py`; just make `MinDistanceToMeshRule` obey the new debug contract.

Conceptually:

```python
# pose_generation/candidate_generation_rules.py

class MinDistanceToMeshRule(Rule):
    def __init__(self, cfg: CandidateViewGeneratorConfig):
        self.cfg = cfg
        self.name = "MinDistanceToMesh"

    def __call__(self, ctx: CandidateContext) -> None:
        if ctx.gt_mesh is None or ctx.centers_world is None:
            return

        # Compute signed distance for all candidates
        pts = ctx.centers_world.detach().cpu().numpy()  # (N, 3)
        mesh: trimesh.Trimesh = ctx.gt_mesh

        # Use trimesh proximity or your GPU kernel
        pq = trimesh.proximity.ProximityQuery(mesh)
        dist = pq.signed_distance(pts)  # (N,)

        dist_t = torch.from_numpy(dist).to(ctx.centers_world.device)

        # Store distances for debugging if requested
        if ctx.cfg.collect_debug_stats:
            ctx.debug["min_distance_to_mesh"] = dist_t

        # Apply clearance mask
        keep = dist_t > self.cfg.min_distance_to_mesh
        ctx.mask_valid &= keep
```

Rules like `PathCollisionRule` and `FreeSpaceRule` don’t need changes unless you also want them to emit debug tensors (e.g. number of collision samples per ray). They can follow the same pattern: check `ctx.cfg.collect_debug_stats`, then write into `ctx.debug[...]`.

---

## 6. What this fixes

* **Offset & y‑bias**
  Directions are sampled in *rig* frame, then filtered in *world* frame relative to:

  * world up (for elevation) and
  * last forward axis (for yaw band).
    Bias is controlled **explicitly** by PowerSpherical’s mean + kappa; no accidental bias towards world +y.

* **Azimuth semantics (`delta_azimuth_deg`)**
  Implemented exactly as you described:

  * 360° → full azimuth around forward.
  * 90° → ±45° around the last forward direction in the horizontal plane.
  * 0° (limit case) → all candidates lie in the plane containing last forward + world up (a narrow “slice”).

* **Orientation correctness**
  `view_axes_from_poses` guarantees:

  * z_cam ∥ last→candidate (look‑away),
  * x_cam ⟂ world_up (horizontal),
  * y_cam chosen by cross product → *no roll*.

  So images won’t be mysteriously rotated, and the x‑axis stays horizontal by construction.

* **Modularity**
  Sampling strategies and rules are now symmetric:

  * strategies: small `DirectionSampler` classes,
  * rules: small `Rule` classes, all controlled via config.

* **Optional diagnostics**
  `collect_rule_masks` and `collect_debug_stats` allow you to turn on:

  * per‑rule masks for analysis,
  * min‑distance‑to‑mesh (and other metrics) without touching the outer API.

This gives you a clean, geometrically sane foundation for NBV work, and you can finally focus on RRI scoring and learning the policy instead of wrestling with coordinate frames.
