I’ll treat this as a design/analysis doc for the whole candidate‑generation stack: what it’s doing geometrically and probabilistically, how HypersphericalUniform / PowerSpherical fit in, and how we might do better for “realistic but diverse” next poses.

---

## 0. Problem: what we’re actually sampling

Abstractly you’re trying to approximate a conditional distribution

$$
p(\text{next pose } T_{w\to c}' \mid \text{state } s)
$$

with two constraints:

* **Realism**: poses look like something a handheld camera could actually reach in one step (bounded translation, bounded rotation, no collisions).
* **Diversity**: candidates cover the space of plausible next views, so the NBV network has good choices to rank.

The current generator splits the pose into:

* **Position**: camera center in world frame $p' \in \mathbb{R}^3$.
* **Orientation**: rotation $R'_{wc} \in SO(3)$.

and then:

1. Samples directions on the **unit sphere** in the *rig frame*.
2. Applies geometric filters (elevation, delta‑azimuth, range).
3. Converts to world‑space positions using the last pose.
4. Builds orientations using a “look‑away” helper that enforces LUF + zero roll.
5. Applies rules (min distance to mesh, collision, free space).

So mathematically you’re building some distribution over directions on $S^2$, mapping it to a distribution over positions in an annulus, and then deterministically mapping positions to orientations.

The choices of distributions and transforms strongly shape what “kind” of views the network ever sees.

---

## 1. Geometry and transforms

### 1.1 Frames

* **Camera / rig frame** (LUF):

  * x: left,
  * y: up,
  * z: forward.
* **World frame** (VIO):

  * some inertial frame; we treat `world_up_tensor()` as the up direction (e.g. ([0,0,1])).

The last pose is a transform $T_{wc}^{\text{last}} \in SE(3)$:

$$
T_{wc}^{\text{last}} =
\begin{bmatrix}
R_{wc}^{\text{last}} & t_{w}^{\text{last}} [2pt]
0 & 1
\end{bmatrix}.
$$

A candidate direction in rig frame $d_{\text{rig}} \in S^2$ is mapped to world frame by

$$
d_{\text{world}} = R_{wc}^{\text{last}} , d_{\text{rig}}.
$$

A sampled radius $r \in [r_{\min}, r_{\max}]$ gives a world‑space center

$$
p' = t_w^{\text{last}} + r , d_{\text{world}}.
$$

So the “position” side is:

* sample $d_{\text{rig}} \sim q_d$ on $S^2$,
* sample $r \sim \text{Uniform}[r_{\min}, r_{\max}]$,
* map to $p'$.

The orientation is then chosen via `view_axes_from_poses` so that:

* the **forward axis** $z_c'$ aligns with the vector from the last pose to the candidate (look‑away), and
* the **up axis** $y_c'$ is aligned with world up, giving zero roll.

I’ll formalise that when we get to orientations.

---

## 2. Direction distributions on the sphere

You’re using two kinds of distributions on the 2‑sphere (S^2):

* **HypersphericalUniform** → area‑uniform sampling.
* **PowerSpherical** → concentrated around a mean direction.

Both live on the unit sphere in $\mathbb{R}^3$.

### 2.1 Hyperspherical uniform

Conceptually:

* Sample $z \sim \mathcal{N}(0, I_3)$.
* Set $x = z / |z|$.

Because the multivariate Gaussian is rotationally symmetric, normalising gives a uniform distribution on the sphere – all directions are equally likely and the density is constant with respect to surface area.

Geometrically:

* Every direction is equally probable.
* There is no privileged direction, so **before filtering**, your candidates have isotropic distribution around the last pose.

In practice, the effective distribution is **uniform on the allowed cap**, because after sampling you:

1. Rotate to world.
2. Throw away directions with elevation outside $[e_{\min}, e_{\max}]$.
3. Throw away directions with yaw deviation outside $[-\Delta\psi/2, +\Delta\psi/2]$.

So the actual distribution is:

> HypersphericalUniform **truncated** to a spherical cap (elevation band) and a yaw band around forward.

### 2.2 PowerSpherical (forward‑biased)

The PowerSpherical distribution is a distribution on the sphere with parameters:

* a **mean direction** $\mu \in S^2$, and
* a **concentration parameter** $\kappa \ge 0$.

Density is a function of the dot product $\mu^\top x$:

$$
p(x \mid \mu, \kappa) = C(\kappa) , g(\mu^\top x, \kappa),
$$

for some monotone function (g). The exact form isn’t important; the key properties are:

* $\mu$ sets the direction around which mass is concentrated.
* Larger $\kappa$ → more mass near $\mu$, less in the tails.
* $\kappa = 0$ → uniform on the sphere.

Compared to the classic von Mises–Fisher distribution, PowerSpherical usually has **heavier tails** for the same concentration – it doesn’t completely annihilate off‑axis directions. That’s good if you want a forward bias but still want occasional “off‑forward” samples to maintain diversity.

In your use:

* $\mu = (0,0,1)$ in rig frame (camera forward).
* So directions are biased towards “straight ahead” in the rig frame.
* After rotation into world frame by $R_{wc}^{\text{last}}$, the bias aligns with the last pose’s forward axis.

Again, after you throw away directions outside the allowed elevation/yaw band, you effectively have a **truncated, forward‑biased distribution**.

### 2.3 Elevation and delta‑azimuth filters

This step is important to understand the shape of the final distribution.

Let:

* $u = \text{world\_up} \in S^2$.
* $f$ = last forward direction in world frame (the image of $(0,0,1)$ through $R_{wc}^{\text{last}}$).

Given a world‑direction (d), you compute:

1. **Elevation** (pitch) w.r.t. horizontal plane:

  * vertical component $h = u^\top d$,
  * elevation $\theta = \arcsin(h)$.

  So $\theta = 0$ means perfectly horizontal, positive = up, negative = down.

   You keep directions with:

   $$
   \theta \in [\theta_{\min}, \theta_{\max}].
   $$

2. **Yaw** w.r.t. last forward direction in the horizontal plane:

   * project both vectors into the horizontal plane:

     $$
     d_h = d - (u^\top d) u,\quad
     f_h = f - (u^\top f) u,
     $$

   * normalise,

  * compute signed angle between $f_h$ and $d_h$ using atan2 on sine and cosine.

   You keep directions with:

   $$
   |\psi| \le \Delta\psi/2,
   $$
  where $\Delta\psi = \text{delta\_azimuth\_deg}$.

So the final (direction) distribution, after all filters, is:

$$
q_d(d) \propto
\underbrace{p_{\text{Hyperspherical or PowerSpherical}}(d_{\text{rig}})}*{\text{initial sampling}}
\cdot \underbrace{\mathbb{1}*{\theta_{\min} \le \theta(d) \le \theta_{\max}}}*{\text{elevation band}}
\cdot \underbrace{\mathbb{1}*{|\psi(d)| \le \Delta\psi/2}}_{\text{yaw band}}.
$$

That’s why:

* `delta_azimuth = 360°` → no yaw restriction; only elevation restricts you.
* `delta_azimuth = 90°` → you get a wedge ±45° around forward in the horizontal plane.
* `delta_azimuth = 0°` → everything lies in the plane spanned by $f$ and $u$ (a single great circle).

Net effect:

* HypersphericalUniform gives a *uniform* distribution in that wedge.
* PowerSpherical gives a *forward‑peaked* distribution in that wedge, controlled by $\kappa$.

This is exactly the behaviour you want *if* you think “good” NBV moves mostly live in a truncated forward cone.

---

## 3. Orientation: “look away from last pose” with zero roll

Once you have the world‑space candidate centers $p_i'$, you construct orientations via `view_axes_from_poses(last_pose, candidate_pose, look_away=True)`.

Let:

* $c_{\text{last}} = t_w^{\text{last}}$,
* $c_i = p_i'$,
* world up $u$.

The helper:

1. Defines the **forward axis** (z_i) as the **normalised displacement**:

   $$
   z_i = \frac{c_i - c_{\text{last}}}{|c_i - c_{\text{last}}|}.
   $$

   This is the “look‑away” behaviour: each camera looks in the direction of travel, away from where it came from.

2. Defines the **horizontal left axis** $x_i$ as the cross product of world‑up with forward:

   $$
   x_i \propto u \times z_i,
   $$
   normalised.

   This guarantees:

  * $x_i \perp u$ → x is horizontal,
  * as long as $z_i$ is not parallel to $u$.

3. Defines the **up axis** $y_i$ as

   $$
   y_i = z_i \times x_i.
   $$

   This ensures ${x_i, y_i, z_i}$ is an orthonormal right‑handed frame – LUF.

So the orientation is $[x_i, y_i, z_i]$ as columns in $R_{wc}'$.

Properties worth noting:

* **Zero roll**: if you think of roll as rotation around $z_i$, this construction pins roll by demanding that “up is as aligned as possible with world up”. There’s no free yaw around the view direction.
* **Deterministic**: orientation is a deterministic function of position and last pose; you’re not sampling rotations separately.

This is why all your candidates tend to have very similar roll and similar “feeling” of orientation: all the stochasticity is in where the camera is placed, not how it’s rotated around the view axis.

---

## 4. Interpreting the whole candidate distribution

Putting it together, you’re effectively imposing:

$$
p(T_{wc}' \mid s) \approx
p_{\text{pos}}(p' \mid s) ,
\delta(R_{wc}' - f(p', T_{wc}^{\text{last}}, u)),
$$

where:

* $p_{\text{pos}}$ is induced by:

  * PowerSpherical/Hyperspherical sampling in rig frame,
  * rotation to world,
  * elevation / yaw truncation,
  * uniform radius sampling,
* $f$ is the deterministic mapping from positions to orientations.

So:

* Diversity comes from the variety of **directions and radii** you sample.
* “Realism” (in the sense of mechanical plausibility) is imposed by:

  * range constraints (radii),
  * elevation/yaw constraints (you don’t move 180° behind or straight up),
  * path and mesh rules (no collisions, keep free space).

It’s roughly “a truncated, slightly forward‑biased annulus”.

Is this *optimal*? Depends what “optimal” means:

* For an NBV **ranking** network, you want candidate proposals that:

  * contain **all high‑quality moves** with decent probability,
  * contain enough **lower‑quality but plausible moves** so the model sees counterexamples,
  * don’t waste too many samples on geometrically impossible stuff.

The current distribution is a reasonable hand‑crafted prior, but:

* it assumes that “good” NBVs lie in a forward cone at moderate elevation,
* it forces the camera always to look exactly in the direction of motion,
* it has no notion of scene layout beyond the coarse rules.

That’s fine as a starting point, but you can do better.

---

## 5. How HypersphericalUniform and PowerSpherical actually behave (intuition)

### 5.1 HypersphericalUniform on S²

Key facts:

* The distribution is invariant under every rotation; any patch of the sphere with the same area has the same probability.
* If you look at the distribution of elevation:

  * $\theta$ (angle from north pole) has density $\tfrac{1}{2}\sin\theta$ on $[0,\pi]$.
  * Equivalently, the latitude $\varphi = \pi/2 - \theta$ doesn’t have a uniform density; more samples around the equator than near the poles, consistent with equal area.

In your truncated setting, you’re discarding most of the sphere anyway, so making it globally uniform doesn’t buy much; it just means you’re not biasing any direction within that cap.

### 5.2 PowerSpherical on S²

Intuition:

* Let $\mu$ be the forward direction $(0,0,1)$ in rig frame.
* Let $\alpha = \arccos(\mu^\top x)$ be the angular distance between $x$ and $\mu$.

Then:

* Small $\kappa$ → broad angular distribution; many samples at moderate angles, some far off.
* Large $\kappa$ → samples concentrated at small $\alpha$. For very large $\kappa$, you’re basically sampling near a narrow forward cone.

For NBV this is quite natural:

* You want a **forward bias** because many good viewpoints for on‑the‑move scanning are not 180° behind you.
* You don’t want to completely kill side views; sometimes the best view to resolve an object is ~90° off your motion direction.

HypersphericalUniform is the special κ→0 case: everything is equally likely.

Whether PowerSpherical is “optimal” is debatable. It’s a simple, isotropic distribution around a mean direction; it ignores anisotropy of the environment. But as a base prior to be modulated by geometric constraints, it’s fine.

---

## 6. Is “look away from last pose” a good orientation policy?

Right now, you:

* choose positions $p'$ on the shell,
* set forward axis $z_c'$ to point from last pose to $p'$.

That means:

* Your motion vector and gaze direction are aligned.
* You never turn your head independently of where you step.

Pros:

* It’s simple, and it matches a “walk and look where you’re going” behaviour.
* It encourages exploration: you rarely look back at already‑seen surfaces.

Cons:

* It hard‑codes a very specific scanning style.
* You **never** generate candidates where you step sideways but keep looking at the same object (which is important for multi‑view reconstruction).
* Because orientation is deterministic given position, the variety of *viewing directions* for a given candidate radius / yaw band is limited.

If the goal is “generate a variety of realistic candidate views”, this is too rigid.

---

## 7. What might be better?

### 7.1 Decouple position and orientation

Right now:

* direction sampling → position,
* orientation is a deterministic function of position + last pose.

Alternative: **two‑stage sampling**:

1. Sample **motion direction** (where to move) – basically what you do now.
2. Sample **view direction** relative to motion and/or last gaze.

For example:

  * Let $v_{\text{move}} = p' - c_{\text{last}}$ (direction of travel).
  * Let $f_{\text{last}}$ be last forward axis.
  * Sample a **view direction** $d_{\text{view}}$ on the sphere using a distribution that depends on both:

  $$
  d_{\text{view}} \sim p(d \mid v_{\text{move}}, f_{\text{last}}).
  $$

Concrete choices:

* A PowerSpherical distribution centerd on $v_{\text{move}}$, with $\kappa$ controlling how tightly you look along your step.
* A mixture: one component around $v_{\text{move}}$ (explore ahead), one around $f_{\text{last}}$ (keep tracking object).
* Optional constraint: bound the change in viewing direction $\angle(f_{\text{last}}, d_{\text{view}})$ to be $\le$ some $\Delta$, matching handheld ergonomics.

Then build orientation with:

* forward axis $z_c' = d_{\text{view}}$,
* x/y from world up as before.

This immediately gives you more variety:

* step left but still look at the sofa,
* step forward and slightly look back for parallax,
* etc.

All remain “realistic” if you bound the view‑change angle.

### 7.2 Use distributions on **relative angles**, not absolute directions

Instead of sampling on the full sphere and then clamping, you can work directly in angular coordinates:

* yaw offset $\Delta\psi \sim \text{some distribution on} [-\psi_{\max}, \psi_{\max}]$,
* pitch offset $\Delta\theta \sim \text{some distribution on } [\theta_{\min}, \theta_{\max}]$.

Examples:

* Uniform in both → pure coverage of cone.
* Cosine‑shaped / Beta in yaw (more likely near forward, fewer extreme side looks).
* Truncated normal in pitch (more likely around horizontal gaze).

This decomposes the power‑spherical behaviour into something easier to tune:

* “I want 70% of views within ±30° horizontally, 20% between 30° and 60°, 10% beyond.”

You can still implement this using PowerSpherical if you like (because PowerSpherical + rejection ≈ some angular law), but writing it explicitly in angular terms gives you better intuition and easier calibration.

### 7.3 Environment‑aware proposals

Right now all structure comes from:

* last pose,
* static angular bounds,
* mesh/free‑space rules.

More realistic NBV proposals would:

* Aim view directions towards **informative parts of the scene**:

  * near the frontier of the current map,
  * near high depth‑uncertainty regions,
  * near object OBBs with poor coverage.

You can encode this as an additive bias on your angular distribution, e.g.:

* Use PowerSpherical centerd at a direction predicted by some heuristic or small network $\hat{d}(s)$.
* Or use a **mixture** with:

  * one component around motion,
  * one component around “interesting region” direction.

This is on the path towards a learned proposal distribution.

### 7.4 Motion realism constraints

You can also filter candidate orientations using a **motion model**:

* Limit the change in orientation between steps:

  * yaw change ≤ 60°, pitch change ≤ 30° per step for a handheld.
* Reject candidates that would require a 180° spin while moving forward half a metre.

The orientation construction via `view_axes_from_poses` gives you a canonical orientation; you could then **perturb it**:

* sample a small roll‑free yaw/pitch offset around that canonical view, within physiological bounds.

This preserves the “look roughly where you’re going” structure but yields a richer variety of views.

### 7.5 Radius distribution

Right now you sample radius uniformly in $[r_{\min}, r_{\max}]$.

For handheld, more realistic might be:

* A Beta‑shaped distribution peaked around a comfortable step size (say 0.5–0.8 m),
* With short steps and long strides less frequent.

That’s cheap and changes the radial density to better match actual AR usage, which should also improve the match between training distribution and deployment.

---

## 8. Putting this into a concrete recommendation

If I had to propose a “v2” candidate generator that aligns with your goals (“diverse but realistic”), I’d do:

1. **Position sampling**:

   * Keep the current shell (min/max radius).
   * Sample motion direction using:

     * PowerSpherical around last forward axis with κ moderate (say 5–10),
     * Elevation band + delta_azimuth as now, but calibrated from real trajectories.
   * Use a **Beta** radius distribution centerd around typical step size.

2. **Orientation sampling** (decoupled):

   For each candidate position:

   * Compute motion direction $v_{\text{move}}$.

   * Compute last forward $f_{\text{last}}$.

   * Define a **mixture**:

     * Component A (explore): PowerSpherical around $v_{\text{move}}$.
     * Component B (revisit): PowerSpherical around $f_{\text{last}}$.

   * Sample one component with probabilities $p_A, p_B$ (tunable).

   * From the drawn component, sample view direction $d_{\text{view}}$.

   * Build orientation:

     * forward $z_c' = d_{\text{view}}$,
     * x/y via world up as now.

  * Reject if $\angle(f_{\text{last}}, d_{\text{view}})$ exceeds a human‑plausible bound.

3. **Rules and diagnostics**:

   * Keep MinDistanceToMeshRule / PathCollisionRule / FreeSpaceRule.
   * Make MinDistanceToMeshRule always compute distances, but only apply the hard threshold if `min_distance_to_mesh > 0`:

     * store `dist_i` in `extras` for analysis.
   * Use `collect_rule_masks` during dev to understand how much each rule prunes.

This would give you:

* A **broad but controlled** set of candidate positions,
* A **richer variety of gaze directions** (side looks, partial look‑backs),
* Realistic per‑step motion and gaze deltas,
* And still a very clean, modular rule‑based architecture.

---

## 9. Summary

* HypersphericalUniform gives you isotropic directions; PowerSpherical adds a tunable forward bias.
* After rotation to world and elevation / yaw truncation, you effectively have a truncated cone distribution around the last forward axis.
* The current orientation policy (look away along displacement, zero roll) is geometrically clean but too rigid: orientation is deterministic given position and last pose.
* For NBV, you want a distribution over poses that:

  * covers good candidates (forward exploration + multi‑view baselines),
  * remains physically plausible,
  * and provides enough variation for the ranking network.

The main structural improvement is to **decouple position and orientation sampling**, using spherical distributions for both, and to parameterise things in terms that correspond more directly to human motion: bounds on step size and bounds on gaze change. Then the rule system stays basically the same, but the candidate cloud becomes a much more expressive approximation of “all realistic things a camera could do next.”
