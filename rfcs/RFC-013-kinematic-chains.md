# RFC 013: Kinematic Chains, IK, and Motor Splines

**Author:** garust maintainers
**Status:** Draft / Request for Comments
**Target:** `garust` (Geometric Algebra in Rust)
**Builds on:** RFC-009 (PGA kernel), RFC-010 (physics), RFC-012 (anim)
**Numbering note:** outside the reserved R-0008→R-0011 program block.

## 1. Context and motivation

The robotics gap analysis after RFC-012 found that garust already holds
every *ingredient* of a robot kinematics library — motors, twists,
wrenches, `exp`/`log`, `commutator`, autodiff, `Prismatic`/`Hinge` joints
in the dynamics layer (#57) — but no *composition* of them: no chain
type, no Jacobian, no IK, and no path richer than two-pose `slerp`.
This RFC adds that composition layer to `garust-geo`, keeping the crate's
zero-dependency, `no_std`, allocation-free discipline.

The GA payoff is the same one RFC-010 claimed for dynamics: **one
representation end to end.** A joint is a screw axis (a PGA line), a link
offset is a motor, forward kinematics is a motor product, the pose error
that IK drives to zero is `log` of a motor — a bivector, i.e. a twist.
There is no quaternion/vector split anywhere, and therefore no
frame-convention seams for sign errors to hide in (the class of bug the
`pga_axis` fix in #49 just removed from the tests).

## 2. Goals and non-goals

**Goals**

1. `Motor` Bézier splines — smooth many-pose paths for robot trajectories
   *and* RFC-012 animation tracks, one primitive serving both.
2. A `Chain` type: links as motor offsets + screw-axis joints, FK by
   product, joint limits out of scope for v1.
3. Damped-least-squares IK on the motor manifold, with the Jacobian taken
   by central finite differences of the log-space error (convention-proof
   by construction).
4. Zero deps, `no_std`, no allocation: slices in, fixed-size stack
   scratch inside, documented caps.

**Non-goals (v1)**

- URDF or any file-format ingest (a downstream crate's job).
- Joint limits, self-collision, redundancy resolution beyond the damping
  term.
- Analytic geometric Jacobians — a follow-up once the numeric one has
  golden tests to validate against.
- Dynamics: `garust-physics` owns forces; this layer is kinematics only.

## 3. Design

### 3.1 Motor Bézier splines (`garust-geo::motor`)

```rust
impl<T: Real> Motor<T> {
    /// Evaluate the Bézier curve on the motor manifold defined by
    /// `ctrl` at parameter `t`, by de Casteljau over `slerp`.
    /// Endpoints are interpolated exactly. Panics if `ctrl` is empty
    /// or longer than 8 (fixed scratch; cubic is the working case).
    pub fn bezier(ctrl: &[Self], t: T) -> Self;
}
```

De Casteljau with `slerp` as the lerp generalizes Bézier to the group:
each round replaces adjacent pairs with their screw interpolant. The
result is C∞ in `t`, endpoint-exact, and — because every step is a
geodesic blend of unit versors — a unit motor at every `t`, with no
renormalization step. Scratch is a stack array `[Motor; 8]`.

### 3.2 Chain and FK (`garust-geo::chain`, new module)

```rust
/// One degree of freedom: how a joint variable becomes a motor.
pub enum ChainJoint {
    /// Rotation by `q` about a fixed line in the parent frame
    /// (unit Euclidean-weight PGA 2-blade, as `Motor::rotation_about`).
    Revolute(Pga3),
    /// Translation by `q` along a unit direction in the parent frame.
    Prismatic([f64; 3]),
}

/// A link: rigid offset from the parent, then the joint's motion.
pub struct Link { pub offset: Motor3, pub joint: ChainJoint }

/// An open kinematic chain over borrowed links (alloc-free).
pub struct Chain<'a> { links: &'a [Link] }

impl<'a> Chain<'a> {
    pub fn new(links: &'a [Link]) -> Self;          // caps len at 16
    /// Base-to-end-effector pose: Π offsetᵢ · jointᵢ(qᵢ).
    pub fn fk(&self, q: &[f64]) -> Motor3;
    /// Damped-least-squares IK from seed `q0` toward `target`.
    pub fn ik_dls(&self, target: &Motor3, q0: &[f64], out: &mut [f64],
                  params: IkParams) -> IkResult;
}

pub struct IkParams { pub damping: f64, pub tol: f64, pub max_iter: usize }
pub struct IkResult { pub converged: bool, pub iters: usize, pub err: f64 }
```

FK is one motor product per link — the composition operator the library
already proves associative and unit-norm-preserving in `tests/laws.rs`.
`f64`-only, matching the `frechet_mean` precedent (iterative numerics
pin the scalar type).

### 3.3 IK: damped least squares in log space

Error 6-vector: `e(q) = coords(log(fk(q)⁻¹ · target))` — the body-frame
twist that carries the current pose to the target, coordinates read off
the bivector blades `[e12, e13, e23, e01, e02, e03]`. Iterate:

```text
J        = ∂e/∂q         (6×N, central finite differences, h = 1e-6)
Δq       = −Jᵀ (J Jᵀ + λ² I₆)⁻¹ e
q        ← q + Δq        until ‖e‖ < tol or max_iter
```

The 6×6 solve is Gaussian elimination with partial pivoting on the
stack; the damping term `λ²I` keeps the system full-rank at
singularities (the straight-arm pose every 2-link reaches). Finite
differences rather than an analytic Jacobian is a deliberate v1 choice:
it is immune to the blade-ordering/sign conventions that produced the
`apply_point_fast` translation bug, and the analytic version can later
be validated *against* it. Caps: N ≤ 16 joints (stack scratch
`[[f64; 16]; 6]`).

### 3.4 What this unlocks downstream

- **RFC-012 A1**: `Track` can offer `Ease`-timed Bézier tracks through
  many keys, not just pairwise slerp spans.
- **goose-rover / goose-ferrum**: `no_std` FK + IK runs on the embedded
  boards as-is — the chain is `&[Link]` in flash, scratch on stack.
- **garust-physics**: a solved `q` from IK seeds joint targets for the
  sequential-impulse solver (#57's `Prismatic`, hinges).

## 4. Milestones

| # | Deliverable | Acceptance |
|---|---|---|
| R1 | `Motor::bezier` | endpoints exact (proptest); 2-point Bézier equals `slerp` (proptest); unit norm along the curve; midpoint of symmetric translation pair is the half translation |
| R2 | `ChainJoint`/`Link`/`Chain::fk` | 2-link planar arm at known angles hits the textbook end-effector pose; FK of all-zero `q` is the product of offsets; revolute+prismatic mixed chain sanity |
| R3 | `Chain::ik_dls` | 2-link arm reaches a reachable target from a generic seed, `fk(ik(target)) ≈ target` within tol (proptest over the reachable annulus, elbow-flip-agnostic via pose error, not `q` equality); singular straight-arm seed still converges (damping); unreachable target reports `converged: false` with best-effort `err` |

## 5. Open questions

1. Should `bezier` clamp `t` to `[0, 1]` or extrapolate like `slerp`
   does? Draft: extrapolate — consistency with `slerp` wins.
2. `ChainJoint::Prismatic` as `[f64; 3]` direction vs. an ideal PGA
   line: the array is friendlier at call sites; the line is more
   uniform. Draft: array, revisit if the analytic Jacobian wants the
   line form.
3. Cap sizes (8 control points, 16 joints): raise, or const-generic
   them later without breaking the slice API? Draft: document and hold.
