# RFC 015: garust-dyn — Physics Backend Roadmap, adjusted to what is built

**Author:** garust maintainers
**Status:** Draft / Request for Comments
**Target:** `garust-physics` (evolving into the backend this RFC calls `garust-dyn`), plus two new crates
**Supersedes the milestone lists of:** RFC-010 §4 (gaps to build), RFC-013 §2 (goals) — their designs stand; their to-do lists are folded in here.
**Source:** the 2026-10-23 roadmap "garust-dyn: Physics Backend Roadmap", rebased on `main` at 9b43a15.

## Goal and scope

Build a rigid-body physics backend written in PGA on top of garust, good enough for robotics work on arms and mobile bases, and validated claim by claim the way physics-lab validates its lessons. MuJoCo stays in the stack permanently as the oracle and as the fallback for scenes the engine does not yet handle well.

In scope: free rigid bodies, articulated chains with revolute and prismatic joints, convex collision, contact with Coulomb friction, URDF and MJCF loading, actuator and sensor models, and sim-to-real tooling.

Out of scope: deformables, cloth, fluids, GPU-parallel RL at thousands of environments, and photoreal rendering. Each is its own project; the trait boundary in the architecture section leaves room for them later.

**The claim discipline**, adapted from physics-lab. physics-lab's rule is that nothing is stepped, only closed forms. A simulator steps, so claims come in three tiers instead of one, and every milestone ships only when its claims pass:

| Tier | Form of the claim | Example |
| --- | --- | --- |
| Closed form | the simulation equals an analytic solution within a stated tolerance | pendulum period, Dzhanibekov flip period, incline slip angle |
| Invariant | a conserved quantity drifts less than a bound, or error falls at the integrator's order as dt halves | energy, angular momentum, convergence order |
| Oracle | an independent simulator agrees on the same scene | inverse dynamics vs Pinocchio, trajectories vs MuJoCo |

Each claim gets a name (`M1-C3`), a test, and where it teaches something, a lesson page in physics-lab. Every tolerance in a claim is a named constant with a comment saying what it protects against (AGENTS.md G5); golden fixtures are trig-free.

## What changed against the original roadmap

The original listed five M0 additions and three new crates. Against `main`:

- **Three of the five M0 items already exist**: `Multivector::log` in closed form via the invariant decomposition (`try_bivector_split`, with the #70/#71 threshold fixes), `Motor::slerp` (plus `slerp_unwrapped`, `bezier`, `frechet_mean`, `geodesic_distance`), and `Multivector::commutator`. Two remain: the 6-component bivector newtype and `Motor::sqrt`.
- **Specialized kernels are half there**: `apply_point_fast`, `to_matrix`, `apply_each_simd`. Motor·line and motor·motor fast paths are missing.
- **`garust-physics` already exists** (RFC-010 slices 1–3): 6-DOF `RigidBody` with a symplectic Lie-group splitting integrator, `Inertia`, sphere/plane/AABB contacts with restitution and Coulomb friction, `World::step` with `Hinge`/`Fixed`/`Prismatic` joints (positional part, Baumgarte), and quasi-static inverse dynamics on a `Tree` (`multibody.rs`, `load.rs`). This RFC keeps the crate and grows it; the name `garust-dyn` in the original is read as "what `garust-physics` becomes". Renaming is a separate decision (see open questions).
- **Kinematics exists in `garust-geo`** (RFC-013): `Chain` (FK, damped-least-squares IK), `Tree` (parent array, FK), `twolink` (closed-form two-link IK).
- **Autodiff for M5 exists in `garust-core`**: forward-mode `Dual<T>` and reverse-mode `Var`. M5 inherits it instead of building it.
- **Symplectic integrators on flat phase space exist** (`Phase::leapfrog`, `ExtendedPhase::tao_step` in `garust-core::dynamics`); they are not the Lie-group integrator the body uses, but they are the reference for the convergence-order claims.

Everything below lists, per milestone, what is built, then only the remaining work and its claims. A claim marked **(have)** already has a test on `main`; it is kept so the claim table stays complete and is renamed to the `M*-C*` scheme when the milestone ships.

## Architecture

Three crates above garust (one existing, two new), and one trait that lets this engine and MuJoCo be swapped or run side by side on the same scene.

```
garust-core / garust-geo   existing: Multivector, Motor, exp/log, commutator, Chain, Tree, Dual, Var, no_std
garust-physics             existing, grows: bodies, inertia, integrators, joints, articulated dynamics, contact solver
garust-collide             new: broadphase, narrowphase primitives, GJK/EPA, contact manifolds
                           (seeded by moving contact.rs's Sphere/Aabb/plane queries out of garust-physics)
garust-robot               new: URDF/MJCF import, actuators, sensors, controller tick, PhysicsBackend impls
physics-lab                lessons (wasm) that state and test the claims; a native lab app later
```

```rust
pub trait PhysicsBackend {
    fn load(&mut self, scene: &Scene) -> Result<(), LoadError>;   // Scene built from MJCF or URDF
    fn set_controls(&mut self, u: &[f64]);
    fn step(&mut self, dt: f64);
    fn state(&self) -> SimState;                                 // q, qdot, body motors, contacts, sensors
}

pub struct GarustDyn { /* garust-physics World + Tree */ }
#[cfg(feature = "mujoco")] pub struct MujocoBackend { /* via the MuJoCo C API */ }
```

The scene format is MJCF from the start, so every scene loads into both backends unchanged; URDF is imported by converting to the same internal `Scene`. `SimState` exposes body poses as garust `Motor`s regardless of backend, so everything above the trait (sensors, controllers, the cone-display camera model, lessons) is written once.

MuJoCo is Apache-2.0 licensed and ships a C library, so the binding is a thin `-sys` crate. It sits behind a feature flag in `garust-robot` and is never a dependency of `garust-physics` itself, keeping the core zero-dependency like garust (G8).

`garust-physics` keeps its allocation-free, `no_std` discipline: `World` holds settings only, bodies and joints are caller-owned slices. `garust-collide` follows the same rule; `garust-robot` is the first crate allowed `alloc` (scene graphs from files are variable-size), behind the usual feature gate.

## M0: what garust needs first

**Built**: `Multivector::log` / `exp` / `commutator`, `Motor::slerp` family, `apply_point_fast`, `to_matrix`, `apply_each_simd`.

Remaining, all in garust itself (the cone display and the game scripting work need them too):

- [ ] A **`Bivector` (or `Line`) newtype** over the six grade-2 components of `Pga3`, so velocities, momenta and forques are typed and the sparse product paths are used automatically. `RigidBody` currently carries `angular_momentum` as a full `Pga3`; the newtype replaces that field. Same layout rule as `Motor` (INV-2 in `docs/invariants.md`).
- [ ] **`Motor::sqrt`**, a thin wrapper over `log`/`exp` (`exp(log(M)/2)`), with the half-turn case pinned by a test.
- [ ] **Motor·line and motor·motor hand-expanded kernels**, the remaining klein-style paths. The README benchmark shows 74 ns per generic sandwich versus 1.5 ns for a matrix; dynamics does many of these per body per step.
- [ ] **README fix**: the implemented list says the regressive product `∨` is the *meet*. In plane-based PGA `∧` is the meet (the README's own plane example intersects three planes with `wedge`) and `∨` is the join (`line_through`). Fix before physics code and lesson pages quote it.

Claims:

| Claim | Tier | Status | Statement |
| --- | --- | --- | --- |
| M0-C1 | invariant | have (unit + proptest in `transform.rs`, `motor.rs`); extend to 100k random motors behind `--ignored` | `exp(log(M)) == M` to 1e-12 (f64), including pure translations and half-turns |
| M0-C2 | invariant | have (`motor.rs` slerp tests) | `slerp(M, N, 0) == M`, `slerp(M, N, 1) == N`, constant angular and linear speed along the path |
| M0-C3 | closed form | have (`products.rs`); add the sign-pinning case against the 3D cross product | `commutator` of two bivectors through the origin equals the cross product of their axis vectors, sign pinned once by the basis convention |
| M0-C4 | invariant | have for point (`apply_point_fast`, `to_matrix`, SIMD); add line and motor | each specialized kernel equals the generic sandwich to 1e-12 on random inputs |
| M0-C5 | invariant | new | `sqrt(M) * sqrt(M) == M` as motions to 1e-12, including the half-turn |

## M1: free rigid body

**Built** (`garust-physics/src/lib.rs`): `RigidBody` with pose as `Motor` and angular momentum as a bivector; `Inertia::principal`; Euler's equations in bivector form; a **symplectic Lie-group splitting integrator** (exact rotation per principal axis conserving `‖Π‖`, orientation transported on the group by `exp`, linear leapfrog on the centre of mass); gravity, a constant world force at the centre of mass, a body-frame torque. Tests on `main`: energy and `‖Π‖` conservation (unit + proptest), world angular momentum constant, parabola under gravity, Dzhanibekov flip *occurs*.

This is further than the original roadmap's baseline (its integrator #2, not #1). The remaining work is the general-position formulation and the integrator ladder that the convergence claims need.

Remaining:

1. [ ] **Single-motor step** `dM/dt = -½ M B` with `B` the full velocity bivector and `I` a bivector→bivector map, replacing the rotation/translation split. The split is exact only for forces at the centre of mass; off-centre forces and the articulated algorithms in M2 need the coupled form (De Keninck, "May the Forque be with you").
2. [ ] **Point force at a body point** converted to a forque by joining point and direction; linear and angular damping.
3. [ ] **RKMK4** (fourth-order Runge–Kutta–Munthe-Kaas in the Lie algebra), for the convergence-order claim and long free-flight scenes. The existing splitting integrator stays as the symplectic option.
4. [ ] A variational integrator later, only if long-horizon energy behaviour matters more than accuracy per step.

Claims:

| Claim | Tier | Status | Statement |
| --- | --- | --- | --- |
| M1-C1 | closed form | have (`constant_force_accelerates_the_centre_of_mass`); tighten to 1e-10 and any dt | under gravity alone, the centre of mass follows the exact parabola with the Lie integrator |
| M1-C2 | invariant | have at 1e-6 over short runs; add the 1e5-step `--ignored` run | torque-free: kinetic energy drifts under 1e-6 relative over 1e5 steps (dt = 1 ms) |
| M1-C3 | invariant | have | torque-free: world-frame angular momentum constant to the same bound; this pins the commutator sign |
| M1-C4 | closed form | new (flip exists; period is not checked) | Dzhanibekov: spin near the intermediate axis flips with the period given by the Jacobi elliptic solution of Euler's equations, to 0.1% |
| M1-C5 | closed form | new | spin about the major or minor axis stays stable: perturbation bounded for 1e5 steps |
| M1-C6 | invariant | new | halving dt reduces the error by about 2^p, p = 1 for Euler, 2 for the splitting integrator, 4 for RKMK4 |

Lesson page: "The tennis racket", the Dzhanibekov flip derived, then shown, with a control that changes the moments of inertia and the closed-form period on screen next to the stepped one.

## M2: joints and articulated chains

**Built**: in `garust-geo`, `Chain` (screw-axis joints `Revolute`/`Prismatic`, FK by motor product, `ik_dls`), `Tree` (parent array, FK, `is_distal_to`), `twolink` (closed-form two-link IK). In `garust-physics`, `World` joints `Hinge`/`Fixed`/`Prismatic` as positional constraints with Baumgarte stabilisation (rotational part stored, not enforced), and **quasi-static inverse dynamics**: `joint_torque`, `generalized_torque` (geometric and variational, cross-checked), `load_work`, external `Load`s. Tests: small-angle pendulum period to 2% via the constraint solver, FK tree≡chain, geometric≡variational torque.

What exists is statics on a tree plus a constraint-based pendulum; the recursive dynamics algorithms are all missing.

Remaining:

1. [ ] **RNEA** (inverse dynamics with accelerations): given `q, q̇, q̈`, return joint torques. `multibody.rs` is its quasi-static limit (`q̇ = q̈ = 0`), which becomes a regression test. Deterministic, so the best first oracle target.
2. [ ] **CRBA**: the joint-space mass matrix, O(n²). For checking and for implicit integration.
3. [ ] **ABA** (forward dynamics): given torques, return accelerations. The one the simulator steps with; replaces the Baumgarte joint path in `World` for tree-structured scenes.
4. [ ] **Floating base**: full motor + bivector state from M1 as the root of a `Tree`.
5. [ ] Joint limits and joint damping as soft constraints; the rotational part of `Hinge`/`Prismatic` enforced. Closed kinematic loops deferred to M3's constraint solver. Spherical joints later (need a rotor-valued `q`).

Joint table (unchanged from the original; `Chain`'s `Revolute`/`Prismatic` already follow it):

| Joint | Motor as a function of q | Status |
| --- | --- | --- |
| Revolute | `exp(-q/2 · L)`, L a normalized Euclidean line | have (`ChainJoint::Revolute`) |
| Prismatic | `exp(-q/2 · L)`, L an ideal line | have (`ChainJoint::Prismatic`) |
| Fixed | constant motor | have (`Link.offset`) |
| Floating base | full motor + bivector state from M1 | new |
| Spherical | rotor-valued q | later |

Claims:

| Claim | Tier | Status | Statement |
| --- | --- | --- | --- |
| M2-C1 | closed form | have at 2% via constraints; re-pin at 0.1% on ABA | simple pendulum, small angle: period τ·√(l/g) |
| M2-C2 | closed form | new | simple pendulum, large angle: period from the complete elliptic integral of the first kind to 0.1% |
| M2-C3 | invariant | new | double pendulum, no damping: energy drift under 1e-5 relative over 60 s at dt = 1 ms |
| M2-C4 | invariant | new | closure: `ABA(q, q̇, RNEA(q, q̇, q̈)) == q̈` to 1e-10 on random states |
| M2-C5 | invariant | new | CRBA mass matrix symmetric positive definite and equal to the RNEA-derived one column by column |
| M2-C6 | oracle | new | RNEA torques match Pinocchio (and MuJoCo inverse dynamics) to 1e-9 on a 7-DoF arm URDF, 10k random states |
| M2-C7 | oracle | new | forward-simulated joint trajectories under identical torques within 1e-6 rad of MuJoCo for 1 s, no contact |
| M2-C8 | invariant | have (`multibody.rs`) | RNEA at `q̇ = q̈ = 0` equals the quasi-static joint torque |

Lesson pages: the large-angle pendulum (stepped period next to the elliptic integral), and the double pendulum (energy readout, integrator switch).

## M3: contacts and friction

**Built** (`garust-physics/src/contact.rs`, `world.rs`): `Sphere` vs sphere / plane, `Aabb` vs AABB as a broadphase bound, `point_plane_distance`, `line_line_closest` (PGA common perpendicular), impulse response with restitution and Coulomb friction (`resolve_pair`, `resolve_static`), `World::step` as `integrate → detect → resolve → constrain` with ground plane. Tests: momentum conservation head-on, KE conserved when elastic, a ball bounces and never tunnels, perpendicular escape and recentering cases.

That is a sphere-only M3a without a solver: each contact is resolved sequentially with no persistence, no manifold and no stacking. The split M3a/M3b stands; M3a ships before M3b starts.

Remaining:

**Collision detection** (new crate `garust-collide`, seeded from `contact.rs`):

1. [ ] Move `Sphere`, `Aabb`, the plane/line queries out of `garust-physics` into `garust-collide`; `garust-physics` depends on it.
2. [ ] Broadphase: sweep-and-prune over the existing `Aabb`; a BVH later if scenes grow past a few hundred bodies.
3. [ ] Narrowphase closed forms for capsule and box, and all pairs among sphere, capsule, box, plane.
4. [ ] Convex hulls through GJK (distance) and EPA (penetration depth), for M3b.
5. [ ] Contact manifolds: up to four persistent points per pair, warm-started across steps.

**Contact solver** (`garust-physics`), a deliberate choice:

| Option | Strengths | Weaknesses | When |
| --- | --- | --- | --- |
| XPBD (Müller et al. 2020) | one framework for contacts, friction and joint limits; stable at large steps; simple | stiffness depends on iteration count; less exact than velocity-level | M3a, likely enough for mobile robots and coarse manipulation |
| Soft convex contact (MuJoCo's formulation) | smooth, well-posed, matches the oracle by construction | a convex optimizer per step | only if M3b oracle claims fail with XPBD |

The current sequential impulse code is the M3a baseline the XPBD solver is compared against, then retired.

Friction: Coulomb with separate static and kinetic coefficients (one `mu` today), and a pyramidal or elliptic friction cone. Restitution per pair (per body today).

Claims:

| Claim | Tier | Status | Statement |
| --- | --- | --- | --- |
| M3-C1 | closed form | new | box on an incline stays put below arctan(μₛ) and slides above it with acceleration g(sin θ − μₖ cos θ), to 1% |
| M3-C2 | closed form | have qualitatively (bounce, no tunnelling); pin the e² height | ball dropped with restitution e rebounds to e² times the drop height, to 1% |
| M3-C3 | invariant | new | stack of 10 boxes rests with total drift under 1 mm over 10 s |
| M3-C4 | invariant | have (`contact.rs`) | head-on collisions conserve linear momentum to 1e-9; energy loss matches the restitution coefficient |
| M3-C5 | invariant | new | sphere rolling without slipping keeps v = ωr to 1e-6 |
| M3-C6 | oracle | new | drop, slide and topple scenes: final poses within a tolerance band of MuJoCo across 100 randomized runs |
| M3-C7 | oracle | new (M3b) | a parallel gripper lifts and holds a box in both engines; slip under 1 mm over 5 s |

Lesson pages: the incline, the bouncing ball, Newton's cradle.

## M4: robot loading, actuators, sensors

**Built**: nothing in this milestone. `garust-anim` (RFC-012) has a scene/camera model the camera sensor can reuse, and `tracking-ghost-pepper` owns the cone-display ray pipeline.

All of the original M4 remains, in the new crate `garust-robot`:

- [ ] MJCF subset first (bodies, joints, inertials, geoms, actuators, sensors), so every scene runs in both backends unchanged.
- [ ] URDF import converted to the same internal `Scene`. Start with a common 6-DoF or 7-DoF arm description and a simple differential-drive base.
- [ ] Actuators: torque source, PD position servo, DC motor + gearbox, rotor inertia (armature), joint friction (Coulomb + viscous, later Stribeck), command latency.
- [ ] Sensors with noise, bias and quantization: joint encoders, IMU (gyro from the body velocity bivector, accelerometer from its derivative plus gravity, bias random walk), force/torque at a joint or wrist from the constraint forces, contact sensors, cameras from the cone-display ray pipeline via wgpu.
- [ ] Controller tick separate from the physics substep (e.g. 1 kHz controller over a 4 kHz physics step).
- [ ] `PhysicsBackend` trait, `GarustDyn` impl, `MujocoBackend` behind the `mujoco` feature.

Claims (all new):

| Claim | Tier | Statement |
| --- | --- | --- |
| M4-C1 | closed form | a PD servo on a single joint without gravity follows the second-order step response (overshoot, settling time) to 1% |
| M4-C2 | closed form | a stationary IMU reads gravity (9.81 m/s²) on the accelerometer and zero on the gyroscope, plus the configured noise statistics |
| M4-C3 | invariant | encoder readings are quantized at exactly the configured resolution |
| M4-C4 | oracle | the same MJCF arm under the same torque sequence matches MuJoCo joint trajectories within a band for 5 s, contact-free |
| M4-C5 | oracle | a URDF arm and its MJCF conversion produce identical RNEA torques to 1e-9 |

## M5: sim-to-real tooling

**Built**: the autodiff substrate. `garust-core` already has forward-mode `Dual<T>` and reverse-mode `Var`, and the kernel is generic over the scalar, so the M1–M4 pipeline runs under `Dual<f64>` with no code changes once every physics path is written against `T: Real` rather than `f64`. **That is the one prerequisite this milestone adds to earlier ones**: `garust-physics` is `f64`-only today (`RigidBody`, `World`, `contact.rs`); M1's single-motor step and M2's recursions are written scalar-generic from the start, and the existing `f64` code is generalised as it is touched.

Remaining:

1. [ ] System identification: log hardware (q, q̇, commanded and measured torques, at the controller rate); replay in simulation with `Dual` seeded on the unknown parameters; minimise trajectory error with Gauss–Newton or Levenberg–Marquardt; write fitted parameters back into the MJCF so the oracle uses them too. Contact-free motions first; contact parameters separately with a derivative-free search.
2. [ ] Domain randomization: per-scene parameter distributions sampled each episode.
3. [ ] Hardware in the loop: a `Robot` trait with simulated and real implementations, one binary drives both.
4. [ ] Replay and compare: overlay real and simulated traces per joint, with error statistics, as a physics-lab page.
5. [ ] Optional Python bindings (pyo3) exposing a Gymnasium-style environment.

Claims (all new):

| Claim | Tier | Statement |
| --- | --- | --- |
| M5-C1 | invariant | dual-number gradients match central finite differences to 1e-6 relative on contact-free trajectories |
| M5-C2 | closed form | system identification recovers known masses, friction and armature within 1% from synthetic logs with sensor noise |
| M5-C3 | oracle | after identification, simulated and real joint trajectories on held-out motions agree within a band set from the hardware's repeatability |

## Claims and oracle protocol

Every claim is a named Rust test in the crate it belongs to; oracle claims additionally have a Python twin in physics-lab's `checks/`, matching the lab's independent-implementation pattern, with the `mujoco` Python package as the oracle.

| Tier | Where | When in CI | Failure means |
| --- | --- | --- | --- |
| Closed form | `cargo test` in `garust-physics` | every push | a physics or sign bug |
| Invariant | `cargo test`, longer runs behind `--ignored` | every push (short), nightly (long) | integrator or numerical drift |
| Oracle | `checks/oracle/*.py` driving MuJoCo, plus `cargo test --features mujoco` in `garust-robot` | nightly | a model mismatch, or a real disagreement to investigate |

Oracle comparisons use the same MJCF file for both engines, a fixed seed, and a recorded tolerance per claim. For contact scenes the comparison is statistical, because two correct contact models need not produce identical trajectories.

A disagreement with the oracle is not automatically our bug. When a claim fails, the first step is a closed-form or invariant check that isolates which engine is wrong; each claim also has a "break it on purpose" test confirming it fails when the physics is deliberately broken (the pattern of #70 and #71).

## Order of work

M0 and M1 are small enough to run together; the first real PR series is M1's single-motor step, because M2's recursions and M5's generic scalar both depend on it.

1. M0: `Bivector` newtype, `Motor::sqrt`, line/motor kernels, README fix. One PR each.
2. M1: single-motor step over `T: Real`, point forces, RKMK4, convergence claim, Dzhanibekov period.
3. M2: RNEA → CRBA → ABA, floating base, Pinocchio oracle.
4. M3a: `garust-collide` split, sweep-and-prune, box/capsule narrowphase, XPBD, incline and stack claims.
5. M4: `garust-robot`, MJCF, `PhysicsBackend`, MuJoCo backend, actuators, sensors.
6. M3b: GJK/EPA, manifolds, gripper claim.
7. M5.

Performance target to check at the end of M2: a 7-DoF arm with ABA at 1 kHz needs each step under 1 ms on one core. A few hundred motor operations per step at 74 ns each lands well under that; measure before relying on it.

## Risks and open questions

The largest risk is M3: contact quality decides whether the engine is useful for manipulation, and it can absorb unbounded time. Mitigation: the M3a/M3b split and MuJoCo as a permanent fallback.

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Contact solver never matches the oracle on manipulation | engine limited to arms and mobile bases without grasping | XPBD first, soft convex contact only if needed; MuJoCo backend for grasping scenes |
| Sandwich cost (74 ns vs 1.5 ns matrix) | slow stepping for many bodies | M0 kernels; `to_matrix()` for bulk geometry; profile before optimizing |
| Sign and convention errors between M1 equations, joints and sensors | silently wrong physics | invariant claims catch sign errors; one conventions module shared by all crates |
| Generalising `garust-physics` from `f64` to `T: Real` late | M5 blocked on a rewrite | write M1/M2 generic from the start; generalise the rest as touched |
| Scope creep into deformables, fluids, GPU RL | milestones never close | out of scope by charter; revisit after M5 |
| physics-lab's "nothing stepped" rule | stepped lessons conflict with the site's stated method | decide below |

Open questions:

- [ ] Keep the crate name `garust-physics`, or rename to `garust-dyn` when `garust-collide` is split out? Renaming is cheap now and costly after physics-lab lessons import it.
- [ ] Does physics-lab relax its "nothing is stepped" rule site-wide, or do stepped lessons live in a separate section with the three-tier claims stated on each page?
- [ ] f64 everywhere, or f32 for the wasm lessons? f64 is the safe default for dynamics; `T: Real` keeps both open.
- [ ] Which robot is the reference platform for M4 and M5: a specific arm we own, or a common open description to start?
- [ ] Is a GPU path (many environments in parallel) ever a goal? If so, the data layout of the M1 single-motor step should be structure-of-arrays from the start.
