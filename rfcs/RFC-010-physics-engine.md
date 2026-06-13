# RFC 010: GA-Native Rigid-Body Physics Engine

**Author:** garust maintainers
**Status:** Draft / Request for Comments
**Target:** `garust` (Geometric Algebra in Rust)
**Builds on:** RFC-009 (PGA kernel `Cl(3,0,1)`)

## 1. Context and motivation

`garust` is meant to be a foundation for, among other things, a physics
engine. After the R-0009 work that foundation is unusually complete for this
purpose: rigid motions are first-class (`Motor`), the Lie bridge between the
motion group and its algebra is closed (`exp` / `log` / `slerp`), symplectic
integrators exist (`Phase::leapfrog`, `ExtendedPhase::tao_step`), and
incidence geometry (meet / join, `Cl(4,1,0)` spheres) is available for
contact. This RFC specifies how those pieces assemble into a rigid-body
engine, and pins the few primitives still missing.

The case for doing rigid-body dynamics in PGA rather than the usual
quaternion-plus-vector split:

- **One state, one update.** A body's pose is a single motor `M`; its
  velocity and momentum are single bivectors. Rotational and translational
  dynamics stop being two coupled subsystems and become one bivector
  equation — no quaternion/translation bookkeeping, no separate angular and
  linear integrators to keep in sync.
- **Structure preserved by construction.** Integrating on the motor group
  (via `exp`) keeps the pose a *unit* versor automatically — no
  renormalization drift, the GA analogue of quaternion normalization but
  exact in the algebra.
- **Contacts are incidence.** Lines and planes meet via the regressive
  product; the same operators that already do PGA incidence give closest
  features and separating planes.

## 2. Architecture

### 2.1 State

| Quantity        | GA object                         | garust type        |
|-----------------|-----------------------------------|--------------------|
| Pose            | motor `M ∈ Spin(3,0,1)`           | `Motor<f64>`       |
| Body velocity   | twist bivector `B` (grade 2)      | `Pga3` (grade 2)   |
| Momentum        | bivector `P` (grade 2)            | `Pga3` (grade 2)   |
| Applied wrench  | bivector `W` (force + torque)     | `Pga3` (grade 2)   |
| Inertia         | linear map `𝓘: grade-2 → grade-2` | **new** (§4)       |

The 6 rigid degrees of freedom live in the six grade-2 blades: the three
Euclidean bivectors (`e23, e31, e12`) carry rotation, the three ideal
bivectors (`e01, e02, e03`) carry translation. One object, both halves.

### 2.2 Equations of motion

The GA form of the rigid-body equations (Hestenes; Gunn; Hadfield &
Lasenby, *Rigid Body Dynamics in a Constrained Setting using GA*, 2019),
in the body frame:

```text
kinematics:   Ṁ = -½ · M · B            (B the body-frame twist)
momentum:     P = 𝓘(B)                   (inertia maps twist → momentum)
dynamics:     Ṗ = W + P × B              (× the bivector commutator ½(ab − ba))
```

`P × B` is the coadjoint ("gyroscopic") term — the single bivector commutator
that produces precession, the tennis-racket instability, and every other
coupling that the quaternion formulation splits across Euler's three scalar
equations.

### 2.3 Integration — the one real gap

`Phase::leapfrog` assumes a **flat** phase space: it advances position by
`q += v·dt`. A motor lives on a curved group; `M += Ṁ·dt` leaves the group
(the result is no longer a unit versor). The correct update transports along
the group with the exponential map:

```text
M ← M · exp(-½ · B · dt)
```

So the engine needs a **Lie-group symplectic integrator** — a variational /
Munthe-Kaas leapfrog that kicks momentum in the algebra and drifts pose by
`exp`, preserving both the group constraint and the symplectic structure
(no energy drift, exactly as flat leapfrog does for separable systems). All
the ingredients (`exp`, the bivector commutator, `Motor::compose`) exist;
the stepper itself is new (§4). This is the single most important deliverable
of this RFC.

The pose-update *substrate* already works today:

```rust
use garust::Pga3;

// One pose increment from a body twist: exponentiate the (half) twist to a
// motor and transport state by the sandwich. Here: spin by dθ about z.
let dtheta = 0.05_f64;
let increment = (Pga3::basis(0b011) * (-0.5 * dtheta)).exp(); // rotor for dθ in e12
let p = increment.sandwich(&Pga3::point(1.0, 0.0, 0.0));
// p has rotated to (cos dθ, sin dθ, 0) — the per-step pose update, exact.
```

### 2.4 Collision and contact

- **Flats (PGA).** Closest points / distances between lines and planes come
  from the regressive product and the commutator, already in `garust-core`.
  Good for box/polytope and joint geometry.
- **Rounds (CGA).** Spheres and their intersections need `Cl(4,1,0)`;
  `Cga3` + `Conformal` already model spheres and conformal transforms, so
  sphere–sphere and sphere–plane contact are a CGA-side add. A broadphase
  bounding-sphere test is the natural first use.

Contact resolution feeds an impulse (a wrench bivector) back into §2.2.

### 2.5 Constraints and joints

A joint constrains the relative motor between two bodies to a subgroup
(hinge = rotation about one axis, prismatic = translation along one, etc.).
Expressed as a constraint bivector whose value the solver drives to zero;
the Lagrange multiplier is, again, a wrench bivector. A sequential-impulse
or projected-Gauss-Seidel solver iterates these — standard engine
machinery, GA-flavored only in that the constraint algebra is bivector
arithmetic.

## 3. What's already in `garust`

| Need                              | Status | API                                            |
|-----------------------------------|--------|------------------------------------------------|
| Pose as motor                     | ✅     | `Motor` (compose, apply, inverse)              |
| Algebra ↔ group                   | ✅     | `Multivector::exp` / `log`, `Motor::log`       |
| Pose interpolation                | ✅     | `Motor::slerp`                                 |
| Twist / wrench as bivector        | ✅     | grade-2 `Pga3` (`.grade(2)`)                   |
| Bivector commutator `P × B`       | ◑      | `½(a*b − b*a)` from `*` — a named helper helps |
| Symplectic integration (flat)     | ✅     | `Phase::leapfrog`, `ExtendedPhase::tao_step`   |
| Incidence / closest features      | ✅     | `wedge`, `regressive`, complements             |
| Spheres / rounds                  | ✅     | `Cga3::sphere`, `Conformal`                    |

## 4. Gaps to build

Ordered by dependency:

1. **Inertia operator** — a grade-2 → grade-2 linear map type (six diagonal
   entries for a principal-axis body; general symmetric otherwise), with
   `apply` and `inverse`. Lives in `garust-core` (pure algebra) or the new
   physics crate (§5).
2. **Twist/wrench + commutator helpers** — named constructors for body
   twist and wrench bivectors, and the bivector commutator `×`, so call
   sites read as dynamics, not raw blade arithmetic.
3. **Lie-group symplectic integrator** — the `exp`-map leapfrog of §2.3, the
   keystone. Reference target: a free rigid body conserves energy and
   angular momentum over long runs (the validation gate, §6).
4. **Contact / distance helpers** — closest-feature and penetration queries
   on PGA flats and CGA rounds.
5. **Constraint solver** — sequential-impulse over joint/contact bivectors.

## 5. Boundary and crate placement

Following RFC-009's split between kernel primitives and the consuming layer:

- **`garust-core` / `garust-geo`** gain the reusable *primitives* — the
  inertia operator, twist/wrench/commutator helpers, the Lie-group
  integrator, and the contact queries. These are pure GA, signature-generic
  where possible, and useful beyond physics.
- A new **`garust-physics`** workspace crate hosts the *engine loop* — the
  `World`, broadphase, the constraint/contact solver iteration, and the
  per-step `integrate → detect → resolve` schedule.

**Open question (see §7):** whether `garust-physics` belongs in this
workspace or is a downstream consumer like the R-0010/R-0011 flow.

## 6. Validation gate

Known-answer physics, in increasing order of discriminating power:

1. **Free rigid body** conserves energy and angular momentum over a long
   run — the direct test of the Lie-group symplectic integrator (§4.3).
2. **The Dzhanibekov / tennis-racket effect** — an intermediate-axis spin
   flips periodically. A dramatic, qualitative known-answer test that the
   gyroscopic `P × B` term and the integrator are both right; almost every
   naive integrator gets it wrong.
3. **Pendulum period** matches the small-angle analytic value (`τ/√(g/ℓ)`,
   τ-only).
4. **A box settles on a plane** under gravity + contact without jitter or
   sink — the end-to-end contact + solver test.

## 7. Open questions

1. **Crate placement.** `garust-physics` in-workspace, or a separate
   downstream project consuming `garust`? (RFC-009 put the engine-equivalent
   downstream; physics may warrant in-workspace because it needs primitives
   that are awkward to add piecemeal from outside.)
2. **PGA-only v1, or CGA contacts from the start?** A PGA-flats-only first
   cut (boxes, planes, joints) is simpler; sphere/round contact via CGA can
   follow.
3. **Which Lie-group integrator?** A variational integrator (Lee–Leok–
   McClamroch) is the gold standard for long-run conservation; a
   Munthe-Kaas RKMK step is simpler to implement. Recommend starting with
   the variational leapfrog and measuring against gate §6.1–6.2.
4. **Inertia operator location** — `garust-core` (as a general grade-2
   linear map, reusable) or `garust-physics` (as a physics concept)?
