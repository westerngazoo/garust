//! The simulation drive loop: integrate, detect, resolve, constrain.
//!
//! [`World`] ties the dynamics ([`RigidBody::step`](crate::RigidBody::step))
//! and the collision layer ([`crate::contact`]) into one `step`: advance every
//! body under gravity, then find and resolve sphere–sphere and sphere–ground
//! contacts, then enforce joint constraints.
//!
//! To keep the crate **allocation-free**, `World` holds only the global
//! settings — it does *not* own the bodies or joints. You pass a `&mut [Body]`
//! and a `&[Joint]` you own, so there is no hidden heap use.
//!
//! ```
//! use garust_physics::world::{Body, World};
//!
//! // A ball dropped onto the ground in a gravity well.
//! let world = World::new().with_ground(0.0); // floor at y = 0
//! let mut bodies = [Body::ball(1.0, 0.5)];
//! bodies[0].rigid.position = [0.0, 5.0, 0.0];
//! bodies[0].restitution = 0.8;
//!
//! let mut lowest = f64::MAX;
//! for _ in 0..1000 {
//!     world.step(&mut bodies, &[], 1.0 / 120.0);
//!     lowest = lowest.min(bodies[0].rigid.position[1]);
//! }
//! // It never tunnels through the floor (centre stays a radius above it).
//! assert!(lowest > 0.5 - 1e-2);
//! ```

use garust_core::Pga3;

use crate::contact::{resolve_pair, resolve_static, Contact, Sphere};
use crate::{Inertia, RigidBody};

/// A simulated body: its dynamics state, inertia, sphere-collider radius, and
/// restitution (bounciness, `0` dead … `1` perfectly elastic).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body {
    /// Rigid-body dynamics state (pose + momenta + mass).
    pub rigid: RigidBody,
    /// Rotational inertia.
    pub inertia: Inertia,
    /// Collision-sphere radius, centred on the body's centre of mass.
    pub radius: f64,
    /// Restitution used when this body collides (`1` = elastic, `0` = inelastic).
    pub restitution: f64,
    /// Coulomb friction coefficient (`0` = frictionless, `1` = moderate, …).
    pub mu: f64,
}

impl Body {
    /// A uniform solid ball of the given `mass` and `radius`, at the origin and
    /// at rest, perfectly elastic and frictionless. Its inertia is the
    /// solid-sphere value `⅖·m·r²` about every axis. Set the public fields to
    /// place, spin, damp, or friction it.
    pub fn ball(mass: f64, radius: f64) -> Self {
        let i = 0.4 * mass * radius * radius;
        Self {
            rigid: RigidBody::new(mass),
            inertia: Inertia::principal([i; 3]),
            radius,
            restitution: 1.0,
            mu: 0.0,
        }
    }
}

/// Global simulation settings and the `step` drive loop. Holds no bodies — see
/// the [module docs](crate::world).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct World {
    /// Uniform acceleration applied to every body (a force `m·g` at the centre
    /// of mass).
    pub gravity: [f64; 3],
    /// Height `y` of a horizontal ground plane (normal `+y`); `None` for none.
    pub ground: Option<f64>,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    /// Earth-like downward gravity, no ground.
    pub fn new() -> Self {
        Self {
            gravity: [0.0, -9.81, 0.0],
            ground: None,
        }
    }

    /// Set the gravity vector.
    pub fn with_gravity(mut self, gravity: [f64; 3]) -> Self {
        self.gravity = gravity;
        self
    }

    /// Add a horizontal ground plane at height `y`.
    pub fn with_ground(mut self, y: f64) -> Self {
        self.ground = Some(y);
        self
    }

    /// Advance every body by `dt`, then detect and resolve all contacts, then
    /// enforce `joints`: `integrate → detect → resolve → constrain`.
    ///
    /// Gravity is applied as a centre-of-mass force (no torque), so the
    /// symplectic integrator runs per body; then every sphere–sphere pair and,
    /// if present, every sphere–ground overlap is resolved with an impulse
    /// plus a positional correction; finally joint constraints are solved via
    /// ten sequential-impulse iterations with Baumgarte stabilization.
    ///
    /// Pass `&[]` for `joints` to run the classic contact-only loop.
    pub fn step(&self, bodies: &mut [Body], joints: &[Joint], dt: f64) {
        // 1. Integrate each body under gravity.
        for body in bodies.iter_mut() {
            let m = body.rigid.mass;
            let force = [
                self.gravity[0] * m,
                self.gravity[1] * m,
                self.gravity[2] * m,
            ];
            body.rigid = body.rigid.step(dt, &body.inertia, force, Pga3::zero());
        }

        // 2. Resolve every unordered pair exactly once (slice-pattern walk —
        //    no indexing, no allocation).
        let mut rest: &mut [Body] = bodies;
        while let [a, tail @ ..] = rest {
            let sa = Sphere {
                center: a.rigid.position,
                radius: a.radius,
            };
            for b in tail.iter_mut() {
                let sb = Sphere {
                    center: b.rigid.position,
                    radius: b.radius,
                };
                if let Some(hit) = sa.vs_sphere(&sb) {
                    let e = a.restitution.min(b.restitution);
                    let mu = a.mu.min(b.mu);
                    resolve_pair(&mut a.rigid, &mut b.rigid, &hit, e, mu);
                    correct_pair(&mut a.rigid, &mut b.rigid, &hit);
                }
            }
            rest = tail;
        }

        // 3. Resolve the ground.
        if let Some(y) = self.ground {
            for body in bodies.iter_mut() {
                let s = Sphere {
                    center: body.rigid.position,
                    radius: body.radius,
                };
                if let Some(hit) = s.vs_plane([0.0, 1.0, 0.0], y) {
                    resolve_static(&mut body.rigid, &hit, body.restitution, body.mu);
                    for (p, &n) in body.rigid.position.iter_mut().zip(hit.normal.iter()) {
                        *p += n * hit.depth; // lift out of the floor
                    }
                }
            }
        }

        // 4. Enforce joint constraints.
        if !joints.is_empty() {
            solve_joints(bodies, joints, dt);
        }
    }
}

// ── Joints ──────────────────────────────────────────────────────────────────

/// Body index sentinel meaning "the static world frame" (infinite mass).
///
/// Use this as the first body index in a [`Joint`] variant to anchor one
/// end to a fixed world-frame point rather than to a simulated body.
pub const STATIC: usize = usize::MAX;

/// A holonomic constraint between two bodies.
///
/// All variants store body indices into the caller's `&mut [Body]` slice.
/// Use [`STATIC`] as the first index to pin to a world-frame fixture.
///
/// The v1 solver enforces the **positional** part of each joint via
/// sequential impulse: ball-in-socket for `Hinge`/`Fixed`, and the two
/// perpendicular-to-axis directions for `Prismatic` (sliding stays free).
/// The `axis` field on `Hinge` records the hinge direction for future
/// rotational-limit work; it is stored but not yet enforced, and relative
/// rotation is likewise not yet locked for `Prismatic`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Joint {
    /// Connects body `b` to body `a` at a shared anchor, allowing rotation
    /// only about `axis` (the axis is in body `a`'s local frame, or in the
    /// world frame when `a == STATIC`).
    Hinge {
        /// First body index, or [`STATIC`] for a fixed world anchor.
        a: usize,
        /// Second body index.
        b: usize,
        /// Attachment in body `a`'s local frame (world-frame if `a == STATIC`).
        anchor_a: [f64; 3],
        /// Attachment in body `b`'s local frame.
        anchor_b: [f64; 3],
        /// Hinge axis in body `a`'s local frame (world-frame if `a == STATIC`).
        axis: [f64; 3],
    },
    /// Locks the relative pose between body `a` and body `b` entirely.
    Fixed {
        /// First body index, or [`STATIC`] for a fixed world anchor.
        a: usize,
        /// Second body index.
        b: usize,
        /// Attachment in body `a`'s local frame (world-frame if `a == STATIC`).
        anchor_a: [f64; 3],
        /// Attachment in body `b`'s local frame.
        anchor_b: [f64; 3],
    },
    /// Lets body `b`'s anchor slide along the `axis` line through body
    /// `a`'s anchor while never leaving it. The two directions
    /// perpendicular to the axis are constrained; motion along it is
    /// free. Like `Hinge`, the v1 solver enforces the positional part
    /// only — relative rotation is not yet locked.
    Prismatic {
        /// First body index, or [`STATIC`] for a fixed world anchor.
        a: usize,
        /// Second body index.
        b: usize,
        /// Attachment in body `a`'s local frame (world-frame if `a == STATIC`).
        anchor_a: [f64; 3],
        /// Attachment in body `b`'s local frame.
        anchor_b: [f64; 3],
        /// Slide axis in body `a`'s local frame (world-frame if `a == STATIC`).
        /// Must be non-zero; it is normalized internally.
        axis: [f64; 3],
    },
}

// ── Joint helpers ─────────────────────────────────────────────────────────

#[inline]
fn jsub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn jdot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn jcross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Extract `[x, y, z]` from a PGA grade-3 point. Divides by the e123
/// coefficient so ideal points (w = 0) return junk — callers must only pass
/// finite Euclidean points.
fn pga_point_xyz(p: &Pga3) -> [f64; 3] {
    // Pga3::point(x,y,z) = e123 - x·e023 + y·e013 - z·e012
    // blade indices: e123=7, e023=14 (sign -x), e013=13 (sign +y), e012=11 (sign -z)
    let w = p.coeffs[7];
    [-p.coeffs[14] / w, p.coeffs[13] / w, -p.coeffs[11] / w]
}

/// World-frame position of a body-local anchor point, or the anchor itself
/// when `idx == STATIC`.
fn world_anchor(bodies: &[Body], idx: usize, anchor_local: [f64; 3]) -> [f64; 3] {
    if idx == STATIC {
        return anchor_local;
    }
    let body = &bodies[idx];
    let p = body
        .rigid
        .pose()
        .apply(&Pga3::point(anchor_local[0], anchor_local[1], anchor_local[2]));
    pga_point_xyz(&p)
}

/// Velocity of a body at a world-frame point (linear + angular contribution).
///
/// Angular velocity is computed in body frame, then rotated to world frame
/// before evaluating `ω_world × r_world`.
fn body_vel_at(body: &Body, world_pt: [f64; 3]) -> [f64; 3] {
    let v = body.rigid.velocity();
    let r = jsub(world_pt, body.rigid.position);
    // Angular velocity in body frame, then rotate to world frame.
    let omega_body = body.rigid.angular_velocity(&body.inertia);
    let omega_world = body.rigid.orientation.apply(&omega_body);
    let planes = Inertia::principal_planes();
    let mut wx = 0.0_f64;
    let mut wy = 0.0_f64;
    let mut wz = 0.0_f64;
    for (k, b) in planes.iter().enumerate() {
        let c = -omega_world.scalar_product(b);
        match k {
            0 => wx = c,
            1 => wy = c,
            _ => wz = c,
        }
    }
    let omega_cross_r = jcross([wx, wy, wz], r);
    [
        v[0] + omega_cross_r[0],
        v[1] + omega_cross_r[1],
        v[2] + omega_cross_r[2],
    ]
}

/// Effective inverse mass for an impulse along `n` applied at lever arm `r`
/// from the body's centre of mass, using the principal-frame inertia.
fn eff_inv_mass(body: &Body, r: [f64; 3], n: [f64; 3]) -> f64 {
    let lin = 1.0 / body.rigid.mass;
    let rxn = jcross(r, n);
    let m = body.inertia.moments();
    let ang = rxn[0] * rxn[0] / m[0] + rxn[1] * rxn[1] / m[1] + rxn[2] * rxn[2] / m[2];
    lin + ang
}

/// Apply a linear-momentum impulse along `n` of magnitude `lambda` to the
/// body at index `idx`, plus the resulting angular impulse from lever arm `r`.
///
/// The torque is computed in world frame, then rotated to body frame via
/// `R⁻¹` before being added to the body-frame angular momentum Π.
fn apply_joint_impulse(bodies: &mut [Body], idx: usize, n: [f64; 3], lambda: f64, world_pt: [f64; 3]) {
    if idx == STATIC {
        return;
    }
    let body = &mut bodies[idx];
    for (p, &ni) in body.rigid.linear_momentum.iter_mut().zip(n.iter()) {
        *p += ni * lambda;
    }
    let r = jsub(world_pt, body.rigid.position);
    let tau_world_3d = jcross(r, [n[0] * lambda, n[1] * lambda, n[2] * lambda]);
    // Build world-frame torque bivector and rotate to body frame.
    let planes = Inertia::principal_planes();
    let tau_world_biv = planes[0] * tau_world_3d[0]
        + planes[1] * tau_world_3d[1]
        + planes[2] * tau_world_3d[2];
    let tau_body_biv = body.rigid.orientation.inverse().apply(&tau_world_biv);
    body.rigid.angular_momentum += tau_body_biv;
}

/// Sequential-impulse ball-in-socket constraint between body `a` and body `b`
/// (or a static world anchor when `a == STATIC`). Processes each Cartesian
/// axis independently (3 scalar constraints).
///
/// `β = baumgarte / dt` is the position-correction bias rate.
fn solve_ball_socket(
    bodies: &mut [Body],
    a: usize,
    b: usize,
    anchor_a: [f64; 3],
    anchor_b: [f64; 3],
    dt: f64,
    baumgarte: f64,
) {
    let wa = world_anchor(bodies, a, anchor_a);
    let wb = world_anchor(bodies, b, anchor_b);
    let err = jsub(wb, wa); // should be [0,0,0] when satisfied

    for axis in 0..3_usize {
        let n: [f64; 3] = if axis == 0 {
            [1.0, 0.0, 0.0]
        } else if axis == 1 {
            [0.0, 1.0, 0.0]
        } else {
            [0.0, 0.0, 1.0]
        };

        // Relative velocity at the constraint point along this axis
        let vel_b = if b == STATIC { [0.0; 3] } else { body_vel_at(&bodies[b], wb) };
        let vel_a = if a == STATIC { [0.0; 3] } else { body_vel_at(&bodies[a], wa) };
        let v_rel = jdot(jsub(vel_b, vel_a), n);

        // Effective inverse mass
        let r_b = jsub(wb, if b == STATIC { wb } else { bodies[b].rigid.position });
        let r_a = jsub(wa, if a == STATIC { wa } else { bodies[a].rigid.position });
        let inv_b = if b == STATIC { 0.0 } else { eff_inv_mass(&bodies[b], r_b, n) };
        let inv_a = if a == STATIC { 0.0 } else { eff_inv_mass(&bodies[a], r_a, n) };
        let inv_total = inv_a + inv_b;
        if inv_total < 1e-30 {
            continue;
        }

        // Impulse: velocity correction + Baumgarte position bias
        let bias = baumgarte / dt * err[axis];
        let lambda = -(v_rel + bias) / inv_total;

        apply_joint_impulse(bodies, b, n, lambda, wb);
        apply_joint_impulse(bodies, a, n, -lambda, wa);
    }
}

/// Sequential-impulse prismatic constraint: body `b`'s anchor may slide
/// along the axis line through body `a`'s anchor but never leave it. The
/// two directions perpendicular to the world-frame axis get the same
/// velocity-plus-Baumgarte treatment as [`solve_ball_socket`]; the axis
/// direction itself is left unconstrained.
fn solve_prismatic(
    bodies: &mut [Body],
    a: usize,
    b: usize,
    anchor_a: [f64; 3],
    anchor_b: [f64; 3],
    axis: [f64; 3],
    dt: f64,
    baumgarte: f64,
) {
    let wa = world_anchor(bodies, a, anchor_a);
    // World-frame axis: transform a second anchor offset by `axis` and
    // subtract — reuses the point path, so it is uniform over `STATIC`.
    let wa2 = world_anchor(
        bodies,
        a,
        [anchor_a[0] + axis[0], anchor_a[1] + axis[1], anchor_a[2] + axis[2]],
    );
    let d = jsub(wa2, wa);
    let len = jdot(d, d).sqrt();
    if len < 1e-12 {
        return; // degenerate axis: nothing well-defined to constrain
    }
    let d = [d[0] / len, d[1] / len, d[2] / len];

    // Orthonormal pair spanning the plane perpendicular to the axis.
    let pick = if d[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let c = jcross(d, pick);
    let clen = jdot(c, c).sqrt();
    let n1 = [c[0] / clen, c[1] / clen, c[2] / clen];
    let n2 = jcross(d, n1);

    let wb = world_anchor(bodies, b, anchor_b);
    let err = jsub(wb, wa);

    for n in [n1, n2] {
        let vel_b = if b == STATIC { [0.0; 3] } else { body_vel_at(&bodies[b], wb) };
        let vel_a = if a == STATIC { [0.0; 3] } else { body_vel_at(&bodies[a], wa) };
        let v_rel = jdot(jsub(vel_b, vel_a), n);

        let r_b = jsub(wb, if b == STATIC { wb } else { bodies[b].rigid.position });
        let r_a = jsub(wa, if a == STATIC { wa } else { bodies[a].rigid.position });
        let inv_b = if b == STATIC { 0.0 } else { eff_inv_mass(&bodies[b], r_b, n) };
        let inv_a = if a == STATIC { 0.0 } else { eff_inv_mass(&bodies[a], r_a, n) };
        let inv_total = inv_a + inv_b;
        if inv_total < 1e-30 {
            continue;
        }

        let bias = baumgarte / dt * jdot(err, n);
        let lambda = -(v_rel + bias) / inv_total;

        apply_joint_impulse(bodies, b, n, lambda, wb);
        apply_joint_impulse(bodies, a, n, -lambda, wa);
    }
}

/// Run N_ITER sequential-impulse iterations over all joints.
fn solve_joints(bodies: &mut [Body], joints: &[Joint], dt: f64) {
    const N_ITER: usize = 10;
    const BAUMGARTE: f64 = 0.2;

    for _ in 0..N_ITER {
        for joint in joints {
            match *joint {
                Joint::Hinge { a, b, anchor_a, anchor_b, .. }
                | Joint::Fixed { a, b, anchor_a, anchor_b } => {
                    solve_ball_socket(bodies, a, b, anchor_a, anchor_b, dt, BAUMGARTE);
                }
                Joint::Prismatic { a, b, anchor_a, anchor_b, axis } => {
                    solve_prismatic(bodies, a, b, anchor_a, anchor_b, axis, dt, BAUMGARTE);
                }
            }
        }
    }
}

/// Push two overlapping bodies apart along the normal, split by inverse mass.
/// Position-only, so it changes no momentum and injects no energy.
fn correct_pair(a: &mut RigidBody, b: &mut RigidBody, hit: &Contact) {
    let inv = 1.0 / a.mass + 1.0 / b.mass;
    let da = hit.depth / a.mass / inv;
    let db = hit.depth / b.mass / inv;
    for ((pa, pb), &n) in a
        .position
        .iter_mut()
        .zip(b.position.iter_mut())
        .zip(hit.normal.iter())
    {
        *pa -= n * da;
        *pb += n * db;
    }
}

#[cfg(test)]
mod tests {
    use super::{Body, World};

    fn total_momentum(bodies: &[Body]) -> [f64; 3] {
        bodies.iter().fold([0.0; 3], |acc, b| {
            let p = b.rigid.linear_momentum;
            [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]]
        })
    }
    fn total_ke(bodies: &[Body]) -> f64 {
        bodies
            .iter()
            .map(|b| b.rigid.kinetic_energy(&b.inertia))
            .sum()
    }

    #[test]
    fn gravity_makes_a_body_fall_exactly() {
        // No ground: free fall. Leapfrog is exact for constant force.
        let world = World::new().with_gravity([0.0, -10.0, 0.0]);
        let mut bodies = [Body::ball(2.0, 0.5)];
        bodies[0].rigid.position = [0.0, 100.0, 0.0];
        let dt = 0.001;
        let n = 1000;
        for _ in 0..n {
            world.step(&mut bodies, &[], dt);
        }
        let t = dt * n as f64; // 1.0 s
        let expected = 100.0 - 0.5 * 10.0 * t * t; // 100 − 5 = 95
        assert!((bodies[0].rigid.position[1] - expected).abs() < 1e-9);
    }

    #[test]
    fn collisions_conserve_total_momentum() {
        // Gravity-free, no ground: collisions are internal forces.
        let world = World {
            gravity: [0.0; 3],
            ground: None,
        };
        let mut a = Body::ball(1.0, 0.5);
        let mut b = Body::ball(2.0, 0.5);
        a.rigid.position = [-2.0, 0.0, 0.0];
        b.rigid.position = [2.0, 0.0, 0.0];
        a.rigid.linear_momentum = [3.0, 0.0, 0.0]; // closing
        b.rigid.linear_momentum = [-1.0, 0.0, 0.0];
        let mut bodies = [a, b];
        let p0 = total_momentum(&bodies);

        for _ in 0..2000 {
            world.step(&mut bodies, &[], 0.005);
        }
        let p1 = total_momentum(&bodies);
        for k in 0..3 {
            assert!(
                (p1[k] - p0[k]).abs() < 1e-9,
                "axis {k}: {} vs {}",
                p1[k],
                p0[k]
            );
        }
    }

    #[test]
    fn elastic_collision_conserves_total_energy() {
        let world = World {
            gravity: [0.0; 3],
            ground: None,
        };
        let mut a = Body::ball(1.0, 0.5); // restitution 1.0
        let mut b = Body::ball(3.0, 0.5);
        a.rigid.position = [-2.0, 0.0, 0.0];
        b.rigid.position = [2.0, 0.0, 0.0];
        a.rigid.linear_momentum = [4.0, 0.0, 0.0];
        let mut bodies = [a, b];
        let ke0 = total_ke(&bodies);

        for _ in 0..2000 {
            world.step(&mut bodies, &[], 0.005);
        }
        assert!((total_ke(&bodies) - ke0).abs() < 1e-9);
        // They must actually have interacted (b is now moving).
        assert!(bodies[1].rigid.linear_momentum[0] > 1e-6);
    }

    #[test]
    fn a_dropped_ball_bounces_and_never_tunnels() {
        let world = World::new().with_ground(0.0); // g = −9.81, floor at 0
        let mut bodies = [Body::ball(1.0, 0.5)];
        bodies[0].rigid.position = [0.0, 4.0, 0.0]; // drop from y = 4

        let mut lowest = f64::MAX;
        let mut sign_changes = 0;
        let mut prev_vy = 0.0;
        for _ in 0..3000 {
            world.step(&mut bodies, &[], 1.0 / 240.0);
            let vy = bodies[0].rigid.velocity()[1];
            if prev_vy < 0.0 && vy > 0.0 {
                sign_changes += 1; // a bounce
            }
            prev_vy = vy;
            lowest = lowest.min(bodies[0].rigid.position[1]);
        }
        // Bounced several times, never sank through the floor, never exploded.
        assert!(sign_changes >= 2, "too few bounces: {sign_changes}");
        assert!(lowest > 0.5 - 1e-2, "tunnelled: lowest = {lowest}");
        assert!(bodies[0].rigid.position[1] < 4.5, "gained energy");
    }

    #[test]
    fn a_dead_ball_settles_on_the_ground() {
        let world = World::new().with_ground(0.0);
        let mut bodies = [Body::ball(1.0, 0.5)];
        bodies[0].rigid.position = [0.0, 3.0, 0.0];
        bodies[0].restitution = 0.0; // no bounce

        for _ in 0..4000 {
            world.step(&mut bodies, &[], 1.0 / 240.0);
        }
        // Comes to rest sitting on the floor (centre a radius above it).
        assert!(
            (bodies[0].rigid.position[1] - 0.5).abs() < 1e-2,
            "y = {}",
            bodies[0].rigid.position[1]
        );
        assert!(bodies[0].rigid.velocity()[1].abs() < 0.2, "still moving");
    }

    // --- Joint constraints (issue #45) -------------------------------------

    /// A simple pendulum: unit-mass bob pinned to a fixed world anchor via a
    /// hinge joint with the attachment point ℓ above the bob's CM.
    ///
    /// Setup: pivot at the world origin `[0, 0, 0]`; bob CM at
    /// `[ℓ·sin θ, −ℓ·cos θ, 0]` displaced 5° from vertical; body orientation
    /// rotated by θ about z so that `pose.apply([0, ℓ, 0]) = [0, 0, 0]`
    /// (constraint satisfied at t = 0, no initial kick).
    ///
    /// The analytic small-angle period is `T = τ / √(g / ℓ)` (RFC-010 §6
    /// gate 3). We allow 2 % tolerance (small-angle error + numerical).
    #[test]
    fn pendulum_period_matches_analytic_value() {
        use super::{Joint, STATIC};
        use core::f64::consts::TAU;
        use garust_geo::Motor;
        use garust_core::Pga3;

        let g = 9.81_f64;
        let ell = 1.0_f64;
        let dt = 1.0 / 1200.0_f64;
        let theta = 5.0_f64 * TAU / 360.0; // 5° displacement from vertical

        let analytic_period = TAU / (g / ell).sqrt();

        let world = World {
            gravity: [0.0, -g, 0.0],
            ground: None,
        };

        // Orientation: CCW rotation by θ about z (e12 plane).
        // Position: [ℓ·sin θ, −ℓ·cos θ, 0] so pose.apply([0,ℓ,0]) = [0,0,0].
        let e1 = Pga3::basis(1_usize);
        let e2 = Pga3::basis(2_usize);
        let e12 = e1 * e2;
        let orient = Motor::rotor(theta, e12);

        let mut bodies = [Body::ball(1.0, 0.05)];
        bodies[0].rigid.orientation = orient;
        bodies[0].rigid.position = [ell * theta.sin(), -ell * theta.cos(), 0.0];

        // attachment point ℓ above the bob's CM in body-local frame.
        let joints = [Joint::Hinge {
            a: STATIC,
            b: 0,
            anchor_a: [0.0, 0.0, 0.0],
            anchor_b: [0.0, ell, 0.0],
            axis: [0.0, 0.0, 1.0],
        }];

        let mut zero_crossings = 0_u32;
        let mut prev_x = bodies[0].rigid.position[0];
        let mut last_cross_t = 0.0_f64;
        let mut periods = [0.0_f64; 4];
        let mut t = 0.0_f64;
        let max_t = analytic_period * 8.0;

        while t < max_t {
            world.step(&mut bodies, &joints, dt);
            t += dt;
            let x = bodies[0].rigid.position[0];
            if prev_x > 0.0 && x <= 0.0 || prev_x < 0.0 && x >= 0.0 {
                if zero_crossings > 0 {
                    let half = t - last_cross_t;
                    if (zero_crossings as usize - 1) < periods.len() {
                        periods[zero_crossings as usize - 1] = half * 2.0;
                    }
                }
                last_cross_t = t;
                zero_crossings += 1;
                if zero_crossings > 4 {
                    break;
                }
            }
            prev_x = x;
        }

        assert!(
            zero_crossings >= 4,
            "pendulum did not oscillate: {zero_crossings} zero-crossings after {t:.2} s \
             (period ≈ {analytic_period:.3} s)"
        );

        let count = periods.iter().filter(|&&p| p > 0.0).count();
        let measured = periods.iter().filter(|&&p| p > 0.0).sum::<f64>() / count as f64;
        let err = ((measured - analytic_period) / analytic_period).abs();
        assert!(
            err < 0.02,
            "period: measured={measured:.4} s, analytic={analytic_period:.4} s, err={:.1} %",
            err * 100.0
        );
    }

    // ── Prismatic joint ─────────────────────────────────────────────────────

    /// Sliding along the axis must be unconstrained: a body launched along
    /// the slide direction keeps its momentum and travels the free-particle
    /// distance.
    #[test]
    fn prismatic_slides_freely_along_axis() {
        use super::{Joint, STATIC};

        let world = World::new().with_gravity([0.0, 0.0, 0.0]);
        let mut bodies = [Body::ball(2.0, 0.1)];
        bodies[0].rigid.linear_momentum = [2.0, 0.0, 0.0]; // v = 1 m/s along x

        let joints = [Joint::Prismatic {
            a: STATIC,
            b: 0,
            anchor_a: [0.0, 0.0, 0.0],
            anchor_b: [0.0, 0.0, 0.0],
            axis: [1.0, 0.0, 0.0],
        }];

        let dt = 0.001;
        for _ in 0..1000 {
            world.step(&mut bodies, &joints, dt);
        }
        let p = bodies[0].rigid.position;
        assert!((p[0] - 1.0).abs() < 1e-9, "x = {} (want 1.0)", p[0]);
        assert!(p[1].abs() < 1e-9 && p[2].abs() < 1e-9, "drifted off axis: {p:?}");
        assert!(
            (bodies[0].rigid.linear_momentum[0] - 2.0).abs() < 1e-9,
            "axis momentum was not conserved"
        );
    }

    /// Motion perpendicular to the axis must be killed: a body launched
    /// sideways stays on the slide line.
    #[test]
    fn prismatic_blocks_perpendicular_motion() {
        use super::{Joint, STATIC};

        let world = World::new().with_gravity([0.0, 0.0, 0.0]);
        let mut bodies = [Body::ball(2.0, 0.1)];
        bodies[0].rigid.linear_momentum = [0.0, 2.0, 0.0]; // v = 1 m/s along y

        let joints = [Joint::Prismatic {
            a: STATIC,
            b: 0,
            anchor_a: [0.0, 0.0, 0.0],
            anchor_b: [0.0, 0.0, 0.0],
            axis: [1.0, 0.0, 0.0],
        }];

        let dt = 0.001;
        for _ in 0..1000 {
            world.step(&mut bodies, &joints, dt);
        }
        let p = bodies[0].rigid.position;
        assert!(p[1].abs() < 1e-3, "escaped perpendicular: y = {}", p[1]);
        assert!(
            bodies[0].rigid.linear_momentum[1].abs() < 1e-6,
            "perpendicular momentum survived: {}",
            bodies[0].rigid.linear_momentum[1]
        );
    }

    /// A body placed off the slide line is pulled back onto it by the
    /// Baumgarte bias, without picking up motion along the axis.
    #[test]
    fn prismatic_recenters_offset_body() {
        use super::{Joint, STATIC};

        let world = World::new().with_gravity([0.0, 0.0, 0.0]);
        let mut bodies = [Body::ball(1.0, 0.1)];
        bodies[0].rigid.position = [0.3, 0.1, 0.0]; // 0.1 off the x-axis line

        let joints = [Joint::Prismatic {
            a: STATIC,
            b: 0,
            anchor_a: [0.0, 0.0, 0.0],
            anchor_b: [0.0, 0.0, 0.0],
            axis: [1.0, 0.0, 0.0],
        }];

        let dt = 0.001;
        for _ in 0..2000 {
            world.step(&mut bodies, &joints, dt);
        }
        let p = bodies[0].rigid.position;
        assert!(p[1].abs() < 1e-3, "not recentered: y = {}", p[1]);
        assert!((p[0] - 0.3).abs() < 1e-6, "picked up axis drift: x = {}", p[0]);
    }

    /// Gravity along the slide axis passes straight through: the body falls
    /// as if free. Gravity perpendicular to it is fully supported.
    #[test]
    fn prismatic_supports_perpendicular_gravity_only() {
        use super::{Joint, STATIC};

        // Slide axis along y, gravity along -y: free fall.
        let world = World::new().with_gravity([0.0, -10.0, 0.0]);
        let joints_y = [Joint::Prismatic {
            a: STATIC,
            b: 0,
            anchor_a: [0.0, 0.0, 0.0],
            anchor_b: [0.0, 0.0, 0.0],
            axis: [0.0, 1.0, 0.0],
        }];
        let mut bodies = [Body::ball(1.0, 0.1)];
        let dt = 0.001;
        for _ in 0..1000 {
            world.step(&mut bodies, &joints_y, dt);
        }
        assert!(
            (bodies[0].rigid.position[1] + 5.0).abs() < 0.02,
            "did not fall freely along the axis: y = {}",
            bodies[0].rigid.position[1]
        );

        // Slide axis along x, same gravity: the joint carries the weight.
        let joints_x = [Joint::Prismatic {
            a: STATIC,
            b: 0,
            anchor_a: [0.0, 0.0, 0.0],
            anchor_b: [0.0, 0.0, 0.0],
            axis: [1.0, 0.0, 0.0],
        }];
        let mut bodies = [Body::ball(1.0, 0.1)];
        for _ in 0..1000 {
            world.step(&mut bodies, &joints_x, dt);
        }
        assert!(
            bodies[0].rigid.position[1].abs() < 1e-2,
            "sagged under supported gravity: y = {}",
            bodies[0].rigid.position[1]
        );
    }
}
