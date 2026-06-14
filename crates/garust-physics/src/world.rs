//! The simulation drive loop: integrate, detect, resolve.
//!
//! [`World`] ties the dynamics ([`RigidBody::step`](crate::RigidBody::step))
//! and the collision layer ([`crate::contact`]) into one `step`: advance every
//! body under gravity, then find and resolve sphere–sphere and sphere–ground
//! contacts.
//!
//! To keep the crate **allocation-free**, `World` holds only the global
//! settings — it does *not* own the bodies. You pass a `&mut [Body]` you own
//! (a `Vec` under `std`, a fixed array on bare metal), so there is no hidden
//! heap use.
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
//!     world.step(&mut bodies, 1.0 / 120.0);
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
    /// Restitution used when this body collides.
    pub restitution: f64,
}

impl Body {
    /// A uniform solid ball of the given `mass` and `radius`, at the origin and
    /// at rest, perfectly elastic. Its inertia is the solid-sphere value
    /// `⅖·m·r²` about every axis. Set the public fields to place, spin, or
    /// damp it.
    pub fn ball(mass: f64, radius: f64) -> Self {
        let i = 0.4 * mass * radius * radius;
        Self {
            rigid: RigidBody::new(mass),
            inertia: Inertia::principal([i; 3]),
            radius,
            restitution: 1.0,
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

    /// Advance every body by `dt`, then detect and resolve all contacts:
    /// `integrate → detect → resolve`.
    ///
    /// Gravity is applied as a centre-of-mass force (no torque), so the
    /// symplectic integrator runs per body; then every sphere–sphere pair and,
    /// if present, every sphere–ground overlap is resolved with a frictionless
    /// impulse plus a positional correction that pushes the overlap out
    /// (without touching momentum, so it injects no kinetic energy).
    pub fn step(&self, bodies: &mut [Body], dt: f64) {
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
                    resolve_pair(&mut a.rigid, &mut b.rigid, &hit, e);
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
                    resolve_static(&mut body.rigid, &hit, body.restitution);
                    for (p, &n) in body.rigid.position.iter_mut().zip(hit.normal.iter()) {
                        *p += n * hit.depth; // lift out of the floor
                    }
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
            world.step(&mut bodies, dt);
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
            world.step(&mut bodies, 0.005);
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
            world.step(&mut bodies, 0.005);
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
            world.step(&mut bodies, 1.0 / 240.0);
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
            world.step(&mut bodies, 1.0 / 240.0);
        }
        // Comes to rest sitting on the floor (centre a radius above it).
        assert!(
            (bodies[0].rigid.position[1] - 0.5).abs() < 1e-2,
            "y = {}",
            bodies[0].rigid.position[1]
        );
        assert!(bodies[0].rigid.velocity()[1].abs() < 0.2, "still moving");
    }
}
