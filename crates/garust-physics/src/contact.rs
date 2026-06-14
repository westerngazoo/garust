//! Collision detection and frictionless impulse response.
//!
//! The next RFC-010 slice on top of the 6-DOF dynamics: **sphere** colliders,
//! the universal broadphase bound and the simplest exact narrowphase, plus the
//! impulse that resolves a contact back into the bodies' momenta.
//!
//! Spheres earn the first slice because a frictionless sphere contact is
//! *exactly* a one-dimensional problem: the contact normal runs along the line
//! of centres, through both centres, so the normal impulse produces **no
//! torque** — collision response is pure linear-momentum exchange, with
//! nothing approximated. That makes it verifiable against the textbook
//! conservation laws (see the tests). Spin from impact needs a *tangential*
//! (friction) impulse, and contacts between flats (PGA planes/edges) need
//! their own narrowphase; both are later slices.
//!
//! ```
//! use garust_physics::{RigidBody, contact::{resolve_pair, Sphere}};
//!
//! // Two unit balls, unit mass, closing head-on along x.
//! let mut a = RigidBody::new(1.0);
//! let mut b = RigidBody::new(1.0);
//! a.position = [0.0, 0.0, 0.0];
//! b.position = [1.5, 0.0, 0.0]; // overlapping (radii sum 2 > 1.5)
//! a.linear_momentum = [2.0, 0.0, 0.0]; // moving toward b
//!
//! let ca = Sphere { center: a.position, radius: 1.0 };
//! let cb = Sphere { center: b.position, radius: 1.0 };
//! let hit = ca.vs_sphere(&cb).unwrap();
//! resolve_pair(&mut a, &mut b, &hit, 1.0); // elastic
//!
//! // Equal masses exchange velocity: a stops, b moves off at 2.
//! assert!(a.linear_momentum[0].abs() < 1e-12);
//! assert!((b.linear_momentum[0] - 2.0).abs() < 1e-12);
//! ```

use crate::RigidBody;

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// A collision sphere in world coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    /// Centre in the world frame (typically a body's [`RigidBody::position`]).
    pub center: [f64; 3],
    /// Radius.
    pub radius: f64,
}

/// A detected contact: a unit `normal` and a positive penetration `depth`.
///
/// For [`Sphere::vs_sphere`] the normal points from the first sphere toward
/// the second; for [`Sphere::vs_plane`] it points away from the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// Unit contact normal.
    pub normal: [f64; 3],
    /// Penetration depth (overlap along the normal), always `> 0`.
    pub depth: f64,
}

impl Sphere {
    /// Contact with another sphere, or `None` if they don't overlap. The
    /// normal points from `self` toward `other`.
    pub fn vs_sphere(&self, other: &Sphere) -> Option<Contact> {
        let d = sub(other.center, self.center);
        let dist_sq = dot(d, d);
        let sum = self.radius + other.radius;
        if dist_sq >= sum * sum || dist_sq == 0.0 {
            return None;
        }
        // Route through the kernel's `Real::sqrt` so this stays no_std-clean
        // (inherent `f64::sqrt` is std-only; the `libm` backend supplies this).
        let dist = garust_core::Real::sqrt(dist_sq);
        Some(Contact {
            normal: [d[0] / dist, d[1] / dist, d[2] / dist],
            depth: sum - dist,
        })
    }

    /// Contact with the half-space below the plane `{ x : n·x = offset }`
    /// (with unit normal `n` pointing out of the surface, e.g. `[0,1,0]` for
    /// a floor at height `offset`), or `None` if the sphere is clear of it.
    pub fn vs_plane(&self, normal: [f64; 3], offset: f64) -> Option<Contact> {
        let signed = dot(self.center, normal) - offset; // centre's height above the plane
        let depth = self.radius - signed;
        if depth <= 0.0 {
            return None;
        }
        Some(Contact { normal, depth })
    }
}

/// Resolve a frictionless collision between two bodies, exchanging linear
/// momentum along the contact normal with restitution `e ∈ [0, 1]`
/// (`1` = perfectly elastic, `0` = perfectly inelastic).
///
/// The impulse is along `contact.normal`, which for spheres passes through
/// both centres, so only linear momentum changes — no torque. Total linear
/// momentum is conserved; with `e = 1` so is kinetic energy. A no-op if the
/// bodies are already separating.
pub fn resolve_pair(a: &mut RigidBody, b: &mut RigidBody, contact: &Contact, restitution: f64) {
    let n = contact.normal;
    let v_rel = dot(sub(b.velocity(), a.velocity()), n);
    if v_rel >= 0.0 {
        return; // separating (or grazing) — no impulse
    }
    let inv_mass = 1.0 / a.mass + 1.0 / b.mass;
    let j = -(1.0 + restitution) * v_rel / inv_mass;
    let impulse = [n[0] * j, n[1] * j, n[2] * j];
    for ((pa, pb), &imp) in a
        .linear_momentum
        .iter_mut()
        .zip(b.linear_momentum.iter_mut())
        .zip(impulse.iter())
    {
        *pa -= imp;
        *pb += imp;
    }
}

/// Resolve a frictionless collision against an **immovable** surface (the
/// ground, a wall — effectively infinite mass), reflecting the body's normal
/// velocity with restitution `e`. `contact.normal` points away from the
/// surface. A no-op if the body is already moving away.
pub fn resolve_static(body: &mut RigidBody, contact: &Contact, restitution: f64) {
    let n = contact.normal;
    let v_rel = dot(body.velocity(), n);
    if v_rel >= 0.0 {
        return;
    }
    let j = -(1.0 + restitution) * v_rel * body.mass; // inverse mass = 1/m
    let impulse = [n[0] * j, n[1] * j, n[2] * j];
    for (p, &imp) in body.linear_momentum.iter_mut().zip(impulse.iter()) {
        *p += imp;
    }
}

#[cfg(test)]
mod tests {
    use super::{resolve_pair, resolve_static, Contact, Sphere};
    use crate::RigidBody;

    fn momentum(b: &RigidBody) -> [f64; 3] {
        b.linear_momentum
    }
    fn ke(b: &RigidBody) -> f64 {
        let p = b.linear_momentum;
        (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) / (2.0 * b.mass)
    }

    // --- Detection ---------------------------------------------------------

    #[test]
    fn spheres_detect_overlap_and_clearance() {
        let a = Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        let near = Sphere {
            center: [1.5, 0.0, 0.0],
            radius: 1.0,
        };
        let far = Sphere {
            center: [3.0, 0.0, 0.0],
            radius: 1.0,
        };
        let c = a.vs_sphere(&near).unwrap();
        assert_eq!(c.normal, [1.0, 0.0, 0.0]);
        assert!((c.depth - 0.5).abs() < 1e-12);
        assert!(a.vs_sphere(&far).is_none());
    }

    #[test]
    fn sphere_detects_the_floor() {
        let s = Sphere {
            center: [0.0, 0.5, 0.0],
            radius: 1.0,
        };
        let c = s.vs_plane([0.0, 1.0, 0.0], 0.0).unwrap();
        assert_eq!(c.normal, [0.0, 1.0, 0.0]);
        assert!((c.depth - 0.5).abs() < 1e-12);
        // Clear of the floor.
        let high = Sphere {
            center: [0.0, 5.0, 0.0],
            radius: 1.0,
        };
        assert!(high.vs_plane([0.0, 1.0, 0.0], 0.0).is_none());
    }

    // --- Response: pairs ---------------------------------------------------

    #[test]
    fn equal_mass_elastic_collision_swaps_velocity() {
        let mut a = RigidBody::new(1.0);
        let mut b = RigidBody::new(1.0);
        a.position = [0.0, 0.0, 0.0];
        b.position = [1.5, 0.0, 0.0];
        a.linear_momentum = [2.0, 0.0, 0.0]; // toward b
        let p_total = [momentum(&a)[0] + momentum(&b)[0], 0.0, 0.0];
        let ke0 = ke(&a) + ke(&b);

        let hit = Sphere {
            center: a.position,
            radius: 1.0,
        }
        .vs_sphere(&Sphere {
            center: b.position,
            radius: 1.0,
        })
        .unwrap();
        resolve_pair(&mut a, &mut b, &hit, 1.0);

        assert!(a.linear_momentum[0].abs() < 1e-12); // a stopped
        assert!((b.linear_momentum[0] - 2.0).abs() < 1e-12); // b took the velocity
                                                             // Conservation.
        assert!((momentum(&a)[0] + momentum(&b)[0] - p_total[0]).abs() < 1e-12);
        assert!((ke(&a) + ke(&b) - ke0).abs() < 1e-12);
    }

    #[test]
    fn unequal_mass_elastic_conserves_momentum_and_energy() {
        let mut a = RigidBody::new(3.0);
        let mut b = RigidBody::new(1.0);
        a.position = [0.0, 0.0, 0.0];
        b.position = [1.5, 0.0, 0.0];
        a.linear_momentum = [6.0, 0.0, 0.0]; // v = 2
        b.linear_momentum = [-1.0, 0.0, 0.0]; // v = -1, closing
        let p0 = momentum(&a)[0] + momentum(&b)[0];
        let ke0 = ke(&a) + ke(&b);

        let hit = Sphere {
            center: a.position,
            radius: 1.0,
        }
        .vs_sphere(&Sphere {
            center: b.position,
            radius: 1.0,
        })
        .unwrap();
        resolve_pair(&mut a, &mut b, &hit, 1.0);

        assert!((momentum(&a)[0] + momentum(&b)[0] - p0).abs() < 1e-12);
        assert!((ke(&a) + ke(&b) - ke0).abs() < 1e-12);
    }

    #[test]
    fn inelastic_collision_conserves_momentum_and_unifies_velocity() {
        let mut a = RigidBody::new(1.0);
        let mut b = RigidBody::new(2.0);
        a.position = [0.0, 0.0, 0.0];
        b.position = [1.5, 0.0, 0.0];
        a.linear_momentum = [4.0, 0.0, 0.0];
        let p0 = momentum(&a)[0] + momentum(&b)[0];

        let hit = Sphere {
            center: a.position,
            radius: 1.0,
        }
        .vs_sphere(&Sphere {
            center: b.position,
            radius: 1.0,
        })
        .unwrap();
        resolve_pair(&mut a, &mut b, &hit, 0.0); // perfectly inelastic

        // Momentum conserved; the two now share one velocity along the normal.
        assert!((momentum(&a)[0] + momentum(&b)[0] - p0).abs() < 1e-12);
        assert!((a.velocity()[0] - b.velocity()[0]).abs() < 1e-12);
    }

    #[test]
    fn separating_bodies_get_no_impulse() {
        let mut a = RigidBody::new(1.0);
        let mut b = RigidBody::new(1.0);
        a.position = [0.0, 0.0, 0.0];
        b.position = [1.5, 0.0, 0.0];
        b.linear_momentum = [3.0, 0.0, 0.0]; // b moving away from a
        let (pa, pb) = (a.linear_momentum, b.linear_momentum);
        let hit = Sphere {
            center: a.position,
            radius: 1.0,
        }
        .vs_sphere(&Sphere {
            center: b.position,
            radius: 1.0,
        })
        .unwrap();
        resolve_pair(&mut a, &mut b, &hit, 1.0);
        assert_eq!(a.linear_momentum, pa); // untouched
        assert_eq!(b.linear_momentum, pb);
    }

    // --- Response: static surface -----------------------------------------

    #[test]
    fn ball_bounces_off_the_floor() {
        let floor = ([0.0, 1.0, 0.0], 0.0);
        let mut ball = RigidBody::new(2.0);
        ball.position = [0.0, 0.5, 0.0];
        ball.linear_momentum = [0.0, -4.0, 0.0]; // falling, v = -2

        let hit = Sphere {
            center: ball.position,
            radius: 1.0,
        }
        .vs_plane(floor.0, floor.1)
        .unwrap();

        // Elastic: speed reflected.
        let mut elastic = ball;
        resolve_static(&mut elastic, &hit, 1.0);
        assert!((elastic.linear_momentum[1] - 4.0).abs() < 1e-12); // v = +2

        // Half restitution: half the speed comes back.
        let mut damped = ball;
        resolve_static(&mut damped, &hit, 0.5);
        assert!((damped.velocity()[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn resting_contact_is_stable() {
        // A ball moving away from the floor keeps its velocity.
        let mut ball = RigidBody::new(1.0);
        ball.position = [0.0, 0.5, 0.0];
        ball.linear_momentum = [0.0, 1.0, 0.0]; // rising
        let hit = Contact {
            normal: [0.0, 1.0, 0.0],
            depth: 0.5,
        };
        resolve_static(&mut ball, &hit, 1.0);
        assert_eq!(ball.linear_momentum, [0.0, 1.0, 0.0]);
    }
}
