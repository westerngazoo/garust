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
//! resolve_pair(&mut a, &mut b, &hit, 1.0, 0.0); // elastic, frictionless
//!
//! // Equal masses exchange velocity: a stops, b moves off at 2.
//! assert!(a.linear_momentum[0].abs() < 1e-12);
//! assert!((b.linear_momentum[0] - 2.0).abs() < 1e-12);
//! ```

use garust_core::Pga3;

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

/// A detected contact: a unit `normal`, a positive penetration `depth`, and
/// the world-space contact `point`.
///
/// For [`Sphere::vs_sphere`] the normal points from the first sphere toward
/// the second; for [`Sphere::vs_plane`] it points away from the surface.
/// The contact point is on the surface between the two objects.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// Unit contact normal.
    pub normal: [f64; 3],
    /// Penetration depth (overlap along the normal), always `> 0`.
    pub depth: f64,
    /// World-space contact point (surface between the two objects).
    pub point: [f64; 3],
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
        let n = [d[0] / dist, d[1] / dist, d[2] / dist];
        // Contact point: on the surface of self, halfway into the overlap.
        let pt = [
            self.center[0] + n[0] * (self.radius - 0.5 * (sum - dist)),
            self.center[1] + n[1] * (self.radius - 0.5 * (sum - dist)),
            self.center[2] + n[2] * (self.radius - 0.5 * (sum - dist)),
        ];
        Some(Contact {
            normal: n,
            depth: sum - dist,
            point: pt,
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
        // Contact point: on the sphere surface toward the plane.
        let pt = [
            self.center[0] - normal[0] * self.radius,
            self.center[1] - normal[1] * self.radius,
            self.center[2] - normal[2] * self.radius,
        ];
        Some(Contact {
            normal,
            depth,
            point: pt,
        })
    }
}

/// Resolve a collision between two bodies: exchanges linear momentum and
/// (when contact point is available) applies angular impulse from the
/// off-centre leverage. Optionally applies Coulomb tangential friction.
///
/// Parameters:
/// - `restitution` ∈ [0, 1]: bounciness (`1` = elastic, `0` = inelastic).
/// - `mu` ∈ [0, ∞): coefficient of friction; `0` = frictionless.
///
/// The normal impulse magnitude uses the linear-only formula (inverse-mass
/// sum as denominator); friction and angular transfer use the same scalar.
/// Total linear momentum is conserved; with `e = 1` and `mu = 0` so is KE.
/// A no-op if the bodies are already separating.
pub fn resolve_pair(
    a: &mut RigidBody,
    b: &mut RigidBody,
    contact: &Contact,
    restitution: f64,
    mu: f64,
) {
    let n = contact.normal;
    let v_rel = dot(sub(b.velocity(), a.velocity()), n);
    if v_rel >= 0.0 {
        return;
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

    // Angular impulse: lever arm × normal impulse, applied to both bodies.
    let pt = contact.point;
    let ra = sub(pt, a.position);
    let rb = sub(pt, b.position);
    let torque_a = cross(ra, [-impulse[0], -impulse[1], -impulse[2]]);
    let torque_b = cross(rb, impulse);
    apply_angular_impulse(a, torque_a);
    apply_angular_impulse(b, torque_b);

    // Coulomb friction: tangential impulse clamped to μ|j|.
    if mu > 0.0 {
        let v_full = sub(b.velocity(), a.velocity());
        let v_tang = [
            v_full[0] - n[0] * dot(v_full, n),
            v_full[1] - n[1] * dot(v_full, n),
            v_full[2] - n[2] * dot(v_full, n),
        ];
        let v_tang_len_sq = dot(v_tang, v_tang);
        if v_tang_len_sq > 1e-24 {
            let v_tang_len = garust_core::Real::sqrt(v_tang_len_sq);
            let t = [
                v_tang[0] / v_tang_len,
                v_tang[1] / v_tang_len,
                v_tang[2] / v_tang_len,
            ];
            let jt = (j * mu).min(inv_mass * v_tang_len / inv_mass);
            let ft = [t[0] * jt, t[1] * jt, t[2] * jt];
            for ((pa, pb), &f) in a
                .linear_momentum
                .iter_mut()
                .zip(b.linear_momentum.iter_mut())
                .zip(ft.iter())
            {
                *pa += f;
                *pb -= f;
            }
            // Friction angular impulse.
            let at_a = cross(ra, ft);
            let at_b = cross(rb, [-ft[0], -ft[1], -ft[2]]);
            apply_angular_impulse(a, at_a);
            apply_angular_impulse(b, at_b);
        }
    }
}

/// Resolve a collision against an **immovable** surface (the ground, a wall —
/// effectively infinite mass), reflecting the body's normal velocity with
/// restitution `e` and optional Coulomb friction `mu`. `contact.normal` points
/// away from the surface. A no-op if the body is already moving away.
pub fn resolve_static(body: &mut RigidBody, contact: &Contact, restitution: f64, mu: f64) {
    let n = contact.normal;
    let v_rel = dot(body.velocity(), n);
    if v_rel >= 0.0 {
        return;
    }
    let j = -(1.0 + restitution) * v_rel * body.mass;
    let impulse = [n[0] * j, n[1] * j, n[2] * j];
    for (p, &imp) in body.linear_momentum.iter_mut().zip(impulse.iter()) {
        *p += imp;
    }

    // Angular impulse from lever arm × normal impulse.
    let pt = contact.point;
    let r = sub(pt, body.position);
    let torque = cross(r, impulse);
    apply_angular_impulse(body, torque);

    // Coulomb friction.
    if mu > 0.0 {
        let v = body.velocity();
        let v_tang = [
            v[0] - n[0] * dot(v, n),
            v[1] - n[1] * dot(v, n),
            v[2] - n[2] * dot(v, n),
        ];
        let v_tang_len_sq = dot(v_tang, v_tang);
        if v_tang_len_sq > 1e-24 {
            let v_tang_len = garust_core::Real::sqrt(v_tang_len_sq);
            let t = [
                v_tang[0] / v_tang_len,
                v_tang[1] / v_tang_len,
                v_tang[2] / v_tang_len,
            ];
            let jt = (j * mu).min(body.mass * v_tang_len);
            let ft = [t[0] * jt, t[1] * jt, t[2] * jt];
            for (p, &f) in body.linear_momentum.iter_mut().zip(ft.iter()) {
                *p -= f;
            }
            let at = cross(r, [-ft[0], -ft[1], -ft[2]]);
            apply_angular_impulse(body, at);
        }
    }
}

/// Apply a 3D torque impulse (world-frame `[τx, τy, τz]`) to a body's
/// angular momentum bivector, mapping via the Euclidean principal planes.
fn apply_angular_impulse(body: &mut RigidBody, torque: [f64; 3]) {
    use crate::Inertia;
    let planes = Inertia::principal_planes();
    for (k, &tau) in planes.iter().zip(torque.iter()) {
        body.angular_momentum += *k * tau;
    }
}

// ── PGA flat-contact geometry (RFC-010 §2.4, §4.4) ──────────────────────────

/// Signed distance from the Euclidean point `pt` to the PGA plane `plane`.
///
/// `plane` is a grade-1 vector `a·e1 + b·e2 + c·e3 + d·e0` encoding
/// `ax + by + cz + d = 0`. The sign is positive on the side the normal
/// `(a, b, c)` points toward. Returns `0` if the plane normal is zero.
///
/// ```
/// use garust_core::Pga3;
/// use garust_physics::contact::point_plane_distance;
///
/// // x = 1 plane, normal e1, offset -1 → point (2,0,0) is 1 unit out.
/// let floor = Pga3::plane(1.0, 0.0, 0.0, -1.0);
/// assert!((point_plane_distance([2.0, 0.0, 0.0], &floor) - 1.0).abs() < 1e-12);
/// assert!((point_plane_distance([0.0, 0.0, 0.0], &floor) + 1.0).abs() < 1e-12);
/// ```
pub fn point_plane_distance(pt: [f64; 3], plane: &Pga3) -> f64 {
    // Grade-1 PGA: a=coeffs[1] (e1), b=coeffs[2] (e2), c=coeffs[4] (e3),
    // d=coeffs[8] (e0, the null/ideal generator).
    let a = plane.coeffs[1];
    let b = plane.coeffs[2];
    let c = plane.coeffs[4];
    let d = plane.coeffs[8];
    let len_sq = a * a + b * b + c * c;
    if len_sq < 1e-24 {
        return 0.0;
    }
    (a * pt[0] + b * pt[1] + c * pt[2] + d) / garust_core::Real::sqrt(len_sq)
}

/// Closest points between two PGA grade-2 lines and their Euclidean distance.
///
/// Returns `(distance, point_on_l1, point_on_l2)`. Uses the Plücker
/// representation of the PGA lines: the Euclidean bivector part (e23, e31, e12)
/// carries the direction; the ideal part (e01, e02, e03) carries the moment.
/// The closest-point formula is the standard skew-line algorithm.
///
/// If the lines are parallel (or one is degenerate) the returned points are
/// the closest approach under a least-squares reading of the direction vectors.
pub fn line_line_closest(l1: &Pga3, l2: &Pga3) -> (f64, [f64; 3], [f64; 3]) {
    // Direction from the Euclidean bivector part (blade indices 6, 5, 3).
    // Sign convention (from blade_product / Cayley table):
    //   e23 = basis(6)  → coeffs[6] = +d_x
    //   e31 = -basis(5) → coeffs[5] = -d_y  (e3∧e1 reverses to -e13)
    //   e12 = basis(3)  → coeffs[3] = +d_z
    let d1 = [l1.coeffs[6], -l1.coeffs[5], l1.coeffs[3]];
    let d2 = [l2.coeffs[6], -l2.coeffs[5], l2.coeffs[3]];

    // Moment from the ideal bivector part (blade indices 9, 10, 12).
    //   e01 = -basis(9)  → coeffs[9]  = -m_x
    //   e02 = -basis(10) → coeffs[10] = -m_y
    //   e03 = -basis(12) → coeffs[12] = -m_z
    let m1 = [-l1.coeffs[9], -l1.coeffs[10], -l1.coeffs[12]];
    let m2 = [-l2.coeffs[9], -l2.coeffs[10], -l2.coeffs[12]];

    // A point on each line: p = d × m / |d|²
    let p1 = cross_div(d1, m1);
    let p2 = cross_div(d2, m2);

    // Standard skew-line closest-point formula.
    skew_closest(p1, d1, p2, d2)
}

/// Cross product `a × b`.
#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `a × b / |a|²`, returning the zero vector when `|a|` is degenerate.
#[inline]
fn cross_div(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    let len_sq = dot(a, a);
    if len_sq < 1e-24 {
        return [0.0; 3];
    }
    let c = cross(a, b);
    [c[0] / len_sq, c[1] / len_sq, c[2] / len_sq]
}

/// Closest-point algorithm for two lines given as (point, direction) pairs.
/// Returns (distance, point_on_line1, point_on_line2).
fn skew_closest(
    p1: [f64; 3],
    d1: [f64; 3],
    p2: [f64; 3],
    d2: [f64; 3],
) -> (f64, [f64; 3], [f64; 3]) {
    let w = sub(p1, p2);
    let a = dot(d1, d1);
    let b = dot(d1, d2);
    let c = dot(d2, d2);
    let d = dot(d1, w);
    let e = dot(d2, w);
    let denom = a * c - b * b;

    let (s, t) = if denom.abs() < 1e-12 {
        // Parallel lines: project onto the other line's direction.
        (0.0, d / a.max(1e-24))
    } else {
        ((b * e - c * d) / denom, (a * e - b * d) / denom)
    };

    let q1 = [p1[0] + s * d1[0], p1[1] + s * d1[1], p1[2] + s * d1[2]];
    let q2 = [p2[0] + t * d2[0], p2[1] + t * d2[1], p2[2] + t * d2[2]];
    let diff = sub(q1, q2);
    (garust_core::Real::sqrt(dot(diff, diff)), q1, q2)
}

/// An axis-aligned bounding box (AABB) centred at `center` with half-extents
/// `half` along each axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// World-space centre.
    pub center: [f64; 3],
    /// Half-widths along x, y, z.
    pub half: [f64; 3],
}

impl Aabb {
    /// Test for overlap with another AABB using the Separating Axis Theorem.
    ///
    /// Returns a `Contact` if the boxes overlap, with normal pointing from
    /// `self` toward `other` along the axis of minimum penetration.
    /// Returns `None` if they are separated on any axis.
    pub fn vs_aabb(&self, other: &Self) -> Option<Contact> {
        let mut min_depth = f64::MAX;
        let mut min_axis = 0;
        let mut min_sign = 1.0_f64;

        for i in 0..3 {
            let delta = other.center[i] - self.center[i];
            let overlap = self.half[i] + other.half[i] - delta.abs();
            if overlap <= 0.0 {
                return None;
            }
            if overlap < min_depth {
                min_depth = overlap;
                min_axis = i;
                min_sign = if delta >= 0.0 { 1.0 } else { -1.0 };
            }
        }

        let mut normal = [0.0; 3];
        normal[min_axis] = min_sign;
        // Contact point: midpoint of the two closest face centres.
        let pt = [
            0.5 * (self.center[0] + other.center[0]),
            0.5 * (self.center[1] + other.center[1]),
            0.5 * (self.center[2] + other.center[2]),
        ];
        Some(Contact {
            normal,
            depth: min_depth,
            point: pt,
        })
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
        resolve_pair(&mut a, &mut b, &hit, 1.0, 0.0);

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
        resolve_pair(&mut a, &mut b, &hit, 1.0, 0.0);

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
        resolve_pair(&mut a, &mut b, &hit, 0.0, 0.0); // perfectly inelastic

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
        resolve_pair(&mut a, &mut b, &hit, 1.0, 0.0);
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
        resolve_static(&mut elastic, &hit, 1.0, 0.0);
        assert!((elastic.linear_momentum[1] - 4.0).abs() < 1e-12); // v = +2

        // Half restitution: half the speed comes back.
        let mut damped = ball;
        resolve_static(&mut damped, &hit, 0.5, 0.0);
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
            point: [0.0, 0.0, 0.0],
        };
        resolve_static(&mut ball, &hit, 1.0, 0.0);
        assert_eq!(ball.linear_momentum, [0.0, 1.0, 0.0]);
    }

    // --- PGA flat geometry ------------------------------------------------

    #[test]
    fn point_plane_distance_positive_side() {
        use super::point_plane_distance;
        use garust_core::Pga3;
        // Plane x = 1: normal e1, offset -1 → Pga3::plane(1,0,0,-1)
        let pl = Pga3::plane(1.0, 0.0, 0.0, -1.0);
        assert!((point_plane_distance([2.0, 0.0, 0.0], &pl) - 1.0).abs() < 1e-12);
        assert!((point_plane_distance([1.0, 0.0, 0.0], &pl)).abs() < 1e-12);
        assert!((point_plane_distance([0.0, 0.0, 0.0], &pl) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn point_plane_distance_unnormalized_normal() {
        use super::point_plane_distance;
        use garust_core::Pga3;
        // Plane 2x = 2 → normal (2,0,0), offset -2. Distance = |2*3 - 2| / 2 = 2.
        let pl = Pga3::plane(2.0, 0.0, 0.0, -2.0);
        assert!((point_plane_distance([3.0, 0.0, 0.0], &pl) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn line_line_closest_orthogonal_skew_lines() {
        use super::line_line_closest;
        use garust_core::Pga3;
        // x-axis and the y-axis shifted 3 units along z: distance should be 3.
        let x_axis = Pga3::point(0.0, 0.0, 0.0).line_through(&Pga3::point(1.0, 0.0, 0.0));
        let y_at_z3 = Pga3::point(0.0, 0.0, 3.0).line_through(&Pga3::point(0.0, 1.0, 3.0));
        let (dist, _, _) = line_line_closest(&x_axis, &y_at_z3);
        assert!((dist - 3.0).abs() < 1e-10, "dist = {dist}");
    }

    #[test]
    fn line_line_closest_same_line_gives_zero() {
        use super::line_line_closest;
        use garust_core::Pga3;
        let l = Pga3::point(1.0, 2.0, 3.0).line_through(&Pga3::point(4.0, 5.0, 6.0));
        let (dist, _, _) = line_line_closest(&l, &l);
        assert!(dist < 1e-10, "self-distance = {dist}");
    }

    #[test]
    fn aabb_vs_aabb_overlap_and_clearance() {
        use super::Aabb;
        let a = Aabb {
            center: [0.0, 0.0, 0.0],
            half: [1.0, 1.0, 1.0],
        };
        let near = Aabb {
            center: [1.5, 0.0, 0.0],
            half: [1.0, 1.0, 1.0],
        };
        let far = Aabb {
            center: [3.0, 0.0, 0.0],
            half: [1.0, 1.0, 1.0],
        };
        let hit = a.vs_aabb(&near).unwrap();
        assert_eq!(hit.normal, [1.0, 0.0, 0.0]);
        assert!((hit.depth - 0.5).abs() < 1e-12, "depth = {}", hit.depth);
        assert!(a.vs_aabb(&far).is_none());
    }

    #[test]
    fn aabb_vs_aabb_minimum_axis_wins() {
        use super::Aabb;
        // Slight y-overlap vs larger x-overlap: normal should be along y.
        let a = Aabb {
            center: [0.0, 0.0, 0.0],
            half: [2.0, 1.0, 1.0],
        };
        let b = Aabb {
            center: [0.5, 1.8, 0.0],
            half: [2.0, 1.0, 1.0],
        };
        let hit = a.vs_aabb(&b).unwrap();
        assert_eq!(hit.normal, [0.0, 1.0, 0.0]);
        assert!((hit.depth - 0.2).abs() < 1e-12, "depth = {}", hit.depth);
    }

    // --- Friction (issue #44) -----------------------------------------------

    /// A ball moving toward the ground with lateral (x) velocity hits with
    /// μ > 0. Friction should:
    ///   – reduce the lateral linear momentum
    ///   – transfer it to angular momentum via the lever arm r × J
    #[test]
    fn friction_transfers_lateral_velocity_to_angular_momentum() {
        let mass = 1.0_f64;
        let radius = 0.5_f64;
        let mut body = RigidBody::new(mass);
        body.position = [0.0, radius, 0.0];
        // Falling and sliding: negative vy, positive vx.
        body.linear_momentum = [2.0 * mass, -3.0 * mass, 0.0]; // px=2, py=-3

        // Measure initial angular momentum magnitude (should be zero).
        let am_norm_before = |b: &RigidBody| -> f64 {
            let c = &b.angular_momentum.coeffs;
            c.iter().map(|x| x * x).sum::<f64>()
        };
        let ang_before = am_norm_before(&body);

        let hit = Contact {
            normal: [0.0, 1.0, 0.0],
            depth: 0.001,
            // Contact at base of ball (lever arm = –radius in y)
            point: [0.0, 0.0, 0.0],
        };
        let mu = 0.4_f64;

        resolve_static(&mut body, &hit, 0.0, mu);

        let p_x_after = body.linear_momentum[0];
        let ang_after = am_norm_before(&body); // reuse helper

        // Friction must have reduced lateral momentum.
        assert!(
            p_x_after < 2.0 * mass,
            "lateral momentum not reduced: p_x = {p_x_after}"
        );
        // Lever arm must have induced angular momentum.
        assert!(
            ang_after > ang_before,
            "no angular momentum induced: before={ang_before}, after={ang_after}"
        );
    }
}
