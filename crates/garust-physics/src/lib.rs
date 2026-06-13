//! # garust-physics — rigid-body dynamics on the PGA kernel
//!
//! The first slice of RFC-010: a free / torqued **rigid body** whose
//! orientation is a `Motor` (a versor of `Cl(3,0,1)`) and whose angular
//! momentum is a single grade-2 **bivector**. The equations of motion are
//! the geometric form of Euler's equations — rotation lives entirely in
//! bivector arithmetic, no quaternion/axis bookkeeping.
//!
//! ## What this version covers
//!
//! Full **6-DOF** motion of a free or forced rigid body: rotation about the
//! centre of mass and translation of it. The rotational half is the keystone
//! of RFC-010, the **symplectic Lie-group integrator** — momentum is advanced
//! by an explicit symplectic *splitting* (exact rotation per principal axis,
//! conserving `‖Π‖` exactly and energy without secular drift) and the
//! orientation is transported *on the group* by `exp`, staying a unit versor
//! by construction. The translational half is a linear leapfrog on the centre
//! of mass; the two are independent for a force applied at the centre of
//! mass, so the composed step is symplectic in all six degrees of freedom.
//!
//! Contacts and constraints (a `World` with collision response) are the next
//! slices of RFC-010.
//!
//! ```
//! use garust_physics::{Inertia, RigidBody};
//! use garust_core::Pga3;
//!
//! // An asymmetric body, spun about its first principal axis and drifting.
//! let inertia = Inertia::principal([2.0, 3.0, 4.0]);
//! let mut body = RigidBody::new(2.0); // mass 2
//! body.angular_momentum = Pga3::basis(0b0110); // Π = e23  (spin about x)
//! body.linear_momentum = [4.0, 0.0, 0.0]; // drifting along +x
//!
//! let e0 = body.kinetic_energy(&inertia);
//! for _ in 0..1000 {
//!     body = body.step(0.01, &inertia, [0.0; 3], Pga3::zero()); // free
//! }
//! // Symplectic ⇒ total (rotational + translational) energy stays put,
//! assert!((body.kinetic_energy(&inertia) - e0).abs() < 1e-6);
//! // and the centre of mass has drifted ~40 units along x.
//! assert!((body.position[0] - 20.0).abs() < 1e-9);
//! ```

#![cfg_attr(not(test), no_std)]
#![deny(missing_docs)]

use garust_core::Pga3;
use garust_geo::Motor;

/// A rigid body's **inertia**, as principal moments about its three body
/// axes — the diagonal of the inertia tensor in the principal frame.
///
/// Maps an angular-velocity bivector to an angular-momentum bivector and
/// back. The three moments correspond to the principal planes `e23`, `e31`,
/// `e12` (rotation about the body x-, y-, z-axes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Inertia {
    moments: [f64; 3],
}

impl Inertia {
    /// Build an inertia from the three principal moments `[Ix, Iy, Iz]`
    /// (each must be positive).
    pub fn principal(moments: [f64; 3]) -> Self {
        Self { moments }
    }

    /// The three principal moments `[Ix, Iy, Iz]`.
    pub fn moments(&self) -> [f64; 3] {
        self.moments
    }

    /// The three right-handed principal planes as unit bivectors — `e23`,
    /// `e31`, `e12` — each squaring to `−1`. Built from generator products
    /// so their orientation is correct by construction.
    pub fn principal_planes() -> [Pga3; 3] {
        [
            Pga3::basis(2) * Pga3::basis(4), // e2 e3 = e23  (about x)
            Pga3::basis(4) * Pga3::basis(1), // e3 e1 = e31  (about y)
            Pga3::basis(1) * Pga3::basis(2), // e1 e2 = e12  (about z)
        ]
    }

    /// Angular momentum `Π = 𝓘(ω)` from an angular-velocity bivector.
    pub fn momentum_of(&self, omega: &Pga3) -> Pga3 {
        let mut out = Pga3::zero();
        for (b, &i) in Self::principal_planes().iter().zip(self.moments.iter()) {
            // Component of ω along the unit plane b (b² = −1).
            let c = -omega.scalar_product(b);
            out += *b * (i * c);
        }
        out
    }

    /// Angular velocity `ω = 𝓘⁻¹(Π)` from an angular-momentum bivector.
    pub fn velocity_of(&self, momentum: &Pga3) -> Pga3 {
        let mut out = Pga3::zero();
        for (b, &i) in Self::principal_planes().iter().zip(self.moments.iter()) {
            let c = -momentum.scalar_product(b);
            out += *b * (c / i);
        }
        out
    }
}

/// A rigid body with full 6-DOF state: an orientation (`Motor`) and
/// body-frame angular momentum about the centre of mass, plus the centre of
/// mass's world position and linear momentum.
///
/// Momentum is stored rather than velocity because momentum is what the
/// symplectic integrator advances (and what an impulse adds to); recover the
/// velocities with [`RigidBody::angular_velocity`] / [`RigidBody::velocity`],
/// and the full body→world transform with [`RigidBody::pose`].
///
/// Linear quantities are plain world-frame 3-vectors by design: the centre of
/// mass *is* a point and its momentum a vector, with no Clifford subtlety to
/// model — the geometric-algebra structure is carried by the orientation
/// `Motor`, the angular-momentum bivector, and the `exp`-based rotation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidBody {
    /// Orientation of the body frame about its centre of mass.
    pub orientation: Motor<f64>,
    /// Body-frame angular momentum `Π` about the centre of mass, a grade-2
    /// bivector.
    pub angular_momentum: Pga3,
    /// Centre-of-mass position in the world frame.
    pub position: [f64; 3],
    /// Centre-of-mass linear momentum in the world frame.
    pub linear_momentum: [f64; 3],
    /// Total mass (must be positive for the linear dynamics).
    pub mass: f64,
}

impl RigidBody {
    /// A body of the given `mass` at the world origin, identity orientation,
    /// and completely at rest.
    pub fn new(mass: f64) -> Self {
        Self {
            orientation: Motor::identity(),
            angular_momentum: Pga3::zero(),
            position: [0.0; 3],
            linear_momentum: [0.0; 3],
            mass,
        }
    }

    /// A unit-mass body at the origin, at rest — `new(1.0)`.
    pub fn at_rest() -> Self {
        Self::new(1.0)
    }

    /// A unit-mass body with the given orientation and body-frame angular
    /// velocity (converted to momentum via the inertia), at the origin with
    /// no linear motion. Set [`mass`](RigidBody::mass) /
    /// [`linear_momentum`](RigidBody::linear_momentum) afterwards for the
    /// general case.
    pub fn spinning(orientation: Motor<f64>, inertia: &Inertia, omega: Pga3) -> Self {
        Self {
            orientation,
            angular_momentum: inertia.momentum_of(&omega),
            ..Self::new(1.0)
        }
    }

    /// The full body→world transform `T(position) ∘ orientation`: rotate
    /// about the centre of mass, then translate it to its world position.
    pub fn pose(&self) -> Motor<f64> {
        Motor::translator(self.position[0], self.position[1], self.position[2]) * self.orientation
    }

    /// The centre-of-mass velocity `v = p/m`.
    pub fn velocity(&self) -> [f64; 3] {
        [
            self.linear_momentum[0] / self.mass,
            self.linear_momentum[1] / self.mass,
            self.linear_momentum[2] / self.mass,
        ]
    }

    /// The body-frame angular velocity `ω = 𝓘⁻¹(Π)`.
    pub fn angular_velocity(&self, inertia: &Inertia) -> Pga3 {
        inertia.velocity_of(&self.angular_momentum)
    }

    /// Rotational kinetic energy `½ ⟨Π, 𝓘⁻¹Π⟩ = ½ Σ Πₖ²/Iₖ`.
    pub fn rotational_kinetic_energy(&self, inertia: &Inertia) -> f64 {
        let mut e = 0.0;
        for (b, &i) in Inertia::principal_planes()
            .iter()
            .zip(inertia.moments().iter())
        {
            let c = -self.angular_momentum.scalar_product(b);
            e += c * c / i;
        }
        0.5 * e
    }

    /// Translational kinetic energy `‖p‖²/(2m)`.
    pub fn translational_kinetic_energy(&self) -> f64 {
        let p = &self.linear_momentum;
        (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) / (2.0 * self.mass)
    }

    /// Total kinetic energy — rotational plus translational. Conserved (no
    /// secular drift) for a free body, since both halves are integrated
    /// symplectically.
    pub fn kinetic_energy(&self, inertia: &Inertia) -> f64 {
        self.rotational_kinetic_energy(inertia) + self.translational_kinetic_energy()
    }

    /// The squared norm of the angular momentum, `‖Π‖² = Σ Πₖ²` — a Casimir
    /// the free-body integrator conserves exactly.
    pub fn angular_momentum_squared(&self) -> f64 {
        -self.angular_momentum.scalar_product(&self.angular_momentum)
    }

    /// Angular momentum in the **world** frame, `M Π M̃` — conserved (in the
    /// world frame) for a torque-free body.
    pub fn world_angular_momentum(&self) -> Pga3 {
        self.orientation.apply(&self.angular_momentum)
    }

    /// The rotational half-step: the symplectic free-body splitting about the
    /// centre of mass, returning the advanced orientation and body-frame
    /// angular momentum.
    fn step_rotational(&self, dt: f64, inertia: &Inertia, torque: Pga3) -> (Motor<f64>, Pga3) {
        let planes = Inertia::principal_planes();
        let moments = inertia.moments();

        let mut pi = self.angular_momentum + torque * (0.5 * dt); // half-kick
        let mut versor = self.orientation.versor();

        // Symmetric (Strang) split over the three axes: x½ y½ z₁ y½ x½.
        for (k, frac) in [
            (0usize, 0.5),
            (1usize, 0.5),
            (2usize, 1.0),
            (1usize, 0.5),
            (0usize, 0.5),
        ] {
            let b = planes[k];
            let comp = -pi.scalar_product(&b); // Πₖ along this principal plane
            let phi = (comp / moments[k]) * frac * dt; // rotation angle this sub-step
            let r = (b * (-0.5 * phi)).exp(); // rotor through +phi about axis k
                                              // Orientation transports by r (M ← M·r); the body momentum
                                              // precesses by the *reverse* rotor, the pairing that keeps the
                                              // world momentum M Π M̃ invariant under each sub-step.
            pi = r.reverse().sandwich(&pi); // r̃ Π r
            versor = versor * r;
        }

        pi += torque * (0.5 * dt); // half-kick
        (Motor::from_versor(versor.normalized()), pi)
    }

    /// Advance the body by `dt` under a constant world-frame `force` (applied
    /// at the centre of mass) and body-frame `torque` bivector. Pass
    /// `[0.0; 3]` and `Pga3::zero()` for free motion.
    ///
    /// Rotation about the centre of mass and translation of it are
    /// **independent** for a force applied at the centre of mass, so the step
    /// composes two symplectic integrators on disjoint state: the rotational
    /// splitting (orientation transported on the group by `exp`, conserving
    /// `‖Π‖` and energy) and a linear leapfrog (kick–drift–kick) on the
    /// centre of mass. (Model a force applied off-centre as a force at the
    /// centre of mass plus the torque it induces.)
    pub fn step(&self, dt: f64, inertia: &Inertia, force: [f64; 3], torque: Pga3) -> Self {
        let (orientation, angular_momentum) = self.step_rotational(dt, inertia, torque);

        // Linear leapfrog on the centre of mass: kick, drift, kick.
        let mut linear_momentum = self.linear_momentum;
        let mut position = self.position;
        for ((p, x), &f) in linear_momentum
            .iter_mut()
            .zip(position.iter_mut())
            .zip(force.iter())
        {
            *p += f * (0.5 * dt);
            *x += *p / self.mass * dt;
            *p += f * (0.5 * dt);
        }

        Self {
            orientation,
            angular_momentum,
            position,
            linear_momentum,
            mass: self.mass,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Inertia, RigidBody};
    use garust_core::Pga3;
    use proptest::prelude::*;

    fn max_coeff_diff(a: &Pga3, b: &Pga3) -> f64 {
        a.coeffs
            .iter()
            .zip(b.coeffs.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    // Component of a bivector along a principal plane.
    fn comp(pi: &Pga3, k: usize) -> f64 {
        -pi.scalar_product(&Inertia::principal_planes()[k])
    }

    #[test]
    fn momentum_velocity_round_trip() {
        let inertia = Inertia::principal([2.0, 3.0, 4.0]);
        let omega = Pga3::basis(0b0110) * 1.5 + Pga3::basis(0b0011) * -0.7;
        let back = inertia.velocity_of(&inertia.momentum_of(&omega));
        assert!(max_coeff_diff(&omega, &back) < 1e-12);
    }

    #[test]
    fn free_body_conserves_energy_and_momentum_norm() {
        let inertia = Inertia::principal([2.0, 3.0, 4.0]);
        let mut body = RigidBody::at_rest();
        // A generic spin touching all three axes.
        body.angular_momentum =
            Pga3::basis(0b0110) * 1.3 + Pga3::basis(0b0101) * -0.9 + Pga3::basis(0b0011) * 0.5;
        let e0 = body.kinetic_energy(&inertia);
        let l0 = body.angular_momentum_squared();
        let world0 = body.world_angular_momentum();

        for _ in 0..10_000 {
            body = body.step(0.01, &inertia, [0.0; 3], Pga3::zero());
        }

        // ‖Π‖² is a Casimir — conserved to machine precision.
        assert!(
            (body.angular_momentum_squared() - l0).abs() < 1e-9,
            "‖Π‖² drift: {} vs {l0}",
            body.angular_momentum_squared()
        );
        // Energy is bounded (symplectic), no secular drift.
        assert!(
            (body.kinetic_energy(&inertia) - e0).abs() < 1e-4,
            "energy drift: {} vs {e0}",
            body.kinetic_energy(&inertia)
        );
        // World-frame angular momentum is conserved for a free body.
        assert!(
            max_coeff_diff(&body.world_angular_momentum(), &world0) < 1e-3,
            "world angular momentum drifted"
        );
    }

    #[test]
    fn orientation_stays_a_unit_versor() {
        let inertia = Inertia::principal([1.0, 2.0, 3.0]);
        let mut body = RigidBody::at_rest();
        body.angular_momentum = Pga3::basis(0b0101) * 2.0 + Pga3::basis(0b0011) * 0.3;
        for _ in 0..5_000 {
            body = body.step(0.01, &inertia, [0.0; 3], Pga3::zero());
        }
        let v = body.orientation.versor();
        // R ~R = 1.
        let rr = (v * v.reverse()).scalar_part();
        assert!((rr - 1.0).abs() < 1e-9, "‖R‖² = {rr}");
    }

    #[test]
    fn dzhanibekov_intermediate_axis_flips() {
        // Ix < Iy < Iz ⇒ the y (intermediate) axis is unstable: a body
        // spun about it periodically flips. Spin mostly about y, with a
        // tiny perturbation, and watch the y component reverse sign.
        let inertia = Inertia::principal([2.0, 3.0, 4.0]);
        let planes = Inertia::principal_planes();
        let mut body = RigidBody::at_rest();
        body.angular_momentum = planes[1] * 5.0  // big spin about the y (intermediate) axis
            + planes[0] * 0.01; // tiny nudge about x
        let l0 = body.angular_momentum_squared();

        let start = comp(&body.angular_momentum, 1); // Π_y > 0
        let mut min_y = start;
        let mut max_y = start;
        for _ in 0..20_000 {
            body = body.step(0.005, &inertia, [0.0; 3], Pga3::zero());
            let y = comp(&body.angular_momentum, 1);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        // The intermediate-axis component swings through both signs — the
        // flip — while ‖Π‖² stays fixed.
        assert!(start > 0.0);
        assert!(min_y < -4.0, "no flip: min Π_y = {min_y}");
        assert!(max_y > 4.0, "did not return: max Π_y = {max_y}");
        assert!((body.angular_momentum_squared() - l0).abs() < 1e-6);
    }

    // --- 6-DOF: translation ------------------------------------------------

    #[test]
    fn free_body_drifts_in_a_straight_line() {
        let inertia = Inertia::principal([2.0, 3.0, 4.0]);
        let mut body = RigidBody::new(2.0);
        body.angular_momentum = Pga3::basis(0b0110) * 1.1; // also spinning
        body.linear_momentum = [4.0, -2.0, 1.0];
        let e0 = body.kinetic_energy(&inertia);
        let p0 = body.linear_momentum;

        for _ in 0..1000 {
            body = body.step(0.01, &inertia, [0.0; 3], Pga3::zero());
        }

        // Linear momentum is conserved exactly (no force).
        assert_eq!(body.linear_momentum, p0);
        // CoM has drifted by velocity × time = (p/m)·10.
        let t = 10.0;
        for (i, (&p, &x)) in p0.iter().zip(body.position.iter()).enumerate() {
            let expect = p / 2.0 * t;
            assert!((x - expect).abs() < 1e-9, "axis {i}: {x}");
        }
        // Total (rotational + translational) energy is conserved.
        assert!((body.kinetic_energy(&inertia) - e0).abs() < 1e-6);
    }

    #[test]
    fn constant_force_accelerates_the_centre_of_mass() {
        // Leapfrog is exact for a constant force: from rest, x = ½ a t².
        let inertia = Inertia::principal([1.0, 1.0, 1.0]);
        let mut body = RigidBody::new(2.0); // m = 2
        let force = [6.0, 0.0, 0.0]; // a = F/m = 3
        for _ in 0..1000 {
            body = body.step(0.01, &inertia, force, Pga3::zero());
        }
        let t = 10.0;
        let a = 3.0;
        assert!((body.position[0] - 0.5 * a * t * t).abs() < 1e-9); // 150
        assert!((body.linear_momentum[0] - body.mass * a * t).abs() < 1e-9); // 60
    }

    #[test]
    fn pose_places_a_body_point_at_the_centre_of_mass() {
        let mut body = RigidBody::new(1.0);
        body.position = [1.0, 2.0, 3.0];
        let here = body
            .pose()
            .apply(&Pga3::point(0.0, 0.0, 0.0))
            .cleaned(1e-10);
        assert!(max_coeff_diff(&here, &Pga3::point(1.0, 2.0, 3.0)) < 1e-9);
    }

    #[test]
    fn rotation_and_translation_are_independent() {
        // A body that both spins and drifts: each invariant holds as if the
        // other motion were absent.
        let inertia = Inertia::principal([2.0, 3.0, 4.0]);
        let mut body = RigidBody::new(1.5);
        body.angular_momentum =
            Pga3::basis(0b0110) * 1.3 + Pga3::basis(0b0101) * -0.7 + Pga3::basis(0b0011) * 0.4;
        body.linear_momentum = [2.0, 5.0, -1.0];
        let l0 = body.angular_momentum_squared();
        let world0 = body.world_angular_momentum();

        for _ in 0..5000 {
            body = body.step(0.01, &inertia, [0.0; 3], Pga3::zero());
        }
        // Rotational invariants survive the simultaneous translation.
        assert!((body.angular_momentum_squared() - l0).abs() < 1e-9);
        assert!(max_coeff_diff(&body.world_angular_momentum(), &world0) < 1e-3);
        // And the orientation is still a unit versor.
        let v = body.orientation.versor();
        assert!(((v * v.reverse()).scalar_part() - 1.0).abs() < 1e-9);
    }

    // --- Property-based conservation ---------------------------------------

    proptest! {
        // Over random inertias, spins, drifts, masses, and step sizes, the
        // integrator's exact invariants hold: ‖Π‖² and the orientation norm
        // are preserved, and a free body's linear momentum never changes.
        #[test]
        fn free_motion_preserves_the_invariants(
            ix in 0.5_f64..5.0, iy in 0.5_f64..5.0, iz in 0.5_f64..5.0,
            a in -3.0_f64..3.0, b in -3.0_f64..3.0, c in -3.0_f64..3.0,
            px in -3.0_f64..3.0, py in -3.0_f64..3.0, pz in -3.0_f64..3.0,
            mass in 0.5_f64..5.0,
            dt in 0.001_f64..0.02,
        ) {
            let inertia = Inertia::principal([ix, iy, iz]);
            let planes = Inertia::principal_planes();
            let mut body = RigidBody::new(mass);
            body.angular_momentum = planes[0] * a + planes[1] * b + planes[2] * c;
            body.linear_momentum = [px, py, pz];
            let l0 = body.angular_momentum_squared();

            for _ in 0..500 {
                body = body.step(dt, &inertia, [0.0; 3], Pga3::zero());
            }

            // ‖Π‖² is a Casimir (each sub-step is a norm-preserving sandwich).
            prop_assert!((body.angular_momentum_squared() - l0).abs() < 1e-7);
            // The orientation stays a unit versor.
            let v = body.orientation.versor();
            prop_assert!(((v * v.reverse()).scalar_part() - 1.0).abs() < 1e-7);
            // Free linear momentum is untouched (force is zero throughout).
            prop_assert_eq!(body.linear_momentum, [px, py, pz]);
        }
    }
}
