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
//! Rotational dynamics about the centre of mass: a spinning body, free or
//! under an applied torque. This is the classic test bed — it is where
//! energy/momentum conservation and the Dzhanibekov (tennis-racket) effect
//! live — and it exercises the keystone of RFC-010, the **symplectic
//! Lie-group integrator**: momentum is advanced by an explicit symplectic
//! *splitting* method (exact rotation per principal axis, so it conserves
//! `‖Π‖` exactly and energy without secular drift), and the orientation is
//! transported *on the group* by the exponential map, so it stays a unit
//! versor by construction.
//!
//! Full 6-DOF motion (coupled translation), contacts, and constraints are
//! the next slices of RFC-010.
//!
//! ```
//! use garust_physics::{Inertia, RigidBody};
//! use garust_core::Pga3;
//!
//! // An asymmetric body, spun mostly about its first principal axis.
//! let inertia = Inertia::principal([2.0, 3.0, 4.0]);
//! let mut body = RigidBody::at_rest();
//! body.angular_momentum = Pga3::basis(0b0110); // Π = e23  (spin about x)
//!
//! let e0 = body.kinetic_energy(&inertia);
//! for _ in 0..1000 {
//!     body = body.step(0.01, &inertia, Pga3::zero()); // free: zero torque
//! }
//! // Symplectic ⇒ energy and ‖Π‖ stay put.
//! assert!((body.kinetic_energy(&inertia) - e0).abs() < 1e-6);
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

/// A rigid body: an orientation (`Motor`) and a body-frame angular-momentum
/// bivector.
///
/// The momentum is stored rather than the velocity because momentum is what
/// the symplectic integrator advances (and what an impulse adds to); recover
/// the angular velocity with [`RigidBody::angular_velocity`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidBody {
    /// Orientation of the body frame in the world frame.
    pub orientation: Motor<f64>,
    /// Body-frame angular momentum `Π`, a grade-2 bivector.
    pub angular_momentum: Pga3,
}

impl RigidBody {
    /// A body at the identity orientation with zero angular momentum.
    pub fn at_rest() -> Self {
        Self {
            orientation: Motor::identity(),
            angular_momentum: Pga3::zero(),
        }
    }

    /// Build a body from an orientation and a body-frame angular velocity
    /// (converted to momentum via the inertia).
    pub fn spinning(orientation: Motor<f64>, inertia: &Inertia, omega: Pga3) -> Self {
        Self {
            orientation,
            angular_momentum: inertia.momentum_of(&omega),
        }
    }

    /// The body-frame angular velocity `ω = 𝓘⁻¹(Π)`.
    pub fn angular_velocity(&self, inertia: &Inertia) -> Pga3 {
        inertia.velocity_of(&self.angular_momentum)
    }

    /// Rotational kinetic energy `½ ⟨Π, 𝓘⁻¹Π⟩ = ½ Σ Πₖ²/Iₖ`.
    pub fn kinetic_energy(&self, inertia: &Inertia) -> f64 {
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

    /// Advance the body by `dt` under a constant body-frame `torque`
    /// bivector (pass `Pga3::zero()` for free motion).
    ///
    /// Strang composition of a torque half-kick, the symplectic free-body
    /// splitting (exact rotation about each principal axis, second order),
    /// and a second half-kick. The orientation is transported on the group
    /// by `exp`, so it stays a unit versor; the splitting conserves `‖Π‖`
    /// exactly and the energy without secular drift.
    pub fn step(&self, dt: f64, inertia: &Inertia, torque: Pga3) -> Self {
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
        Self {
            orientation: Motor::from_versor(versor.normalized()),
            angular_momentum: pi,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Inertia, RigidBody};
    use garust_core::Pga3;

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
            body = body.step(0.01, &inertia, Pga3::zero());
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
            body = body.step(0.01, &inertia, Pga3::zero());
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
            body = body.step(0.005, &inertia, Pga3::zero());
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
}
