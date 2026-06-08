//! Hamiltonian dynamics on multivector phase space — symplectic integration.
//!
//! A [`Phase`] is a pair `(q, p)` of multivectors — position and momentum.
//! For a **separable** Hamiltonian `H(q, p) = T(p) + V(q)`, one
//! [`Phase::leapfrog`] step advances the state by the Störmer–Verlet
//! (leapfrog / velocity-Verlet) scheme:
//!
//! ```text
//! p½    = p − (dt/2) ∂V/∂q(q)
//! q'    = q + dt     ∂T/∂p(p½)
//! p'    = p½ − (dt/2) ∂V/∂q(q')
//! ```
//!
//! It is **symplectic**: it conserves a shadow Hamiltonian exactly, so the
//! true energy stays *bounded* over arbitrarily long integrations rather than
//! drifting — the property that makes "conserve by construction" real, and
//! that plain explicit Euler / RK4 lack.
//!
//! The gradients `∂T/∂p` and `∂V/∂q` are supplied as closures: write them in
//! closed form, or obtain them from a Hamiltonian with
//! [`multivector_derivative`](crate::multivector_derivative) for AD-driven
//! dynamics. General, *non-separable* Hamiltonians need an implicit symplectic
//! method, left to a future round.
//!
//! ```
//! use garust_core::{dynamics::Phase, Vga3};
//!
//! // A unit-mass harmonic oscillator, spring constant k, along e1.
//! let k = 0.75_f64;
//! let mut s = Phase::new(Vga3::basis(1), Vga3::zero()); // q = e1, p = 0
//! let (grad_t, grad_v) = (|p: &Vga3| *p, |q: &Vga3| *q * k);
//! for _ in 0..1000 {
//!     s = s.leapfrog(0.01, grad_t, grad_v);
//! }
//! // Energy ½‖p‖² + ½k‖q‖² is conserved to O(dt²).
//! let e = 0.5 * s.p.scalar_product(&s.p) + 0.5 * k * s.q.scalar_product(&s.q);
//! assert!((e - 0.5 * k).abs() < 1e-3);
//! ```

use crate::algebra::Algebra;
use crate::multivector::Multivector;
use crate::scalar::Real;

/// A point in multivector phase space: position `q` and momentum `p`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Phase<A: Algebra, T: Real> {
    /// Position multivector.
    pub q: Multivector<A, T>,
    /// Momentum multivector.
    pub p: Multivector<A, T>,
}

impl<A: Algebra, T: Real> Phase<A, T> {
    /// A phase point with position `q` and momentum `p`.
    pub fn new(q: Multivector<A, T>, p: Multivector<A, T>) -> Self {
        Self { q, p }
    }

    /// One symplectic leapfrog (Störmer–Verlet) step of size `dt` for a
    /// separable Hamiltonian `H = T(p) + V(q)`, given `grad_t = ∂T/∂p` and
    /// `grad_v = ∂V/∂q`.
    ///
    /// Returns the advanced state; `q` and `p` end the step synchronized in
    /// time (velocity-Verlet form). Being symplectic, repeated steps keep the
    /// energy bounded rather than drifting.
    pub fn leapfrog<FT, FV>(&self, dt: T, grad_t: FT, grad_v: FV) -> Self
    where
        FT: Fn(&Multivector<A, T>) -> Multivector<A, T>,
        FV: Fn(&Multivector<A, T>) -> Multivector<A, T>,
    {
        let half = dt * T::from_f64(0.5);
        let p_half = self.p - grad_v(&self.q) * half;
        let q_next = self.q + grad_t(&p_half) * dt;
        let p_next = p_half - grad_v(&q_next) * half;
        Self {
            q: q_next,
            p: p_next,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Phase;
    use crate::Vga3;

    // Unit-mass harmonic oscillator along e1: H = ½‖p‖² + ½k‖q‖²,
    // so ∂T/∂p = p and ∂V/∂q = k·q.
    const K: f64 = 0.75;

    fn grad_t(p: &Vga3) -> Vga3 {
        *p
    }
    fn grad_v(q: &Vga3) -> Vga3 {
        *q * K
    }
    fn energy(s: &Phase<crate::Vga3Sig, f64>) -> f64 {
        0.5 * s.p.scalar_product(&s.p) + 0.5 * K * s.q.scalar_product(&s.q)
    }

    #[test]
    fn leapfrog_conserves_energy_over_a_long_run() {
        let mut s = Phase::new(Vga3::basis(1), Vga3::zero());
        let e0 = energy(&s);
        for _ in 0..10_000 {
            s = s.leapfrog(0.01, grad_t, grad_v);
        }
        // Symplectic ⇒ energy stays bounded (here to O(dt²)), no secular drift.
        assert!(
            (energy(&s) - e0).abs() < 1e-3,
            "energy drifted: {} vs {e0}",
            energy(&s)
        );
    }

    #[test]
    fn leapfrog_actually_integrates_the_oscillator() {
        // Start at q = e1, p = 0. Half a period later the oscillator has
        // swung to the opposite extreme, q ≈ −e1, p ≈ 0.
        let mut s = Phase::new(Vga3::basis(1), Vga3::zero());
        // Half period = (τ/2)/√k for a unit-mass oscillator (τ-only convention).
        let half_period_steps = (core::f64::consts::TAU / (2.0 * K.sqrt()) / 0.01) as usize;
        for _ in 0..half_period_steps {
            s = s.leapfrog(0.01, grad_t, grad_v);
        }
        assert!(s.q.coeffs[1] < -0.9, "q did not invert: {}", s.q.coeffs[1]);
        assert!(
            s.p.coeffs[1].abs() < 0.1,
            "p not near rest: {}",
            s.p.coeffs[1]
        );
    }
}
