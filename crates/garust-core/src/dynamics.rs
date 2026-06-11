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
//! dynamics.
//!
//! General, **non-separable** Hamiltonians `H(q, p)` — where position and
//! momentum couple, as in relativistic motion, rotating frames, or learned
//! Hamiltonians — classically force a choice between explicit-but-drifting
//! (RK4) and symplectic-but-implicit (midpoint with a nonlinear solve).
//! [`ExtendedPhase::tao_step`] is the third way (Tao 2016): double the phase
//! space with a shadow copy `(x, y)` of `(q, p)`, integrate the augmented
//! Hamiltonian
//!
//! ```text
//! H̄(q, p, x, y) = H(q, y) + H(x, p) + ω·½(‖q − x‖² + ‖p − y‖²)
//! ```
//!
//! whose three pieces each have an *exact, explicit* flow (each `H` copy
//! freezes the variables it differentiates; the binding term is a rotation
//! of the difference coordinates), and compose them in a Strang splitting.
//! The result is explicit, second-order, and symplectic in the extended
//! space — so the true energy `H(q, p)` stays bounded for arbitrarily long
//! runs, exactly like leapfrog in the separable case.
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

/// A point in Tao's *doubled* phase space: the physical pair `(q, p)` plus
/// its shadow copy `(x, y)`, the auxiliary coordinates that make explicit
/// symplectic integration of a **non-separable** Hamiltonian possible (see
/// [`ExtendedPhase::tao_step`] and the module docs).
///
/// Lift a physical state in with [`ExtendedPhase::new`] (which sets
/// `x = q`, `y = p`) and read the physical state back out with
/// [`ExtendedPhase::phase`]. The shadow pair must be *carried between
/// steps* — re-lifting each step would discard the symplectic structure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExtendedPhase<A: Algebra, T: Real> {
    /// Position multivector.
    pub q: Multivector<A, T>,
    /// Momentum multivector.
    pub p: Multivector<A, T>,
    /// Shadow position — Tao's auxiliary copy of `q`.
    pub x: Multivector<A, T>,
    /// Shadow momentum — Tao's auxiliary copy of `p`.
    pub y: Multivector<A, T>,
}

impl<A: Algebra, T: Real> ExtendedPhase<A, T> {
    /// Lift a physical state `(q, p)` into the doubled phase space, with
    /// the shadow copy starting exactly on top of it (`x = q`, `y = p`).
    pub fn new(q: Multivector<A, T>, p: Multivector<A, T>) -> Self {
        Self { q, p, x: q, y: p }
    }

    /// Lift an existing [`Phase`] (same as [`ExtendedPhase::new`]).
    pub fn from_phase(s: &Phase<A, T>) -> Self {
        Self::new(s.q, s.p)
    }

    /// The physical state `(q, p)`, dropping the shadow copy.
    pub fn phase(&self) -> Phase<A, T> {
        Phase {
            q: self.q,
            p: self.p,
        }
    }

    /// One explicit symplectic step of size `dt` for a **non-separable**
    /// Hamiltonian `H(q, p)`, by Tao's extended-phase-space splitting
    /// (M. Tao, *Phys. Rev. E* **94**, 043303, 2016).
    ///
    /// `grad_q = ∂H/∂q` and `grad_p = ∂H/∂p`, each a closure of *both*
    /// arguments `(q, p)` — closed-form or via
    /// [`multivector_derivative`](crate::multivector_derivative). The five
    /// sub-flows (each exact, none implicit) are composed in the
    /// second-order Strang pattern
    /// `φ_A(dt/2) φ_B(dt/2) φ_C(dt) φ_B(dt/2) φ_A(dt/2)`, where `φ_A`/`φ_B`
    /// are the two Hamiltonian copies evaluated at the mixed pairs
    /// `(q, y)` / `(x, p)` and `φ_C` rotates the difference coordinates
    /// `(q − x, p − y)` by the angle `2ω·dt`.
    ///
    /// `omega` (`ω > 0`) is the **binding strength** tying the shadow to
    /// the physical pair: the copies track each other to `O(1/ω)`, so pick
    /// `ω` large against the system's fastest frequency while keeping the
    /// per-step binding rotation `2ω·dt` well under a quarter turn (τ/4).
    /// When `H` *is* separable, this reduces to (twice the work of)
    /// leapfrog — use [`Phase::leapfrog`] there.
    ///
    /// ```
    /// use garust_core::{dynamics::ExtendedPhase, Vga3};
    ///
    /// // Tao's benchmark H = ½(q² + 1)(p² + 1): genuinely non-separable —
    /// // each gradient needs both q and p.
    /// let gq = |q: &Vga3, p: &Vga3| *q * (p.scalar_part().powi(2) + 1.0);
    /// let gp = |q: &Vga3, p: &Vga3| *p * (q.scalar_part().powi(2) + 1.0);
    /// let h = |s: &ExtendedPhase<garust_core::Vga3Sig, f64>| {
    ///     0.5 * (s.q.scalar_part().powi(2) + 1.0) * (s.p.scalar_part().powi(2) + 1.0)
    /// };
    ///
    /// let mut s = ExtendedPhase::new(Vga3::scalar(-3.0), Vga3::zero());
    /// let e0 = h(&s);
    /// for _ in 0..2_000 {
    ///     s = s.tao_step(1e-3, 20.0, gq, gp);
    /// }
    /// assert!((h(&s) - e0).abs() < 1e-4); // energy bounded, no drift
    /// ```
    pub fn tao_step<Gq, Gp>(&self, dt: T, omega: T, grad_q: Gq, grad_p: Gp) -> Self
    where
        Gq: Fn(&Multivector<A, T>, &Multivector<A, T>) -> Multivector<A, T>,
        Gp: Fn(&Multivector<A, T>, &Multivector<A, T>) -> Multivector<A, T>,
    {
        let half = dt * T::from_f64(0.5);
        let mut s = *self;

        // φ_A(dt/2): H(q, y) — its flow freezes (q, y), kicks p, drifts x.
        s.p -= grad_q(&s.q, &s.y) * half;
        s.x += grad_p(&s.q, &s.y) * half;

        // φ_B(dt/2): H(x, p) — freezes (x, p), drifts q, kicks y.
        s.q += grad_p(&s.x, &s.p) * half;
        s.y -= grad_q(&s.x, &s.p) * half;

        // φ_C(dt): the binding term ω·½(‖q−x‖² + ‖p−y‖²) rotates the
        // difference coordinates by 2ω·dt and leaves the sums invariant —
        // solved exactly, no force evaluation.
        let angle = T::from_f64(2.0) * omega * dt;
        let (cos, sin) = (angle.cos(), angle.sin());
        let u = s.q - s.x;
        let v = s.p - s.y;
        let u_rot = u * cos + v * sin;
        let v_rot = v * cos - u * sin;
        let mid = T::from_f64(0.5);
        let sum_q = s.q + s.x;
        let sum_p = s.p + s.y;
        s.q = (sum_q + u_rot) * mid;
        s.x = (sum_q - u_rot) * mid;
        s.p = (sum_p + v_rot) * mid;
        s.y = (sum_p - v_rot) * mid;

        // φ_B(dt/2), φ_A(dt/2): mirror the opening half-flows.
        s.q += grad_p(&s.x, &s.p) * half;
        s.y -= grad_q(&s.x, &s.p) * half;
        s.p -= grad_q(&s.q, &s.y) * half;
        s.x += grad_p(&s.q, &s.y) * half;

        s
    }
}

#[cfg(test)]
mod tests {
    use super::{ExtendedPhase, Phase};
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

    // --- Tao's method: non-separable Hamiltonians -------------------------

    // Tao's own benchmark, H = ½(q² + 1)(p² + 1) on scalar coefficients:
    // genuinely non-separable (∂H/∂q depends on p and vice versa), so
    // leapfrog does not apply at all.
    fn ns_grad_q(q: &Vga3, p: &Vga3) -> Vga3 {
        let (q0, p0) = (q.scalar_part(), p.scalar_part());
        Vga3::scalar(q0 * (p0 * p0 + 1.0))
    }
    fn ns_grad_p(q: &Vga3, p: &Vga3) -> Vga3 {
        let (q0, p0) = (q.scalar_part(), p.scalar_part());
        Vga3::scalar(p0 * (q0 * q0 + 1.0))
    }
    fn ns_energy(s: &ExtendedPhase<crate::Vga3Sig, f64>) -> f64 {
        let (q0, p0) = (s.q.scalar_part(), s.p.scalar_part());
        0.5 * (q0 * q0 + 1.0) * (p0 * p0 + 1.0)
    }

    #[test]
    fn tao_conserves_nonseparable_energy_over_a_long_run() {
        let mut s = ExtendedPhase::new(Vga3::scalar(-3.0), Vga3::zero());
        let e0 = ns_energy(&s);
        for _ in 0..50_000 {
            s = s.tao_step(1e-3, 20.0, ns_grad_q, ns_grad_p);
        }
        // Symplectic in the extended space ⇒ H(q, p) bounded, no drift.
        assert!(
            (ns_energy(&s) - e0).abs() < 1e-3,
            "energy drifted: {} vs {e0}",
            ns_energy(&s)
        );
    }

    #[test]
    fn tao_shadow_copy_tracks_the_physical_pair() {
        let mut s = ExtendedPhase::new(Vga3::scalar(-3.0), Vga3::zero());
        for _ in 0..50_000 {
            s = s.tao_step(1e-3, 20.0, ns_grad_q, ns_grad_p);
        }
        let dq = (s.q - s.x).scalar_part().abs();
        let dp = (s.p - s.y).scalar_part().abs();
        assert!(dq < 1e-2 && dp < 1e-2, "copies separated: dq {dq}, dp {dp}");
    }

    #[test]
    fn tao_reduces_to_the_separable_oscillator() {
        // Feed the separable oscillator through the non-separable stepper:
        // same half-period inversion leapfrog shows (τ-only convention).
        let mut s = ExtendedPhase::new(Vga3::basis(1), Vga3::zero());
        let gq = |q: &Vga3, _: &Vga3| *q * K;
        let gp = |_: &Vga3, p: &Vga3| *p;
        let half_period_steps = (core::f64::consts::TAU / (2.0 * K.sqrt()) / 0.01) as usize;
        for _ in 0..half_period_steps {
            s = s.tao_step(0.01, 5.0, gq, gp);
        }
        assert!(s.q.coeffs[1] < -0.9, "q did not invert: {}", s.q.coeffs[1]);
        assert!(
            s.p.coeffs[1].abs() < 0.1,
            "p not near rest: {}",
            s.p.coeffs[1]
        );
        // And the lift/read-out round-trip is lossless on (q, p).
        let ph = s.phase();
        assert_eq!(ExtendedPhase::from_phase(&ph).phase(), ph);
    }
}
