//! Versor algebra: inversion, the sandwich product, and the closed-form
//! exponential of elements with scalar squares (the bridge from
//! bivectors to rotors).
//!
//! These three operations together are what make GA into a practical
//! tool for rotations, reflections, and motors:
//!
//! ```text
//! bivector  ──exp──▶  rotor  ──sandwich──▶  rotation of any object
//! ```
//!
//! A **versor** is a multivector `M` for which `M * ~M` is a pure
//! scalar. That includes scalars, vectors, simple bivectors, rotors,
//! and any product of vectors. It excludes things like `1 + e1`, whose
//! reverse-product `(1+e1)(1+e1) = 2 + 2e1` is not a scalar.

use crate::algebra::{Algebra, BladeStore};
use crate::multivector::Multivector;
use crate::scalar::{max, Real, Scalar};

impl<A: Algebra, T: Scalar> Multivector<A, T> {
    /// Inverse `M⁻¹ = ~M / ⟨M ~M⟩_0`, defined for versors.
    ///
    /// A multivector is a versor iff `M * ~M` is a pure scalar (after
    /// floating-point tolerance). Returns `None` when:
    /// - `M * ~M` has any non-scalar component above tolerance, or
    /// - `M * ~M` is itself zero (the versor has zero "norm").
    pub fn try_versor_inverse(&self) -> Option<Self> {
        let prod = *self * self.reverse();
        let scalar = prod.scalar_part();
        // The tolerance lives in the real magnitude type, so this works even
        // when the coefficient field (e.g. Complex) has no order of its own.
        let one = <T::Magnitude as Scalar>::ONE;
        let tol = <T::Magnitude as Scalar>::from_f64(1e-10) * max(scalar.abs(), one);
        for i in 1..A::DIM {
            if prod.coeffs[i].abs() > tol {
                return None;
            }
        }
        if scalar == T::ZERO {
            return None;
        }
        Some(self.reverse() * (T::ONE / scalar))
    }

    /// Inverse of a versor. Panics if `self` is not a versor (in the
    /// sense of [`Multivector::try_versor_inverse`]) or has zero norm.
    pub fn versor_inverse(&self) -> Self {
        self.try_versor_inverse().expect(
            "versor_inverse: multivector is not a versor (M·~M not a scalar) \
             or has zero norm; use try_versor_inverse to handle this",
        )
    }

    /// Sandwich product `self · x · ~self`.
    ///
    /// The universal GA transformation:
    /// - For a unit rotor `R`, `R.sandwich(v)` rotates `v`.
    /// - For a unit vector `n`, `n.sandwich(v)` reflects `v` in the
    ///   line/hyperplane through `n` (perpendicular components flip,
    ///   `n`-direction component is preserved).
    /// - For a product of unit vectors, you get the composition of
    ///   those reflections — which by the Cartan–Dieudonné theorem can
    ///   represent any orthogonal transformation.
    ///
    /// For non-unit versors, sandwich with `~self` differs from the
    /// "true" conjugation `self · x · self⁻¹` by a positive scalar
    /// factor. If you need the precise conjugation, compose with
    /// [`Multivector::versor_inverse`] explicitly.
    ///
    /// Computed as two *sparse* geometric products, so it skips the all-zero
    /// blades that dominate a sandwich's operands — an even-grade versor
    /// (half its blades zero) and a single-grade object (a point, line, or
    /// plane is mostly zero). The result is identical to `self * x * ~self`;
    /// only the wasted multiplies are gone.
    pub fn sandwich(&self, x: &Self) -> Self {
        let rev = self.reverse();
        self.sparse_product(x).sparse_product(&rev)
    }

    /// Sandwich `self` over every element of `xs`, in place — the batch
    /// form of [`Multivector::sandwich`] for transforming a whole point
    /// cloud (or set of lines/planes) by one versor.
    ///
    /// The reverse `~self` is computed once and reused across the batch, so
    /// this is cheaper than calling [`Multivector::sandwich`] in a loop
    /// while giving bit-identical results. It is also the natural shape for
    /// vectorization: each element's transform is independent.
    pub fn sandwich_each(&self, xs: &mut [Self]) {
        let rev = self.reverse();
        for x in xs.iter_mut() {
            *x = self.sparse_product(x).sparse_product(&rev);
        }
    }

    /// Geometric product that skips blade pairs with a zero coefficient on
    /// either side. Bit-identical to `*` (a zero coefficient contributes a
    /// `0` term either way), but for the graded, mostly-zero operands of a
    /// sandwich it does far less work. `*` itself stays dense — the per-pair
    /// zero test would only slow the dense case it is tuned for.
    fn sparse_product(&self, rhs: &Self) -> Self {
        let mut out = Self::zero();
        let table = A::CAYLEY;
        for (i, &ca) in self.coeffs.as_slice().iter().enumerate() {
            if ca == T::ZERO {
                continue;
            }
            let row = i * A::DIM;
            for (j, &cb) in rhs.coeffs.as_slice().iter().enumerate() {
                if cb == T::ZERO {
                    continue;
                }
                let (idx, sign) = table[row + j];
                let term = ca * cb;
                if sign > 0 {
                    out.coeffs[idx as usize] += term;
                } else if sign < 0 {
                    out.coeffs[idx as usize] -= term;
                }
            }
        }
        out
    }
}

impl<A: Algebra, T: Real> Multivector<A, T> {
    /// Closed-form exponential `exp(self)` for an element whose square
    /// is a scalar.
    ///
    /// The headline use is **simple bivectors** — that's how rotors get
    /// built — but the same formula works for any blade with scalar
    /// square (vectors, pseudoscalars in odd-dimensional algebras, ...).
    /// Three cases by `self² = c`:
    ///
    /// ```text
    /// c < 0:  exp(self) = cos(√−c) + (sin(√−c) / √−c) · self
    /// c > 0:  exp(self) = cosh(√c)  + (sinh(√c)  / √c)  · self
    /// c = 0:  exp(self) = 1 + self
    /// ```
    ///
    /// Constructing a rotor: in Euclidean space a unit bivector `B̂`
    /// satisfies `B̂² = −1`, so
    /// `R = exp(−θ/2 · B̂) = cos(θ/2) − sin(θ/2) · B̂`
    /// is the unit rotor that rotates by angle `θ` in the plane `B̂`.
    ///
    /// Panics in debug builds if `self²` isn't (approximately) scalar
    /// — the formula doesn't apply to non-simple bivectors like
    /// `e12 + e34` in 4D, where exp decomposes into a product over
    /// commuting parts instead.
    pub fn exp(&self) -> Self {
        let sq = *self * *self;
        debug_assert!(
            {
                let scalar = sq.scalar_part();
                let tol = T::from_f64(1e-9) * max(scalar.abs(), T::ONE);
                (1..A::DIM).all(|i| sq.coeffs[i].abs() <= tol)
            },
            "Multivector::exp: self² is not a scalar; the closed-form \
             formula only applies to elements with scalar square",
        );
        let c = sq.scalar_part();
        if c < T::ZERO {
            let s = (-c).sqrt();
            Self::scalar(s.cos()) + *self * (s.sin() / s)
        } else if c > T::ZERO {
            let s = c.sqrt();
            Self::scalar(s.cosh()) + *self * (s.sinh() / s)
        } else {
            Self::one() + *self
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Vga2, Vga3};
    use std::f64::consts::TAU;

    fn approx_eq(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tol, "index {i}: {x} vs {y}");
        }
    }

    // --- Versor inverse -------------------------------------------------

    #[test]
    fn vector_times_its_inverse_is_one() {
        let v = Vga2 {
            coeffs: [0.0, 3.0, 4.0, 0.0],
        };
        let inv = v.versor_inverse();
        let prod = v * inv;
        approx_eq(&prod.coeffs, &Vga2::one().coeffs, 1e-12);
    }

    #[test]
    fn try_versor_inverse_returns_none_for_non_versor() {
        // 1 + e1 isn't a versor: (1+e1)(1+e1) = 2 + 2 e1 (not scalar).
        let m = Vga3::scalar(1.0) + Vga3::basis(1);
        assert!(m.try_versor_inverse().is_none());
    }

    #[test]
    fn try_versor_inverse_returns_none_for_zero() {
        assert!(Vga2::zero().try_versor_inverse().is_none());
    }

    #[test]
    fn rotor_inverse_round_trips() {
        let theta = 0.7;
        let e12 = Vga2::basis(3);
        let r = (e12 * (-theta / 2.0)).exp();
        let prod = r * r.versor_inverse();
        approx_eq(&prod.coeffs, &Vga2::one().coeffs, 1e-12);
    }

    // --- Sandwich product ----------------------------------------------

    #[test]
    fn rotor_90_in_e12_sends_e1_to_e2() {
        let e12 = Vga2::basis(3);
        let r = (e12 * (-TAU / 8.0)).exp();
        let rotated = r.sandwich(&Vga2::basis(1));
        approx_eq(&rotated.coeffs, &Vga2::basis(2).coeffs, 1e-12);
    }

    #[test]
    fn rotor_90_in_e12_sends_e2_to_minus_e1() {
        let e12 = Vga2::basis(3);
        let r = (e12 * (-TAU / 8.0)).exp();
        let rotated = r.sandwich(&Vga2::basis(2));
        approx_eq(&rotated.coeffs, &(-Vga2::basis(1)).coeffs, 1e-12);
    }

    #[test]
    fn rotation_axis_is_fixed_in_3d() {
        // Any rotation in the e23 plane leaves e1 untouched.
        let e23 = Vga3::basis(6);
        let r = (e23 * (-0.73 / 2.0)).exp();
        let rotated = r.sandwich(&Vga3::basis(1));
        approx_eq(&rotated.coeffs, &Vga3::basis(1).coeffs, 1e-12);
    }

    #[test]
    fn unit_vector_sandwich_flips_perpendicular_component() {
        // n = e1, v = e1 + e2 ⇒ n.sandwich(v) = e1 - e2.
        let n = Vga3::basis(1);
        let v = Vga3::basis(1) + Vga3::basis(2);
        let reflected = n.sandwich(&v);
        let expected = Vga3::basis(1) - Vga3::basis(2);
        approx_eq(&reflected.coeffs, &expected.coeffs, 1e-12);
    }

    #[test]
    fn two_reflections_compose_into_a_rotation() {
        // Reflecting across n1 then across n2 = (e1+e2)/√2 rotates by
        // 2·45° = 90° in the e12 plane.
        let n1 = Vga2::basis(1);
        let n2 = (Vga2::basis(1) + Vga2::basis(2)) * (1.0 / 2.0_f64.sqrt());
        let v = Vga2::basis(2);
        let after_first = n1.sandwich(&v);
        let after_both = n2.sandwich(&after_first);
        // e2 → −e2 → −e1
        approx_eq(&after_both.coeffs, &(-Vga2::basis(1)).coeffs, 1e-12);
    }

    // --- Exponential ----------------------------------------------------

    #[test]
    fn exp_of_zero_is_one() {
        assert_eq!(Vga3::zero().exp().coeffs, Vga3::one().coeffs);
    }

    #[test]
    fn exp_of_quarter_tau_e12_is_e12() {
        // exp((τ/4) e12) where e12² = -1
        //   = cos(τ/4) + sin(τ/4) (e12 / 1) = e12
        let b = Vga2::basis(3) * (TAU / 4.0);
        approx_eq(&b.exp().coeffs, &Vga2::basis(3).coeffs, 1e-12);
    }

    #[test]
    fn rotor_built_via_exp_has_unit_norm() {
        let r = (Vga3::basis(3) * (-1.234 / 2.0)).exp();
        let n2 = r.norm_squared();
        assert!((n2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn exp_of_vector_in_p_group_uses_cosh_sinh() {
        // In Cl(1,0,0): e1² = +1, so exp(t e1) = cosh(t) + sinh(t) e1.
        use crate::Multivector;
        crate::define_algebra!(Cl100 = Cl(1, 0, 0));
        type Cl10 = Multivector<Cl100, f64>;
        let t = 0.5;
        let v = Cl10 { coeffs: [0.0, t] };
        let e = v.exp();
        assert!((e.coeffs[0] - t.cosh()).abs() < 1e-12);
        assert!((e.coeffs[1] - t.sinh()).abs() < 1e-12);
    }

    // --- PGA translators (Cl(3,0,1)) -----------------------------------

    #[test]
    fn null_bivector_squares_to_zero_in_pga3() {
        // e0·e1 lives at bit positions {0, 3} (bit 3 is the R generator).
        // It's a null bivector, so its square must be zero.
        use crate::Pga3;
        let e0 = Pga3::basis(8); // bit 3
        let e1 = Pga3::basis(1);
        let b = e0 * e1;
        let sq = b * b;
        approx_eq(&sq.coeffs, &Pga3::zero().coeffs, 1e-12);
    }

    #[test]
    fn exp_of_null_bivector_is_one_plus_bivector() {
        // c = 0 branch of exp: exp(B) = 1 + B when B² = 0.
        use crate::Pga3;
        let e0 = Pga3::basis(8);
        let e1 = Pga3::basis(1);
        let b = (e0 * e1) * 0.42;
        let expected = Pga3::one() + b;
        approx_eq(&b.exp().coeffs, &expected.coeffs, 1e-12);
    }

    #[test]
    fn pga_translator_moves_origin_to_target_point() {
        // Translator T = exp(-d/2 · e0·e1) translates by d in e1.
        // Origin point P = e1·e2·e3 (trivector convention).
        // Target point at (d, 0, 0): P_d = e1·e2·e3 − d · e0·e2·e3.
        use crate::Pga3;
        let d = 3.0;
        let e0 = Pga3::basis(8);
        let e1 = Pga3::basis(1);
        let e2 = Pga3::basis(2);
        let e3 = Pga3::basis(4);
        let t = ((e0 * e1) * (-d / 2.0)).exp();
        let origin = e1 * e2 * e3;
        let translated = t.sandwich(&origin);
        let expected = e1 * e2 * e3 - (e0 * e2 * e3) * d;
        approx_eq(&translated.coeffs, &expected.coeffs, 1e-12);
    }

    #[test]
    fn pga_translator_inverse_equals_reverse_translator() {
        // Translators are versors: T · ~T = 1, so ~T = T⁻¹.
        // For null bivectors this is also exp(+d/2 e0e1), i.e. negate
        // the displacement.
        use crate::Pga3;
        let d = 1.7;
        let e0 = Pga3::basis(8);
        let e1 = Pga3::basis(1);
        let t = ((e0 * e1) * (-d / 2.0)).exp();
        let t_inv = ((e0 * e1) * (d / 2.0)).exp();
        let prod = t * t_inv;
        approx_eq(&prod.coeffs, &Pga3::one().coeffs, 1e-12);
        // And ~T equals the inverse-translator we just built.
        approx_eq(&t.reverse().coeffs, &t_inv.coeffs, 1e-12);
    }
}
