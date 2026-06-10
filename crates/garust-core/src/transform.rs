//! Versor algebra: inversion, the sandwich product, and the exp/log
//! pair between bivectors and versors (the Lie algebra ↔ Lie group
//! bridge).
//!
//! These operations together are what make GA into a practical
//! tool for rotations, reflections, and motors:
//!
//! ```text
//! bivector  ──exp──▶  rotor  ──sandwich──▶  rotation of any object
//!           ◀──log──
//! ```
//!
//! [`exp`](Multivector::exp) is total (closed forms where they exist, a
//! convergent series everywhere else); [`log`](Multivector::log) is its
//! principal inverse on normalized even versors below six dimensions,
//! via the invariant decomposition
//! ([`try_bivector_split`](Multivector::try_bivector_split)) of a
//! bivector into commuting simple parts. Together they enable versor
//! interpolation (`Motor::slerp` in `garust-geo`) and optimization in
//! the bivector (Lie-algebra) parameters of a transformation.
//!
//! A **versor** is a multivector `M` for which `M * ~M` is a pure
//! scalar. That includes scalars, vectors, simple bivectors, rotors,
//! and any product of vectors. It excludes things like `1 + e1`, whose
//! reverse-product `(1+e1)(1+e1) = 2 + 2e1` is not a scalar.

use crate::algebra::{Algebra, BladeStore};
use crate::multivector::Multivector;
use crate::scalar::{max, Real, Ring, Scalar};
use crate::signature::grade_of;

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
        let one = <T::Magnitude as Ring>::ONE;
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
}

// The sandwich is a pair of geometric products — pure ring arithmetic, so it
// (and the batch form) is available over any `Ring` coefficient, not just a
// field.
impl<A: Algebra, T: Ring> Multivector<A, T> {
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
    /// Exponential `exp(self)` — **total**: defined for every multivector,
    /// by the fastest method that applies.
    ///
    /// 1. **Scalar square** `self² = c` (simple bivectors — how rotors are
    ///    built — but also vectors, pseudoscalars, ...): closed form
    ///
    ///    ```text
    ///    c < 0:  exp(self) = cos(√−c) + (sin(√−c) / √−c) · self
    ///    c > 0:  exp(self) = cosh(√c)  + (sinh(√c)  / √c)  · self
    ///    c = 0:  exp(self) = 1 + self
    ///    ```
    ///
    /// 2. **General bivector** in `N < 6` (screws in PGA, mixed
    ///    rotation–boost in STA, ...): the invariant decomposition
    ///    ([`try_bivector_split`](Self::try_bivector_split)) writes it as a
    ///    sum of two *commuting* simple bivectors, so
    ///    `exp(B) = exp(B₁) · exp(B₂)` with each factor closed-form.
    ///
    /// 3. **Everything else** (non-bivectors with non-scalar square, and
    ///    the rare bivectors the split can't separate, e.g. isoclinic
    ///    `e12 + e34`): a scaled-and-squared Taylor series — slower, but
    ///    convergent for every element of a finite-dimensional algebra.
    ///
    /// Constructing a rotor: in Euclidean space a unit bivector `B̂`
    /// satisfies `B̂² = −1`, so
    /// `R = exp(−θ/2 · B̂) = cos(θ/2) − sin(θ/2) · B̂`
    /// is the unit rotor that rotates by angle `θ` in the plane `B̂`.
    pub fn exp(&self) -> Self {
        let sq = *self * *self;
        let c = sq.scalar_part();
        let scalar_square = {
            let tol = T::from_f64(1e-9) * max(c.abs(), T::ONE);
            (1..A::DIM).all(|i| sq.coeffs[i].abs() <= tol)
        };
        if scalar_square {
            return self.exp_scalar_square(c);
        }
        let is_bivector = (0..A::DIM).all(|i| grade_of(i) == 2 || self.coeffs[i] == T::ZERO);
        if is_bivector && A::N < 6 {
            if let Some((b1, b2)) = self.try_bivector_split() {
                let c1 = (b1 * b1).scalar_part();
                let c2 = (b2 * b2).scalar_part();
                // The parts commute, so the product order is irrelevant.
                return b1.exp_scalar_square(c1) * b2.exp_scalar_square(c2);
            }
        }
        self.exp_series()
    }

    /// The closed-form exponential for an element with scalar square `c`
    /// (`self² = c`). Private: callers guarantee the precondition, so the
    /// split path can't recurse through [`exp`](Self::exp).
    fn exp_scalar_square(&self, c: T) -> Self {
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

    /// Exponential by scaled-and-squared Taylor series: halve `self` until
    /// every coefficient is at most `1/(16·DIM)` (so the operator norm of
    /// the geometric product keeps each Taylor term shrinking 16-fold),
    /// sum 12 terms, and square back. Works for any element; the closed
    /// forms in [`exp`](Self::exp) are preferred where they apply.
    fn exp_series(&self) -> Self {
        let mut norm = T::ZERO;
        for &c in self.coeffs.as_slice() {
            norm = max(norm, c.abs());
        }
        let target = T::from_f64(1.0 / (16.0 * A::DIM as f64));
        let half = T::from_f64(0.5);
        let mut x = *self;
        let mut squarings = 0u32;
        while norm > target && squarings < 64 {
            x = x * half;
            norm *= half;
            squarings += 1;
        }
        let mut sum = Self::one();
        let mut term = Self::one();
        for n in 1..=12u32 {
            term = (term * x) * (T::ONE / T::from_f64(f64::from(n)));
            sum += term;
        }
        for _ in 0..squarings {
            sum = sum * sum;
        }
        sum
    }

    /// Invariant decomposition of a **bivector** (`N < 6`): write
    /// `self = B₁ + B₂` with `B₁`, `B₂` *commuting simple* bivectors
    /// (each squares to a scalar), the split that makes the general
    /// [`exp`](Self::exp) and [`log`](Self::log) closed-form.
    ///
    /// With `self² = s + F` (`s` scalar, `F` the grade-4 part — there is
    /// nothing higher below six dimensions), the squares `β₁,₂` of the two
    /// parts are the roots of `β² − sβ + F²/4`, and
    /// `B₁ = self · (F − 2β₁) / (2(β₂ − β₁))`, `B₂ = self − B₁`.
    ///
    /// Returns:
    /// - `Some((self, 0))` when `self` is already simple (`F ≈ 0`);
    /// - `None` when the roots coincide (`s² ≈ F²`) — the *isoclinic* case
    ///   (e.g. `e12 + e34` with equal weights), where the decomposition is
    ///   not unique — or are complex, so no real split exists.
    ///
    /// Debug-asserts that `self` is a bivector and `N < 6`.
    pub fn try_bivector_split(&self) -> Option<(Self, Self)> {
        debug_assert!(
            A::N < 6 && (0..A::DIM).all(|i| grade_of(i) == 2 || self.coeffs[i] == T::ZERO),
            "try_bivector_split: defined for bivectors in algebras with \
             fewer than six generators",
        );
        let sq = *self * *self;
        let s = sq.scalar_part();
        let f = sq.grade(4);
        let mut f_norm = T::ZERO;
        for &c in f.coeffs.as_slice() {
            f_norm = max(f_norm, c.abs());
        }
        let scale = max(s.abs(), T::ONE);
        if f_norm <= T::from_f64(1e-12) * scale {
            // self² is scalar: already simple.
            return Some((*self, Self::zero()));
        }
        let f2 = (f * f).scalar_part();
        let disc = s * s - f2;
        if disc <= T::from_f64(1e-12) * scale * scale {
            // Coincident (isoclinic) or complex roots: no canonical real split.
            return None;
        }
        let rad = disc.sqrt();
        let two = T::from_f64(2.0);
        let beta1 = (s + rad) / two;
        // B(F − 2β₁) = 2(β₂ − β₁)B₁ and β₂ − β₁ = −rad; the geometric
        // product B·F is grade-2 in exact arithmetic, so project away the
        // numerical grade-4 dust.
        let b1 = ((*self * f) - *self * (two * beta1)) * (T::ONE / -(two * rad));
        let b1 = b1.grade(2);
        Some((b1, *self - b1))
    }

    /// Principal logarithm of a **normalized even versor** (`N < 6`):
    /// the bivector `B` with `exp(B) = ±self`, the inverse of
    /// [`exp`](Self::exp) up to the sign double-cover.
    ///
    /// `self` must satisfy `self · ~self = 1` with even grades only
    /// (`⟨self⟩₀ + ⟨self⟩₂ + ⟨self⟩₄` below six dimensions) — rotors,
    /// PGA motors, STA boosts, CGA transformations. Since `−R` acts
    /// identically to `R` in the sandwich, the sign is first folded to
    /// `⟨self⟩₀ ≥ 0`; the returned bivector is the *short way* — exactly
    /// what motor blending wants (cf. quaternion slerp's sign flip).
    ///
    /// The recovery mirrors the exponential's invariant split: writing
    /// `R = (c₁ + S₁)(c₂ + S₂)` over commuting simple bivectors,
    /// `⟨R⟩₂ = c₂S₁ + c₁S₂` splits into `W₁ + W₂` by
    /// [`try_bivector_split`](Self::try_bivector_split), and with
    /// `Q = ⟨R⟩₄` the identities `Sᵢ² = Wᵢ² − Q²` and `cᵢ² = 1 + Sᵢ²`
    /// pin every factor; each part is then `atan2`/`asinh`-rescaled from
    /// `(Sᵢ, cᵢ)` back to angle × unit bivector.
    ///
    /// Debug-asserts the versor contract. Outside it — most notably
    /// **isoclinic** rotors (equal-angle double rotations, where the split
    /// is non-unique) and pure grade-4 versors — debug builds panic and
    /// release builds return a best-effort `⟨self⟩₂`.
    pub fn log(&self) -> Self {
        let r = if self.scalar_part() < T::ZERO {
            -*self
        } else {
            *self
        };
        debug_assert!(
            A::N < 6 && {
                let n = r * r.reverse();
                let tol = T::from_f64(1e-6);
                (n.scalar_part() - T::ONE).abs() <= tol
                    && (1..A::DIM).all(|i| n.coeffs[i].abs() <= tol)
                    && (0..A::DIM).all(|i| grade_of(i) % 2 == 0 || r.coeffs[i].abs() <= tol)
            },
            "Multivector::log: input is not a normalized even versor \
             (R·~R must be 1) in an algebra with fewer than six generators",
        );
        let w = r.grade(2);
        let q = r.grade(4);
        let Some((w1, w2)) = w.try_bivector_split() else {
            debug_assert!(
                false,
                "Multivector::log: isoclinic versor — the invariant split \
                 of ⟨R⟩₂ is not unique; no principal logarithm",
            );
            return w;
        };
        let q2 = (q * q).scalar_part();
        // Sᵢ² = Wᵢ² − Q² and cᵢ² = 1 + Sᵢ² (cos²+sin² or cosh²−sinh², both
        // read `c² − S² = 1`); the cosines cross-multiply: Wᵢ = cⱼ Sᵢ.
        let s1_sq = (w1 * w1).scalar_part() - q2;
        let s2_sq = (w2 * w2).scalar_part() - q2;
        let c1 = max(T::ONE + s1_sq, T::ZERO).sqrt();
        let c2 = max(T::ONE + s2_sq, T::ZERO).sqrt();
        let eps = T::from_f64(1e-6);
        let (s1, s2) = if c1 > eps && c2 > eps {
            (w1 * (T::ONE / c2), w2 * (T::ONE / c1))
        } else if c1 > eps {
            // c₂ ≈ 0: part 1 is a quarter turn and W₁ = c₂S₁ vanishes —
            // recover S₁ through the grade-4 channel Q = S₁S₂ instead.
            let s2 = w2 * (T::ONE / c1);
            ((q * s2).grade(2) * (T::ONE / s2_sq), s2)
        } else if c2 > eps {
            let s1 = w1 * (T::ONE / c2);
            let s2 = (q * s1).grade(2) * (T::ONE / s1_sq);
            (s1, s2)
        } else {
            debug_assert!(
                false,
                "Multivector::log: both factors are quarter turns (⟨R⟩₀ ≈ 0 \
                 ≈ ⟨R⟩₂); no principal logarithm",
            );
            return w;
        };
        Self::simple_log_part(s1, s1_sq, c1) + Self::simple_log_part(s2, s2_sq, c2)
    }

    /// One commuting factor of the versor log: rescale `S = sin(θ)·B̂`
    /// (or its hyperbolic / null analogue) with cosine `c` back to the
    /// bivector `θ·B̂`. `s_sq` is `S²`, whose sign picks the conic.
    fn simple_log_part(s: Self, s_sq: T, c: T) -> Self {
        let eps = T::from_f64(1e-9);
        if s_sq < -eps {
            // Elliptic: S² = −sin²θ; θ = atan2(sin, cos) ∈ [0, τ/4].
            let m = (-s_sq).sqrt();
            s * (m.atan2(c) / m)
        } else if s_sq > eps {
            // Hyperbolic: S² = sinh²φ; φ = asinh(m) = ln(m + √(m² + 1)).
            let m = s_sq.sqrt();
            s * ((m + (s_sq + T::ONE).sqrt()).ln() / m)
        } else {
            // Parabolic (null, c = 1) and the tiny-angle limit of the other
            // two conics (θ/sin θ → 1/c as both → 0).
            s * (T::ONE / c)
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
