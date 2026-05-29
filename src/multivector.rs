//! The [`Multivector`] type — an element of the algebra `Cl(P, Q, R)`.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use crate::signature::blade_product;

/// A multivector in the Clifford algebra `Cl(P, Q, R)`.
///
/// - `P` = number of basis vectors that square to `+1`
/// - `Q` = number of basis vectors that square to `-1`
/// - `R` = number of basis vectors that square to `0` (degenerate / null)
/// - `DIM` = `2^(P+Q+R)`, the number of basis blades
///
/// `DIM` is a fourth const parameter only because stable Rust can't yet
/// evaluate `1 << (P+Q+R)` in a const-generic array length position.
/// A const assertion below catches any mismatch at compile time, and the
/// type aliases in [`crate`] (`Vga2`, `Vga3`, `Pga3`, …) mean end users
/// almost never need to type the redundant `DIM`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Multivector<const P: usize, const Q: usize, const R: usize, const DIM: usize> {
    /// Coefficient of each basis blade, indexed by the bitmask convention
    /// in [`crate::signature`]. `coeffs[0]` is always the scalar part.
    pub coeffs: [f64; DIM],
}

impl<const P: usize, const Q: usize, const R: usize, const DIM: usize>
    Multivector<P, Q, R, DIM>
{
    /// Compile-time check that `DIM == 2^(P+Q+R)`. Referencing this in
    /// every constructor makes the check actually fire when the type is
    /// used, not just when it's declared somewhere.
    const _DIM_CHECK: () = assert!(
        DIM == 1 << (P + Q + R),
        "garust: Multivector<P,Q,R,DIM> requires DIM == 2^(P+Q+R)",
    );

    /// The zero multivector.
    pub fn zero() -> Self {
        let _ = Self::_DIM_CHECK;
        Self { coeffs: [0.0; DIM] }
    }

    /// A pure scalar `s + 0·e1 + 0·e2 + …`.
    pub fn scalar(s: f64) -> Self {
        let mut m = Self::zero();
        m.coeffs[0] = s;
        m
    }

    /// The multiplicative identity (scalar `1`). Named `one` because
    /// it's also `1` under the (not-yet-implemented) geometric product.
    pub fn one() -> Self {
        Self::scalar(1.0)
    }

    /// The basis blade at the given index, with coefficient `1.0`.
    /// See [`crate::signature`] for the indexing convention. Panics if
    /// `index >= DIM`.
    pub fn basis(index: usize) -> Self {
        assert!(
            index < DIM,
            "basis blade index {index} out of range for DIM = {DIM}",
        );
        let mut m = Self::zero();
        m.coeffs[index] = 1.0;
        m
    }

    /// The scalar (grade-0) part of the multivector.
    pub fn scalar_part(&self) -> f64 {
        self.coeffs[0]
    }

    /// Returns a copy of `self` with every coefficient whose magnitude
    /// is below `tol` set to zero.
    ///
    /// Useful for suppressing floating-point dust before printing or
    /// comparing. Rotor sandwiches and exp products in particular leak
    /// `~1e-16`-magnitude noise into blades that should be exactly
    /// zero by symmetry; passing `1e-10` knocks those out while
    /// leaving any real-magnitude coefficient untouched.
    pub fn cleaned(&self, tol: f64) -> Self {
        let mut out = *self;
        for i in 0..DIM {
            if out.coeffs[i].abs() < tol {
                out.coeffs[i] = 0.0;
            }
        }
        out
    }
}

impl<const P: usize, const Q: usize, const R: usize, const DIM: usize> Default
    for Multivector<P, Q, R, DIM>
{
    fn default() -> Self {
        Self::zero()
    }
}

// --- Linear arithmetic ---------------------------------------------------
//
// `+`, `-`, and unary `-` on multivectors are componentwise on the
// coefficient array — exactly the same operations you'd do on a vector
// in `R^DIM`. None of the *geometric* part of geometric algebra shows
// up yet; that's all hiding in the product, which is coming next.

impl<const P: usize, const Q: usize, const R: usize, const DIM: usize> Add
    for Multivector<P, Q, R, DIM>
{
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        for i in 0..DIM {
            self.coeffs[i] += rhs.coeffs[i];
        }
        self
    }
}

impl<const P: usize, const Q: usize, const R: usize, const DIM: usize> AddAssign
    for Multivector<P, Q, R, DIM>
{
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..DIM {
            self.coeffs[i] += rhs.coeffs[i];
        }
    }
}

impl<const P: usize, const Q: usize, const R: usize, const DIM: usize> Sub
    for Multivector<P, Q, R, DIM>
{
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self {
        for i in 0..DIM {
            self.coeffs[i] -= rhs.coeffs[i];
        }
        self
    }
}

impl<const P: usize, const Q: usize, const R: usize, const DIM: usize> SubAssign
    for Multivector<P, Q, R, DIM>
{
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..DIM {
            self.coeffs[i] -= rhs.coeffs[i];
        }
    }
}

impl<const P: usize, const Q: usize, const R: usize, const DIM: usize> Neg
    for Multivector<P, Q, R, DIM>
{
    type Output = Self;
    fn neg(mut self) -> Self {
        for i in 0..DIM {
            self.coeffs[i] = -self.coeffs[i];
        }
        self
    }
}

// --- Geometric product --------------------------------------------------
//
// The first place the signature `(P, Q, R)` actually shows up. The
// product distributes over the sum of basis blades:
//
//     (Σ aᵢ Eᵢ) * (Σ bⱼ Eⱼ) = Σᵢⱼ aᵢ bⱼ (Eᵢ * Eⱼ)
//
// and each single-blade product `Eᵢ * Eⱼ` is computed in one shot by
// `signature::blade_product` (target index = `i XOR j`, sign comes from
// blade-reordering parity times the metric of every shared generator).
//
// Cost: `O(DIM²)` per multiplication — fine for the algebras a human
// would write by hand (≤ 1024 ops for `Cga3`).

impl<const P: usize, const Q: usize, const R: usize, const DIM: usize> Mul
    for Multivector<P, Q, R, DIM>
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut out = Self::zero();
        for a in 0..DIM {
            for b in 0..DIM {
                let (idx, sign) = blade_product(a, b, P, Q);
                if sign != 0 {
                    out.coeffs[idx] += (sign as f64) * self.coeffs[a] * rhs.coeffs[b];
                }
            }
        }
        out
    }
}

// --- Scalar multiplication ----------------------------------------------
//
// Linear scaling — nothing geometric — but worth defining on both sides
// so `2.0 * v` and `v * 2.0` both work.

impl<const P: usize, const Q: usize, const R: usize, const DIM: usize> Mul<f64>
    for Multivector<P, Q, R, DIM>
{
    type Output = Self;
    fn mul(mut self, rhs: f64) -> Self {
        for i in 0..DIM {
            self.coeffs[i] *= rhs;
        }
        self
    }
}

impl<const P: usize, const Q: usize, const R: usize, const DIM: usize>
    Mul<Multivector<P, Q, R, DIM>> for f64
{
    type Output = Multivector<P, Q, R, DIM>;
    fn mul(self, rhs: Multivector<P, Q, R, DIM>) -> Multivector<P, Q, R, DIM> {
        rhs * self
    }
}

// --- Display ------------------------------------------------------------
//
// Formats as `s + a·e1 + b·e2 + c·e12 + …` skipping zeros and omitting
// the `1·` for unit-coefficient blades. The zero multivector prints as
// `0`.
//
// Blade labels are generated mechanically from the bit-mask index:
// bit `k` → `e_{k+1}`. This is signature-agnostic, so in PGA `Cl(3,0,1)`
// the null generator prints as `e4` rather than the conventional `e0`.
// We'll fix that when we add per-algebra wrapper types.

impl<const P: usize, const Q: usize, const R: usize, const DIM: usize> fmt::Display
    for Multivector<P, Q, R, DIM>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = P + Q + R;
        let mut first = true;
        for i in 0..DIM {
            let c = self.coeffs[i];
            if c == 0.0 {
                continue;
            }
            let neg = c < 0.0;
            if first {
                first = false;
                if neg {
                    write!(f, "-")?;
                }
            } else if neg {
                write!(f, " - ")?;
            } else {
                write!(f, " + ")?;
            }
            let a = c.abs();
            // Suppress "1·" for non-scalar blades.
            if i == 0 || a != 1.0 {
                write!(f, "{a}")?;
                if i != 0 {
                    write!(f, "·")?;
                }
            }
            if i != 0 {
                write!(f, "e")?;
                for k in 0..n {
                    if i & (1 << k) != 0 {
                        write!(f, "{}", k + 1)?;
                    }
                }
            }
        }
        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{Pga3, Vga2, Vga3};

    #[test]
    fn vga2_has_four_blades() {
        assert_eq!(Vga2::zero().coeffs.len(), 4);
    }

    #[test]
    fn vga3_has_eight_blades() {
        assert_eq!(Vga3::zero().coeffs.len(), 8);
    }

    #[test]
    fn pga3_has_sixteen_blades() {
        assert_eq!(Pga3::zero().coeffs.len(), 16);
    }

    #[test]
    fn scalar_lives_at_index_zero() {
        let s = Vga2::scalar(7.0);
        assert_eq!(s.coeffs, [7.0, 0.0, 0.0, 0.0]);
        assert_eq!(s.scalar_part(), 7.0);
    }

    #[test]
    fn basis_e1_in_vga2() {
        // In Cl(2,0,0) the blade layout is [1, e1, e2, e12].
        let e1 = Vga2::basis(1);
        assert_eq!(e1.coeffs, [0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn addition_is_componentwise() {
        // a = 1 + 2 e1 + 3 e12
        let a = Vga2 { coeffs: [1.0, 2.0, 0.0, 3.0] };
        // b = 10 + 20 e2
        let b = Vga2 { coeffs: [10.0, 0.0, 20.0, 0.0] };
        assert_eq!((a + b).coeffs, [11.0, 2.0, 20.0, 3.0]);
    }

    #[test]
    fn subtraction_is_componentwise() {
        let a = Vga3 { coeffs: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] };
        let b = Vga3 { coeffs: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] };
        assert_eq!((a - b).coeffs, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn neg_flips_every_coefficient() {
        let a = Vga2 { coeffs: [1.0, 2.0, -3.0, 4.0] };
        assert_eq!((-a).coeffs, [-1.0, -2.0, 3.0, -4.0]);
    }

    #[test]
    fn add_assign_and_sub_assign() {
        let mut a = Vga2::scalar(1.0);
        a += Vga2::basis(1);
        a -= Vga2::scalar(0.5);
        assert_eq!(a.coeffs, [0.5, 1.0, 0.0, 0.0]);
    }

    // --- Geometric product (round 2) ------------------------------------

    #[test]
    fn e1_squares_to_one_in_vga2() {
        let e1 = Vga2::basis(1);
        assert_eq!((e1 * e1).coeffs, Vga2::one().coeffs);
    }

    #[test]
    fn e2_squares_to_one_in_vga2() {
        let e2 = Vga2::basis(2);
        assert_eq!((e2 * e2).coeffs, Vga2::one().coeffs);
    }

    #[test]
    fn e1_times_e2_is_e12() {
        let e1 = Vga2::basis(1);
        let e2 = Vga2::basis(2);
        let e12 = Vga2::basis(3); // 0b11
        assert_eq!((e1 * e2).coeffs, e12.coeffs);
    }

    #[test]
    fn vectors_anticommute() {
        let e1 = Vga2::basis(1);
        let e2 = Vga2::basis(2);
        assert_eq!((e2 * e1).coeffs, (-(e1 * e2)).coeffs);
    }

    #[test]
    fn pseudoscalar_squares_to_minus_one_in_vga2() {
        let e12 = Vga2::basis(3);
        let minus_one = Vga2::scalar(-1.0);
        assert_eq!((e12 * e12).coeffs, minus_one.coeffs);
    }

    #[test]
    fn cross_terms_cancel_in_sum_of_vectors_squared() {
        // (e1 + e2)² = e1·e1 + e1·e2 + e2·e1 + e2·e2 = 1 - e12 + e12 + 1 = 2
        let v = Vga2::basis(1) + Vga2::basis(2);
        assert_eq!((v * v).coeffs, Vga2::scalar(2.0).coeffs);
    }

    #[test]
    fn null_generator_squares_to_zero_in_pga3() {
        // In Cl(3,0,1), bit 3 is the R-group generator.
        let e4 = Pga3::basis(0b1000);
        assert_eq!((e4 * e4).coeffs, Pga3::zero().coeffs);
    }

    #[test]
    fn geometric_product_is_left_distributive() {
        // a * (b + c) == a*b + a*c, exercised on a non-trivial pair.
        let a = Vga3 { coeffs: [1.0, 2.0, 0.0, -1.0, 0.0, 3.0, 0.0, 0.0] };
        let b = Vga3 { coeffs: [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0] };
        let c = Vga3 { coeffs: [2.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0] };
        let lhs = a * (b + c);
        let rhs = (a * b) + (a * c);
        assert_eq!(lhs.coeffs, rhs.coeffs);
    }

    // --- Scalar multiplication ------------------------------------------

    #[test]
    fn scalar_mul_right() {
        let v = Vga2::basis(1) + Vga2::basis(2);
        assert_eq!((v * 3.0).coeffs, [0.0, 3.0, 3.0, 0.0]);
    }

    #[test]
    fn scalar_mul_left_matches_right() {
        let v = Vga2 { coeffs: [1.0, -2.0, 0.5, 4.0] };
        assert_eq!((2.5 * v).coeffs, (v * 2.5).coeffs);
    }

    // --- Display --------------------------------------------------------

    #[test]
    fn display_zero() {
        assert_eq!(format!("{}", Vga2::zero()), "0");
    }

    #[test]
    fn display_pure_scalar() {
        assert_eq!(format!("{}", Vga2::scalar(3.5)), "3.5");
    }

    #[test]
    fn display_unit_blade_omits_one_coefficient() {
        assert_eq!(format!("{}", Vga2::basis(1)), "e1");
        assert_eq!(format!("{}", Vga2::basis(3)), "e12");
    }

    #[test]
    fn display_mixed_grade_with_signs() {
        // 2 + 3·e1 - e2 + e12
        let m = Vga2 { coeffs: [2.0, 3.0, -1.0, 1.0] };
        assert_eq!(format!("{m}"), "2 + 3·e1 - e2 + e12");
    }

    #[test]
    fn display_starts_with_minus_when_first_term_negative() {
        let m = Vga2 { coeffs: [0.0, -2.0, 0.0, 1.0] };
        assert_eq!(format!("{m}"), "-2·e1 + e12");
    }

    // --- cleaned() ------------------------------------------------------

    #[test]
    fn cleaned_zeros_subthreshold_coefficients() {
        let m = Vga2 { coeffs: [1e-15, 1.0, -1e-13, 2.0] };
        assert_eq!(m.cleaned(1e-12).coeffs, [0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn cleaned_preserves_significant_coefficients() {
        let m = Vga2 { coeffs: [0.5, -0.3, 0.1, 1e-5] };
        assert_eq!(m.cleaned(1e-12).coeffs, m.coeffs);
    }

    #[test]
    fn cleaned_at_zero_tolerance_is_identity() {
        let m = Vga2 { coeffs: [1e-300, 1.0, -2.0, 0.0] };
        assert_eq!(m.cleaned(0.0).coeffs, m.coeffs);
    }
}
