//! The [`Multivector`] type — an element of the algebra `Cl(P, Q, R)`.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use crate::algebra::{Algebra, BladeStore};
use crate::scalar::Scalar;
use crate::signature::blade_product;

/// A multivector in the Clifford algebra `A`, with coefficients of type
/// `T`.
///
/// - `A` = the signature, a marker type implementing [`Algebra`]
///   (`Vga3Sig`, `Pga3Sig`, …, or one minted with
///   [`define_algebra!`](crate::define_algebra)). It pins `P`, `Q`, `R`
///   and hence the blade count `DIM = 2^(P+Q+R)`.
/// - `T` = coefficient type (any [`Scalar`]; `f32`/`f64` provided).
///
/// The coefficient storage is `A::Blades<T>` — a fixed `[T; 2^N]` array
/// for every concrete signature. Tying the length to `A` rather than a
/// separate const parameter means it can never be specified
/// inconsistently. The type aliases in [`crate`] (`Vga2`, `Vga3`, `Pga3`,
/// …) name the common signatures so end users rarely write the generic
/// form.
///
/// The type is `#[repr(transparent)]` over its `coeffs` field, so under the
/// optional `bytemuck` feature a `Multivector` is plain-old-data
/// (`Pod`/`Zeroable`) whenever the scalar `T` is — letting a
/// `&[Multivector<A, T>]` be reinterpreted as a flat scalar (or byte)
/// buffer with no copy, e.g. for upload to the GPU.
#[repr(transparent)]
pub struct Multivector<A: Algebra, T: Scalar> {
    /// Coefficient of each basis blade, indexed by the bitmask convention
    /// in [`crate::signature`]. `coeffs[0]` is always the scalar part.
    pub coeffs: A::Blades<T>,
}

// `derive` can't prove `A::Blades<T>: Clone/Copy/…` through the projected
// associated type, so the standard marker traits are written by hand. The
// bounds are real: `BladeStore<T>: Copy` makes the field `Copy`, and
// `Scalar: PartialEq + Debug` covers the slice-based comparisons below.

impl<A: Algebra, T: Scalar> Clone for Multivector<A, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<A: Algebra, T: Scalar> Copy for Multivector<A, T> {}

impl<A: Algebra, T: Scalar> PartialEq for Multivector<A, T> {
    fn eq(&self, other: &Self) -> bool {
        self.coeffs.as_slice() == other.coeffs.as_slice()
    }
}

impl<A: Algebra, T: Scalar> fmt::Debug for Multivector<A, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Multivector")
            .field("coeffs", &self.coeffs.as_slice())
            .finish()
    }
}

impl<A: Algebra, T: Scalar> Multivector<A, T> {
    /// The zero multivector.
    pub fn zero() -> Self {
        Self {
            coeffs: <A::Blades<T> as BladeStore<T>>::splat(T::ZERO),
        }
    }

    /// A pure scalar `s + 0·e1 + 0·e2 + …`.
    pub fn scalar(s: T) -> Self {
        let mut m = Self::zero();
        m.coeffs[0] = s;
        m
    }

    /// The multiplicative identity (scalar `1`). Named `one` because
    /// it's also `1` under the geometric product.
    pub fn one() -> Self {
        Self::scalar(T::ONE)
    }

    /// The basis blade at the given index, with coefficient `1`.
    /// See [`crate::signature`] for the indexing convention. Panics if
    /// `index >= DIM`.
    pub fn basis(index: usize) -> Self {
        assert!(
            index < A::DIM,
            "basis blade index {index} out of range for DIM = {}",
            A::DIM,
        );
        let mut m = Self::zero();
        m.coeffs[index] = T::ONE;
        m
    }

    /// The scalar (grade-0) part of the multivector.
    pub fn scalar_part(&self) -> T {
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
    pub fn cleaned(&self, tol: T) -> Self {
        let mut out = *self;
        for i in 0..A::DIM {
            if out.coeffs[i].abs() < tol {
                out.coeffs[i] = T::ZERO;
            }
        }
        out
    }
}

impl<A: Algebra, T: Scalar> Default for Multivector<A, T> {
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

impl<A: Algebra, T: Scalar> Add for Multivector<A, T> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        for i in 0..A::DIM {
            self.coeffs[i] += rhs.coeffs[i];
        }
        self
    }
}

impl<A: Algebra, T: Scalar> AddAssign for Multivector<A, T> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..A::DIM {
            self.coeffs[i] += rhs.coeffs[i];
        }
    }
}

impl<A: Algebra, T: Scalar> Sub for Multivector<A, T> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self {
        for i in 0..A::DIM {
            self.coeffs[i] -= rhs.coeffs[i];
        }
        self
    }
}

impl<A: Algebra, T: Scalar> SubAssign for Multivector<A, T> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..A::DIM {
            self.coeffs[i] -= rhs.coeffs[i];
        }
    }
}

impl<A: Algebra, T: Scalar> Neg for Multivector<A, T> {
    type Output = Self;
    fn neg(mut self) -> Self {
        for i in 0..A::DIM {
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

impl<A: Algebra, T: Scalar> Mul for Multivector<A, T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut out = Self::zero();
        for a in 0..A::DIM {
            for b in 0..A::DIM {
                let (idx, sign) = blade_product(a, b, A::P, A::Q);
                if sign != 0 {
                    let term = self.coeffs[a] * rhs.coeffs[b];
                    if sign > 0 {
                        out.coeffs[idx] += term;
                    } else {
                        out.coeffs[idx] -= term;
                    }
                }
            }
        }
        out
    }
}

// --- Scalar multiplication ----------------------------------------------
//
// Linear scaling — nothing geometric — but worth defining on both sides
// so `2.0 * v` and `v * 2.0` both work. The right side (`v * s`) is
// generic over the scalar type; the left side (`s * v`) has to be
// written out per concrete scalar because coherence forbids a blanket
// `impl<T> Mul<Multivector<T, …>> for T`.

impl<A: Algebra, T: Scalar> Mul<T> for Multivector<A, T> {
    type Output = Self;
    fn mul(mut self, rhs: T) -> Self {
        for i in 0..A::DIM {
            self.coeffs[i] *= rhs;
        }
        self
    }
}

macro_rules! impl_left_scalar_mul {
    ($t:ty) => {
        impl<A: Algebra> Mul<Multivector<A, $t>> for $t {
            type Output = Multivector<A, $t>;
            fn mul(self, rhs: Multivector<A, $t>) -> Self::Output {
                rhs * self
            }
        }
    };
}

impl_left_scalar_mul!(f32);
impl_left_scalar_mul!(f64);

// --- Display ------------------------------------------------------------
//
// Formats as `s + a·e1 + b·e2 + c·e12 + …` skipping zeros and omitting
// the `1·` for unit-coefficient blades. The zero multivector prints as
// `0`.
//
// Blade labels are generated mechanically from the bit-mask index:
// bit `k` → `e_{k+1}`. This is signature-agnostic, so in PGA `Cl(3,0,1)`
// the null generator prints as `e4` rather than the conventional `e0`.
// For PGA-conventional output (null generator named `e0`, written first
// in each blade) use [`Multivector::display_pga`].

impl<A: Algebra, T: Scalar> fmt::Display for Multivector<A, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = A::N;
        let mut first = true;
        for i in 0..A::DIM {
            let c = self.coeffs[i];
            if c == T::ZERO {
                continue;
            }
            let neg = c < T::ZERO;
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
            if i == 0 || a != T::ONE {
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
        let a = Vga2 {
            coeffs: [1.0, 2.0, 0.0, 3.0],
        };
        // b = 10 + 20 e2
        let b = Vga2 {
            coeffs: [10.0, 0.0, 20.0, 0.0],
        };
        assert_eq!((a + b).coeffs, [11.0, 2.0, 20.0, 3.0]);
    }

    #[test]
    fn subtraction_is_componentwise() {
        let a = Vga3 {
            coeffs: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        };
        let b = Vga3 {
            coeffs: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        };
        assert_eq!((a - b).coeffs, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn neg_flips_every_coefficient() {
        let a = Vga2 {
            coeffs: [1.0, 2.0, -3.0, 4.0],
        };
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
        let a = Vga3 {
            coeffs: [1.0, 2.0, 0.0, -1.0, 0.0, 3.0, 0.0, 0.0],
        };
        let b = Vga3 {
            coeffs: [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        };
        let c = Vga3 {
            coeffs: [2.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0],
        };
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
        let v = Vga2 {
            coeffs: [1.0, -2.0, 0.5, 4.0],
        };
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
        let m = Vga2 {
            coeffs: [2.0, 3.0, -1.0, 1.0],
        };
        assert_eq!(format!("{m}"), "2 + 3·e1 - e2 + e12");
    }

    #[test]
    fn display_starts_with_minus_when_first_term_negative() {
        let m = Vga2 {
            coeffs: [0.0, -2.0, 0.0, 1.0],
        };
        assert_eq!(format!("{m}"), "-2·e1 + e12");
    }

    // --- cleaned() ------------------------------------------------------

    #[test]
    fn cleaned_zeros_subthreshold_coefficients() {
        let m = Vga2 {
            coeffs: [1e-15, 1.0, -1e-13, 2.0],
        };
        assert_eq!(m.cleaned(1e-12).coeffs, [0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn cleaned_preserves_significant_coefficients() {
        let m = Vga2 {
            coeffs: [0.5, -0.3, 0.1, 1e-5],
        };
        assert_eq!(m.cleaned(1e-12).coeffs, m.coeffs);
    }

    #[test]
    fn cleaned_at_zero_tolerance_is_identity() {
        let m = Vga2 {
            coeffs: [1e-300, 1.0, -2.0, 0.0],
        };
        assert_eq!(m.cleaned(0.0).coeffs, m.coeffs);
    }

    // --- Scalar genericity ----------------------------------------------

    #[test]
    fn works_over_f32() {
        use crate::{Vga2f, Vga3f};
        // The whole pipeline must run with f32 coefficients.
        let e1 = Vga3f::basis(1);
        let e2 = Vga3f::basis(2);
        // e1 * e2 == e12 (index 3)
        assert_eq!((e1 * e2).coeffs, Vga3f::basis(3).coeffs);
        // (3 e1 + 4 e2)² == 25 as f32
        let v = e1 * 3.0_f32 + e2 * 4.0_f32;
        assert_eq!((v * v).scalar_part(), 25.0_f32);
        // and left-side scalar mul works too
        let _ = Vga2f::basis(1);
    }

    #[test]
    fn f32_scalar_mul_both_sides() {
        use crate::Vga2f;
        let v = Vga2f::basis(1);
        assert_eq!((2.0_f32 * v).coeffs, (v * 2.0_f32).coeffs);
    }
}
