//! The [`Multivector`] type — an element of the algebra `Cl(P, Q, R)`.

use core::ops::{Add, AddAssign, Neg, Sub, SubAssign};

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
}
