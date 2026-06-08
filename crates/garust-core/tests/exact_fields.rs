//! Proof that the relaxed `Scalar::Magnitude` bound (`Scalar + PartialOrd`,
//! no longer `Real`) admits **exact** and **finite-field** coefficients —
//! not just `f32`/`f64`/`Complex`/`Dual`.
//!
//! The geometric product, wedge, grades, and involutions are pure ring
//! arithmetic — they never divide, take a norm, or compare for order — so
//! they work over any field. Two external coefficient types demonstrate it:
//!
//! * `Gf2` — the finite field GF(2). Over it the metric signs `±1` collapse
//!   (`−1 = 1`), so generators *commute* and nilpotents appear.
//! * `Zint` — exact integers `ℤ`. Reconstructions are bit-exact, no float
//!   dust. (`ℤ` is a ring, not a field; its truncating `Div` makes the
//!   field-only ops — `versor_inverse`, `norm` — meaningless, so just don't
//!   call them. The product/wedge/grade core is exact.)
//!
//! Neither is `Real`, so neither gets `norm`/`exp`/`Display` on multivectors.

// In GF(2), addition *is* XOR and multiplication *is* AND — so the lint that
// flags "suspicious" operators in arithmetic impls is a false positive here.
#![allow(clippy::suspicious_arithmetic_impl)]

use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use garust_core::{Multivector, Scalar, Vga2Sig};

// --- GF(2): the two-element finite field --------------------------------

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
struct Gf2(u8);
impl fmt::Display for Gf2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Add for Gf2 {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Gf2(self.0 ^ o.0) // addition mod 2 = XOR
    }
}
impl Sub for Gf2 {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Gf2(self.0 ^ o.0) // −x = x, so subtraction = XOR too
    }
}
impl Mul for Gf2 {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Gf2(self.0 & o.0) // multiplication mod 2 = AND
    }
}
impl Div for Gf2 {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        Gf2(self.0 / o.0) // only 1 is invertible; /0 panics, as a field demands
    }
}
impl Neg for Gf2 {
    type Output = Self;
    fn neg(self) -> Self {
        self
    }
}
impl AddAssign for Gf2 {
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}
impl SubAssign for Gf2 {
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}
impl MulAssign for Gf2 {
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}
impl Scalar for Gf2 {
    type Magnitude = Gf2;
    const ZERO: Self = Gf2(0);
    const ONE: Self = Gf2(1);
    fn from_f64(x: f64) -> Self {
        Gf2((x as i64).rem_euclid(2) as u8)
    }
    fn abs(self) -> Gf2 {
        self
    }
}

#[test]
fn finite_field_gf2_through_the_product() {
    // Over GF(2) the metric signs collapse (−1 = 1), so e1 and e2 commute:
    // (e1+e2)² = e1² + e1e2 + e2e1 + e2² = 1 + e12 + e12 + 1 = 0.
    let mut v = Multivector::<Vga2Sig, Gf2>::zero();
    v.coeffs[1] = Gf2(1); // e1
    v.coeffs[2] = Gf2(1); // e2
    assert_eq!(v * v, Multivector::<Vga2Sig, Gf2>::zero());

    // e1² = 1 still holds.
    let mut e1 = Multivector::<Vga2Sig, Gf2>::zero();
    e1.coeffs[1] = Gf2(1);
    assert_eq!((e1 * e1).scalar_part(), Gf2(1));
}

// --- ℤ: exact integers ---------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
struct Zint(i64);
impl fmt::Display for Zint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Add for Zint {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Zint(self.0 + o.0)
    }
}
impl Sub for Zint {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Zint(self.0 - o.0)
    }
}
impl Mul for Zint {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Zint(self.0 * o.0)
    }
}
impl Div for Zint {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        Zint(self.0 / o.0) // truncating: ℤ is a ring, not a field
    }
}
impl Neg for Zint {
    type Output = Self;
    fn neg(self) -> Self {
        Zint(-self.0)
    }
}
impl AddAssign for Zint {
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}
impl SubAssign for Zint {
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}
impl MulAssign for Zint {
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}
impl Scalar for Zint {
    type Magnitude = Zint;
    const ZERO: Self = Zint(0);
    const ONE: Self = Zint(1);
    fn from_f64(x: f64) -> Self {
        Zint(x as i64)
    }
    fn abs(self) -> Zint {
        Zint(self.0.abs())
    }
}

#[test]
fn exact_integer_coefficients_through_product_and_wedge() {
    // (2e1 + 3e2)² = 4·e1² + 6·e1e2 + 6·e2e1 + 9·e2² = 13 (e12 cancels), in ℤ.
    let mut v = Multivector::<Vga2Sig, Zint>::zero();
    v.coeffs[1] = Zint(2);
    v.coeffs[2] = Zint(3);
    let sq = v * v;
    assert_eq!(sq.scalar_part(), Zint(13));
    assert_eq!(sq.coeffs[3], Zint(0));

    // e1 ∧ e2 = e12, exactly.
    let mut e1 = Multivector::<Vga2Sig, Zint>::zero();
    e1.coeffs[1] = Zint(1);
    let mut e2 = Multivector::<Vga2Sig, Zint>::zero();
    e2.coeffs[2] = Zint(1);
    assert_eq!(e1.wedge(&e2).coeffs[3], Zint(1));
}
