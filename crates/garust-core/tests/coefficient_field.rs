//! Proof that the `Scalar` / `Magnitude` split admits **non-real** coefficient
//! fields — the refactor the EML–GA program (RFC: "Computing as Geometry")
//! needs for R-0002.
//!
//! Two external types implement `Scalar` here without being ordered or being
//! their own magnitude:
//!
//! * `Complex` — the field whose `PartialOrd`-lessness was the blocker. Its
//!   `Magnitude` is `f64` (the modulus), so it can be a multivector
//!   coefficient even though it has no order.
//! * `Dual` — a forward-mode AD number (`a + b·ε`, `ε² = 0`). Flowing it
//!   through the geometric product carries derivatives for free (§4.3).
//!
//! Neither implements `Real`, so neither gets `norm`/`exp`/`Display` — exactly
//! the operations that genuinely need an ordered, real field.

use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use garust_core::{Multivector, Ring, Scalar, Vga2Sig};

// --- A minimal complex field --------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
struct Complex {
    re: f64,
    im: f64,
}
impl Complex {
    const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
}
impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{:+}i", self.re, self.im)
    }
}
impl Add for Complex {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
}
impl Sub for Complex {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
}
impl Mul for Complex {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}
impl Div for Complex {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        let d = o.re * o.re + o.im * o.im;
        Self::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
}
impl Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}
impl AddAssign for Complex {
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}
impl SubAssign for Complex {
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}
impl MulAssign for Complex {
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}
impl Ring for Complex {
    const ZERO: Self = Self::new(0.0, 0.0);
    const ONE: Self = Self::new(1.0, 0.0);
}
impl Scalar for Complex {
    type Magnitude = f64;
    fn from_f64(x: f64) -> Self {
        Self::new(x, 0.0)
    }
    fn abs(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

#[test]
fn complex_coefficients_flow_through_the_geometric_product() {
    // i·e1 in Vga2 = Cl(2,0,0). Since e1² = +1 and i² = -1,
    // (i·e1)·(i·e1) = i²·(e1·e1) = -1 (a real scalar).
    let mut v = Multivector::<Vga2Sig, Complex>::zero();
    v.coeffs[1] = Complex::new(0.0, 1.0); // i · e1
    let sq = v * v;
    assert_eq!(sq.scalar_part(), Complex::new(-1.0, 0.0));
    for i in 1..4 {
        assert_eq!(sq.coeffs[i], Complex::new(0.0, 0.0));
    }
}

// --- A forward-mode AD (dual) number -------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
struct Dual {
    re: f64, // value
    du: f64, // derivative (coefficient of ε, with ε² = 0)
}
impl Dual {
    const fn new(re: f64, du: f64) -> Self {
        Self { re, du }
    }
}
impl fmt::Display for Dual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{:+}ε", self.re, self.du)
    }
}
impl Add for Dual {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.du + o.du)
    }
}
impl Sub for Dual {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.du - o.du)
    }
}
impl Mul for Dual {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        // (a + bε)(c + dε) = ac + (ad + bc)ε
        Self::new(self.re * o.re, self.re * o.du + self.du * o.re)
    }
}
impl Div for Dual {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        // (a + bε)/(c + dε) = a/c + (bc − ad)/c² · ε
        Self::new(
            self.re / o.re,
            (self.du * o.re - self.re * o.du) / (o.re * o.re),
        )
    }
}
impl Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.du)
    }
}
impl AddAssign for Dual {
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}
impl SubAssign for Dual {
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}
impl MulAssign for Dual {
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}
impl Ring for Dual {
    const ZERO: Self = Self::new(0.0, 0.0);
    const ONE: Self = Self::new(1.0, 0.0);
}
impl Scalar for Dual {
    type Magnitude = f64;
    fn from_f64(x: f64) -> Self {
        Self::new(x, 0.0)
    }
    fn abs(self) -> f64 {
        self.re.abs()
    }
}

#[test]
fn dual_coefficients_carry_derivatives_through_the_product() {
    // Seed x with ε on the e1 coefficient. With e1² = +1 in Vga2,
    // (x + ε)·e1 squared has scalar part (x + ε)² = x² + 2x·ε — so the dual
    // part of the result is d/dx(x²) = 2x. Forward-mode AD, for free, through
    // the geometric product.
    let x = 3.0;
    let mut v = Multivector::<Vga2Sig, Dual>::zero();
    v.coeffs[1] = Dual::new(x, 1.0); // value x, derivative seed 1
    let s = (v * v).scalar_part();
    assert!((s.re - x * x).abs() < 1e-12, "value: {} vs {}", s.re, x * x);
    assert!(
        (s.du - 2.0 * x).abs() < 1e-12,
        "derivative: {} vs {}",
        s.du,
        2.0 * x
    );
}
