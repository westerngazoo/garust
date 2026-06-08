//! Forward-mode automatic differentiation via **dual numbers**.
//!
//! A [`Dual`] number is `value + deriv·ε` with `ε² = 0`. For any analytic
//! `f`, the Taylor expansion truncates after one term:
//!
//! ```text
//! f(a + b·ε) = f(a) + f'(a)·b·ε
//! ```
//!
//! so the ε-part carries the derivative *exactly* — no symbolic
//! differentiation, no finite differences.
//!
//! [`Dual`] implements [`Scalar`] and [`Real`], so it drops in as a
//! [`Multivector`](crate::Multivector) coefficient: evaluating once over
//! `Dual<f64>` coefficients yields both a value and a derivative through the
//! *entire* algebra — the geometric product, the involutions, `exp`, the
//! sandwich, norms. Seed the variable you want to differentiate with
//! [`Dual::variable`] and read the result's [`deriv`](Dual::deriv).
//!
//! ```
//! use garust_core::Dual;
//!
//! // d/dx (x²) at x = 3, straight through multiplication.
//! let x = Dual::variable(3.0_f64);
//! let y = x * x;
//! assert_eq!(y.value, 9.0);
//! assert_eq!(y.deriv, 6.0); // 2x = 6
//! ```
//!
//! Dual numbers are also the geometric algebra `G(0,0,1)` — the algebra of a
//! single *null* direction — so forward-mode AD is, structurally, "adding a
//! null dimension to the metric." Nesting (`Dual<Dual<f64>>`) gives
//! higher-order derivatives, since `Dual<T>` is itself a [`Real`] field.

use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::scalar::{Real, Ring, Scalar};

/// A forward-mode AD dual number: `value + deriv·ε`, with `ε² = 0`.
///
/// Build a constant with [`Dual::constant`], the variable you're
/// differentiating against with [`Dual::variable`], or both parts directly
/// with [`Dual::new`]. The fields are public for ergonomic access.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual<T> {
    /// The value (real part), `f(a)`.
    pub value: T,
    /// The first derivative (coefficient of `ε`), `f'(a)`.
    pub deriv: T,
}

impl<T: Real> Dual<T> {
    /// A dual with explicit value and derivative parts.
    pub fn new(value: T, deriv: T) -> Self {
        Self { value, deriv }
    }

    /// A constant — derivative zero. Use for inputs you are *not*
    /// differentiating against.
    pub fn constant(value: T) -> Self {
        Self {
            value,
            deriv: T::ZERO,
        }
    }

    /// The variable being differentiated — derivative one. Seed exactly the
    /// input whose derivative you want; the ε-part of the result is `df/dx`.
    pub fn variable(value: T) -> Self {
        Self {
            value,
            deriv: T::ONE,
        }
    }
}

impl<T: Real> Add for Dual<T> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.value + o.value, self.deriv + o.deriv)
    }
}

impl<T: Real> Sub for Dual<T> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.value - o.value, self.deriv - o.deriv)
    }
}

impl<T: Real> Mul for Dual<T> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        // (a + bε)(c + dε) = ac + (ad + bc)ε  — the product rule.
        Self::new(
            self.value * o.value,
            self.value * o.deriv + self.deriv * o.value,
        )
    }
}

impl<T: Real> Div for Dual<T> {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        // (a + bε)/(c + dε) = a/c + (bc − ad)/c² · ε  — the quotient rule.
        Self::new(
            self.value / o.value,
            (self.deriv * o.value - self.value * o.deriv) / (o.value * o.value),
        )
    }
}

impl<T: Real> Neg for Dual<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.value, -self.deriv)
    }
}

impl<T: Real> AddAssign for Dual<T> {
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

impl<T: Real> SubAssign for Dual<T> {
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}

impl<T: Real> MulAssign for Dual<T> {
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}

impl<T: Real> fmt::Display for Dual<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}ε", self.value, self.deriv)
    }
}

/// Ordered by value (the ε-part doesn't change which side of a tolerance a
/// magnitude falls on), so duals work as a [`Scalar::Magnitude`].
impl<T: Real> PartialOrd for Dual<T> {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&o.value)
    }
}

impl<T: Real> Ring for Dual<T> {
    const ZERO: Self = Self {
        value: T::ZERO,
        deriv: T::ZERO,
    };
    const ONE: Self = Self {
        value: T::ONE,
        deriv: T::ZERO,
    };
}

impl<T: Real> Scalar for Dual<T> {
    type Magnitude = Self;
    fn from_f64(x: f64) -> Self {
        Self {
            value: T::from_f64(x),
            deriv: T::ZERO,
        }
    }
    fn abs(self) -> Self::Magnitude {
        // d/dx |x| = sign(x); at x = 0 we take the right derivative.
        if self.value < T::ZERO {
            Self::new(-self.value, -self.deriv)
        } else {
            self
        }
    }
}

impl<T: Real> Real for Dual<T> {
    fn sqrt(self) -> Self {
        let root = self.value.sqrt();
        // d/dx √x = 1/(2√x)
        Self::new(root, self.deriv / (T::from_f64(2.0) * root))
    }
    fn sin(self) -> Self {
        Self::new(self.value.sin(), self.deriv * self.value.cos())
    }
    fn cos(self) -> Self {
        Self::new(self.value.cos(), -(self.deriv * self.value.sin()))
    }
    fn sinh(self) -> Self {
        Self::new(self.value.sinh(), self.deriv * self.value.cosh())
    }
    fn cosh(self) -> Self {
        Self::new(self.value.cosh(), self.deriv * self.value.sinh())
    }
    fn ln(self) -> Self {
        // d/dx ln x = 1/x
        Self::new(self.value.ln(), self.deriv / self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::Dual;
    use crate::scalar::Real;
    use crate::{Multivector, Vga2Sig};

    type DualF = Dual<f64>;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }

    #[test]
    fn product_rule_through_the_geometric_product() {
        // Coefficient (x + ε) on e1 in Vga2; e1² = +1, so the scalar part of
        // the square is (x + ε)² = x² + 2x·ε.
        let x = 3.0;
        let mut v = Multivector::<Vga2Sig, DualF>::zero();
        v.coeffs[1] = Dual::variable(x);
        let s = (v * v).scalar_part();
        close(s.value, x * x);
        close(s.deriv, 2.0 * x);
    }

    #[test]
    fn chain_rule_through_transcendentals() {
        let x = Dual::variable(0.7_f64);
        close(x.sin().deriv, 0.7_f64.cos());
        close(x.cos().deriv, -0.7_f64.sin());
        close(x.ln().deriv, 1.0 / 0.7);
        close(
            Dual::variable(2.0_f64).sqrt().deriv,
            1.0 / (2.0 * 2.0_f64.sqrt()),
        );
    }

    #[test]
    fn derivative_of_a_rotor_through_multivector_exp() {
        // R(θ) = exp(−θ/2 · e12) = cos(θ/2) − sin(θ/2)·e12 in Vga2.
        // Differentiate w.r.t. θ straight through Multivector::exp.
        let theta = 0.8_f64;
        let half = Dual::variable(theta) * Dual::constant(-0.5); // −θ/2, dθ = −0.5
        let mut b = Multivector::<Vga2Sig, DualF>::zero();
        b.coeffs[3] = half; // (−θ/2)·e12
        let r = b.exp();

        // scalar part cos(θ/2): d/dθ = −½ sin(θ/2)
        close(r.coeffs[0].value, (theta / 2.0).cos());
        close(r.coeffs[0].deriv, -0.5 * (theta / 2.0).sin());
        // e12 part −sin(θ/2): d/dθ = −½ cos(θ/2)
        close(r.coeffs[3].value, -(theta / 2.0).sin());
        close(r.coeffs[3].deriv, -0.5 * (theta / 2.0).cos());
    }

    #[test]
    fn second_order_via_nested_duals() {
        // Dual<Dual<f64>> gives the second derivative: d²/dx²(x³) = 6x.
        // The outer `.deriv` is d/dx; its inner `.deriv` is d²/dx².
        let x = Dual::<DualF>::new(Dual::variable(2.0), Dual::constant(1.0));
        let y = x * x * x;
        close(y.deriv.deriv, 6.0 * 2.0);
    }
}
