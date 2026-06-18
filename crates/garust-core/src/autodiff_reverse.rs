//! Reverse-mode automatic differentiation — the whole gradient in **one**
//! backward pass.
//!
//! Forward-mode ([`Dual`](crate::Dual)) carries one derivative through a
//! computation, so [`partials`](crate::partials) needs `DIM` evaluations of
//! `f` to fill a multivector gradient (one ε-seed per coefficient). Reverse
//! mode instead *records* the computation as it runs (a Wengert tape) and then
//! propagates adjoints backward, yielding **every** coefficient's derivative
//! from a single forward evaluation plus a single backward sweep — the win
//! grows with `DIM` (16 for PGA, 32 for CGA) and with how expensive `f` is.
//!
//! [`Var`] is the reverse-mode scalar. It implements [`Ring`] / [`Scalar`] /
//! [`Real`], so it drops into `Multivector<A, Var>` exactly like `f64` or
//! `Dual` — write `f` once, generic over the scalar, and differentiate it
//! either way. [`gradient`] is the reverse-mode analogue of
//! [`partials`](crate::partials).
//!
//! ```
//! use garust_core::{autodiff_reverse::gradient, Vga3, Vga3Sig, Multivector, Var};
//!
//! // f(X) = x0·x1 (scalar coefficient times the e1 coefficient).
//! let mut x = Vga3::zero();
//! x.coeffs[0] = 5.0;
//! x.coeffs[1] = 7.0;
//! let g = gradient(&x, |m: &Multivector<Vga3Sig, Var>| m.coeffs[0] * m.coeffs[1]);
//! assert_eq!(g.coeffs[0], 7.0); // ∂/∂x0 = x1
//! assert_eq!(g.coeffs[1], 5.0); // ∂/∂x1 = x0
//! ```
//!
//! ## Mechanism and limits
//!
//! The tape is **thread-local**: each thread differentiates independently, and
//! [`Var`] stays a plain `Copy` value (`{value, node-index}`) with `const`
//! [`ZERO`](Ring::ZERO) / [`ONE`](Ring::ONE) (a sentinel index marks a
//! constant). [`gradient`] resets the thread's tape, so a single
//! differentiation runs at a time per thread — do **not** call [`gradient`]
//! re-entrantly (e.g. from inside the `f` passed to another `gradient`).
//! Requires the `std` feature (the tape is a heap `Vec`); the default build has
//! it.

use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use std::cell::RefCell;
use std::thread_local; // the crate is #![no_std]; bring the std macro into scope
use std::vec::Vec;

use crate::algebra::Algebra;
use crate::multivector::Multivector;
use crate::scalar::{Real, Ring, Scalar};

/// Sentinel node index marking a *constant* — a [`Var`] not recorded on the
/// tape, so it seeds no adjoint.
const CONST: u32 = u32::MAX;

/// One tape entry: up to two parents, each `(node index, ∂self/∂parent)`.
/// Unused slots and constant parents use [`CONST`] and are skipped on the
/// backward sweep. Leaves have both slots `CONST`.
#[derive(Clone, Copy)]
struct Node {
    parents: [(u32, f64); 2],
}

thread_local! {
    static TAPE: RefCell<Vec<Node>> = const { RefCell::new(Vec::new()) };
}

/// Push a node and return its index.
fn record(parents: [(u32, f64); 2]) -> u32 {
    TAPE.with(|t| {
        let mut t = t.borrow_mut();
        let idx = t.len() as u32;
        t.push(Node { parents });
        idx
    })
}

const NONE: (u32, f64) = (CONST, 0.0);

/// A reverse-mode AD scalar: a value plus its position on the thread-local
/// tape. `Copy`, and a drop-in coefficient type (`Multivector<A, Var>`) via
/// its [`Ring`] / [`Scalar`] / [`Real`] impls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Var {
    /// The forward value.
    pub value: f64,
    /// Tape index, or [`CONST`] for a constant.
    idx: u32,
}

impl Var {
    /// A constant — contributes a value but no gradient.
    pub fn constant(value: f64) -> Self {
        Var { value, idx: CONST }
    }

    /// A tape leaf (an independent variable). [`gradient`] seeds these.
    fn leaf(value: f64) -> Self {
        Var {
            value,
            idx: record([NONE, NONE]),
        }
    }

    fn unary(value: f64, p: u32, partial: f64) -> Self {
        Var {
            value,
            idx: record([(p, partial), NONE]),
        }
    }

    fn binary(value: f64, a: u32, pa: f64, b: u32, pb: f64) -> Self {
        Var {
            value,
            idx: record([(a, pa), (b, pb)]),
        }
    }
}

// --- Ring / Scalar / Real -------------------------------------------------

impl Ring for Var {
    const ZERO: Self = Var {
        value: 0.0,
        idx: CONST,
    };
    const ONE: Self = Var {
        value: 1.0,
        idx: CONST,
    };
}

impl Add for Var {
    type Output = Var;
    fn add(self, rhs: Var) -> Var {
        Var::binary(self.value + rhs.value, self.idx, 1.0, rhs.idx, 1.0)
    }
}
impl Sub for Var {
    type Output = Var;
    fn sub(self, rhs: Var) -> Var {
        Var::binary(self.value - rhs.value, self.idx, 1.0, rhs.idx, -1.0)
    }
}
impl Mul for Var {
    type Output = Var;
    fn mul(self, rhs: Var) -> Var {
        // product rule: ∂(ab)/∂a = b, ∂(ab)/∂b = a
        Var::binary(
            self.value * rhs.value,
            self.idx,
            rhs.value,
            rhs.idx,
            self.value,
        )
    }
}
impl Neg for Var {
    type Output = Var;
    fn neg(self) -> Var {
        Var::unary(-self.value, self.idx, -1.0)
    }
}
impl AddAssign for Var {
    fn add_assign(&mut self, rhs: Var) {
        *self = *self + rhs;
    }
}
impl SubAssign for Var {
    fn sub_assign(&mut self, rhs: Var) {
        *self = *self - rhs;
    }
}
impl MulAssign for Var {
    fn mul_assign(&mut self, rhs: Var) {
        *self = *self * rhs;
    }
}

impl Div for Var {
    type Output = Var;
    fn div(self, rhs: Var) -> Var {
        let inv = 1.0 / rhs.value;
        // ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = −a/b²
        Var::binary(
            self.value * inv,
            self.idx,
            inv,
            rhs.idx,
            -self.value * inv * inv,
        )
    }
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

// Ordered by forward value (PartialEq stays structural so a seeded
// zero-valued leaf is *not* equal to the constant ZERO, and so keeps its
// gradient through the products' zero-skip).
impl PartialOrd for Var {
    fn partial_cmp(&self, other: &Var) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl Scalar for Var {
    type Magnitude = Var;
    fn from_f64(x: f64) -> Self {
        Var::constant(x)
    }
    fn abs(self) -> Var {
        let s = if self.value < 0.0 { -1.0 } else { 1.0 };
        Var::unary(self.value.abs(), self.idx, s)
    }
}

impl Real for Var {
    fn sqrt(self) -> Self {
        let r = self.value.sqrt();
        Var::unary(r, self.idx, 0.5 / r) // d√x = 1/(2√x)
    }
    fn sin(self) -> Self {
        Var::unary(self.value.sin(), self.idx, self.value.cos())
    }
    fn cos(self) -> Self {
        Var::unary(self.value.cos(), self.idx, -self.value.sin())
    }
    fn sinh(self) -> Self {
        Var::unary(self.value.sinh(), self.idx, self.value.cosh())
    }
    fn cosh(self) -> Self {
        Var::unary(self.value.cosh(), self.idx, self.value.sinh())
    }
    fn ln(self) -> Self {
        Var::unary(self.value.ln(), self.idx, 1.0 / self.value)
    }
    fn atan2(self, x: Self) -> Self {
        let d = self.value * self.value + x.value * x.value;
        // ∂atan2(y,x)/∂y = x/(x²+y²), ∂/∂x = −y/(x²+y²)
        Var::binary(
            self.value.atan2(x.value),
            self.idx,
            x.value / d,
            x.idx,
            -self.value / d,
        )
    }
}

// --- The reverse sweep ----------------------------------------------------

/// Propagate the adjoint of node `out` (seeded to 1) backward over the tape,
/// returning the adjoint of every node — `adj[i] = ∂out/∂node_i`.
fn backward(out: u32) -> Vec<f64> {
    TAPE.with(|t| {
        let tape = t.borrow();
        let n = tape.len();
        let mut adj: Vec<f64> = Vec::new();
        adj.resize(n, 0.0); // (no `vec!` macro under no_std)
        if (out as usize) < n {
            adj[out as usize] = 1.0;
        }
        // Reverse sweep: a node's parents always precede it on the tape, so
        // walking high→low means each node's adjoint is final before it feeds
        // its parents. This needs indexed access with cross-index accumulation
        // into `adj` (`adj[p] += …` for p < i), which no iterator expresses.
        #[allow(clippy::needless_range_loop)]
        for i in (0..n).rev() {
            let a = adj[i];
            if a == 0.0 {
                continue;
            }
            for (p, partial) in tape[i].parents {
                if p != CONST {
                    adj[p as usize] += a * partial;
                }
            }
        }
        adj
    })
}

/// The coefficient gradient of a scalar function `f`, by **reverse-mode** AD —
/// the same result as [`partials`](crate::partials), in one forward pass plus
/// one backward sweep instead of `DIM` forward passes.
///
/// `f` is evaluated once over [`Var`] coefficients (seeded as the tape's first
/// `DIM` leaves), then the adjoints are swept back; `grad.coeffs[J]` is
/// `∂f/∂x_J`. For the coordinate-free `∂_X f`, scale by the reciprocal-frame
/// metric sign exactly as [`multivector_derivative`](crate::multivector_derivative)
/// does to [`partials`](crate::partials).
pub fn gradient<A, F>(x: &Multivector<A, f64>, f: F) -> Multivector<A, f64>
where
    A: Algebra,
    F: Fn(&Multivector<A, Var>) -> Var,
{
    // Reset this thread's tape, then seed coefficient leaves. `zero()` records
    // nothing (its ZEROs are constants), so leaf J lands at tape index J.
    TAPE.with(|t| t.borrow_mut().clear());
    let mut xv = Multivector::<A, Var>::zero();
    for i in 0..A::DIM {
        xv.coeffs[i] = Var::leaf(x.coeffs[i]);
    }
    let out = f(&xv);
    let adj = backward(out.idx);

    let mut grad = Multivector::<A, f64>::zero();
    // Leaf J is tape node J, so its adjoint is ∂f/∂x_J.
    for (i, &a) in adj.iter().take(A::DIM).enumerate() {
        grad.coeffs[i] = a;
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::{gradient, Var};
    use crate::autodiff::Dual;
    use crate::calculus::partials;
    use crate::scalar::Real;
    use crate::{Multivector, Vga3, Vga3Sig};

    // A scalar function of a multivector, written once and generic over the
    // scalar so the *same* code differentiates via forward and reverse modes.
    fn energy<S: Real>(m: &Multivector<Vga3Sig, S>) -> S {
        // ⟨X X⟩₀ + x1·x2  — products + a cross term, exercising the chain.
        (*m * *m).scalar_part() + m.coeffs[1] * m.coeffs[2]
    }

    fn random_vga3() -> Vga3 {
        Vga3 {
            coeffs: [0.3, -1.2, 2.5, 0.7, -0.4, 1.1, -2.0, 0.9],
        }
    }

    #[test]
    fn hand_checked_product_gradient() {
        // f(X) = x0·x1 ⇒ ∂/∂x0 = x1, ∂/∂x1 = x0, rest 0.
        let mut x = Vga3::zero();
        x.coeffs[0] = 5.0;
        x.coeffs[1] = 7.0;
        let g = gradient(&x, |m: &Multivector<Vga3Sig, Var>| {
            m.coeffs[0] * m.coeffs[1]
        });
        assert_eq!(g.coeffs[0], 7.0);
        assert_eq!(g.coeffs[1], 5.0);
        for i in 2..8 {
            assert_eq!(g.coeffs[i], 0.0);
        }
    }

    #[test]
    fn reverse_matches_forward_on_a_quadratic() {
        let x = random_vga3();
        let fwd = partials(&x, |m: &Multivector<Vga3Sig, Dual<f64>>| energy(m));
        let rev = gradient(&x, |m: &Multivector<Vga3Sig, Var>| energy(m));
        for i in 0..8 {
            assert!(
                (fwd.coeffs[i] - rev.coeffs[i]).abs() < 1e-9,
                "coeff {i}: fwd {} vs rev {}",
                fwd.coeffs[i],
                rev.coeffs[i]
            );
        }
    }

    #[test]
    fn reverse_matches_forward_through_transcendentals() {
        // A scalar built through sqrt/sin/ln of coefficient combinations.
        fn g<S: Real>(m: &Multivector<Vga3Sig, S>) -> S {
            let a = m.coeffs[1] * m.coeffs[1] + m.coeffs[2] * m.coeffs[2] + S::from_f64(1.0);
            a.sqrt().sin() + (m.coeffs[4] * m.coeffs[4] + S::from_f64(2.0)).ln()
        }
        let x = random_vga3();
        let fwd = partials(&x, |m: &Multivector<Vga3Sig, Dual<f64>>| g(m));
        let rev = gradient(&x, |m: &Multivector<Vga3Sig, Var>| g(m));
        for i in 0..8 {
            assert!(
                (fwd.coeffs[i] - rev.coeffs[i]).abs() < 1e-9,
                "coeff {i}: fwd {} vs rev {}",
                fwd.coeffs[i],
                rev.coeffs[i]
            );
        }
    }

    #[test]
    fn gradient_of_a_zero_valued_input_is_kept() {
        // The classic reverse-AD trap: a zero-valued input must still get its
        // (generally nonzero) gradient — f = x0·x1 at x0 = 0 has ∂/∂x0 = x1.
        let mut x = Vga3::zero();
        x.coeffs[0] = 0.0;
        x.coeffs[1] = 4.0;
        let g = gradient(&x, |m: &Multivector<Vga3Sig, Var>| {
            m.coeffs[0] * m.coeffs[1]
        });
        assert_eq!(g.coeffs[0], 4.0); // not skipped despite x0 == 0
        assert_eq!(g.coeffs[1], 0.0);
    }
}
