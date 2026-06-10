//! The numeric backbone: the [`Scalar`] and [`Real`] traits that let
//! [`Multivector`](crate::Multivector) be generic over its coefficient
//! type.
//!
//! garust takes **zero external dependencies**, so rather than pulling
//! in `num-traits` we define the minimal interface the algebra needs
//! and implement it for `f32` and `f64`. Anything that behaves like an
//! ordered field — fixed-point types, `f16` wrappers, dual numbers for
//! autodiff — can opt in by implementing these traits.
//!
//! The split is deliberate, in three layers:
//!
//! - [`Ring`] is the **division-free core**: `+`, `−`, `×`, and the
//!   `ZERO`/`ONE` identities. The geometric product, the derived products,
//!   grade projection, and the involutions need only this — so exact,
//!   division-free types (the integers `ℤ`, …) can be coefficients, and you
//!   simply can't take a versor inverse over them (a compile error, by
//!   design, not a silent wrong answer).
//! - [`Scalar`] adds **division**, an `f64` conversion, and a real-valued
//!   [`abs`](Scalar::abs) whose type [`Magnitude`](Scalar::Magnitude) carries
//!   the ordering. It is the field interface — what the versor inverse and
//!   normalization require. The field itself need not be ordered, so
//!   `Complex<f64>` and dual numbers (no order, but a real modulus) qualify.
//! - [`Real`] adds **ordering** (`PartialOrd`) and the **transcendentals**
//!   (`sqrt`, `sin`, `cos`, `sinh`, `cosh`, `ln`) used by
//!   [`Multivector::exp`](crate::Multivector::exp) and the norm / `Display`
//!   paths, and is its own magnitude (`Magnitude = Self`).

use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// A commutative ring — the arithmetic the algebra *core* needs.
///
/// The geometric product, the wedge / inner / scalar products, grade
/// projection, and the involutions are built entirely from `+`, `−`, `×`, and
/// the `0`/`1` identities — **no division**. So they are defined for any
/// `Ring`, which lets exact, division-free coefficient types (the integers
/// `ℤ`, polynomials, …) be multivector coefficients. Division-only operations
/// (the versor inverse, normalization) live on [`Scalar`], so they are simply
/// unavailable — a compile error, not a silent wrong answer — over a ring
/// that is not a field.
pub trait Ring:
    Copy
    + fmt::Debug
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
{
    /// The additive identity.
    const ZERO: Self;
    /// The multiplicative identity.
    const ONE: Self;
}

/// A field: a [`Ring`] with division, a real-valued magnitude, and an `f64`
/// conversion for tolerances and literals.
///
/// Implemented for `f32`/`f64`, and satisfiable by `Complex<f64>`, dual
/// numbers, and exact fields (a finite field like GF(2), the rationals). The
/// field itself need not be ordered — only its
/// [`Magnitude`](Scalar::Magnitude). `Scalar` is what the **division-based**
/// operations (versor inverse, normalization) require.
pub trait Scalar: Ring + Div<Output = Self> + fmt::Display {
    /// The ordered magnitude type returned by [`abs`](Scalar::abs), used for
    /// tolerance and zero-threshold comparisons.
    ///
    /// It need only be an *ordered* [`Scalar`] (`Scalar + PartialOrd`), not a
    /// [`Real`] one — so the field itself may be unordered (`Complex<f64>`,
    /// whose magnitude is the real modulus) or exact (a finite field, the
    /// rationals — each its own magnitude). For an ordered real scalar it is
    /// simply `Self`.
    type Magnitude: Scalar + PartialOrd;

    /// Convert from an `f64`. Used for tolerances and scalar literals.
    /// May lose precision (e.g. for `f32`); that's expected.
    fn from_f64(x: f64) -> Self;
    /// Real-valued magnitude (absolute value / modulus), used for tolerance
    /// and zero-threshold comparisons.
    fn abs(self) -> Self::Magnitude;
}

/// An *ordered, real* [`Scalar`] that also provides the transcendental
/// functions used by [`Multivector::exp`](crate::Multivector::exp).
///
/// `Real` is the magnitude end of the [`Scalar`] split: it is comparable
/// (`PartialOrd`) and is its own [`Magnitude`](Scalar::Magnitude)
/// (`Scalar<Magnitude = Self>`), so norms and tolerances stay in one type.
/// `f32` and `f64` implement it; `Complex<f64>` does not (it has no order),
/// which is exactly why the ordering-dependent, transcendental operations
/// live here rather than on [`Scalar`].
pub trait Real: Scalar<Magnitude = Self> + PartialOrd {
    /// Square root.
    fn sqrt(self) -> Self;
    /// Sine (radians).
    fn sin(self) -> Self;
    /// Cosine (radians).
    fn cos(self) -> Self;
    /// Hyperbolic sine.
    fn sinh(self) -> Self;
    /// Hyperbolic cosine.
    fn cosh(self) -> Self;
    /// Natural logarithm. Used to turn a scaling *factor* into the
    /// log-scale that drives a conformal dilator.
    fn ln(self) -> Self;
    /// Four-quadrant arctangent of `self / x` (radians, in `(−τ/2, τ/2]`).
    /// Used by the versor logarithm to recover a rotation angle from its
    /// sine and cosine.
    fn atan2(self, x: Self) -> Self;
}

// `abs` and the transcendentals are the only operations that need a math
// backend — everything else is plain `core` arithmetic. We therefore take
// the backend functions as macro arguments and stamp the impls out once per
// backend (the standard library or `libm`), each gated by feature below. A
// custom `Scalar`/`Real` type needs neither feature.
macro_rules! impl_scalar_real {
    ($t:ty, $abs:path, $sqrt:path, $sin:path, $cos:path, $sinh:path, $cosh:path, $ln:path, $atan2:path) => {
        impl Ring for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
        }

        impl Scalar for $t {
            type Magnitude = $t;
            #[inline]
            fn from_f64(x: f64) -> Self {
                x as $t
            }
            #[inline]
            fn abs(self) -> Self::Magnitude {
                $abs(self)
            }
        }

        impl Real for $t {
            #[inline]
            fn sqrt(self) -> Self {
                $sqrt(self)
            }
            #[inline]
            fn sin(self) -> Self {
                $sin(self)
            }
            #[inline]
            fn cos(self) -> Self {
                $cos(self)
            }
            #[inline]
            fn sinh(self) -> Self {
                $sinh(self)
            }
            #[inline]
            fn cosh(self) -> Self {
                $cosh(self)
            }
            #[inline]
            fn ln(self) -> Self {
                $ln(self)
            }
            #[inline]
            fn atan2(self, x: Self) -> Self {
                $atan2(self, x)
            }
        }
    };
}

// The standard-library backend: the inherent float methods. `std` is linked
// at the crate root under the `std` feature. Takes precedence when both
// `std` and `libm` are enabled.
#[cfg(feature = "std")]
mod float_backend {
    use super::{Real, Ring, Scalar};

    impl_scalar_real!(
        f32,
        f32::abs,
        f32::sqrt,
        f32::sin,
        f32::cos,
        f32::sinh,
        f32::cosh,
        f32::ln,
        f32::atan2
    );
    impl_scalar_real!(
        f64,
        f64::abs,
        f64::sqrt,
        f64::sin,
        f64::cos,
        f64::sinh,
        f64::cosh,
        f64::ln,
        f64::atan2
    );
}

// The `libm` backend: the same functions without the standard library, for
// `no_std` builds. Used when `std` is off and `libm` is on. `libm::log` is
// the natural logarithm.
#[cfg(all(not(feature = "std"), feature = "libm"))]
mod float_backend {
    use super::{Real, Ring, Scalar};

    impl_scalar_real!(
        f32,
        libm::fabsf,
        libm::sqrtf,
        libm::sinf,
        libm::cosf,
        libm::sinhf,
        libm::coshf,
        libm::logf,
        libm::atan2f
    );
    impl_scalar_real!(
        f64,
        libm::fabs,
        libm::sqrt,
        libm::sin,
        libm::cos,
        libm::sinh,
        libm::cosh,
        libm::log,
        libm::atan2
    );
}

/// Returns the larger of two ordered values (`PartialOrd`-based, NaN-naive).
/// A small helper since `Ord::max` isn't available for floats. Called on
/// magnitudes ([`Scalar::Magnitude`]), which are always [`Real`] and hence
/// ordered — so it works even when the coefficient field itself is not.
#[inline]
pub(crate) fn max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b {
        a
    } else {
        b
    }
}
