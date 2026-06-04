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
//! The split is deliberate:
//!
//! - [`Scalar`] is the **ring/field interface**: the arithmetic, the
//!   `ZERO`/`ONE` identities, ordering, `abs`, and a conversion from
//!   `f64` for tolerances and literals. Everything needed for the
//!   geometric product, the derived products, the involutions, the
//!   versor inverse, and `Display`.
//! - [`Real`] adds the **transcendental functions** (`sqrt`, `sin`,
//!   `cos`, `sinh`, `cosh`) needed only by
//!   [`Multivector::exp`](crate::Multivector::exp). Keeping these out of
//!   `Scalar` means a coefficient type that can't define `sin` (a
//!   rational, say) can still drive the whole product algebra.

use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// An ordered field suitable as a multivector coefficient type.
///
/// Implemented for `f32` and `f64`. The required `AddAssign` /
/// `SubAssign` / `MulAssign` bounds let the linear-algebra loops stay
/// written in their natural `+=` / `-=` form.
pub trait Scalar:
    Copy
    + fmt::Debug
    + fmt::Display
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
{
    /// The additive identity.
    const ZERO: Self;
    /// The multiplicative identity.
    const ONE: Self;
    /// Convert from an `f64`. Used for tolerances and scalar literals.
    /// May lose precision (e.g. for `f32`); that's expected.
    fn from_f64(x: f64) -> Self;
    /// Absolute value.
    fn abs(self) -> Self;
}

/// A [`Scalar`] that also provides the transcendental functions used by
/// [`Multivector::exp`](crate::Multivector::exp).
pub trait Real: Scalar {
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
}

// `abs` and the transcendentals are the only operations that need a math
// backend — everything else is plain `core` arithmetic. We therefore take
// the backend functions as macro arguments and stamp the impls out once per
// backend (the standard library or `libm`), each gated by feature below. A
// custom `Scalar`/`Real` type needs neither feature.
macro_rules! impl_scalar_real {
    ($t:ty, $abs:path, $sqrt:path, $sin:path, $cos:path, $sinh:path, $cosh:path, $ln:path) => {
        impl Scalar for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            #[inline]
            fn from_f64(x: f64) -> Self {
                x as $t
            }
            #[inline]
            fn abs(self) -> Self {
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
        }
    };
}

// The standard-library backend: the inherent float methods. `std` is linked
// at the crate root under the `std` feature. Takes precedence when both
// `std` and `libm` are enabled.
#[cfg(feature = "std")]
mod float_backend {
    use super::{Real, Scalar};

    impl_scalar_real!(
        f32,
        f32::abs,
        f32::sqrt,
        f32::sin,
        f32::cos,
        f32::sinh,
        f32::cosh,
        f32::ln
    );
    impl_scalar_real!(
        f64,
        f64::abs,
        f64::sqrt,
        f64::sin,
        f64::cos,
        f64::sinh,
        f64::cosh,
        f64::ln
    );
}

// The `libm` backend: the same functions without the standard library, for
// `no_std` builds. Used when `std` is off and `libm` is on. `libm::log` is
// the natural logarithm.
#[cfg(all(not(feature = "std"), feature = "libm"))]
mod float_backend {
    use super::{Real, Scalar};

    impl_scalar_real!(
        f32,
        libm::fabsf,
        libm::sqrtf,
        libm::sinf,
        libm::cosf,
        libm::sinhf,
        libm::coshf,
        libm::logf
    );
    impl_scalar_real!(
        f64,
        libm::fabs,
        libm::sqrt,
        libm::sin,
        libm::cos,
        libm::sinh,
        libm::cosh,
        libm::log
    );
}

/// Returns the larger of two scalars (`PartialOrd`-based, NaN-naive).
/// A small helper since `Ord::max` isn't available for floats.
#[inline]
pub(crate) fn max<T: Scalar>(a: T, b: T) -> T {
    if a > b {
        a
    } else {
        b
    }
}
