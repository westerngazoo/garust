//! The inverse of a versor that is a versor by construction.
//!
//! [`Motor`](crate::Motor) and [`Conformal`](crate::Conformal) hold a
//! versor by construction: every public constructor builds one, and
//! products and exponentials of versors stay versors. Their inverse is
//! therefore `~V / ⟨V ~V⟩₀` with no need to *check* that `V ~V` is a
//! scalar. The kernel's `versor_inverse` does check, and panics when the
//! check fails; numerical drift after many compositions can push a
//! genuine motor past that check, which turned an inverse, and every
//! `slerp` built on it, into a crash (rule G3). This helper is the same
//! formula without the check.

use garust_core::algebra::Algebra;
use garust_core::multivector::Multivector;
use garust_core::scalar::Scalar;

/// `~V / ⟨V ~V⟩₀`, the inverse of a versor `V`, computed exactly as the
/// kernel's `try_versor_inverse` computes it, so a valid versor gets the
/// same result to the last bit.
///
/// Never panics for floating-point coefficients. If `⟨V ~V⟩₀` is zero,
/// which a versor built through the public constructors never has, the
/// result has non-finite coefficients rather than a crash.
pub(crate) fn inverse_by_construction<A: Algebra, T: Scalar>(
    v: &Multivector<A, T>,
) -> Multivector<A, T> {
    let rev = v.reverse();
    let norm_sq = (*v * rev).scalar_part();
    rev * (T::ONE / norm_sq)
}
