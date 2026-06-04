//! Property-based **law tests** for the algebra kernel.
//!
//! These check the *algebraic identities* every signature must satisfy —
//! associativity, distributivity, the involution / anti-automorphism laws,
//! the grade decomposition, and meet/join duality — against thousands of
//! random multivectors via `proptest`. They run over two deliberately
//! different signatures: `Vga3` (`Cl(3,0,0)`, non-degenerate Euclidean) and
//! `Pga3` (`Cl(3,0,1)`, whose null generator and null pseudoscalar are
//! exactly where a naive kernel breaks).
//!
//! Exact identities (the involutions, the grade partition) are asserted
//! with `==`; the bilinear-product identities accumulate floating-point
//! round-off and are checked with a mixed absolute/relative tolerance.

use garust_core::{Algebra, Multivector, Pga3, Vga3};
use proptest::prelude::*;

/// Coefficient-wise approximate equality with a mixed absolute/relative
/// tolerance — generic over the signature so one helper serves every
/// algebra under test.
fn close<A: Algebra>(a: &Multivector<A, f64>, b: &Multivector<A, f64>) -> bool {
    (0..A::DIM).all(|i| {
        let (x, y) = (a.coeffs[i], b.coeffs[i]);
        (x - y).abs() <= 1e-9 + 1e-6 * x.abs().max(y.abs())
    })
}

/// `⟨M⟩_0 + ⟨M⟩_1 + … + ⟨M⟩_N` reconstructs `M` exactly: grade projection
/// partitions the blades, so summing every grade copies each coefficient
/// back into place with no arithmetic on the values.
fn grades_partition<A: Algebra>(m: Multivector<A, f64>) -> bool {
    let recon = (0..=A::N).fold(Multivector::<A, f64>::zero(), |acc, k| acc + m.grade(k));
    recon == m
}

/// The geometric product computed straight from `blade_product` — the
/// definitional reference the const-Cayley-table product must reproduce
/// exactly (same blade-pair order ⇒ bit-identical accumulation).
fn reference_product<A: Algebra>(
    a: &Multivector<A, f64>,
    b: &Multivector<A, f64>,
) -> Multivector<A, f64> {
    let mut out = Multivector::<A, f64>::zero();
    for i in 0..A::DIM {
        for j in 0..A::DIM {
            let (idx, sign) = garust_core::signature::blade_product(i, j, A::P, A::Q);
            if sign != 0 {
                let term = a.coeffs[i] * b.coeffs[j];
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

/// A dense random `Vga3` (`Cl(3,0,0)`) — all 8 coefficients drawn from
/// `[-3, 3]`.
fn any_vga3() -> impl Strategy<Value = Vga3> {
    proptest::array::uniform8(-3.0f64..3.0).prop_map(|coeffs| Vga3 { coeffs })
}

/// A dense random `Pga3` (`Cl(3,0,1)`) — all 16 coefficients drawn from
/// `[-3, 3]`. The degenerate metric (`r = 1`) makes this the stricter test
/// of the product and duality kernels.
fn any_pga3() -> impl Strategy<Value = Pga3> {
    proptest::array::uniform16(-3.0f64..3.0).prop_map(|coeffs| Pga3 { coeffs })
}

/// Stamp out the full law-suite for one signature, given a strategy that
/// produces dense random multivectors of it.
macro_rules! algebra_laws {
    ($name:ident, $mv:expr) => {
        mod $name {
            use super::*;

            proptest! {
                // (a b) c = a (b c) — the geometric product is associative.
                #[test]
                fn geometric_product_is_associative(a in $mv, b in $mv, c in $mv) {
                    prop_assert!(close(&((a * b) * c), &(a * (b * c))));
                }

                // The const-Cayley-table product matches the blade_product
                // reference exactly — the fast path is bit-faithful.
                #[test]
                fn geometric_product_matches_reference(a in $mv, b in $mv) {
                    prop_assert_eq!(a * b, reference_product(&a, &b));
                }

                // a (b + c) = a b + a c, and the right-hand analogue.
                #[test]
                fn geometric_product_distributes_over_addition(a in $mv, b in $mv, c in $mv) {
                    prop_assert!(close(&(a * (b + c)), &(a * b + a * c)));
                    prop_assert!(close(&((a + b) * c), &(a * c + b * c)));
                }

                // ~(a b) = ~b ~a — reverse is an anti-automorphism.
                #[test]
                fn reverse_is_an_anti_automorphism(a in $mv, b in $mv) {
                    prop_assert!(close(&(a * b).reverse(), &(b.reverse() * a.reverse())));
                }

                // (a ∧ b) ∧ c = a ∧ (b ∧ c) — the wedge is associative.
                #[test]
                fn wedge_is_associative(a in $mv, b in $mv, c in $mv) {
                    prop_assert!(close(&a.wedge(&b).wedge(&c), &a.wedge(&b.wedge(&c))));
                }

                // ⟨a b⟩_0 = ⟨b a⟩_0 — the scalar product is symmetric.
                #[test]
                fn scalar_product_is_symmetric(a in $mv, b in $mv) {
                    let (ab, ba) = (a.scalar_product(&b), b.scalar_product(&a));
                    prop_assert!((ab - ba).abs() <= 1e-9 + 1e-6 * ab.abs().max(ba.abs()));
                }

                // rc(a ∨ b) = rc(a) ∧ rc(b) — meet/join De Morgan duality.
                // Holds even in degenerate Pga3, where the metric dual fails.
                #[test]
                fn meet_and_join_are_dual(a in $mv, b in $mv) {
                    let lhs = a.regressive(&b).right_complement();
                    let rhs = a.right_complement().wedge(&b.right_complement());
                    prop_assert!(close(&lhs, &rhs));
                }

                // M = Σ_k ⟨M⟩_k exactly — the grades partition the multivector.
                #[test]
                fn grades_partition_the_multivector(a in $mv) {
                    prop_assert!(grades_partition(a));
                }

                // ~~M = M exactly — reverse is an involution.
                #[test]
                fn reverse_is_an_involution(a in $mv) {
                    prop_assert_eq!(a.reverse().reverse(), a);
                }

                // M̂̂ = M exactly — grade involution is an involution.
                #[test]
                fn grade_involution_is_an_involution(a in $mv) {
                    prop_assert_eq!(a.grade_involution().grade_involution(), a);
                }
            }
        }
    };
}

algebra_laws!(vga3_laws, any_vga3());
algebra_laws!(pga3_laws, any_pga3());
