//! Derived products of multivectors: grade projection, the wedge `∧`,
//! the inner product `·`, and the scalar product `⟨ab⟩_0`.
//!
//! All four are *bilinear* and can be computed blade-by-blade from
//! pieces we already have:
//!
//! - **Grade projection** `⟨M⟩_k` keeps only the coefficients of blades
//!   whose index has popcount `k`. Pure linear filter.
//! - **Wedge** `a ∧ b` of two basis blades is nonzero only when they
//!   share no generators (`a & b == 0`); the result is `e_{a|b}` with
//!   the reordering sign. Note: the wedge does **not** depend on the
//!   signature `(P, Q, R)` — no generator ever gets squared.
//! - **Inner product** uses the Hestenes / Doran-Lasenby convention:
//!   `⟨M⟩_i · ⟨N⟩_j = ⟨⟨M⟩_i * ⟨N⟩_j⟩_{|i-j|}` when both `i, j ≥ 1`,
//!   and zero whenever either operand is a scalar. With this convention
//!   the iconic GA identity
//!
//!   ```text
//!   a * b  =  a · b  +  a ∧ b
//!   ```
//!
//!   holds for any two multivectors with no scalar part — in
//!   particular, for any two vectors.
//! - **Scalar product** `⟨ab⟩_0` is the grade-0 part of the geometric
//!   product. It's an `f64`-valued primitive that comes up everywhere
//!   (norms, projections, the kernel of the versor inverse).

use crate::multivector::Multivector;
use crate::scalar::Scalar;
use crate::signature::{blade_product, grade_of, swap_sign};

impl<T: Scalar, const P: usize, const Q: usize, const R: usize, const DIM: usize>
    Multivector<T, P, Q, R, DIM>
{
    /// Grade-`k` projection `⟨M⟩_k`. Zeros every coefficient whose
    /// blade has popcount different from `k`.
    pub fn grade(&self, k: usize) -> Self {
        let mut out = Self::zero();
        for i in 0..DIM {
            if grade_of(i) == k {
                out.coeffs[i] = self.coeffs[i];
            }
        }
        out
    }

    /// Outer (wedge) product `self ∧ rhs`.
    ///
    /// Signature-independent: the wedge of two basis blades is the
    /// union of their generators times the reordering sign, and is
    /// zero whenever they overlap. Result blade index `a | b` (bitwise
    /// OR) is well-defined precisely because the disjointness check
    /// makes `a | b == a + b == a ^ b` for the surviving pairs.
    pub fn wedge(&self, rhs: &Self) -> Self {
        let mut out = Self::zero();
        for a in 0..DIM {
            let ca = self.coeffs[a];
            if ca == T::ZERO {
                continue;
            }
            for b in 0..DIM {
                if a & b != 0 {
                    continue;
                }
                let sign = swap_sign(a, b);
                let term = ca * rhs.coeffs[b];
                if sign > 0 {
                    out.coeffs[a | b] += term;
                } else {
                    out.coeffs[a | b] -= term;
                }
            }
        }
        out
    }

    /// Inner product `self · rhs` (Hestenes / Doran-Lasenby).
    ///
    /// For each pair of homogeneous-grade parts `⟨self⟩_i` and
    /// `⟨rhs⟩_j` with `i, j ≥ 1`, contributes
    /// `⟨⟨self⟩_i * ⟨rhs⟩_j⟩_{|i-j|}`. Scalar parts contribute nothing
    /// — the convention that lets `a * b = a · b + a ∧ b` hold for any
    /// two multivectors without scalar parts.
    ///
    /// Depends on the signature because metric factors enter through
    /// `blade_product`.
    pub fn inner(&self, rhs: &Self) -> Self {
        let mut out = Self::zero();
        for a in 0..DIM {
            let ca = self.coeffs[a];
            if ca == T::ZERO {
                continue;
            }
            let ga = grade_of(a);
            if ga == 0 {
                continue;
            }
            for b in 0..DIM {
                let cb = rhs.coeffs[b];
                if cb == T::ZERO {
                    continue;
                }
                let gb = grade_of(b);
                if gb == 0 {
                    continue;
                }
                let target = (ga as isize - gb as isize).unsigned_abs();
                let (idx, sign) = blade_product(a, b, P, Q);
                if sign != 0 && grade_of(idx) == target {
                    let term = ca * cb;
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

    /// Scalar product `⟨self * rhs⟩_0`.
    ///
    /// A pair of basis blades multiplies to a scalar only when their
    /// indices are equal (so `a ^ b == 0`), so this collapses to a
    /// single diagonal loop. Returns `T` directly — the answer is
    /// always a scalar by construction.
    pub fn scalar_product(&self, rhs: &Self) -> T {
        let mut acc = T::ZERO;
        for i in 0..DIM {
            let (_idx, sign) = blade_product(i, i, P, Q);
            if sign != 0 {
                let term = self.coeffs[i] * rhs.coeffs[i];
                if sign > 0 {
                    acc += term;
                } else {
                    acc -= term;
                }
            }
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use crate::{Vga2, Vga3};

    // --- Grade projection -----------------------------------------------

    #[test]
    fn grade_projection_isolates_each_grade() {
        // m = 1 + 2·e1 + 3·e2 + 4·e12 in Vga2
        let m = Vga2 {
            coeffs: [1.0, 2.0, 3.0, 4.0],
        };
        assert_eq!(m.grade(0).coeffs, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(m.grade(1).coeffs, [0.0, 2.0, 3.0, 0.0]);
        assert_eq!(m.grade(2).coeffs, [0.0, 0.0, 0.0, 4.0]);
    }

    #[test]
    fn grades_sum_to_original() {
        let m = Vga3 {
            coeffs: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        };
        let recon = m.grade(0) + m.grade(1) + m.grade(2) + m.grade(3);
        assert_eq!(recon.coeffs, m.coeffs);
    }

    // --- Wedge ----------------------------------------------------------

    #[test]
    fn wedge_of_distinct_vectors_is_their_bivector() {
        let e1 = Vga2::basis(1);
        let e2 = Vga2::basis(2);
        let e12 = Vga2::basis(3);
        assert_eq!(e1.wedge(&e2).coeffs, e12.coeffs);
    }

    #[test]
    fn wedge_of_a_vector_with_itself_is_zero() {
        let e1 = Vga3::basis(1);
        assert_eq!(e1.wedge(&e1).coeffs, Vga3::zero().coeffs);
    }

    #[test]
    fn wedge_anticommutes_for_vectors() {
        let e1 = Vga3::basis(1);
        let e2 = Vga3::basis(2);
        assert_eq!(e1.wedge(&e2).coeffs, (-(e2.wedge(&e1))).coeffs);
    }

    // --- Inner ----------------------------------------------------------

    #[test]
    fn inner_of_orthonormal_basis_vectors_is_delta() {
        let e1 = Vga3::basis(1);
        let e2 = Vga3::basis(2);
        // e1 · e1 = 1 (scalar)
        assert_eq!(e1.inner(&e1).coeffs, Vga3::scalar(1.0).coeffs);
        // e1 · e2 = 0
        assert_eq!(e1.inner(&e2).coeffs, Vga3::zero().coeffs);
    }

    #[test]
    fn inner_is_signature_aware() {
        // In Cl(0,1,0) the only vector squares to -1, so v · v = -1.
        use crate::Multivector;
        type Anti = Multivector<f64, 0, 1, 0, 2>;
        let e1 = Anti::basis(1);
        assert_eq!(e1.inner(&e1).coeffs, Anti::scalar(-1.0).coeffs);
    }

    // --- The iconic identity --------------------------------------------

    #[test]
    fn vectors_geometric_product_equals_inner_plus_wedge() {
        // a = 2 e1 + 3 e2 - e3,  b = -e1 + 4 e2 + 2 e3
        let a = Vga3 {
            coeffs: [0.0, 2.0, 3.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        };
        let b = Vga3 {
            coeffs: [0.0, -1.0, 4.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        };
        let lhs = a * b;
        let rhs = a.inner(&b) + a.wedge(&b);
        assert_eq!(lhs.coeffs, rhs.coeffs);
    }

    // --- Scalar product -------------------------------------------------

    #[test]
    fn scalar_product_of_basis_blades_is_their_diagonal_metric() {
        let e1 = Vga3::basis(1);
        let e2 = Vga3::basis(2);
        let e12 = Vga3::basis(3);
        // ⟨e1 * e1⟩_0 = 1
        assert_eq!(e1.scalar_product(&e1), 1.0);
        // ⟨e1 * e2⟩_0 = 0  (it's e12)
        assert_eq!(e1.scalar_product(&e2), 0.0);
        // ⟨e12 * e12⟩_0 = -1
        assert_eq!(e12.scalar_product(&e12), -1.0);
    }
}
