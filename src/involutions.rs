//! Sign-flip operations indexed by blade grade: reverse, grade
//! involution, and Clifford conjugation. Plus the squared norm, which
//! is built from them.
//!
//! Each involution flips the sign of basis blades according to a
//! pattern that depends only on the *grade* of the blade. Cheat sheet
//! (`+` = unchanged, `−` = sign flipped):
//!
//! | grade `k`   | reverse `~M` | grade involution `M̂` | conjugation `M̄` |
//! |-------------|--------------|----------------------|-----------------|
//! | 0 (scalar)  | `+`          | `+`                  | `+`             |
//! | 1 (vector)  | `+`          | `−`                  | `−`             |
//! | 2 (bivec)   | `−`          | `+`                  | `−`             |
//! | 3 (trivec)  | `−`          | `−`                  | `+`             |
//! | 4           | `+`          | `+`                  | `+`             |
//!
//! Algebraic forms of the sign exponent (mod 2):
//! - Reverse:           `(k / 2)       mod 2`  — i.e. `(-1)^(k(k-1)/2)`
//! - Grade involution:  `k             mod 2`
//! - Conjugation:       `((k + 1) / 2) mod 2`  — i.e. `(-1)^(k(k+1)/2)`
//!   (equivalently, `reverse ∘ grade_involution`)
//!
//! [`Multivector::norm_squared`] is `⟨M * ~M⟩_0`. For *versors* (unit
//! rotors, vectors, products of vectors) this is the squared magnitude.
//! For arbitrary multivectors it's still a well-defined scalar but
//! isn't necessarily positive and isn't a norm in any geometric sense.

use crate::multivector::Multivector;
use crate::scalar::{Real, Scalar};
use crate::signature::grade_of;

impl<T: Scalar, const P: usize, const Q: usize, const R: usize, const DIM: usize>
    Multivector<T, P, Q, R, DIM>
{
    /// Reverse `~M`. Reverses the order of generators in every blade,
    /// which flips the sign whenever the grade `k` satisfies
    /// `k(k − 1) / 2 ≡ 1 (mod 2)` — grades 2, 3, 6, 7, ...
    pub fn reverse(&self) -> Self {
        let mut out = *self;
        for i in 0..DIM {
            if (grade_of(i) / 2) & 1 == 1 {
                out.coeffs[i] = -out.coeffs[i];
            }
        }
        out
    }

    /// Grade involution `M̂`. Flips the sign of every odd-grade blade.
    /// Equivalent to substituting `−e_k` for every generator `e_k`.
    pub fn grade_involution(&self) -> Self {
        let mut out = *self;
        for i in 0..DIM {
            if grade_of(i) & 1 == 1 {
                out.coeffs[i] = -out.coeffs[i];
            }
        }
        out
    }

    /// Clifford conjugation `M̄ = reverse(grade_involution(M))`.
    /// Flips grades 1, 2, 5, 6, ... — i.e. when `k(k + 1) / 2` is odd.
    pub fn conjugate(&self) -> Self {
        let mut out = *self;
        for i in 0..DIM {
            // The sign exponent is k(k+1)/2 mod 2; (k+1)/2 has the same
            // parity, so we test that. Written as explicit arithmetic
            // rather than div_ceil to mirror the textbook formula.
            #[allow(clippy::manual_div_ceil)]
            let flip = ((grade_of(i) + 1) / 2) & 1 == 1;
            if flip {
                out.coeffs[i] = -out.coeffs[i];
            }
        }
        out
    }

    /// `⟨M * ~M⟩_0`. For versors (vectors, rotors, and their products)
    /// this is the squared magnitude. For other multivectors it's still
    /// a well-defined scalar but its geometric meaning is murkier.
    pub fn norm_squared(&self) -> T {
        self.scalar_product(&self.reverse())
    }
}

impl<T: Real, const P: usize, const Q: usize, const R: usize, const DIM: usize>
    Multivector<T, P, Q, R, DIM>
{
    /// The magnitude `|M| = √|⟨M ~M⟩_0|`.
    ///
    /// The absolute value is taken before the square root: in mixed
    /// signatures (`Q > 0`) `norm_squared` can come out negative, and
    /// `|M|` should still be the real, non-negative length. In a
    /// Euclidean algebra it is just `√(norm_squared)`.
    pub fn norm(&self) -> T {
        self.norm_squared().abs().sqrt()
    }

    /// `M` scaled to unit magnitude, `M / |M|`.
    ///
    /// Intended for elements with non-zero norm — unit vectors, rotors,
    /// and (crucially) the line bivector you feed to a rotation. If the
    /// norm is zero this divides by zero and yields non-finite
    /// coefficients, so guard with [`Multivector::norm`] when the input
    /// might be null.
    pub fn normalized(&self) -> Self {
        *self * (T::ONE / self.norm())
    }
}

#[cfg(test)]
mod tests {
    use crate::{Vga2, Vga3};

    #[test]
    fn reverse_pattern_scalar_vector_bivector_trivector() {
        // m = 1 + 2 e1 + 3 e2 + 5 e3 + 7 e12 + 11 e13 + 13 e23 + 17 e123
        let m = Vga3 {
            coeffs: [1.0, 2.0, 3.0, 7.0, 5.0, 11.0, 13.0, 17.0],
        };
        let r = m.reverse();
        // Grades 0, 1 unchanged; grades 2, 3 flipped.
        assert_eq!(r.coeffs, [1.0, 2.0, 3.0, -7.0, 5.0, -11.0, -13.0, -17.0]);
    }

    #[test]
    fn grade_involution_flips_odd_grades() {
        let m = Vga3 {
            coeffs: [1.0, 2.0, 3.0, 7.0, 5.0, 11.0, 13.0, 17.0],
        };
        let g = m.grade_involution();
        // Flip indices with odd popcount: 1, 2, 4, 7.
        assert_eq!(g.coeffs, [1.0, -2.0, -3.0, 7.0, -5.0, 11.0, 13.0, -17.0]);
    }

    #[test]
    fn conjugation_equals_reverse_of_grade_involution() {
        let m = Vga3 {
            coeffs: [1.0, 2.0, 3.0, 7.0, 5.0, 11.0, 13.0, 17.0],
        };
        assert_eq!(m.conjugate().coeffs, m.grade_involution().reverse().coeffs);
        // And `reverse ∘ grade_involution == grade_involution ∘ reverse`.
        assert_eq!(m.conjugate().coeffs, m.reverse().grade_involution().coeffs);
    }

    #[test]
    fn double_reverse_is_identity() {
        let m = Vga3 {
            coeffs: [1.0, 2.0, 3.0, 7.0, 5.0, 11.0, 13.0, 17.0],
        };
        assert_eq!(m.reverse().reverse().coeffs, m.coeffs);
    }

    #[test]
    fn norm_squared_of_euclidean_vector_is_squared_length() {
        // v = 3 e1 + 4 e2 in Vga2 ⇒ |v|² = 25
        let v = Vga2 {
            coeffs: [0.0, 3.0, 4.0, 0.0],
        };
        assert_eq!(v.norm_squared(), 25.0);
    }

    #[test]
    fn norm_is_the_euclidean_length_of_a_vector() {
        // v = 3 e1 + 4 e2 ⇒ |v| = 5.
        let v = Vga2 {
            coeffs: [0.0, 3.0, 4.0, 0.0],
        };
        assert!((v.norm() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn norm_is_nonnegative_in_mixed_signature() {
        // In Cl(0,1,0) the lone vector squares to -1, so norm_squared is
        // negative — but the norm itself must be the real length 2.
        use crate::Multivector;
        type Anti = Multivector<f64, 0, 1, 0, 2>;
        let v = Anti { coeffs: [0.0, 2.0] };
        assert_eq!(v.norm_squared(), -4.0);
        assert!((v.norm() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn normalized_has_unit_norm() {
        let v = Vga3 {
            coeffs: [0.0, 1.0, 2.0, 0.0, -2.0, 0.0, 0.0, 0.0],
        };
        let u = v.normalized();
        assert!((u.norm() - 1.0).abs() < 1e-12);
        // Direction is preserved: u is a positive multiple of v.
        let k = u.coeffs[1] / v.coeffs[1];
        for i in 0..8 {
            assert!((u.coeffs[i] - k * v.coeffs[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn norm_squared_of_unit_rotor_is_one() {
        // R = cos(θ/2) - sin(θ/2) e12  is a unit Vga2 rotor for any θ.
        let theta: f64 = 0.7;
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let r = Vga2 {
            coeffs: [c, 0.0, 0.0, -s],
        };
        let n2 = r.norm_squared();
        assert!((n2 - 1.0).abs() < 1e-12, "expected 1.0, got {n2}");
    }
}
