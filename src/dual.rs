//! Duality: the unit pseudoscalar, the metric-independent complements,
//! and the regressive product (the *meet*) they generate.
//!
//! Where the wedge `∧` *joins* subspaces (the line through two points,
//! the plane through three), the regressive product `∨` *meets* them
//! (the point where two lines cross, the line where two planes
//! intersect). The two are dual notions, bridged by a complement that
//! swaps each blade for the blade spanning the leftover generators.
//!
//! # Why a *combinatorial* complement, not a metric dual
//!
//! The textbook dual is `M ↦ M * I⁻¹` — multiply by the inverse
//! pseudoscalar. That is fine in a non-degenerate algebra, but it falls
//! apart precisely where GA earns its keep: in PGA `Cl(3, 0, 1)` the
//! pseudoscalar is **null** (`I² = 0`), so `I⁻¹` doesn't exist and the
//! metric dual is undefined.
//!
//! The complements here sidestep the metric entirely. For a unit blade
//! `e_S` (a subset `S` of the generators) the complement is the unit
//! blade on the *complementary* subset `S̄`, with the unique sign that
//! makes the defining wedge equal the pseudoscalar `I`:
//!
//! ```text
//! right complement:  e_S ∧ rc(e_S) = I
//! left  complement:  lc(e_S) ∧ e_S = I
//! ```
//!
//! No generator is ever squared, so `(P, Q, R)` never enters — the
//! complements, and the regressive product built from them, work
//! identically in Euclidean, projective, conformal, and spacetime
//! algebras. `lc` and `rc` are mutual inverses (`lc(rc(M)) = M`).

use crate::multivector::Multivector;
use crate::scalar::Scalar;
use crate::signature::swap_sign;

impl<T: Scalar, const P: usize, const Q: usize, const R: usize, const DIM: usize>
    Multivector<T, P, Q, R, DIM>
{
    /// The unit pseudoscalar `I = e1 ∧ e2 ∧ … ∧ eN` — the single
    /// top-grade blade, living at index `DIM - 1` (all bits set).
    ///
    /// `I` is the identity element of the regressive product
    /// ([`Multivector::regressive`]), mirroring the way the scalar `1`
    /// is the identity of the geometric product.
    pub fn pseudoscalar() -> Self {
        Self::basis(DIM - 1)
    }

    /// Right complement, defined blade-by-blade by `e_S ∧ rc(e_S) = I`.
    ///
    /// Metric-independent: it relabels each blade to its complementary
    /// blade with a reordering sign, never squaring a generator, so it
    /// is well-defined even in degenerate algebras like PGA where the
    /// metric dual is not. Inverse of [`Multivector::left_complement`].
    pub fn right_complement(&self) -> Self {
        let mut out = Self::zero();
        let top = DIM - 1;
        for i in 0..DIM {
            let c = self.coeffs[i];
            if c == T::ZERO {
                continue;
            }
            let comp = top ^ i;
            if swap_sign(i, comp) > 0 {
                out.coeffs[comp] += c;
            } else {
                out.coeffs[comp] -= c;
            }
        }
        out
    }

    /// Left complement, defined blade-by-blade by `lc(e_S) ∧ e_S = I`.
    ///
    /// The inverse of [`Multivector::right_complement`]: applying one
    /// then the other returns the original multivector. The two differ
    /// only by a grade-dependent sign and coincide whenever every blade
    /// present has a self-reversing grade.
    pub fn left_complement(&self) -> Self {
        let mut out = Self::zero();
        let top = DIM - 1;
        for i in 0..DIM {
            let c = self.coeffs[i];
            if c == T::ZERO {
                continue;
            }
            let comp = top ^ i;
            if swap_sign(comp, i) > 0 {
                out.coeffs[comp] += c;
            } else {
                out.coeffs[comp] -= c;
            }
        }
        out
    }

    /// Regressive product `self ∨ rhs` — the *meet* of two subspaces.
    ///
    /// Built from the wedge by De Morgan duality:
    ///
    /// ```text
    /// a ∨ b = lc( rc(a) ∧ rc(b) )
    /// ```
    ///
    /// Geometrically it intersects: in 3D Euclidean GA the meet of the
    /// `xy`-plane (`e12`) and the `xz`-plane (`e13`) is the `x`-axis
    /// (`e1`); in PGA the meet of two planes is their common line. Like
    /// the complements it is metric-independent, so it behaves the same
    /// across every signature.
    ///
    /// The pseudoscalar `I` is its identity: `I ∨ M = M ∨ I = M`.
    pub fn regressive(&self, rhs: &Self) -> Self {
        self.right_complement()
            .wedge(&rhs.right_complement())
            .left_complement()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Pga3, Vga2, Vga3};

    // --- Pseudoscalar ---------------------------------------------------

    #[test]
    fn pseudoscalar_is_the_top_blade() {
        // Vga3: I = e123 at index 0b111 = 7.
        assert_eq!(Vga3::pseudoscalar().coeffs, Vga3::basis(7).coeffs);
        // Vga2: I = e12 at index 0b11 = 3.
        assert_eq!(Vga2::pseudoscalar().coeffs, Vga2::basis(3).coeffs);
    }

    // --- Complements: defining property ---------------------------------

    #[test]
    fn blade_wedge_its_right_complement_is_the_pseudoscalar() {
        // e_S ∧ rc(e_S) = I for every basis blade, by construction.
        for i in 0..8 {
            let e = Vga3::basis(i);
            let got = e.wedge(&e.right_complement());
            assert_eq!(
                got.coeffs,
                Vga3::pseudoscalar().coeffs,
                "blade index {i}"
            );
        }
    }

    #[test]
    fn left_complement_wedge_blade_is_the_pseudoscalar() {
        // lc(e_S) ∧ e_S = I for every basis blade.
        for i in 0..8 {
            let e = Vga3::basis(i);
            let got = e.left_complement().wedge(&e);
            assert_eq!(
                got.coeffs,
                Vga3::pseudoscalar().coeffs,
                "blade index {i}"
            );
        }
    }

    #[test]
    fn right_complement_of_vector_is_perpendicular_plane() {
        // In Euclidean 3D the complement is the Hodge dual: rc(e1) = e23.
        assert_eq!(Vga3::basis(1).right_complement().coeffs, Vga3::basis(6).coeffs);
        // rc(e2) = e31 = -e13 ; e13 is index 0b101 = 5.
        assert_eq!(
            Vga3::basis(2).right_complement().coeffs,
            (-Vga3::basis(5)).coeffs
        );
    }

    #[test]
    fn left_and_right_complements_are_inverse() {
        // lc(rc(M)) == M and rc(lc(M)) == M on a dense multivector.
        let m = Vga3 { coeffs: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] };
        assert_eq!(m.right_complement().left_complement().coeffs, m.coeffs);
        assert_eq!(m.left_complement().right_complement().coeffs, m.coeffs);
    }

    // --- Regressive product (meet) --------------------------------------

    #[test]
    fn meet_of_two_planes_is_their_common_axis() {
        // xy-plane ∨ xz-plane = x-axis :  e12 ∨ e13 = e1.
        let e12 = Vga3::basis(3);
        let e13 = Vga3::basis(5);
        assert_eq!(e12.regressive(&e13).coeffs, Vga3::basis(1).coeffs);
    }

    #[test]
    fn meet_of_xy_and_yz_planes_is_the_y_axis() {
        // e12 ∨ e23 = e2.
        let e12 = Vga3::basis(3);
        let e23 = Vga3::basis(6);
        assert_eq!(e12.regressive(&e23).coeffs, Vga3::basis(2).coeffs);
    }

    #[test]
    fn pseudoscalar_is_the_regressive_identity() {
        // I ∨ M == M ∨ I == M for an arbitrary multivector.
        let m = Vga3 { coeffs: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] };
        let i = Vga3::pseudoscalar();
        assert_eq!(i.regressive(&m).coeffs, m.coeffs);
        assert_eq!(m.regressive(&i).coeffs, m.coeffs);
    }

    #[test]
    fn duality_is_well_defined_in_degenerate_pga() {
        // PGA Cl(3,0,1) has a *null* pseudoscalar (I² = 0), so the
        // metric dual M·I⁻¹ is undefined — yet the combinatorial
        // complements and the regressive identity still hold exactly.
        let m = Pga3 { coeffs: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0] };
        // Pseudoscalar is null here — confirm the metric dual would fail.
        let i = Pga3::pseudoscalar();
        assert_eq!((i * i).cleaned(1e-12).coeffs, Pga3::zero().coeffs);
        // Complements still round-trip, and I is the regressive identity.
        assert_eq!(m.right_complement().left_complement().coeffs, m.coeffs);
        assert_eq!(i.regressive(&m).coeffs, m.coeffs);
        assert_eq!(m.regressive(&i).coeffs, m.coeffs);
    }
}
