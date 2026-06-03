//! 3D Conformal Geometric Algebra constructors, for `Cga3 = Cl(4, 1, 0)`.
//!
//! Where PGA (`Cl(3,0,1)`) linearizes *rigid* motions, CGA linearizes
//! the whole **conformal** group — rotations, translations, *and*
//! uniform scalings and inversions — by embedding Euclidean 3-space as a
//! null cone in a 5D space of signature `(+, +, +, +, −)`. The payoff:
//! points, planes, **spheres, and circles** all become blades, and the
//! same meet/join machinery does real incidence geometry on round
//! objects, not just flat ones.
//!
//! # The two extra null directions
//!
//! On top of the Euclidean `e1, e2, e3` (each squaring to `+1`) the
//! model adds two more generators:
//!
//! - `e+ = e4 = basis(8)` squares to `+1`
//! - `e− = e5 = basis(16)` squares to `−1`
//!
//! From them we build the **null basis** that gives CGA its geometry:
//!
//! ```text
//! no   = (e− − e+) / 2      the point at the origin
//! n∞   = e− + e+            the point at infinity
//! ```
//!
//! Both are null (`no² = n∞² = 0`) and they pair as `no · n∞ = −1`.
//! Every Euclidean point `(x,y,z)` lifts to a *null vector*
//!
//! ```text
//! P = no + x·e1 + y·e2 + z·e3 + ½(x²+y²+z²)·n∞
//! ```
//!
//! and the inner product reads off Euclidean distance:
//! `P · Q = −½ |p − q|²`. A point on the null cone has `P · P = 0`.

use crate::multivector::Multivector;
use crate::scalar::Scalar;
use crate::Cga3Sig;

/// CGA constructors are defined only for the `Cl(4, 1, 0)` signature, so
/// this `impl` is pinned to the `Cga3Sig` marker. It covers both `Cga3`
/// (`f64`) and `Cga3f` (`f32`).
impl<T: Scalar> Multivector<Cga3Sig, T> {
    /// The point at infinity `n∞ = e+ + e−`. A null vector
    /// (`n∞² = 0`) fixed by every Euclidean motion.
    pub fn n_infinity() -> Self {
        Self::basis(8) + Self::basis(16)
    }

    /// The origin point `no = (e− − e+) / 2`. The null vector that
    /// [`Multivector::cga_point`] reduces to at `(0, 0, 0)`.
    pub fn n_origin() -> Self {
        (Self::basis(16) - Self::basis(8)) * T::from_f64(0.5)
    }

    /// A Euclidean point `(x, y, z)` lifted to its null vector in the
    /// conformal model:
    ///
    /// ```text
    /// P = no + x·e1 + y·e2 + z·e3 + ½(x²+y²+z²)·n∞
    /// ```
    ///
    /// The result is a grade-1 null vector: `P · P == 0`.
    pub fn cga_point(x: T, y: T, z: T) -> Self {
        let e1 = Self::basis(1);
        let e2 = Self::basis(2);
        let e3 = Self::basis(4);
        let r2 = x * x + y * y + z * z;
        Self::n_origin() + e1 * x + e2 * y + e3 * z + Self::n_infinity() * (r2 * T::from_f64(0.5))
    }

    /// A sphere with centre `(cx, cy, cz)` and radius `r`, as a grade-1
    /// vector:
    ///
    /// ```text
    /// S = cga_point(c) − ½ r²·n∞
    /// ```
    ///
    /// A Euclidean point `P` lies on the sphere exactly when
    /// `P · S == 0`. Taking `r = 0` recovers the point itself, and a
    /// *negative* `r²` describes an imaginary sphere — both fall out of
    /// the same formula.
    pub fn sphere(cx: T, cy: T, cz: T, r: T) -> Self {
        Self::cga_point(cx, cy, cz) - Self::n_infinity() * (r * r * T::from_f64(0.5))
    }

    /// The plane `a·x + b·y + c·z = d` with normal `(a, b, c)`, as a
    /// grade-1 vector:
    ///
    /// ```text
    /// π = a·e1 + b·e2 + c·e3 + d·n∞
    /// ```
    ///
    /// A plane is a sphere through the point at infinity — same grade as
    /// [`Multivector::sphere`], with `P · π == 0` the on-plane test.
    pub fn cga_plane(a: T, b: T, c: T, d: T) -> Self {
        let e1 = Self::basis(1);
        let e2 = Self::basis(2);
        let e3 = Self::basis(4);
        e1 * a + e2 * b + e3 * c + Self::n_infinity() * d
    }

    /// Read the Euclidean coordinates back out of a conformal point.
    ///
    /// A point may carry an arbitrary positive *weight* `w` (a conformal
    /// dilation, for instance, rescales it), so we divide through by
    /// `w = −P · n∞` before reading the `e1, e2, e3` coordinates. The
    /// inverse of [`Multivector::cga_point`] up to that homogeneity.
    pub fn to_euclidean(&self) -> (T, T, T) {
        let w = -self.scalar_product(&Self::n_infinity());
        (self.coeffs[1] / w, self.coeffs[2] / w, self.coeffs[4] / w)
    }
}

#[cfg(test)]
mod tests {
    use crate::Cga3;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-10, "{a} vs {b}");
    }

    // --- Null structure -------------------------------------------------

    #[test]
    fn the_two_null_vectors_square_to_zero() {
        approx(Cga3::n_infinity().norm_squared(), 0.0);
        approx(Cga3::n_origin().norm_squared(), 0.0);
    }

    #[test]
    fn origin_and_infinity_pair_to_minus_one() {
        // no · n∞ = −1, the defining normalization of the null basis.
        let no = Cga3::n_origin();
        let ninf = Cga3::n_infinity();
        approx(no.scalar_product(&ninf), -1.0);
    }

    #[test]
    fn a_point_is_a_null_vector() {
        // P · P = 0 for every conformal point.
        let p = Cga3::cga_point(1.0, -2.0, 3.0);
        approx(p.norm_squared(), 0.0);
    }

    #[test]
    fn point_at_origin_is_n_origin() {
        approx_eq_coeffs(Cga3::cga_point(0.0, 0.0, 0.0), Cga3::n_origin());
    }

    // --- Distance -------------------------------------------------------

    #[test]
    fn inner_product_of_points_is_minus_half_distance_squared() {
        // P · Q = −½|p − q|².  p=(1,0,0), q=(4,4,0) ⇒ |p−q|²=25.
        let p = Cga3::cga_point(1.0, 0.0, 0.0);
        let q = Cga3::cga_point(4.0, 4.0, 0.0);
        approx(p.scalar_product(&q), -12.5);
    }

    // --- Sphere ---------------------------------------------------------

    #[test]
    fn points_on_a_sphere_have_zero_inner_product_with_it() {
        // Unit sphere at the origin: (1,0,0) and (0,0,1) lie on it,
        // (2,0,0) does not.
        let s = Cga3::sphere(0.0, 0.0, 0.0, 1.0);
        approx(Cga3::cga_point(1.0, 0.0, 0.0).scalar_product(&s), 0.0);
        approx(Cga3::cga_point(0.0, 0.0, 1.0).scalar_product(&s), 0.0);
        assert!(Cga3::cga_point(2.0, 0.0, 0.0).scalar_product(&s).abs() > 1e-6);
    }

    #[test]
    fn sphere_off_origin_contains_the_right_points() {
        // Sphere radius 2 centred at (1,2,3): (3,2,3) is on it.
        let s = Cga3::sphere(1.0, 2.0, 3.0, 2.0);
        approx(Cga3::cga_point(3.0, 2.0, 3.0).scalar_product(&s), 0.0);
    }

    #[test]
    fn a_zero_radius_sphere_is_a_point() {
        approx_eq_coeffs(
            Cga3::sphere(1.0, 2.0, 3.0, 0.0),
            Cga3::cga_point(1.0, 2.0, 3.0),
        );
    }

    // --- Plane ----------------------------------------------------------

    #[test]
    fn points_on_a_plane_have_zero_inner_product_with_it() {
        // Plane x = 1 (normal e1, offset 1). (1,5,-2) on it; (2,0,0) off.
        let pi = Cga3::cga_plane(1.0, 0.0, 0.0, 1.0);
        approx(Cga3::cga_point(1.0, 5.0, -2.0).scalar_product(&pi), 0.0);
        assert!(Cga3::cga_point(2.0, 0.0, 0.0).scalar_product(&pi).abs() > 1e-6);
    }

    fn approx_eq_coeffs(a: Cga3, b: Cga3) {
        for (i, (&x, &y)) in a.coeffs.iter().zip(b.coeffs.iter()).enumerate() {
            assert!((x - y).abs() < 1e-10, "coeff {i}: {x} vs {y}");
        }
    }
}
