//! The [`Conformal`] versor — a conformal transformation in 3D CGA
//! `Cl(4, 1, 0)`.
//!
//! Where a PGA [`Motor`](crate::Motor) covers the *rigid* motions
//! (rotation + translation), a conformal versor covers the strictly
//! larger **conformal group**: rigid motions *plus* uniform scaling
//! (dilation) about the origin. Geometrically these are exactly the
//! angle-preserving maps of Euclidean space.
//!
//! [`Conformal`] is a thin newtype over a `Cga3` multivector, with the
//! same small, total API as `Motor` — build the generators, compose them
//! with `*`, and apply them to any conformal object (point, sphere,
//! plane, …) by the sandwich product:
//!
//! ```text
//! translator ─┐
//! rotor ──────┼─ compose (·) ─▶ versor ─ apply (sandwich) ─▶ moved object
//! dilator ────┘
//! ```
//!
//! Each generator is a unit versor (`V ~V = 1`), so the reverse-based
//! sandwich is the exact conjugation `V x V⁻¹`. A dilation rescales a
//! point's homogeneous *weight*, so read coordinates back with
//! [`Multivector::to_euclidean`](crate::Multivector::to_euclidean),
//! which divides that weight out.

use core::ops::Mul;

use crate::multivector::Multivector;
use crate::scalar::{Real, Scalar};

/// The PGA-sibling multivector type a [`Conformal`] wraps: `Cl(4, 1, 0)`
/// over `T`.
type Cga<T> = Multivector<T, 4, 1, 0, 32>;

/// A conformal transformation in 3D CGA — a versor of `Cl(4, 1, 0)`.
///
/// Construct one with [`Conformal::identity`], [`Conformal::translator`],
/// [`Conformal::rotor`], or [`Conformal::dilator`]; compose with `*` (or
/// [`Conformal::compose`]); and move geometry with [`Conformal::apply`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Conformal<T> {
    versor: Cga<T>,
}

impl<T: Scalar> Conformal<T> {
    /// The identity transformation — leaves every object where it is.
    pub fn identity() -> Self {
        Self {
            versor: Cga::<T>::one(),
        }
    }

    /// Wrap a raw CGA multivector as a conformal versor, unchecked.
    ///
    /// The caller promises `versor` is a genuine conformal versor (a
    /// product of translators, rotors, and dilators). Prefer the named
    /// constructors otherwise.
    pub fn from_versor(versor: Cga<T>) -> Self {
        Self { versor }
    }

    /// The underlying CGA multivector.
    pub fn versor(&self) -> Cga<T> {
        self.versor
    }

    /// Apply the transformation to a CGA object (point, sphere, plane, …)
    /// via the sandwich product `V x ~V`.
    ///
    /// A dilation rescales a point's weight; recover Euclidean
    /// coordinates with
    /// [`Multivector::to_euclidean`](crate::Multivector::to_euclidean).
    pub fn apply(&self, x: &Cga<T>) -> Cga<T> {
        self.versor.sandwich(x)
    }

    /// Compose two transformations: `self.compose(&rhs)` does `rhs`
    /// first, then `self`, like function composition. Equals `self * rhs`.
    pub fn compose(&self, rhs: &Self) -> Self {
        Self {
            versor: self.versor * rhs.versor,
        }
    }

    /// The inverse transformation, undoing this one. Built from the
    /// versor inverse, so `c.compose(&c.inverse())` is the identity.
    pub fn inverse(&self) -> Self {
        Self {
            versor: self.versor.versor_inverse(),
        }
    }

    /// `⟨V ~V⟩_0`. Each generator and their products are unit versors,
    /// so this is `1`.
    pub fn norm_squared(&self) -> T {
        self.versor.norm_squared()
    }
}

impl<T: Real> Conformal<T> {
    /// A pure translation by `(dx, dy, dz)`.
    ///
    /// Built as `exp(½(dx·n∞e1 + dy·n∞e2 + dz·n∞e3))`. As in PGA the
    /// generating bivector is null (it squares to zero, since `n∞² = 0`),
    /// so the exponential truncates after one term. (The sign is `+½`
    /// rather than PGA's `−½` because we write `n∞` *before* the
    /// Euclidean generator, the opposite order to PGA's `e0`-first
    /// convention.)
    pub fn translator(dx: T, dy: T, dz: T) -> Self {
        let ninf = Cga::<T>::n_infinity();
        let b = (ninf * Cga::<T>::basis(1)) * dx
            + (ninf * Cga::<T>::basis(2)) * dy
            + (ninf * Cga::<T>::basis(4)) * dz;
        let versor = (b * T::from_f64(0.5)).exp();
        Self { versor }
    }

    /// A rotation by `radians` in the plane of the unit Euclidean
    /// bivector `plane`, about the line through the origin orthogonal to
    /// it.
    ///
    /// Identical in form to a PGA/VGA rotor — `plane` is built from the
    /// Euclidean generators only (`e23`, `e13`, `e12`), so the origin and
    /// the point at infinity are both fixed. Built as
    /// `exp(−½·radians·plane)`.
    pub fn rotor(radians: T, plane: Cga<T>) -> Self {
        let versor = (plane * (radians * T::from_f64(-0.5))).exp();
        Self { versor }
    }

    /// A uniform scaling about the origin by `factor` (which must be
    /// positive).
    ///
    /// Built as `exp(½·ln(factor)·(no ∧ n∞))`. The bivector `no ∧ n∞`
    /// squares to `+1`, so this is a *hyperbolic* rotor in the
    /// origin–infinity plane; conjugating a point by it multiplies its
    /// Euclidean coordinates by `factor` (and its weight by `1/factor`).
    pub fn dilator(factor: T) -> Self {
        // E = no ∧ n∞ = e− ∧ e+ = e−·e+ (orthogonal ⇒ wedge = product).
        let e = Cga::<T>::basis(16) * Cga::<T>::basis(8);
        let versor = (e * (factor.ln() * T::from_f64(0.5))).exp();
        Self { versor }
    }
}

/// Composition of transformations. `a * b` applies `b` first, then `a`.
impl<T: Scalar> Mul for Conformal<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.compose(&rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::Conformal;
    use crate::Cga3;
    use std::f64::consts::FRAC_PI_2;

    fn approx(a: (f64, f64, f64), b: (f64, f64, f64)) {
        assert!((a.0 - b.0).abs() < 1e-10, "x: {} vs {}", a.0, b.0);
        assert!((a.1 - b.1).abs() < 1e-10, "y: {} vs {}", a.1, b.1);
        assert!((a.2 - b.2).abs() < 1e-10, "z: {} vs {}", a.2, b.2);
    }

    #[test]
    fn identity_leaves_points_untouched() {
        let p = Cga3::cga_point(1.0, 2.0, 3.0);
        approx(
            Conformal::identity().apply(&p).to_euclidean(),
            (1.0, 2.0, 3.0),
        );
    }

    #[test]
    fn translator_moves_a_point() {
        let t = Conformal::translator(3.0, -1.0, 2.0);
        let moved = t.apply(&Cga3::cga_point(1.0, 1.0, 1.0));
        approx(moved.to_euclidean(), (4.0, 0.0, 3.0));
    }

    #[test]
    fn translator_carries_the_origin_to_the_offset() {
        let t = Conformal::translator(5.0, 6.0, 7.0);
        let moved = t.apply(&Cga3::cga_point(0.0, 0.0, 0.0));
        approx(moved.to_euclidean(), (5.0, 6.0, 7.0));
    }

    #[test]
    fn rotor_about_x_axis_sends_y_to_z() {
        // 90° in the e23 plane: (0,1,0) → (0,0,1).
        let r = Conformal::rotor(FRAC_PI_2, Cga3::basis(0b0110));
        let moved = r.apply(&Cga3::cga_point(0.0, 1.0, 0.0));
        approx(moved.to_euclidean(), (0.0, 0.0, 1.0));
    }

    #[test]
    fn dilator_scales_coordinates_by_the_factor() {
        let d = Conformal::dilator(3.0);
        let moved = d.apply(&Cga3::cga_point(1.0, -2.0, 0.5));
        approx(moved.to_euclidean(), (3.0, -6.0, 1.5));
    }

    #[test]
    fn dilator_fixes_the_origin() {
        let d = Conformal::dilator(7.5);
        let moved = d.apply(&Cga3::cga_point(0.0, 0.0, 0.0));
        approx(moved.to_euclidean(), (0.0, 0.0, 0.0));
    }

    #[test]
    fn generators_are_unit_versors() {
        assert!((Conformal::<f64>::translator(2.0, 3.0, -1.0).norm_squared() - 1.0).abs() < 1e-12);
        assert!((Conformal::rotor(1.1, Cga3::basis(0b0110)).norm_squared() - 1.0).abs() < 1e-12);
        assert!((Conformal::<f64>::dilator(2.5).norm_squared() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn compose_then_inverse_is_identity() {
        let c = Conformal::translator(2.0, 0.0, -1.0)
            * Conformal::rotor(0.8, Cga3::basis(0b0110))
            * Conformal::dilator(2.0);
        let p = Cga3::cga_point(1.0, 2.0, 3.0);
        let round = c.inverse().apply(&c.apply(&p));
        approx(round.to_euclidean(), (1.0, 2.0, 3.0));
    }

    #[test]
    fn scale_then_translate_differs_from_translate_then_scale() {
        // S then T: (1,0,0) ×2 = (2,0,0), +(1,0,0) = (3,0,0).
        let s = Conformal::dilator(2.0);
        let t = Conformal::translator(1.0, 0.0, 0.0);
        let st = (t * s).apply(&Cga3::cga_point(1.0, 0.0, 0.0));
        approx(st.to_euclidean(), (3.0, 0.0, 0.0));
        // T then S: +(1,0,0) = (2,0,0), ×2 = (4,0,0).
        let ts = (s * t).apply(&Cga3::cga_point(1.0, 0.0, 0.0));
        approx(ts.to_euclidean(), (4.0, 0.0, 0.0));
    }

    #[test]
    fn dilation_preserves_a_sphere_centered_at_origin_rescaling_its_radius() {
        // Doubling about the origin turns the unit sphere into radius 2:
        // (2,0,0) lands on the image sphere, (1,0,0) no longer does.
        let s = Cga3::sphere(0.0, 0.0, 0.0, 1.0);
        let d = Conformal::dilator(2.0);
        let s2 = d.apply(&s);
        assert!(Cga3::cga_point(2.0, 0.0, 0.0).scalar_product(&s2).abs() < 1e-10);
        assert!(Cga3::cga_point(1.0, 0.0, 0.0).scalar_product(&s2).abs() > 1e-6);
    }
}
