//! The [`Motor`] — a rigid-body motion in 3D PGA `Cl(3, 0, 1)`.
//!
//! A motor is a *versor* living in the even subalgebra of PGA (grades
//! 0, 2, 4). Geometrically it is a screw motion: the composition of a
//! rotation about a line and a translation along it, which between them
//! cover every rigid motion of Euclidean 3-space. Rotors and translators
//! are the two pure special cases.
//!
//! [`Motor`] is a thin newtype over a `Pga3` multivector. Its whole job
//! is to give rigid motions a name and a small, total API — build them,
//! compose them with `*`, and apply them to points/lines/planes — while
//! the heavy lifting stays in the underlying [`Multivector`]:
//!
//! ```text
//! translator ─┐
//!             ├─ compose (·) ─▶ motor ─ apply (sandwich) ─▶ moved object
//! rotor ──────┘
//! ```

use core::ops::Mul;

use crate::multivector::Multivector;
use crate::scalar::{Real, Scalar};

/// The PGA multivector type a [`Motor`] wraps: `Cl(3, 0, 1)` over `T`.
type Pga<T> = Multivector<T, 3, 0, 1, 16>;

/// A rigid-body motion in 3D PGA — an even-grade versor of `Cl(3, 0, 1)`.
///
/// Construct one with [`Motor::identity`], [`Motor::translator`], or
/// [`Motor::rotor`]; compose with `*` (or [`Motor::compose`]); and move
/// geometry with [`Motor::apply`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Motor<T> {
    versor: Pga<T>,
}

impl<T: Scalar> Motor<T> {
    /// The identity motion — leaves every object exactly where it is.
    pub fn identity() -> Self {
        Self { versor: Pga::<T>::one() }
    }

    /// Wrap a raw PGA multivector as a motor, unchecked.
    ///
    /// The caller promises `versor` is an even-grade versor (a product
    /// of rotors and translators). Used internally and for advanced
    /// constructions; prefer the named constructors otherwise.
    pub fn from_versor(versor: Pga<T>) -> Self {
        Self { versor }
    }

    /// The underlying PGA multivector.
    pub fn versor(&self) -> Pga<T> {
        self.versor
    }

    /// Apply the motion to a PGA object (point, line, plane, …) via the
    /// sandwich product `M x ~M`.
    pub fn apply(&self, x: &Pga<T>) -> Pga<T> {
        self.versor.sandwich(x)
    }

    /// Compose two motions: `self.compose(&rhs)` does `rhs` first, then
    /// `self`, exactly like function composition. Equals `self * rhs`.
    pub fn compose(&self, rhs: &Self) -> Self {
        Self { versor: self.versor * rhs.versor }
    }

    /// The inverse motion, undoing this one. Built from the versor
    /// inverse, so `m.compose(&m.inverse())` is the identity.
    pub fn inverse(&self) -> Self {
        Self { versor: self.versor.versor_inverse() }
    }

    /// `⟨M ~M⟩_0`. A unit motor (every rotor/translator and their
    /// products) has `norm_squared() == 1`.
    pub fn norm_squared(&self) -> T {
        self.versor.norm_squared()
    }
}

impl<T: Real> Motor<T> {
    /// A pure translation by `(dx, dy, dz)`.
    ///
    /// Built as `exp(−½(dx·e0e1 + dy·e0e2 + dz·e0e3))`. The generating
    /// bivector is null (it squares to zero), so the exponential series
    /// truncates after one term — translations are "linear" in PGA.
    pub fn translator(dx: T, dy: T, dz: T) -> Self {
        let e0 = Pga::<T>::basis(8);
        let b = (e0 * Pga::<T>::basis(1)) * dx
            + (e0 * Pga::<T>::basis(2)) * dy
            + (e0 * Pga::<T>::basis(4)) * dz;
        let versor = (b * T::from_f64(-0.5)).exp();
        Self { versor }
    }

    /// A rotation by `radians` in the plane of the unit Euclidean
    /// bivector `plane` (about the line through the origin orthogonal to
    /// it).
    ///
    /// `plane` should be a unit bivector with `plane² = −1`, e.g.
    /// `Pga3::basis(0b0110)` (`e23`) to rotate about the x-axis,
    /// `e13` about y, `e12` about z. Built as `exp(−½·radians·plane)`.
    pub fn rotor(radians: T, plane: Pga<T>) -> Self {
        let versor = (plane * (radians * T::from_f64(-0.5))).exp();
        Self { versor }
    }
}

/// Composition of motions. `a * b` applies `b` first, then `a`.
impl<T: Scalar> Mul for Motor<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.compose(&rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::Motor;
    use crate::Pga3;
    use std::f64::consts::FRAC_PI_2;

    fn approx_eq(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tol, "index {i}: {x} vs {y}");
        }
    }

    #[test]
    fn identity_leaves_points_untouched() {
        let p = Pga3::point(1.0, 2.0, 3.0);
        approx_eq(&Motor::identity().apply(&p).coeffs, &p.coeffs, 1e-12);
    }

    #[test]
    fn translator_moves_a_point() {
        let t = Motor::translator(3.0, -1.0, 2.0);
        let moved = t.apply(&Pga3::point(0.0, 0.0, 0.0)).cleaned(1e-10);
        approx_eq(&moved.coeffs, &Pga3::point(3.0, -1.0, 2.0).coeffs, 1e-12);
    }

    #[test]
    fn rotor_about_x_axis_sends_y_to_z() {
        // 90° about the x-axis (e23 plane): (0,1,0) → (0,0,1).
        let r = Motor::rotor(FRAC_PI_2, Pga3::basis(0b0110));
        let moved = r.apply(&Pga3::point(0.0, 1.0, 0.0)).cleaned(1e-10);
        approx_eq(&moved.coeffs, &Pga3::point(0.0, 0.0, 1.0).coeffs, 1e-12);
    }

    #[test]
    fn motor_is_even_grade() {
        let m = Motor::translator(1.0, 2.0, 3.0) * Motor::rotor(0.7, Pga3::basis(0b0110));
        let v = m.versor();
        approx_eq(&v.grade(1).coeffs, &Pga3::zero().coeffs, 1e-12);
        approx_eq(&v.grade(3).coeffs, &Pga3::zero().coeffs, 1e-12);
    }

    #[test]
    fn motors_are_unit_norm() {
        let m = Motor::translator(5.0, -2.0, 0.5) * Motor::rotor(1.23, Pga3::basis(0b0101));
        assert!((m.norm_squared() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn compose_then_inverse_is_identity() {
        let m = Motor::translator(2.0, 3.0, -1.0) * Motor::rotor(0.9, Pga3::basis(0b0011));
        let round = m.compose(&m.inverse());
        // The round trip is the identity versor (the scalar 1).
        approx_eq(&round.versor().cleaned(1e-10).coeffs, &Pga3::one().coeffs, 1e-10);
    }

    #[test]
    fn screw_motion_matches_rotate_then_translate() {
        // Rotate (0,1,0) 90° about x → (0,0,1), then translate +3 in x
        // ⇒ (3,0,1). Order: M = T · R applies R first.
        let r = Motor::rotor(FRAC_PI_2, Pga3::basis(0b0110));
        let t = Motor::translator(3.0, 0.0, 0.0);
        let m = t * r;
        let moved = m.apply(&Pga3::point(0.0, 1.0, 0.0)).cleaned(1e-10);
        approx_eq(&moved.coeffs, &Pga3::point(3.0, 0.0, 1.0).coeffs, 1e-12);
    }

    #[test]
    fn composition_order_matters() {
        // Translate-then-rotate ≠ rotate-then-translate in general.
        // (The translation must not be along the rotation axis, or the
        // two would commute — here we rotate about x and translate in y.)
        let r = Motor::rotor(FRAC_PI_2, Pga3::basis(0b0110)); // about x-axis
        let t = Motor::translator(0.0, 3.0, 0.0); // along y
        let tr = (t * r).apply(&Pga3::point(0.0, 1.0, 0.0)).cleaned(1e-10);
        let rt = (r * t).apply(&Pga3::point(0.0, 1.0, 0.0)).cleaned(1e-10);
        assert_ne!(tr.coeffs, rt.coeffs);
        // Spot-check the values: R-then-T gives (0,3,1); T-then-R gives (0,0,4).
        approx_eq(&tr.coeffs, &Pga3::point(0.0, 3.0, 1.0).coeffs, 1e-12);
        approx_eq(&rt.coeffs, &Pga3::point(0.0, 0.0, 4.0).coeffs, 1e-12);
    }
}
