//! Typed PGA geometry — [`Point`], [`Line`], and [`Plane`] in `Cl(3, 0, 1)`.
//!
//! The kernel represents all three as a bare
//! [`Multivector`] of the appropriate grade
//! (point = grade 3, line = grade 2, plane = grade 1). This module gives
//! each a name and a small, total, *type-checked* incidence API, so the
//! compiler tracks what a construction produces instead of leaving every
//! result an untyped multivector:
//!
//! - **join** (`∨`, the regressive product) builds *up* in dimension: two
//!   [`Point`]s join to the [`Line`] through them; a [`Line`] and a
//!   [`Point`] join to the [`Plane`] they span.
//! - **meet** (`∧`, the wedge) cuts *down*: two [`Plane`]s meet in their
//!   [`Line`] of intersection; a [`Line`] and a [`Plane`] meet in a
//!   [`Point`].
//!
//! ```text
//! Point ─join─▶ Line ─join─▶ Plane     (span / build up)
//! Plane ─meet─▶ Line ─meet─▶ Point     (intersect / cut down)
//! ```
//!
//! Both ladders read as left-to-right method chains — three planes meet at
//! a point with `a.meet(&b).meet(&c)`, three points span a plane with
//! `a.join(&b).join(&c)`. Rigid motions ([`Motor`]) preserve
//! grade, so [`Point::transform`] and its siblings map each type to
//! itself. Drop to the raw blade any time with [`Point::multivector`] /
//! [`Point::from_multivector`] (and the analogues on [`Line`]/[`Plane`]).
//!
//! ```
//! use garust_geo::pga::Plane;
//!
//! // The coordinate-offset planes x = 1, y = 2, z = 3 meet at (1, 2, 3).
//! // `Plane`'s scalar defaults to f64, so one annotation pins the rest.
//! let px: Plane = Plane::new(1.0, 0.0, 0.0, -1.0);
//! let py = Plane::new(0.0, 1.0, 0.0, -2.0);
//! let pz = Plane::new(0.0, 0.0, 1.0, -3.0);
//! let (x, y, z) = px.meet(&py).meet(&pz).to_euclidean();
//! assert!((x - 1.0).abs() < 1e-12);
//! assert!((y - 2.0).abs() < 1e-12);
//! assert!((z - 3.0).abs() < 1e-12);
//! ```

use garust_core::multivector::Multivector;
use garust_core::scalar::Scalar;
use garust_core::Pga3Sig;

use crate::Motor;

/// Re-exported so the umbrella crate's `garust::pga` namespace keeps the
/// PGA-aware [`Display`](core::fmt::Display) adapter alongside the typed
/// objects.
pub use garust_core::pga::PgaDisplay;

/// The PGA multivector type these objects wrap: `Cl(3, 0, 1)` over `T`.
type Pga<T> = Multivector<Pga3Sig, T>;

/// Generate a grade-typed PGA newtype with the access/transform methods
/// every one of them shares: wrap/unwrap the raw blade and apply a motor.
macro_rules! pga_object {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name<T: Scalar = f64> {
            mv: Pga<T>,
        }

        impl<T: Scalar> $name<T> {
            /// Wrap a raw PGA multivector, unchecked.
            ///
            /// The caller promises `mv` is a blade of this object's grade
            /// (point = 3, line = 2, plane = 1). Use the named constructors
            /// and the incidence operators when you can; this is the escape
            /// hatch for blades produced elsewhere.
            pub fn from_multivector(mv: Pga<T>) -> Self {
                Self { mv }
            }

            /// The underlying PGA multivector — the raw-access seam back to
            /// the [`garust_core`] kernel.
            pub fn multivector(&self) -> Pga<T> {
                self.mv
            }

            /// Move this object by a rigid motion. A [`Motor`]
            /// preserves grade, so a point stays a point, a line a line, and
            /// a plane a plane.
            pub fn transform(&self, motor: &Motor<T>) -> Self {
                Self {
                    mv: motor.apply(&self.mv),
                }
            }
        }
    };
}

pga_object!(
    /// A Euclidean point, the grade-3 trivector of PGA.
    Point
);
pga_object!(
    /// A line, the grade-2 bivector of PGA. Build one by joining two
    /// [`Point`]s or meeting two [`Plane`]s.
    Line
);
pga_object!(
    /// A plane, the grade-1 vector of PGA.
    Plane
);

impl<T: Scalar> Point<T> {
    /// The Euclidean point at `(x, y, z)`.
    pub fn new(x: T, y: T, z: T) -> Self {
        Self {
            mv: Pga::point(x, y, z),
        }
    }

    /// Read the Euclidean coordinates back out, dividing through by the
    /// homogeneous weight so the result is independent of scale.
    ///
    /// Meant for finite points; an ideal point (zero weight) divides by
    /// zero and yields non-finite coordinates.
    pub fn to_euclidean(&self) -> (T, T, T) {
        let c = &self.mv.coeffs;
        let w = c[0b0111];
        (-c[0b1110] / w, c[0b1101] / w, -c[0b1011] / w)
    }

    /// The [`Line`] through this point and `other` — their *join*
    /// `self ∨ other`. Two coincident points join to zero.
    pub fn join(&self, other: &Self) -> Line<T> {
        Line {
            mv: self.mv.regressive(&other.mv),
        }
    }
}

impl<T: Scalar> Line<T> {
    /// The line through two points — an alias for [`Point::join`], spelled
    /// from the [`Line`]'s point of view.
    pub fn through(a: &Point<T>, b: &Point<T>) -> Self {
        a.join(b)
    }

    /// The [`Plane`] spanned by this line and a point off it — their
    /// *join* `self ∨ point`. A point *on* the line spans nothing and
    /// joins to zero.
    pub fn join(&self, point: &Point<T>) -> Plane<T> {
        Plane {
            mv: self.mv.regressive(&point.mv),
        }
    }

    /// The [`Point`] where this line pierces a plane — their *meet*
    /// `self ∧ plane`. A line lying in the plane meets it everywhere and
    /// yields zero.
    pub fn meet(&self, plane: &Plane<T>) -> Point<T> {
        Point {
            mv: self.mv.wedge(&plane.mv),
        }
    }
}

impl<T: Scalar> Plane<T> {
    /// The plane `a·x + b·y + c·z + d = 0`. The `(a, b, c)` part is the
    /// (unnormalized) normal direction, `d` the offset.
    pub fn new(a: T, b: T, c: T, d: T) -> Self {
        Self {
            mv: Pga::plane(a, b, c, d),
        }
    }

    /// The plane through three points, `a ∨ b ∨ c`. Collinear points span
    /// no plane and give zero.
    pub fn through(a: &Point<T>, b: &Point<T>, c: &Point<T>) -> Self {
        a.join(b).join(c)
    }

    /// The [`Line`] where this plane meets `other` — their *meet*
    /// `self ∧ other`. Parallel planes meet at infinity (an ideal line).
    pub fn meet(&self, other: &Self) -> Line<T> {
        Line {
            mv: self.mv.wedge(&other.mv),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Line, Plane, Point};
    use crate::Motor;
    use garust_core::Pga3;
    use std::f64::consts::FRAC_PI_2;

    fn approx_xyz(got: (f64, f64, f64), want: (f64, f64, f64)) {
        let (x, y, z) = got;
        let (a, b, c) = want;
        assert!(
            (x - a).abs() < 1e-10 && (y - b).abs() < 1e-10 && (z - c).abs() < 1e-10,
            "got ({x}, {y}, {z}), want ({a}, {b}, {c})"
        );
    }

    #[test]
    fn point_round_trips_through_euclidean() {
        approx_xyz(
            Point::new(1.5, -2.0, 3.25).to_euclidean(),
            (1.5, -2.0, 3.25),
        );
    }

    #[test]
    fn typed_point_wraps_the_raw_constructor() {
        // The newtype is exactly the kernel blade, no re-encoding.
        assert_eq!(
            Point::new(1.0, 2.0, 3.0).multivector(),
            Pga3::point(1.0, 2.0, 3.0)
        );
    }

    #[test]
    fn join_of_two_points_is_a_grade_2_line() {
        let line = Point::new(0.0, 0.0, 0.0).join(&Point::new(1.0, 0.0, 0.0));
        let mv = line.multivector();
        assert_eq!(mv.grade(2), mv); // pure bivector
        assert_ne!(mv, Pga3::zero());
    }

    #[test]
    fn meet_of_two_planes_is_a_grade_2_line() {
        let line = Plane::new(1.0, 0.0, 0.0, 0.0).meet(&Plane::new(0.0, 1.0, 0.0, 0.0));
        let mv = line.multivector();
        assert_eq!(mv.grade(2), mv);
        assert_ne!(mv, Pga3::zero());
    }

    #[test]
    fn three_planes_meet_at_their_common_point() {
        let px = Plane::new(1.0, 0.0, 0.0, -1.0);
        let py = Plane::new(0.0, 1.0, 0.0, -2.0);
        let pz = Plane::new(0.0, 0.0, 1.0, -3.0);
        approx_xyz(px.meet(&py).meet(&pz).to_euclidean(), (1.0, 2.0, 3.0));
    }

    #[test]
    fn line_pierces_plane_at_the_expected_point() {
        // The z-axis (through the origin and (0,0,1)) pierces the plane
        // z = 5 at (0, 0, 5).
        let z_axis = Point::new(0.0, 0.0, 0.0).join(&Point::new(0.0, 0.0, 1.0));
        let plane = Plane::new(0.0, 0.0, 1.0, -5.0);
        approx_xyz(z_axis.meet(&plane).to_euclidean(), (0.0, 0.0, 5.0));
    }

    #[test]
    fn plane_through_three_points_is_grade_1_and_correct() {
        // (1,0,0), (0,1,0), (0,0,1) span the plane x + y + z = 1.
        let plane: Plane = Plane::through(
            &Point::new(1.0, 0.0, 0.0),
            &Point::new(0.0, 1.0, 0.0),
            &Point::new(0.0, 0.0, 1.0),
        );
        let mv = plane.multivector();
        assert_eq!(mv.grade(1), mv); // pure vector
                                     // Proportional to plane(1, 1, 1, -1): all four coeffs share a ratio.
        let k = mv.coeffs[0b0001];
        assert!(k.abs() > 1e-12);
        for &i in &[0b0010usize, 0b0100] {
            assert!((mv.coeffs[i] - k).abs() < 1e-10);
        }
        assert!((mv.coeffs[0b1000] + k).abs() < 1e-10); // d = -k
    }

    #[test]
    fn through_alias_matches_join() {
        let a: Point = Point::new(1.0, 2.0, 3.0);
        let b = Point::new(-1.0, 0.5, 4.0);
        assert_eq!(Line::through(&a, &b), a.join(&b));
    }

    #[test]
    fn transform_moves_a_point_like_the_motor() {
        // 90° about the x-axis sends (0,1,0) → (0,0,1).
        let r = Motor::rotor(FRAC_PI_2, Pga3::basis(0b0110));
        let moved = Point::new(0.0, 1.0, 0.0).transform(&r);
        approx_xyz(moved.to_euclidean(), (0.0, 0.0, 1.0));
        // And it agrees with applying the motor to the raw blade.
        assert_eq!(moved.multivector(), r.apply(&Pga3::point(0.0, 1.0, 0.0)));
    }

    #[test]
    fn transform_is_grade_preserving_for_a_plane() {
        let plane: Plane = Plane::new(0.0, 0.0, 1.0, -2.0); // z = 2
        let t = Motor::translator(0.0, 0.0, 3.0); // lift by 3 ⇒ z = 5
        let moved = plane.transform(&t).multivector().cleaned(1e-10);
        assert_eq!(moved.grade(1), moved); // still a plane
    }
}
