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
/// kernel's PGA-aware [`Display`](core::fmt::Display) adapter and the
/// screw-axis data it decomposes twists into alongside the typed objects.
pub use garust_core::pga::{PgaDisplay, ScrewAxis};

/// The PGA multivector type these objects wrap: `Cl(3, 0, 1)` over `T`.
type Pga<T> = Multivector<Pga3Sig, T>;

/// Generate a grade-typed PGA newtype with the access/transform methods
/// every one of them shares: wrap/unwrap the raw blade and apply a motor.
macro_rules! pga_object {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[cfg_attr(
            feature = "serde",
            derive(serde::Serialize, serde::Deserialize),
            serde(transparent)
        )]
        #[repr(transparent)]
        #[must_use]
        pub struct $name<T: Scalar = f64> {
            mv: Pga<T>,
        }

        // SAFETY: `#[repr(transparent)]` over a `Pga<T>` multivector, so the
        // newtype inherits its plain-old-data layout under the `bytemuck`
        // feature — handy for uploading a point/line/plane buffer to the GPU.
        #[cfg(feature = "bytemuck")]
        unsafe impl<T: Scalar> bytemuck::Zeroable for $name<T> where Pga<T>: bytemuck::Zeroable {}
        #[cfg(feature = "bytemuck")]
        unsafe impl<T: Scalar + 'static> bytemuck::Pod for $name<T> where Pga<T>: bytemuck::Pod {}

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

    /// The line's direction `d`, the Euclidean half of its Plücker
    /// coordinates `(d, m)` — the convention a renderer strokes by.
    ///
    /// Signs are pinned so a join runs from its first point to its second:
    /// [`Line::through(a, b)`](Line::through) has `direction() == b − a`,
    /// and the meet of two planes ([`Plane::meet`]) with normals `n₁`, `n₂`
    /// has `direction() == n₂ × n₁`. Unnormalized: it scales with the
    /// line's weight (join of unit-weight points ⇒ exactly `b − a`), and
    /// an *ideal* line (e.g. the meet of parallel planes) has direction
    /// zero.
    ///
    /// Note the sign seam against the *rotation* machinery: used as a
    /// rotation axis ([`Motor::rotation_about`], twist bivectors), this
    /// same blade spins right-handed about `−direction()` — i.e. about
    /// `a − b` for a join. The two readings differ by exactly a sign;
    /// pick per use and don't mix.
    pub fn direction(&self) -> [T; 3] {
        let c = &self.mv.coeffs;
        [-c[0b0110], c[0b0101], -c[0b0011]]
    }

    /// The line's Plücker moment `m = p × d` (for any point `p` on the
    /// line and `d = `[`direction()`](Line::direction)) — the ideal half
    /// of its Plücker coordinates.
    ///
    /// For a join [`Line::through(a, b)`](Line::through) of unit-weight
    /// points this is exactly `a × b`. Like the direction it scales with
    /// the line's weight; the pair `(d, m)` always satisfies `d · m = 0`.
    pub fn moment(&self) -> [T; 3] {
        let c = &self.mv.coeffs;
        [c[0b1001], c[0b1010], c[0b1100]]
    }

    /// The point on the line closest to the origin, `(d × m) / ‖d‖²` —
    /// [`Line::point_at`] at parameter zero, and the natural anchor for
    /// stroking the line.
    ///
    /// Meant for finite lines: an ideal line (zero direction) divides by
    /// zero and yields non-finite coordinates.
    pub fn point_closest_to_origin(&self) -> Point<T> {
        self.point_at(T::ZERO)
    }

    /// The point `point_closest_to_origin() + t·direction()` — walk the
    /// line by parameter `t`.
    ///
    /// The parametrization is *unnormalized*: `t` is in units of the
    /// line's weight `‖direction()‖`, so for a join of unit-weight points
    /// `point_at(0)` is the closest point to the origin and `t` advances
    /// by whole `b − a` steps. Meant for finite lines; an ideal line
    /// divides by zero.
    pub fn point_at(&self, t: T) -> Point<T> {
        let d = self.direction();
        let m = self.moment();
        let n = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        Point::new(
            (d[1] * m[2] - d[2] * m[1]) / n + d[0] * t,
            (d[2] * m[0] - d[0] * m[2]) / n + d[1] * t,
            (d[0] * m[1] - d[1] * m[0]) / n + d[2] * t,
        )
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

    /// The plane's normal direction `(a, b, c)` — the Euclidean
    /// coefficients of `a·x + b·y + c·z + d = 0`, exactly as passed to
    /// [`Plane::new`].
    ///
    /// Unnormalized: a plane built by join/meet carries its construction's
    /// weight, so normalize before using it as a unit normal.
    pub fn normal(&self) -> [T; 3] {
        let c = &self.mv.coeffs;
        [c[0b0001], c[0b0010], c[0b0100]]
    }

    /// The plane's offset `d` in `a·x + b·y + c·z + d = 0` — the null
    /// (`e0`) coefficient, exactly as passed to [`Plane::new`].
    ///
    /// Shares the plane's weight with [`Plane::normal`]: the origin
    /// distance of the plane is `−d / ‖(a, b, c)‖`.
    pub fn offset(&self) -> T {
        self.mv.coeffs[0b1000]
    }
}

#[cfg(test)]
mod tests {
    use super::{Line, Plane, Point};
    use crate::Motor;
    use garust_core::Pga3;
    use std::f64::consts::TAU;

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
        let r = Motor::rotor(TAU / 4.0, Pga3::basis(0b0110));
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

    // --- Coordinate accessors -------------------------------------------

    #[test]
    fn join_direction_runs_from_the_first_point_to_the_second() {
        let line = Point::new(0.0, 0.0, 0.0).join(&Point::new(1.0, 0.0, 0.0));
        assert_eq!(line.direction(), [1.0, 0.0, 0.0]);
        assert_eq!(line.moment(), [0.0, 0.0, 0.0]); // through the origin
    }

    #[test]
    fn join_direction_and_moment_are_the_plucker_pair() {
        // a = (1,2,3), b = (3,2,4): d = b − a = (2,0,1), m = a × b = (2,5,−4).
        let a: Point = Point::new(1.0, 2.0, 3.0);
        let b = Point::new(3.0, 2.0, 4.0);
        let line = a.join(&b);
        assert_eq!(line.direction(), [2.0, 0.0, 1.0]);
        assert_eq!(line.moment(), [2.0, 5.0, -4.0]);
        // The Plücker incidence invariant d · m = 0.
        let (d, m) = (line.direction(), line.moment());
        assert_eq!(d[0] * m[0] + d[1] * m[1] + d[2] * m[2], 0.0);
    }

    #[test]
    fn point_at_walks_the_line_from_the_closest_point() {
        // The vertical line through (1, 2, 0): closest point to the origin
        // has z = 0, and point_at advances by whole direction steps.
        let line = Point::new(1.0, 2.0, 0.0).join(&Point::new(1.0, 2.0, 1.0));
        approx_xyz(
            line.point_closest_to_origin().to_euclidean(),
            (1.0, 2.0, 0.0),
        );
        approx_xyz(line.point_at(2.5).to_euclidean(), (1.0, 2.0, 2.5));
        approx_xyz(line.point_at(-1.0).to_euclidean(), (1.0, 2.0, -1.0));
    }

    #[test]
    fn point_at_lands_on_the_joined_points() {
        // For a join of unit-weight points the parametrization is exact:
        // some t hits a, t+1 hits b. Here a is 1 step below the closest
        // point (0,0,0) along d = (1,0,0)… check both endpoints directly.
        let a: Point = Point::new(-1.0, 0.0, 0.0);
        let b = Point::new(2.0, 0.0, 0.0);
        let line = a.join(&b); // d = (3,0,0), closest = origin
        approx_xyz(line.point_at(0.0).to_euclidean(), (0.0, 0.0, 0.0));
        // t is in units of ‖d‖ = 3: t = ⅓ steps one unit along x.
        approx_xyz(line.point_at(1.0 / 3.0).to_euclidean(), (1.0, 0.0, 0.0));
    }

    #[test]
    fn meet_direction_is_n2_cross_n1() {
        // x = 1 meets y = 2 in the vertical line through (1, 2, 0); with
        // the join-pinned signs its direction is n₂ × n₁ = (0, 0, −1).
        let px: Plane = Plane::new(1.0, 0.0, 0.0, -1.0);
        let py = Plane::new(0.0, 1.0, 0.0, -2.0);
        let line = px.meet(&py);
        assert_eq!(line.direction(), [0.0, 0.0, -1.0]);
        approx_xyz(
            line.point_closest_to_origin().to_euclidean(),
            (1.0, 2.0, 0.0),
        );
    }

    #[test]
    fn plane_round_trips_normal_and_offset() {
        let plane: Plane = Plane::new(1.5, -2.0, 3.0, 4.25);
        assert_eq!(plane.normal(), [1.5, -2.0, 3.0]);
        assert_eq!(plane.offset(), 4.25);
    }

    #[test]
    fn rotation_about_spins_right_handed_about_minus_direction() {
        // The documented sign seam: a join line's rotation sense is the
        // opposite of its stroke direction. direction() here is +z, and
        // rotation_about spins right-handed about −z: (1,0,0) → (0,−1,0).
        let line = Point::new(0.0, 0.0, 0.0).join(&Point::new(0.0, 0.0, 1.0));
        assert_eq!(line.direction(), [0.0, 0.0, 1.0]);
        let m = Motor::rotation_about(line.multivector(), TAU / 4.0);
        approx_xyz(
            Point::new(1.0, 0.0, 0.0).transform(&m).to_euclidean(),
            (0.0, -1.0, 0.0),
        );
    }
}
