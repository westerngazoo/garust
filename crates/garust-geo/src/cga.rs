//! Typed CGA geometry — [`Point`], [`Sphere`], and [`Plane`] in `Cl(4, 1, 0)`.
//!
//! The kernel represents all three as a grade-1 [`Multivector`] in the
//! conformal model's *inner-product null space* (IPNS): a point is a null
//! vector, and a sphere or plane is the vector whose null space is the
//! round (or flat) surface it bounds. This module gives each a name, a
//! constructor, and a small, total incidence API so the compiler tracks
//! what you are holding instead of leaving every blade an untyped
//! multivector.
//!
//! Unlike the PGA ladder ([`pga`](crate::pga)), where meet and join close
//! over point/line/plane, two CGA spheres meet in a *circle* and three in
//! a *point pair* — objects outside this trio. So the typed CGA surface
//! centers on the relation the conformal model makes effortless:
//! **incidence**. A [`Point`] lies on a [`Sphere`] or [`Plane`] exactly
//! when their scalar product vanishes ([`Sphere::contains`],
//! [`Plane::contains`]); the raw scalar ([`Sphere::incidence`]) carries the
//! sidedness too. Conformal motions ([`Conformal`]) preserve grade, so
//! [`Point::transform`] and its siblings map each type to itself. Drop to
//! the raw blade any time with [`Point::multivector`] /
//! [`Point::from_multivector`] (and the analogues on [`Sphere`]/[`Plane`]).
//!
//! ```
//! use garust_geo::cga::{Point, Sphere};
//!
//! // The unit sphere at the origin: (1, 0, 0) lies on it, (2, 0, 0) does not.
//! // `Sphere`'s scalar defaults to f64, so one annotation pins the rest.
//! let unit: Sphere = Sphere::new(0.0, 0.0, 0.0, 1.0);
//! assert!(unit.contains(&Point::new(1.0, 0.0, 0.0), 1e-12));
//! assert!(!unit.contains(&Point::new(2.0, 0.0, 0.0), 1e-12));
//! ```

use garust_core::multivector::Multivector;
use garust_core::scalar::Scalar;
use garust_core::Cga3Sig;

use crate::Conformal;

/// The CGA multivector type these objects wrap: `Cl(4, 1, 0)` over `T`.
type Cga<T> = Multivector<Cga3Sig, T>;

/// Generate a grade-1 CGA newtype with the access/transform methods every
/// one of them shares: wrap/unwrap the raw blade and apply a conformal
/// versor.
macro_rules! cga_object {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[cfg_attr(
            feature = "serde",
            derive(serde::Serialize, serde::Deserialize),
            serde(transparent)
        )]
        #[repr(transparent)]
        pub struct $name<T: Scalar = f64> {
            mv: Cga<T>,
        }

        // SAFETY: `#[repr(transparent)]` over a `Cga<T>` multivector, so the
        // newtype inherits its plain-old-data layout under the `bytemuck`
        // feature — handy for uploading a point/sphere/plane buffer to the GPU.
        #[cfg(feature = "bytemuck")]
        unsafe impl<T: Scalar> bytemuck::Zeroable for $name<T> where Cga<T>: bytemuck::Zeroable {}
        #[cfg(feature = "bytemuck")]
        unsafe impl<T: Scalar + 'static> bytemuck::Pod for $name<T> where Cga<T>: bytemuck::Pod {}

        impl<T: Scalar> $name<T> {
            /// Wrap a raw CGA multivector, unchecked.
            ///
            /// The caller promises `mv` is a grade-1 conformal blade of
            /// this object's kind. Use the named constructors when you
            /// can; this is the escape hatch for blades produced
            /// elsewhere.
            pub fn from_multivector(mv: Cga<T>) -> Self {
                Self { mv }
            }

            /// The underlying CGA multivector — the raw-access seam back
            /// to the [`garust_core`] kernel.
            pub fn multivector(&self) -> Cga<T> {
                self.mv
            }

            /// Move this object by a conformal transformation. A
            /// [`Conformal`] versor preserves grade, so a point stays a
            /// point, a sphere a sphere, and a plane a plane.
            pub fn transform(&self, versor: &Conformal<T>) -> Self {
                Self {
                    mv: versor.apply(&self.mv),
                }
            }
        }
    };
}

cga_object!(
    /// A Euclidean point, lifted to its grade-1 null vector in the
    /// conformal model.
    Point
);
cga_object!(
    /// A sphere, the grade-1 vector `c − ½r²·n∞`. A [`Point`] lies on it
    /// exactly when [`Sphere::contains`] holds; a zero radius recovers the
    /// centre point, a negative `r²` an imaginary sphere.
    Sphere
);
cga_object!(
    /// A plane, the grade-1 vector `a·e1 + b·e2 + c·e3 + d·n∞` — a sphere
    /// through the point at infinity.
    Plane
);

impl<T: Scalar> Point<T> {
    /// The Euclidean point at `(x, y, z)`, lifted to its conformal null
    /// vector `no + x·e1 + y·e2 + z·e3 + ½(x²+y²+z²)·n∞`.
    pub fn new(x: T, y: T, z: T) -> Self {
        Self {
            mv: Cga::cga_point(x, y, z),
        }
    }

    /// Read the Euclidean coordinates back out, dividing through by the
    /// homogeneous weight so the result is independent of scale (a
    /// [`Conformal`] dilation rescales that weight).
    pub fn to_euclidean(&self) -> (T, T, T) {
        self.mv.to_euclidean()
    }
}

impl<T: Scalar> Sphere<T> {
    /// The sphere with centre `(cx, cy, cz)` and radius `r`, as the
    /// grade-1 vector `cga_point(c) − ½r²·n∞`. A zero radius gives the
    /// point at the centre; a negative `r²` an imaginary sphere.
    pub fn new(cx: T, cy: T, cz: T, r: T) -> Self {
        Self {
            mv: Cga::sphere(cx, cy, cz, r),
        }
    }

    /// The scalar product `p · S` — the fundamental incidence quantity,
    /// **zero exactly when `p` lies on the sphere** (scaled by both
    /// homogeneous weights), with a sign that says which side. Compare it
    /// against a tolerance rather than testing exact equality in floating
    /// point; [`Sphere::contains`] does just that.
    pub fn incidence(&self, p: &Point<T>) -> T {
        self.mv.scalar_product(&p.mv)
    }

    /// Whether `p` lies on this sphere, within `tol` of the exact
    /// zero-[`incidence`](Sphere::incidence) condition.
    pub fn contains(&self, p: &Point<T>, tol: T) -> bool {
        self.incidence(p).abs() <= tol
    }
}

impl<T: Scalar> Plane<T> {
    /// The plane `a·x + b·y + c·z = d` with normal `(a, b, c)`, as the
    /// grade-1 vector `a·e1 + b·e2 + c·e3 + d·n∞`.
    pub fn new(a: T, b: T, c: T, d: T) -> Self {
        Self {
            mv: Cga::cga_plane(a, b, c, d),
        }
    }

    /// The scalar product `p · π` — **zero exactly when `p` lies on the
    /// plane** (scaled by the point's weight and the normal's length),
    /// with a sign that says which side. Compare it against a tolerance;
    /// [`Plane::contains`] does.
    pub fn incidence(&self, p: &Point<T>) -> T {
        self.mv.scalar_product(&p.mv)
    }

    /// Whether `p` lies on this plane, within `tol` of the exact
    /// zero-[`incidence`](Plane::incidence) condition.
    pub fn contains(&self, p: &Point<T>, tol: T) -> bool {
        self.incidence(p).abs() <= tol
    }
}

#[cfg(test)]
mod tests {
    use super::{Plane, Point, Sphere};
    use crate::Conformal;
    use garust_core::Cga3;
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
            Cga3::cga_point(1.0, 2.0, 3.0)
        );
    }

    #[test]
    fn point_lies_on_the_unit_sphere() {
        let unit: Sphere = Sphere::new(0.0, 0.0, 0.0, 1.0);
        assert!(unit.contains(&Point::new(1.0, 0.0, 0.0), 1e-10));
        assert!(unit.contains(&Point::new(0.0, 0.0, 1.0), 1e-10));
        assert!(!unit.contains(&Point::new(2.0, 0.0, 0.0), 1e-10));
    }

    #[test]
    fn sphere_off_origin_contains_the_right_point() {
        // Radius-2 sphere centred at (1, 2, 3): (3, 2, 3) lies on it.
        let s: Sphere = Sphere::new(1.0, 2.0, 3.0, 2.0);
        assert!(s.contains(&Point::new(3.0, 2.0, 3.0), 1e-10));
    }

    #[test]
    fn zero_radius_sphere_is_the_centre_point() {
        let s: Sphere = Sphere::new(1.0, 2.0, 3.0, 0.0);
        assert_eq!(s.multivector(), Point::new(1.0, 2.0, 3.0).multivector());
    }

    #[test]
    fn incidence_vanishes_on_the_sphere_and_not_off_it() {
        let s: Sphere = Sphere::new(0.0, 0.0, 0.0, 1.0);
        assert!(s.incidence(&Point::new(1.0, 0.0, 0.0)).abs() < 1e-10);
        assert!(s.incidence(&Point::new(2.0, 0.0, 0.0)).abs() > 1e-6);
    }

    #[test]
    fn point_lies_on_the_plane() {
        // Plane x = 1 (normal e1, offset 1): (1, 5, -2) on it, (2, 0, 0) off.
        let pi: Plane = Plane::new(1.0, 0.0, 0.0, 1.0);
        assert!(pi.contains(&Point::new(1.0, 5.0, -2.0), 1e-10));
        assert!(!pi.contains(&Point::new(2.0, 0.0, 0.0), 1e-10));
    }

    #[test]
    fn transform_moves_a_point_like_the_versor() {
        let t = Conformal::translator(3.0, -1.0, 2.0);
        let moved = Point::new(1.0, 1.0, 1.0).transform(&t);
        approx_xyz(moved.to_euclidean(), (4.0, 0.0, 3.0));
        // And it agrees with applying the versor to the raw blade.
        assert_eq!(
            moved.multivector(),
            t.apply(&Cga3::cga_point(1.0, 1.0, 1.0))
        );
    }

    #[test]
    fn rotor_about_x_axis_sends_y_to_z() {
        // 90° in the e23 plane: (0, 1, 0) → (0, 0, 1).
        let r = Conformal::rotor(TAU / 4.0, Cga3::basis(0b0110));
        let moved = Point::new(0.0, 1.0, 0.0).transform(&r);
        approx_xyz(moved.to_euclidean(), (0.0, 0.0, 1.0));
    }

    #[test]
    fn dilation_rescales_a_sphere_through_the_typed_api() {
        // Doubling about the origin turns the unit sphere into radius 2:
        // (2, 0, 0) lands on the image, (1, 0, 0) no longer does.
        let unit: Sphere = Sphere::new(0.0, 0.0, 0.0, 1.0);
        let scaled = unit.transform(&Conformal::dilator(2.0));
        assert!(scaled.contains(&Point::new(2.0, 0.0, 0.0), 1e-10));
        assert!(!scaled.contains(&Point::new(1.0, 0.0, 0.0), 1e-10));
    }

    #[test]
    fn transform_is_grade_preserving_for_a_plane() {
        let plane: Plane = Plane::new(0.0, 0.0, 1.0, 2.0); // z = 2
        let t = Conformal::translator(0.0, 0.0, 3.0); // lift by 3 ⇒ z = 5
        let moved = plane.transform(&t).multivector().cleaned(1e-10);
        assert_eq!(moved.grade(1), moved); // still a plane
    }
}
