//! 3D Projective Geometric Algebra constructors, for `Pga3 = Cl(3, 0, 1)`.
//!
//! Round 5 gave us the wedge (`∧`, *join*) and the regressive product
//! (`∨`, *meet*). This module turns those abstract operations into
//! honest geometry by naming the blades you actually care about —
//! points, planes, and the lines between them — in the one signature
//! where they have a Euclidean interpretation.
//!
//! # Conventions (this crate's `Pga3`)
//!
//! Four generators, with the null one at bit 3 (printed `e4`, called
//! `e0` in the PGA literature):
//!
//! - `e1, e2, e3` square to `+1` — the Euclidean directions
//! - `e0` squares to `0` — the projective / ideal direction
//!
//! Grade carries the geometric meaning, dual to the usual vector view:
//!
//! | object | grade | blade form |
//! |--------|-------|------------|
//! | plane  | 1     | `a·e1 + b·e2 + c·e3 + d·e0` for `ax+by+cz+d = 0` |
//! | line   | 2     | join of two points / meet of two planes |
//! | point  | 3     | `e1e2e3 − x·e0e2e3 + y·e0e1e3 − z·e0e1e2` for `(x,y,z)` |
//!
//! With these conventions the two products do exactly what their names
//! promise: the **wedge of two planes is their intersection line**, the
//! **wedge of three planes is their common point**, and the
//! **regressive product of two points is the line through them**.

use core::fmt;

use crate::multivector::Multivector;
use crate::scalar::{Real, Scalar};
use crate::Pga3Sig;

/// PGA constructors are defined only for the `Cl(3, 0, 1)` signature, so
/// this `impl` is pinned to the `Pga3Sig` marker. It covers both `Pga3`
/// (`f64`) and `Pga3f` (`f32`).
impl<T: Scalar> Multivector<Pga3Sig, T> {
    /// A Euclidean point at `(x, y, z)`, as a grade-3 trivector.
    ///
    /// ```
    /// use garust_core::Pga3;
    /// let p = Pga3::point(1.0, 2.0, 3.0);
    /// assert_eq!(p.grade(3).coeffs, p.coeffs); // pure trivector
    /// ```
    pub fn point(x: T, y: T, z: T) -> Self {
        let e0 = Self::basis(8);
        let e1 = Self::basis(1);
        let e2 = Self::basis(2);
        let e3 = Self::basis(4);
        e1 * e2 * e3 - (e0 * e2 * e3) * x + (e0 * e1 * e3) * y - (e0 * e1 * e2) * z
    }

    /// The plane `a·x + b·y + c·z + d = 0`, as a grade-1 vector.
    ///
    /// The `(a, b, c)` part is the (unnormalized) normal direction and
    /// `d` the offset along the null generator `e0`.
    pub fn plane(a: T, b: T, c: T, d: T) -> Self {
        let e0 = Self::basis(8);
        let e1 = Self::basis(1);
        let e2 = Self::basis(2);
        let e3 = Self::basis(4);
        e1 * a + e2 * b + e3 * c + e0 * d
    }

    /// The line through two points — their *join*, i.e. the regressive
    /// product `p ∨ q`. A grade-2 bivector.
    ///
    /// Antisymmetric: `line_through(p, q) == −line_through(q, p)`. Three
    /// collinear points join to zero, which is the incidence test for
    /// "do these points lie on a common line?".
    pub fn line_through(&self, other: &Self) -> Self {
        self.regressive(other)
    }

    /// A body-frame **twist** bivector `B = ω + v`.
    ///
    /// Encodes the instantaneous rigid-body velocity as a single grade-2
    /// element — the natural PGA form of the body Lie algebra element:
    ///
    /// - Angular part `ω = wx·e23 + wy·e31 + wz·e12` (Euclidean bivectors,
    ///   one per rotation axis).
    /// - Translational part `v = vx·e01 + vy·e02 + vz·e03` (ideal bivectors,
    ///   one per translation direction).
    ///
    /// The equations of motion are `Ṁ = -½·M·B` (kinematics) and
    /// `Ṗ = W + P.commutator(B)` (dynamics), where `P = 𝓘(B)` is momentum
    /// and `W` a wrench. See also [`Multivector::wrench`] and
    /// [`Multivector::commutator`].
    pub fn twist(vx: T, vy: T, vz: T, wx: T, wy: T, wz: T) -> Self {
        let e0 = Self::basis(8);
        let e1 = Self::basis(1);
        let e2 = Self::basis(2);
        let e3 = Self::basis(4);
        (e2 * e3) * wx
            + (e3 * e1) * wy
            + (e1 * e2) * wz
            + (e0 * e1) * vx
            + (e0 * e2) * vy
            + (e0 * e3) * vz
    }

    /// A **wrench** bivector `W = τ + f`.
    ///
    /// The PGA dual of a twist: packages torque and force into one grade-2
    /// element for feeding into the equations of motion (`Ṗ = W + P×B`):
    ///
    /// - Torque part `τ = tx·e23 + ty·e31 + tz·e12` (Euclidean bivectors).
    /// - Force part `f = fx·e01 + fy·e02 + fz·e03` (ideal bivectors).
    ///
    /// A wrench applied at the centre of mass produces no angular impulse;
    /// for an off-centre force, add the lever-arm cross-product as torque.
    pub fn wrench(fx: T, fy: T, fz: T, tx: T, ty: T, tz: T) -> Self {
        let e0 = Self::basis(8);
        let e1 = Self::basis(1);
        let e2 = Self::basis(2);
        let e3 = Self::basis(4);
        (e2 * e3) * tx
            + (e3 * e1) * ty
            + (e1 * e2) * tz
            + (e0 * e1) * fx
            + (e0 * e2) * fy
            + (e0 * e3) * fz
    }

    /// The exact inverse of [`Multivector::twist`]: read the six grade-2
    /// components back out as `([vx, vy, vz], [wx, wy, wz])`.
    ///
    /// `twist` writes its arguments onto the six bivector blades with the
    /// signs below. The stored basis keeps generators in bit order with the
    /// null generator *last*, so the PGA-literature blades `e31`, `e01`,
    /// `e02`, `e03` land on their stored slots with a sign flip:
    ///
    /// | argument | blade (PGA order) | stored slot        | stored sign |
    /// |----------|-------------------|--------------------|-------------|
    /// | `wx`     | `e23`             | `0b0110`           | `+`         |
    /// | `wy`     | `e31`             | `0b0101` (`e13`)   | `−`         |
    /// | `wz`     | `e12`             | `0b0011`           | `+`         |
    /// | `vx`     | `e01`             | `0b1001` (`e1∧e0`) | `−`         |
    /// | `vy`     | `e02`             | `0b1010` (`e2∧e0`) | `−`         |
    /// | `vz`     | `e03`             | `0b1100` (`e3∧e0`) | `−`         |
    ///
    /// This method undoes exactly those signs, so
    /// `Pga3::twist(v, w).twist_parts() == (v, w)` bit-for-bit — placing
    /// and negating coefficients is exact in floating point, and no other
    /// arithmetic happens. [`Multivector::wrench`] shares the layout, so
    /// the same call splits a wrench into `([fx, fy, fz], [tx, ty, tz])`.
    /// Only the six grade-2 slots are read; any other grades present are
    /// ignored.
    ///
    /// ```
    /// use garust_core::Pga3;
    /// let b = Pga3::twist(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    /// assert_eq!(b.twist_parts(), ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]));
    /// ```
    pub fn twist_parts(&self) -> ([T; 3], [T; 3]) {
        let c = &self.coeffs;
        (
            [-c[0b1001], -c[0b1010], -c[0b1100]],
            [c[0b0110], -c[0b0101], c[0b0011]],
        )
    }

    /// A PGA-aware [`fmt::Display`] view that prints the null generator
    /// as `e0`, written *first* in each blade, matching the PGA
    /// literature.
    ///
    /// The crate's generic `Display` labels bit `k` as `e{k+1}`, so the
    /// null generator at bit 3 prints as the misleading `e4`. This
    /// wrapper relabels it `e0` and moves it to the front of every blade.
    /// Reordering `e0` past the `m` Euclidean generators ahead of it
    /// costs a sign `(-1)^m`, which is folded into the printed
    /// coefficient — so `basis(9)` (stored as `e1∧e0`) prints as `-e01`.
    ///
    /// ```
    /// use garust_core::Pga3;
    /// let p = Pga3::point(1.0, 2.0, 3.0);
    /// assert_eq!(p.display_pga().to_string(), "e123 - 3·e012 + 2·e013 - e023");
    /// ```
    pub fn display_pga(&self) -> PgaDisplay<T> {
        PgaDisplay { mv: *self }
    }
}

/// The screw-axis decomposition needs square roots and ordering, so it
/// lives in a separate `impl` bounded by [`Real`] rather than [`Scalar`].
/// Like the constructors above it is pinned to the `Cl(3, 0, 1)` signature.
impl<T: Real> Multivector<Pga3Sig, T> {
    /// Decompose a grade-2 **twist** into its instantaneous screw axis —
    /// the drawable form of the motion: a directed axis line in space, the
    /// rotation rate about it, and the slide rate along it.
    ///
    /// A twist `B = twist(v, w)` generates the motion `M(t) = exp(−t·B/2)`,
    /// under which a point `x` moves with velocity `ẋ = w × x + v`. By
    /// Chasles' theorem that velocity field is a **screw**: a right-handed
    /// rotation at `‖w‖` about a unique axis, plus a slide along that same
    /// axis. This method returns the axis and its rates:
    ///
    /// - `direction` — the unit axis `w / ‖w‖`; rotation follows the
    ///   right-hand rule about it.
    /// - `point` — the point on the axis closest to the origin,
    ///   `(w × v) / ‖w‖²`.
    /// - `angle_rate` — `‖w‖`, radians per unit of the twist's scale.
    /// - `pitch` — slide along the axis per **radian** of rotation,
    ///   `(w · v) / ‖w‖²`; zero for a pure (possibly off-axis) rotation.
    ///
    /// Internally the twist is first separated by
    /// [`Multivector::try_bivector_split`] into a *simple* part (the axis
    /// line, weighted by the rotation rate) and an *ideal* null part (the
    /// slide along it); the axis point comes from the simple part and the
    /// pitch from the ideal part. When the twist is already simple — every
    /// join/meet line, every zero-pitch rotation — the split is exact and
    /// `pitch` is exactly `0.0`, so a renderer can branch on it. Like the
    /// split, this expects a bivector (debug-asserted).
    ///
    /// Returns `None` when there is no finite axis to draw:
    ///
    /// - **pure translations** (and the zero twist). An ideal twist's
    ///   "axis" is the ideal line at infinity — equivalently its pitch is
    ///   infinite — so rather than return non-finite components the policy
    ///   here is `None`; recover the translation with
    ///   [`Multivector::twist_parts`] and draw a straight arrow instead.
    ///   The test is relative: a Euclidean part below `1e-12` of the
    ///   twist's total magnitude (norm-wise) counts as ideal.
    /// - the split itself fails (isoclinic bivectors). Unreachable for
    ///   genuine PGA twists — the degenerate metric makes the split total —
    ///   but mapped to `None` rather than a panic.
    ///
    /// Raw components are returned rather than a typed line because the
    /// typed `pga::Line` lives a layer up, in `garust-geo` (which
    /// re-exports [`ScrewAxis`] alongside it); the components feed a
    /// renderer directly. Note the sign seam pinned there: a typed line's
    /// `direction()` is the join's `b − a`, while a twist rotates
    /// right-handed about *its* `direction` — for the same raw bivector
    /// the two conventions differ by a sign.
    ///
    /// ```
    /// use garust_core::Pga3;
    /// // Spin about +z through (1, 0, 0) at 1 rad/s, sliding 0.5/radian.
    /// let b = Pga3::twist(0.0, -1.0, 0.5, 0.0, 0.0, 1.0);
    /// let s = b.screw_axis().unwrap();
    /// assert_eq!(s.direction, [0.0, 0.0, 1.0]);
    /// assert!((s.point[0] - 1.0).abs() < 1e-12);
    /// assert!((s.pitch - 0.5).abs() < 1e-12);
    /// // A pure translation has no finite axis.
    /// assert_eq!(Pga3::twist(1.0, 2.0, 3.0, 0.0, 0.0, 0.0).screw_axis(), None);
    /// ```
    pub fn screw_axis(&self) -> Option<ScrewAxis<T>> {
        let (b1, b2) = self.try_bivector_split()?;
        let (v1, w1) = b1.twist_parts();
        let (v2, w2) = b2.twist_parts();
        let dot = |a: &[T; 3], b: &[T; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let (n1, n2) = (dot(&w1, &w1), dot(&w2, &w2));
        // The split returns its parts in no particular order; the axis line
        // is the Euclidean-carrying one, the slide the ideal one.
        let (v_axis, w, n, v_slide) = if n1 >= n2 {
            (v1, w1, n1, v2)
        } else {
            (v2, w2, n2, v1)
        };
        // Ideal twist (pure translation or zero): the axis is the line at
        // infinity — nothing finite to draw.
        let scale = n1 + n2 + dot(&v1, &v1) + dot(&v2, &v2);
        if n <= T::from_f64(1e-24) * scale {
            return None;
        }
        let rate = n.sqrt();
        let inv = T::ONE / n;
        Some(ScrewAxis {
            direction: [w[0] / rate, w[1] / rate, w[2] / rate],
            point: [
                (w[1] * v_axis[2] - w[2] * v_axis[1]) * inv,
                (w[2] * v_axis[0] - w[0] * v_axis[2]) * inv,
                (w[0] * v_axis[1] - w[1] * v_axis[0]) * inv,
            ],
            angle_rate: rate,
            pitch: dot(&w, &v_slide) * inv,
        })
    }
}

/// A twist bivector's instantaneous **screw axis**, in renderer-ready
/// components — built by [`Multivector::screw_axis`].
///
/// The motion is a right-handed rotation at `angle_rate` about the line
/// `point + t·direction`, plus a slide of `pitch` along `direction` per
/// radian of rotation. All components live in the same frame as the twist
/// itself: decompose a body-frame twist and the axis is in body
/// coordinates; conjugate the twist into another frame first to draw
/// there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScrewAxis<T: Scalar> {
    /// Unit vector along the axis; rotation is right-handed about it.
    pub direction: [T; 3],
    /// The point on the axis closest to the origin.
    pub point: [T; 3],
    /// Rotation rate `‖w‖` about the axis (radians per unit time, for a
    /// velocity twist).
    pub angle_rate: T,
    /// Translation along `direction` per **radian** of rotation; exactly
    /// `0.0` for a simple (pure-rotation) twist.
    pub pitch: T,
}

/// A `Display` adapter for `Pga3` multivectors that names the null
/// generator `e0` and writes it first in each blade. Build one with
/// [`Multivector::display_pga`].
pub struct PgaDisplay<T: Scalar> {
    mv: Multivector<Pga3Sig, T>,
}

impl<T: Real> fmt::Display for PgaDisplay<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for i in 0..16 {
            let raw = self.mv.coeffs[i];
            if raw == T::ZERO {
                continue;
            }
            let has_e0 = i & 0b1000 != 0;
            let others = i & 0b0111;
            // Reordering e0 to the front past the Euclidean generators
            // already in the blade flips the sign once per crossing.
            let reorder_neg = has_e0 && (others.count_ones() & 1 == 1);
            let c = if reorder_neg { -raw } else { raw };

            // Sign / separator.
            let negative = c < T::ZERO;
            if first {
                if negative {
                    write!(f, "-")?;
                }
                first = false;
            } else if negative {
                write!(f, " - ")?;
            } else {
                write!(f, " + ")?;
            }
            let mag = if negative { -c } else { c };

            // Suppress the magnitude when it's exactly 1 (a bare blade),
            // unless this is the scalar term where the number is all there is.
            let is_scalar = i == 0;
            if is_scalar || mag != T::ONE {
                write!(f, "{mag}")?;
                if !is_scalar {
                    write!(f, "·")?;
                }
            }
            if is_scalar {
                continue;
            }

            write!(f, "e")?;
            if has_e0 {
                write!(f, "0")?;
            }
            for bit in 0..3 {
                if others & (1 << bit) != 0 {
                    write!(f, "{}", bit + 1)?;
                }
            }
        }
        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::Pga3;
    use proptest::prelude::*;

    // --- Grades ---------------------------------------------------------

    #[test]
    fn point_is_a_pure_trivector() {
        let p = Pga3::point(1.0, -2.0, 3.5);
        assert_eq!(p.grade(3).coeffs, p.coeffs);
        // The e1e2e3 component is always 1 (the homogeneous weight).
        assert_eq!(p.coeffs[0b0111], 1.0);
    }

    #[test]
    fn plane_is_a_pure_vector() {
        let pi = Pga3::plane(1.0, 2.0, 3.0, 4.0);
        assert_eq!(pi.grade(1).coeffs, pi.coeffs);
        // a,b,c land on e1,e2,e3 ; d lands on the null generator e0.
        assert_eq!(pi.coeffs[0b0001], 1.0);
        assert_eq!(pi.coeffs[0b0010], 2.0);
        assert_eq!(pi.coeffs[0b0100], 3.0);
        assert_eq!(pi.coeffs[0b1000], 4.0);
    }

    // --- Meet of planes -------------------------------------------------

    #[test]
    fn three_coordinate_planes_meet_at_the_origin() {
        // x=0, y=0, z=0 intersect at the origin point.
        let xy0 = Pga3::plane(1.0, 0.0, 0.0, 0.0);
        let yz0 = Pga3::plane(0.0, 1.0, 0.0, 0.0);
        let zx0 = Pga3::plane(0.0, 0.0, 1.0, 0.0);
        let meet = xy0.wedge(&yz0).wedge(&zx0);
        assert_eq!(meet.coeffs, Pga3::point(0.0, 0.0, 0.0).coeffs);
    }

    #[test]
    fn three_planes_meet_at_their_common_point() {
        // x=1, y=2, z=3  ⇒  meet at exactly (1, 2, 3), no tolerance.
        let px = Pga3::plane(1.0, 0.0, 0.0, -1.0);
        let py = Pga3::plane(0.0, 1.0, 0.0, -2.0);
        let pz = Pga3::plane(0.0, 0.0, 1.0, -3.0);
        let meet = px.wedge(&py).wedge(&pz);
        assert_eq!(meet.coeffs, Pga3::point(1.0, 2.0, 3.0).coeffs);
    }

    #[test]
    fn two_planes_meet_in_a_grade_2_line() {
        let px = Pga3::plane(1.0, 0.0, 0.0, 0.0);
        let py = Pga3::plane(0.0, 1.0, 0.0, 0.0);
        let line = px.wedge(&py);
        assert_eq!(line.grade(2).coeffs, line.coeffs);
        assert_ne!(line.coeffs, Pga3::zero().coeffs);
    }

    // --- Join of points -------------------------------------------------

    #[test]
    fn line_through_two_points_is_a_bivector() {
        let line = Pga3::line_through(&Pga3::point(0.0, 0.0, 0.0), &Pga3::point(1.0, 0.0, 0.0));
        assert_eq!(line.grade(2).coeffs, line.coeffs);
        assert_ne!(line.coeffs, Pga3::zero().coeffs);
    }

    #[test]
    fn line_through_is_antisymmetric() {
        let p = Pga3::point(1.0, 2.0, 3.0);
        let q = Pga3::point(-4.0, 5.0, 0.5);
        let pq = Pga3::line_through(&p, &q);
        let qp = Pga3::line_through(&q, &p);
        assert_eq!(pq.coeffs, (-qp).coeffs);
    }

    #[test]
    fn three_collinear_points_join_to_zero() {
        // (0,0,0), (1,0,0), (2,0,0) all lie on the x-axis, so their
        // triple join (the plane they would span) vanishes.
        let a = Pga3::point(0.0, 0.0, 0.0);
        let b = Pga3::point(1.0, 0.0, 0.0);
        let c = Pga3::point(2.0, 0.0, 0.0);
        let join3 = a.line_through(&b).regressive(&c);
        assert_eq!(join3.cleaned(1e-10).coeffs, Pga3::zero().coeffs);
    }

    #[test]
    fn three_non_collinear_points_have_a_nonzero_join() {
        let a = Pga3::point(0.0, 0.0, 0.0);
        let b = Pga3::point(1.0, 0.0, 0.0);
        let c = Pga3::point(0.0, 1.0, 0.0);
        let join3 = a.line_through(&b).regressive(&c);
        assert_ne!(join3.cleaned(1e-10).coeffs, Pga3::zero().coeffs);
    }

    // --- PGA-aware display ----------------------------------------------

    #[test]
    fn display_names_the_null_generator_e0() {
        // basis(8) is the null generator, printed e0 (not the generic e4).
        assert_eq!(Pga3::basis(8).display_pga().to_string(), "e0");
    }

    #[test]
    fn display_folds_the_reorder_sign_into_the_coefficient() {
        // basis(9) is stored as e1∧e0; written e0-first it is −e01.
        assert_eq!(Pga3::basis(9).display_pga().to_string(), "-e01");
    }

    #[test]
    fn display_of_a_point_writes_e0_first_in_every_blade() {
        let p = Pga3::point(1.0, 2.0, 3.0);
        assert_eq!(p.display_pga().to_string(), "e123 - 3·e012 + 2·e013 - e023");
    }

    #[test]
    fn display_of_the_pseudoscalar_carries_its_reorder_sign() {
        // I = e0123 with the e0-first reorder sign for three crossings.
        assert_eq!(Pga3::pseudoscalar().display_pga().to_string(), "-e0123");
    }

    #[test]
    fn display_of_zero_is_zero() {
        assert_eq!(Pga3::zero().display_pga().to_string(), "0");
    }

    #[test]
    fn display_shows_the_scalar_term() {
        assert_eq!(Pga3::one().display_pga().to_string(), "1");
    }

    // --- Twist and wrench ------------------------------------------------

    #[test]
    fn twist_is_pure_grade_2() {
        let b = Pga3::twist(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        assert_eq!(b.grade(2).coeffs, b.coeffs);
    }

    #[test]
    fn wrench_is_pure_grade_2() {
        let w = Pga3::wrench(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        assert_eq!(w.grade(2).coeffs, w.coeffs);
    }

    #[test]
    fn twist_zero_angular_has_only_ideal_blades() {
        // No angular part → only e01, e02, e03 are nonzero.
        let b = Pga3::twist(1.0, 2.0, 3.0, 0.0, 0.0, 0.0);
        // Euclidean bivectors e23=6, e13=5, e12=3 must be zero.
        assert_eq!(b.coeffs[0b0110], 0.0);
        assert_eq!(b.coeffs[0b0101], 0.0);
        assert_eq!(b.coeffs[0b0011], 0.0);
    }

    #[test]
    fn twist_zero_velocity_has_only_euclidean_blades() {
        // No translational part → only e23, e31, e12 are nonzero.
        let b = Pga3::twist(0.0, 0.0, 0.0, 1.0, 2.0, 3.0);
        // Ideal bivectors e01=9, e02=10, e03=12 must be zero.
        assert_eq!(b.coeffs[9], 0.0);
        assert_eq!(b.coeffs[10], 0.0);
        assert_eq!(b.coeffs[12], 0.0);
    }

    #[test]
    fn twist_and_wrench_share_blade_layout() {
        // Same (vx,vy,vz,wx,wy,wz) in twist and (fx,fy,fz,tx,ty,tz) in wrench
        // should produce identical coefficient arrays when corresponding
        // arguments are equal.
        let b = Pga3::twist(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let w = Pga3::wrench(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        assert_eq!(b.coeffs, w.coeffs);
    }

    // --- twist_parts ------------------------------------------------------

    #[test]
    fn twist_parts_inverts_twist_exactly() {
        let (v, w) = ([1.0, -2.5, 3.25], [-4.0, 5.5, -6.125]);
        let b = Pga3::twist(v[0], v[1], v[2], w[0], w[1], w[2]);
        assert_eq!(b.twist_parts(), (v, w));
    }

    #[test]
    fn twist_parts_reads_only_the_bivector_slots() {
        // A point (grade 3) has nothing on the six grade-2 blades.
        assert_eq!(
            Pga3::point(1.0, 2.0, 3.0).twist_parts(),
            ([0.0; 3], [0.0; 3])
        );
    }

    proptest! {
        // The round-trip is exact: construction and extraction only place
        // and negate coefficients, so no floating-point error can appear.
        #[test]
        fn twist_parts_twist_round_trip_is_exact(
            vx in -1e6_f64..1e6, vy in -1e6_f64..1e6, vz in -1e6_f64..1e6,
            wx in -1e6_f64..1e6, wy in -1e6_f64..1e6, wz in -1e6_f64..1e6,
        ) {
            let b = Pga3::twist(vx, vy, vz, wx, wy, wz);
            prop_assert_eq!(b.twist_parts(), ([vx, vy, vz], [wx, wy, wz]));
        }
    }

    // --- Screw axis -------------------------------------------------------

    #[test]
    fn screw_axis_of_rotation_about_z_through_origin() {
        let s = Pga3::twist(0.0, 0.0, 0.0, 0.0, 0.0, 1.5)
            .screw_axis()
            .unwrap();
        assert_eq!(s.direction, [0.0, 0.0, 1.0]);
        assert_eq!(s.point, [0.0, 0.0, 0.0]);
        assert_eq!(s.angle_rate, 1.5);
        assert_eq!(s.pitch, 0.0);
    }

    #[test]
    fn screw_axis_of_offset_rotation_has_zero_pitch_exactly() {
        // Spin about the vertical line through (1, 0, 0): v = −w × p, so
        // the twist is simple and the split's ideal part is exactly zero.
        let s = Pga3::twist(0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
            .screw_axis()
            .unwrap();
        assert_eq!(s.direction, [0.0, 0.0, 1.0]);
        assert_eq!(s.point, [1.0, 0.0, 0.0]);
        assert_eq!(s.angle_rate, 1.0);
        assert_eq!(s.pitch, 0.0);
    }

    #[test]
    fn screw_axis_of_a_known_screw() {
        // Axis +z through (1, 0, 0), rate 1, pitch 0.5: v = −w×p + pitch·w.
        let s = Pga3::twist(0.0, -1.0, 0.5, 0.0, 0.0, 1.0)
            .screw_axis()
            .unwrap();
        for (got, want) in s.direction.iter().zip([0.0, 0.0, 1.0]) {
            assert!((got - want).abs() < 1e-12);
        }
        for (got, want) in s.point.iter().zip([1.0, 0.0, 0.0]) {
            assert!((got - want).abs() < 1e-12);
        }
        assert!((s.angle_rate - 1.0).abs() < 1e-12);
        assert!((s.pitch - 0.5).abs() < 1e-12);
    }

    #[test]
    fn screw_axis_of_pure_translation_is_none() {
        assert_eq!(
            Pga3::twist(1.0, -2.0, 3.0, 0.0, 0.0, 0.0).screw_axis(),
            None
        );
        assert_eq!(Pga3::zero().screw_axis(), None);
    }

    #[test]
    fn screw_axis_treats_a_negligible_euclidean_part_as_translation() {
        // The Euclidean part is 1e-26 of the ideal part: relatively ideal,
        // so no (absurdly distant) axis is invented.
        let b = Pga3::twist(1e6, 0.0, 0.0, 1e-20, 0.0, 0.0);
        assert_eq!(b.screw_axis(), None);
    }

    #[test]
    fn screw_axis_point_slides_along_the_axis() {
        // For any screw, the twist's velocity field w×x + v evaluated at
        // the returned axis point is pitch·angle_rate along direction —
        // pure slide, no swing.
        let b = Pga3::twist(0.7, -0.2, 0.4, 0.3, -1.1, 0.9);
        let (v, w) = b.twist_parts();
        let s = b.screw_axis().unwrap();
        let p = s.point;
        let vel = [
            w[1] * p[2] - w[2] * p[1] + v[0],
            w[2] * p[0] - w[0] * p[2] + v[1],
            w[0] * p[1] - w[1] * p[0] + v[2],
        ];
        for (got, dir) in vel.iter().zip(s.direction.iter()) {
            assert!((got - s.pitch * s.angle_rate * dir).abs() < 1e-12);
        }
    }
}
