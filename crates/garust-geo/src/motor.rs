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

use garust_core::multivector::Multivector;
use garust_core::scalar::{Real, Scalar};
use garust_core::Pga3Sig;

/// The PGA multivector type a [`Motor`] wraps: `Cl(3, 0, 1)` over `T`.
type Pga<T> = Multivector<Pga3Sig, T>;

/// A rigid-body motion in 3D PGA — an even-grade versor of `Cl(3, 0, 1)`.
///
/// Construct one with [`Motor::identity`], [`Motor::translator`], or
/// [`Motor::rotor`]; compose with `*` (or [`Motor::compose`]); and move
/// geometry with [`Motor::apply`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(transparent)
)]
#[repr(transparent)]
#[must_use]
pub struct Motor<T: Scalar> {
    versor: Pga<T>,
}

/// Defaults to the identity motion (like `nalgebra::Isometry3::default()`).
impl<T: Scalar> Default for Motor<T> {
    fn default() -> Self {
        Self::identity()
    }
}

// SAFETY: `Motor` is `#[repr(transparent)]` over its `Pga<T>` versor, so it
// inherits that multivector's plain-old-data layout under the `bytemuck`
// feature — a motor is just its 16 scalar coefficients.
#[cfg(feature = "bytemuck")]
unsafe impl<T: Scalar> bytemuck::Zeroable for Motor<T> where Pga<T>: bytemuck::Zeroable {}
#[cfg(feature = "bytemuck")]
unsafe impl<T: Scalar + 'static> bytemuck::Pod for Motor<T> where Pga<T>: bytemuck::Pod {}

impl<T: Scalar> Motor<T> {
    /// The identity motion — leaves every object exactly where it is.
    pub fn identity() -> Self {
        Self {
            versor: Pga::<T>::one(),
        }
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

    /// Apply the motion to every object in `xs`, in place — the batch form
    /// of [`Motor::apply`] for moving a whole point cloud (or set of lines
    /// or planes) by one motor. The versor is reversed once and reused
    /// across the batch, so this is cheaper than calling [`Motor::apply`] in
    /// a loop, with identical results.
    pub fn apply_each(&self, xs: &mut [Pga<T>]) {
        self.versor.sandwich_each(xs);
    }

    /// Compose two motions: `self.compose(&rhs)` does `rhs` first, then
    /// `self`, exactly like function composition. Equals `self * rhs`.
    pub fn compose(&self, rhs: &Self) -> Self {
        Self {
            versor: self.versor * rhs.versor,
        }
    }

    /// The inverse motion, undoing this one. Built from the versor
    /// inverse, so `m.compose(&m.inverse())` is the identity.
    pub fn inverse(&self) -> Self {
        Self {
            versor: self.versor.versor_inverse(),
        }
    }

    /// `⟨M ~M⟩_0`. A unit motor (every rotor/translator and their
    /// products) has `norm_squared() == 1`.
    pub fn norm_squared(&self) -> T {
        self.versor.norm_squared()
    }

    /// The equivalent **column-major homogeneous 4×4 matrix** (`m[col][row]`),
    /// mapping a point `(x, y, z, 1)` to `Σ_c m[c] · [x, y, z, 1][c]` — the
    /// same Euclidean result as [`apply`](Motor::apply) on a point, as a plain
    /// matrix.
    ///
    /// This is the bridge to matrix throughput: keep poses as motors (no
    /// gimbal lock; clean [`compose`](Motor::compose) / [`log`](Motor::log) /
    /// [`slerp`](Motor::slerp)), convert **once per frame**, then run the bulk
    /// vertex transform as 4×4·vector — a handful of FLOPs per point, far
    /// cheaper than the sandwich for large clouds (see the `vs_nalgebra`
    /// benchmark). It is built from the images of the origin and the three
    /// basis directions, so it reproduces [`apply`](Motor::apply) exactly
    /// (proptested).
    ///
    /// Column-major matches GL and `nalgebra::Matrix4::from_column_slice`.
    pub fn to_matrix(&self) -> [[T; 4]; 4] {
        let img = |x, y, z| {
            crate::pga::Point::new(x, y, z)
                .transform(self)
                .to_euclidean()
        };
        let (z, o) = (T::ZERO, T::ONE);
        let (ox, oy, oz) = img(z, z, z); // image of the origin = translation
        let (xx, xy, xz) = img(o, z, z); // image of the x-axis unit point
        let (yx, yy, yz) = img(z, o, z);
        let (zx, zy, zz) = img(z, z, o);
        [
            [xx - ox, xy - oy, xz - oz, z], // col 0: x-axis direction
            [yx - ox, yy - oy, yz - oz, z], // col 1: y-axis direction
            [zx - ox, zy - oy, zz - oz, z], // col 2: z-axis direction
            [ox, oy, oz, o],                // col 3: translation
        ]
    }
}

#[cfg(feature = "simd")]
impl Motor<f64> {
    /// SIMD batch apply: transform every PGA object in `xs` in place, four
    /// objects per SIMD vector (structure-of-arrays). Behind the `simd`
    /// feature; bit-faithful to [`Motor::apply_each`] (the tail that doesn't
    /// fill a vector uses the scalar path), but several times the throughput
    /// for large point clouds.
    pub fn apply_each_simd(&self, xs: &mut [Pga<f64>]) {
        crate::simd::sandwich_each_pga(&self.versor, xs);
    }
}

impl Motor<f64> {
    /// Build a pure rotor from a unit quaternion `(w, x, y, z)`.
    ///
    /// The quaternion must be unit-norm. The resulting motor has no
    /// translation component — it is a pure rotation. Composition with
    /// a `Motor::translator` gives a full rigid-body pose.
    ///
    /// Blade mapping for Cl(3,0,1) (garust convention):
    ///   scalar ← w
    ///   e23    ← -x
    ///   e13    ← y
    ///   e12    ← -z
    pub fn from_unit_quaternion(w: f64, x: f64, y: f64, z: f64) -> Self {
        let mut versor = Pga::<f64>::zero();
        versor.coeffs[0] = w;
        versor.coeffs[0b0110] = -x;
        versor.coeffs[0b0101] = y;
        versor.coeffs[0b0011] = -z;
        Self { versor }
    }

    /// Fréchet (Riemannian) mean of a non-empty slice of motors.
    ///
    /// Iterates until the tangent-space residual norm drops below `tol`
    /// or `max_iter` is reached. Typical values: tol=1e-8, max_iter=20.
    ///
    /// # Panics
    /// Panics if `motors` is empty.
    pub fn frechet_mean(motors: &[Self], tol: f64, max_iter: usize) -> Self {
        assert!(
            !motors.is_empty(),
            "frechet_mean: motors slice cannot be empty"
        );
        let mut mu = motors[0];
        let n = motors.len() as f64;
        let n_inv = 1.0 / n;
        for _ in 0..max_iter {
            let mut b_bar = Pga::<f64>::zero();
            let mu_inv = mu.inverse();
            for m in motors {
                b_bar += mu_inv.compose(m).log();
            }
            b_bar = b_bar * n_inv;
            if b_bar.norm() < tol {
                break;
            }
            mu = mu.compose(&Self::from_versor(b_bar.exp()));
        }
        mu
    }

    /// Geodesic (bi-invariant) distance between two motors on the motor
    /// manifold.
    ///
    /// Defined as `‖log(self⁻¹ ∘ other)‖`. Returns 0 when `self == other`
    /// as motions, and is symmetric and satisfies the triangle inequality.
    pub fn geodesic_distance(&self, other: &Self) -> f64 {
        self.inverse().compose(other).log().norm()
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

    /// A rotation by `radians` about an arbitrary `line` in space.
    ///
    /// Unlike [`Motor::rotor`], which spins about an origin axis, this
    /// takes a full PGA line bivector — typically from
    /// [`Pga3::line_through`](garust_core::Multivector::line_through) — so
    /// the axis can be anywhere. The line is
    /// normalized internally, then `exp(−½·radians·L̂)` is the rotor
    /// about it. Points on the line are fixed; everything else swings
    /// around it. (Compose with a [`Motor::translator`] along the line
    /// to get a general screw motion.)
    ///
    /// `line` must be a genuine line (a 2-blade) with non-zero
    /// Euclidean weight, so that `L̂² = −1` and the closed-form
    /// exponential applies.
    pub fn rotation_about(line: Pga<T>, radians: T) -> Self {
        let unit = line.normalized();
        let versor = (unit * (radians * T::from_f64(-0.5))).exp();
        Self { versor }
    }

    /// The motor's logarithm: the **screw bivector** `B` (rotation plane +
    /// pitch translation, the motor's Lie-algebra coordinates) with
    /// `exp(B) = ±M`. Inverse of building a motor from
    /// [`rotor`](Motor::rotor) / [`translator`](Motor::translator) /
    /// [`rotation_about`](Motor::rotation_about) products; see
    /// [`Multivector::log`](garust_core::Multivector::log) for the
    /// principal-branch contract.
    pub fn log(&self) -> Pga<T> {
        self.versor.log()
    }

    /// Smooth screw interpolation between two motors — the motor "slerp".
    ///
    /// `t = 0` gives `self`, `t = 1` gives `other` (as motions; the versor
    /// may differ by the irrelevant overall sign), and in between the
    /// motion follows the constant-speed screw connecting them:
    ///
    /// ```text
    /// M(t) = exp(t · log(other ∘ self⁻¹)) ∘ self
    /// ```
    ///
    /// Because [`log`](Motor::log) folds the versor sign, the path takes
    /// the short way around — the motor analogue of quaternion slerp's
    /// antipodal flip. `t` outside `[0, 1]` extrapolates along the same
    /// screw.
    pub fn slerp(&self, other: &Self, t: T) -> Self {
        let delta = (other.versor * self.versor.versor_inverse()).log();
        Self {
            versor: (delta * t).exp() * self.versor,
        }
    }

    /// `norm_squared().sqrt()`. A unit motor has `norm() ≈ 1.0`.
    pub fn norm(&self) -> T {
        self.versor.norm()
    }

    /// Return a unit motor geometrically nearest to `self`.
    ///
    /// Uses the first-order approximation `M / √⟨M ~M⟩₀`, which is
    /// exact for rotors and translators and a good approximation for
    /// general motors close to unit norm. For severely denormalized
    /// motors, iterate.
    pub fn renormalize(&self) -> Self {
        Self {
            versor: self.versor.normalized(),
        }
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
    use garust_core::Pga3;
    use std::f64::consts::TAU;

    fn approx_eq(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tol, "index {i}: {x} vs {y}");
        }
    }

    #[test]
    fn from_unit_quaternion_identity() {
        let m = Motor::from_unit_quaternion(1.0, 0.0, 0.0, 0.0);
        assert_eq!(m, Motor::identity());
    }

    #[test]
    fn from_unit_quaternion_round_trip() {
        // Test rotations of 90 degrees (tau/4) about x, y, and z axes
        // The issue notes garust exp sign convention mapping.
        // Motor::rotor(theta, plane) creates exp(-theta/2 * plane)
        let s = std::f64::consts::FRAC_1_SQRT_2;
        // In quat (w, x, y, z), a 90 deg rotation around x is w=cos(45)=s, x=sin(45)=s.
        // from_unit_quaternion(s, s, 0, 0) maps to versor = s - s*e23.
        // Motor::rotor(90 deg, e23) computes exp(-45 deg * e23) = cos(-45) + sin(-45)*e23 = s - s*e23.
        let cases = [
            (s, s, 0.0, 0.0, Motor::rotor(TAU / 4.0, Pga3::basis(0b0110))), // x-axis
            (
                s,
                0.0,
                s,
                0.0,
                Motor::rotor(TAU / 4.0, -Pga3::basis(0b0101)),
            ), // y-axis
            (s, 0.0, 0.0, s, Motor::rotor(TAU / 4.0, Pga3::basis(0b0011))), // z-axis
            (
                -s,
                s,
                0.0,
                0.0,
                Motor::rotor(-TAU / 4.0, Pga3::basis(0b0110)),
            ), // -x-axis
        ];

        for (w, x, y, z, expected_m) in cases {
            let m = Motor::from_unit_quaternion(w, x, y, z);
            // compare their matrices to ensure they are the same rigid motion
            let m_mat = m.to_matrix();
            let exp_mat = expected_m.to_matrix();
            for c in 0..4 {
                approx_eq(&m_mat[c], &exp_mat[c], 1e-12);
            }
        }
    }

    #[test]
    fn default_motor_is_identity() {
        let p = Pga3::point(1.0, 2.0, 3.0);
        approx_eq(&Motor::default().apply(&p).coeffs, &p.coeffs, 1e-12);
    }

    #[test]
    fn to_matrix_reproduces_apply() {
        use crate::pga::Point;
        // A general screw motion (two rotations + a translation).
        let m = Motor::translator(1.0, -2.0, 0.5)
            * Motor::rotor(0.9, Pga3::basis(0b0110))
            * Motor::rotor(0.4, Pga3::basis(0b1010));
        let mat = m.to_matrix(); // column-major: mat[col][row]
        for &(x, y, z) in &[
            (1.0, 2.0, 3.0),
            (-1.0, 0.5, -2.0),
            (0.0, 0.0, 0.0),
            (4.0, -3.0, 1.0),
        ] {
            // result = x·col0 + y·col1 + z·col2 + col3
            let got = [
                mat[0][0] * x + mat[1][0] * y + mat[2][0] * z + mat[3][0],
                mat[0][1] * x + mat[1][1] * y + mat[2][1] * z + mat[3][1],
                mat[0][2] * x + mat[1][2] * y + mat[2][2] * z + mat[3][2],
            ];
            let (wx, wy, wz) = Point::new(x, y, z).transform(&m).to_euclidean();
            approx_eq(&got, &[wx, wy, wz], 1e-12);
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
        let r = Motor::rotor(TAU / 4.0, Pga3::basis(0b0110));
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
    fn norm_of_identity_and_rotor() {
        assert!((Motor::<f64>::identity().norm() - 1.0).abs() < 1e-12);
        let m = Motor::rotor(1.23, Pga3::basis(0b0101));
        assert!((m.norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn norm_squared_consistency() {
        let m = Motor::translator(5.0, -2.0, 0.5) * Motor::rotor(1.23, Pga3::basis(0b0101));
        let log_m = m.log();
        assert!((log_m.norm().powi(2) - log_m.norm_squared()).abs() < 1e-12);
    }

    #[test]
    fn frechet_mean_single() {
        let m = Motor::translator(1.0, 2.0, 3.0) * Motor::rotor(0.5, Pga3::basis(0b0110));
        let mean = Motor::frechet_mean(&[m], 1e-8, 20);
        approx_eq(&mean.versor().coeffs, &m.versor().coeffs, 1e-12);
    }

    #[test]
    fn frechet_mean_two_motors() {
        let m1 = Motor::translator(1.0, 0.0, 0.0);
        let m2 = Motor::translator(0.0, 2.0, 0.0) * Motor::rotor(0.8, Pga3::basis(0b0110));
        let mean = Motor::frechet_mean(&[m1, m2], 1e-8, 20);
        let slerp = m1.slerp(&m2, 0.5);

        let p = Pga3::point(1.0, 1.0, 1.0);
        approx_eq(&mean.apply(&p).coeffs, &slerp.apply(&p).coeffs, 1e-8);
    }

    #[test]
    fn frechet_mean_subgroup() {
        // N rotors in the same subgroup
        let plane = Pga3::basis(0b0110);
        let m1 = Motor::rotor(0.1, plane);
        let m2 = Motor::rotor(0.5, plane);
        let m3 = Motor::rotor(0.9, plane);
        let mean = Motor::frechet_mean(&[m1, m2, m3], 1e-8, 20);
        let expected = Motor::rotor((0.1 + 0.5 + 0.9) / 3.0, plane);
        approx_eq(&mean.versor().coeffs, &expected.versor().coeffs, 1e-8);
    }

    #[test]
    #[should_panic(expected = "motors slice cannot be empty")]
    fn frechet_mean_panics_on_empty() {
        let _: Motor<f64> = Motor::frechet_mean(&[], 1e-8, 20);
    }

    #[test]
    fn geodesic_distance_properties() {
        let a = Motor::translator(1.0, -2.0, 0.5) * Motor::rotor(0.9, Pga3::basis(0b0110));
        let b = Motor::translator(0.0, 1.0, -1.0) * Motor::rotor(1.2, Pga3::basis(0b1010));
        let c = Motor::translator(-1.0, 0.0, 2.0) * Motor::rotor(0.4, Pga3::basis(0b0011));

        // Self-distance is 0
        assert!(a.geodesic_distance(&a).abs() < 1e-12);

        // Symmetric
        assert!((a.geodesic_distance(&b) - b.geodesic_distance(&a)).abs() < 1e-12);

        // Triangle inequality
        assert!(
            a.geodesic_distance(&c) <= a.geodesic_distance(&b) + b.geodesic_distance(&c) + 1e-12
        );

        // Angle on principal range
        let theta = 0.5_f64;
        let plane = Pga3::basis(0b0110);
        let r = Motor::rotor(theta, plane);
        let id = Motor::identity();
        // Since rotor(theta, plane) builds exp(-theta/2 * plane),
        // the log is -theta/2 * plane. The norm is theta/2.
        assert!((id.geodesic_distance(&r) - theta / 2.0).abs() < 1e-12);
    }

    #[test]
    fn renormalize_fixed_point_and_drift() {
        let id = Motor::<f64>::identity();
        assert_eq!(id.renormalize(), id);

        let mut m = id;
        for _ in 0..1000 {
            // slightly non-unit rotor to simulate drift
            let r = Motor::rotor(0.1, Pga3::basis(0b0110));
            // add some small error to versor
            let mut v = r.versor();
            v.coeffs[0] += 1e-6;
            let m_drift = Motor::from_versor(v);
            m = m * m_drift;
        }

        let m_renorm = m.renormalize();
        assert!((m_renorm.norm_squared() - 1.0).abs() < 1e-12);

        // idempotence
        let m_renorm_twice = m_renorm.renormalize();
        approx_eq(
            &m_renorm.versor().coeffs,
            &m_renorm_twice.versor().coeffs,
            1e-12,
        );
    }

    #[test]
    fn compose_then_inverse_is_identity() {
        let m = Motor::translator(2.0, 3.0, -1.0) * Motor::rotor(0.9, Pga3::basis(0b0011));
        let round = m.compose(&m.inverse());
        // The round trip is the identity versor (the scalar 1).
        approx_eq(
            &round.versor().cleaned(1e-10).coeffs,
            &Pga3::one().coeffs,
            1e-10,
        );
    }

    #[test]
    fn screw_motion_matches_rotate_then_translate() {
        // Rotate (0,1,0) 90° about x → (0,0,1), then translate +3 in x
        // ⇒ (3,0,1). Order: M = T · R applies R first.
        let r = Motor::rotor(TAU / 4.0, Pga3::basis(0b0110));
        let t = Motor::translator(3.0, 0.0, 0.0);
        let m = t * r;
        let moved = m.apply(&Pga3::point(0.0, 1.0, 0.0)).cleaned(1e-10);
        approx_eq(&moved.coeffs, &Pga3::point(3.0, 0.0, 1.0).coeffs, 1e-12);
    }

    #[test]
    fn rotation_about_origin_line_rotates_in_the_yz_plane() {
        // 90° about the x-axis line sends (0,1,0) into the y→z plane:
        // x stays 0 and the result lands on the unit circle there.
        let axis = Pga3::point(0.0, 0.0, 0.0).line_through(&Pga3::point(1.0, 0.0, 0.0));
        let about = Motor::rotation_about(axis, TAU / 4.0);
        let moved = about.apply(&Pga3::point(0.0, 1.0, 0.0)).cleaned(1e-10);
        // Read Euclidean coords back out of the PGA point (weight is 1).
        let (x, y, z) = (-moved.coeffs[14], moved.coeffs[13], -moved.coeffs[11]);
        assert!(x.abs() < 1e-10, "x should stay 0, got {x}");
        assert!(y.abs() < 1e-10, "y should rotate away to 0, got {y}");
        assert!((z.abs() - 1.0).abs() < 1e-10, "|z| should be 1, got {z}");
    }

    #[test]
    fn rotation_about_off_origin_line_is_a_real_screw_axis() {
        // A 180° turn about the vertical line through (1,0,0) carries the
        // origin to (2,0,0) — the hallmark of an *off-origin* rotation
        // that a plain origin rotor cannot express.
        let axis = Pga3::point(1.0, 0.0, 0.0).line_through(&Pga3::point(1.0, 0.0, 1.0));
        let half_turn = Motor::rotation_about(axis, TAU / 2.0);
        let moved = half_turn.apply(&Pga3::point(0.0, 0.0, 0.0)).cleaned(1e-10);
        approx_eq(&moved.coeffs, &Pga3::point(2.0, 0.0, 0.0).coeffs, 1e-10);
    }

    #[test]
    fn points_on_the_axis_are_fixed() {
        let axis = Pga3::point(1.0, 0.0, 0.0).line_through(&Pga3::point(1.0, 0.0, 1.0));
        let m = Motor::rotation_about(axis, 0.9);
        // (1, 0, 0.5) lies on that vertical line, so it must not move.
        let on_axis = Pga3::point(1.0, 0.0, 0.5);
        let moved = m.apply(&on_axis).cleaned(1e-10);
        approx_eq(&moved.coeffs, &on_axis.coeffs, 1e-10);
    }

    #[test]
    fn composition_order_matters() {
        // Translate-then-rotate ≠ rotate-then-translate in general.
        // (The translation must not be along the rotation axis, or the
        // two would commute — here we rotate about x and translate in y.)
        let r = Motor::rotor(TAU / 4.0, Pga3::basis(0b0110)); // about x-axis
        let t = Motor::translator(0.0, 3.0, 0.0); // along y
        let tr = (t * r).apply(&Pga3::point(0.0, 1.0, 0.0)).cleaned(1e-10);
        let rt = (r * t).apply(&Pga3::point(0.0, 1.0, 0.0)).cleaned(1e-10);
        assert_ne!(tr.coeffs, rt.coeffs);
        // Spot-check the values: R-then-T gives (0,3,1); T-then-R gives (0,0,4).
        approx_eq(&tr.coeffs, &Pga3::point(0.0, 3.0, 1.0).coeffs, 1e-12);
        approx_eq(&rt.coeffs, &Pga3::point(0.0, 0.0, 4.0).coeffs, 1e-12);
    }

    #[test]
    fn apply_each_matches_apply_per_element() {
        let m = Motor::translator(1.0, -2.0, 0.5) * Motor::rotor(0.9, Pga3::basis(0b1010));
        let pts = [
            Pga3::point(1.0, 2.0, 3.0),
            Pga3::point(-1.0, 0.5, 2.0),
            Pga3::point(0.0, 0.0, 0.0),
        ];
        let mut batch = pts;
        m.apply_each(&mut batch);
        for (src, got) in pts.iter().zip(batch.iter()) {
            approx_eq(&got.coeffs, &m.apply(src).coeffs, 1e-12);
        }
    }

    #[test]
    fn log_recovers_the_screw_generator() {
        // A screw: rotate about x while translating along it. log must
        // return exactly the bivector exp was fed (principal range).
        let m = Motor::rotor(0.8, Pga3::basis(0b0110)) * Motor::translator(1.5, 0.0, 0.0);
        let b = m.log();
        approx_eq(
            &Motor::from_versor(b.exp()).versor().coeffs,
            &m.versor().coeffs,
            1e-12,
        );
    }

    #[test]
    fn slerp_hits_both_endpoints() {
        let a = Motor::rotor(0.7, Pga3::basis(0b0011)) * Motor::translator(0.0, 2.0, -1.0);
        let b = Motor::rotor(-0.4, Pga3::basis(0b0110)) * Motor::translator(3.0, 0.0, 0.5);
        let p = Pga3::point(1.0, -2.0, 0.25);
        // Compare as *motions* (apply to a point): the versor itself may
        // come back with the opposite, equivalent sign.
        approx_eq(
            &a.slerp(&b, 0.0).apply(&p).coeffs,
            &a.apply(&p).coeffs,
            1e-10,
        );
        approx_eq(
            &a.slerp(&b, 1.0).apply(&p).coeffs,
            &b.apply(&p).coeffs,
            1e-10,
        );
    }

    #[test]
    fn slerp_midpoint_of_translations_is_the_half_translation() {
        let a = Motor::translator(0.0, 0.0, 0.0);
        let b = Motor::translator(4.0, -2.0, 6.0);
        let mid = a.slerp(&b, 0.5);
        let moved = mid.apply(&Pga3::point(0.0, 0.0, 0.0)).cleaned(1e-10);
        approx_eq(&moved.coeffs, &Pga3::point(2.0, -1.0, 3.0).coeffs, 1e-12);
    }

    #[test]
    fn slerp_midpoint_of_rotations_is_the_half_angle() {
        let a = Motor::identity();
        let b = Motor::rotor(TAU / 4.0, Pga3::basis(0b0110));
        let mid = a.slerp(&b, 0.5);
        let expected = Motor::rotor(TAU / 8.0, Pga3::basis(0b0110));
        let p = Pga3::point(0.0, 1.0, 2.0);
        approx_eq(&mid.apply(&p).coeffs, &expected.apply(&p).coeffs, 1e-12);
    }

    #[test]
    fn slerp_of_screws_follows_a_constant_screw() {
        // Interpolating identity → screw must pass through the t-scaled
        // screw at every t (one-parameter subgroup property).
        let screw = Motor::rotor(1.0, Pga3::basis(0b0110)) * Motor::translator(2.0, 0.0, 0.0);
        let gen = screw.log();
        let p = Pga3::point(0.5, 1.0, -1.0);
        for &t in &[0.25, 0.5, 0.75] {
            let direct = Motor::from_versor((gen * t).exp());
            let lerped = Motor::identity().slerp(&screw, t);
            approx_eq(&lerped.apply(&p).coeffs, &direct.apply(&p).coeffs, 1e-10);
        }
    }
}
