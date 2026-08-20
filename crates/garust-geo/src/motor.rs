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
use garust_core::signature::grade_of;
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

    /// Build a pure rotor from a **unit quaternion** `(w, x, y, z)`.
    ///
    /// Produces the same rotation as the standard quaternion formula
    /// `v' = q·v·q̄`. The blade mapping accounts for the sign conventions
    /// in garust's `Motor::rotor` / `exp(-θ/2·B)` convention:
    ///
    /// | quaternion | PGA coeff | index |
    /// |------------|-----------|-------|
    /// | `w`        | scalar    | 0     |
    /// | `x`        | -e23      | 6     |
    /// | `y`        | +e13      | 5     |
    /// | `z`        | -e12      | 3     |
    ///
    /// The resulting motor is a **pure rotation** with no translation
    /// component. Compose with [`Motor::translator`] to add a translation.
    ///
    /// # Precondition
    ///
    /// The input must be a unit quaternion (`w²+x²+y²+z² ≈ 1`).
    /// No normalisation is performed internally.
    ///
    /// ```
    /// use garust_geo::Motor;
    /// let id = Motor::<f64>::from_unit_quaternion(1.0, 0.0, 0.0, 0.0);
    /// assert_eq!(id, Motor::identity());
    /// ```
    pub fn from_unit_quaternion(w: T, x: T, y: T, z: T) -> Self {
        let mut v = Pga::<T>::zero();
        v.coeffs[0] = w;  // scalar
        v.coeffs[6] = -x; // e23 (sign matches exp(-θ/2·e23) convention)
        v.coeffs[5] = y;  // e13 (e13 plane is reversed relative to e23/e12)
        v.coeffs[3] = -z; // e12 (sign matches exp(-θ/2·e12) convention)
        Self { versor: v }
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

    /// Fast single-point sandwich kernel, klein-style.
    ///
    /// Computes `M · p · M̃` analytically — no 16×16 Cayley table traversal.
    /// Exploits the fact that the motor has 8 non-zero coefficients and a point
    /// has 4, reducing the double-product to closed-form rotation + translation
    /// formulas followed by a single `f64x4` matrix-column FMA step.
    ///
    /// **Precondition**: `self` must be a unit motor (`norm() ≈ 1`).
    ///
    /// The result is identical to
    /// ```ignore
    /// let p = Pga3::point(pt[0], pt[1], pt[2]);
    /// let [x, y, z] = Motor::apply_then_extract_xyz(&m, p);
    /// ```
    /// but bypasses the general table-driven path.
    ///
    /// # Motor coefficient layout (non-zero even-grade blades)
    ///
    /// | blade  | index | symbol |
    /// |--------|-------|--------|
    /// | scalar | 0     | a      |
    /// | e12    | 3     | b      |
    /// | e13    | 5     | c      |
    /// | e23    | 6     | d      |
    /// | e01    | 9     | e      |
    /// | e02    | 10    | f      |
    /// | e03    | 12    | g      |
    /// | e0123  | 15    | h      |
    pub fn apply_point_fast(&self, pt: [f64; 3]) -> [f64; 3] {
        use wide::f64x4;
        let v = &self.versor.coeffs;
        let (a, b, c, d) = (v[0], v[3], v[5], v[6]);
        let (e, f, g, h) = (v[9], v[10], v[12], v[15]);
        let (px, py, pz) = (pt[0], pt[1], pt[2]);

        // Quadratic terms of the rotor part.
        let (aa, bb, cc, dd) = (a * a, b * b, c * c, d * d);
        let (ab, ac, ad) = (a * b, a * c, a * d);
        let (bc, bd, cd) = (b * c, b * d, c * d);

        // Cross-terms with the dual (translation) part.
        let (ae, bf, cg, dh) = (a * e, b * f, c * g, d * h);
        let (be, af, dg, ch) = (b * e, a * f, d * g, c * h);
        let (ce, df, ga, bh) = (c * e, d * f, g * a, b * h);

        // Rotation matrix (unit-quaternion formula with q=(a,-d,c,-b)).
        let r00 = aa - bb - cc + dd;
        let r01 = 2.0 * (ab - cd);
        let r02 = 2.0 * (ac + bd);
        let r10 = -2.0 * (ab + cd);
        let r11 = aa - bb + cc - dd;
        let r12 = 2.0 * (ad - bc);
        let r20 = 2.0 * (bd - ac);
        let r21 = -2.0 * (ad + bc);
        let r22 = aa + bb - cc - dd;

        // Translation vector extracted from the dual bivector components.
        // Derived from t = 2·q_d·q̄_r using the motor ↔ dual-quaternion map,
        // where q_r = (a,-d,c,-b) and q_d = (h,e,f,g) for blade-indexed
        // motor coeffs (e01→e, e02→f, e03→g, e0123→h).
        let tx = 2.0 * (ae + bf + cg + dh);
        let ty = 2.0 * (af - be + dg - ch);
        let tz = 2.0 * (ga + bh - ce - df);

        // Matrix-column FMA: result = col0*px + col1*py + col2*pz + col3.
        // Packs (x', y', z', w'=1) into one f64x4 with no horizontal adds.
        let col0 = f64x4::from([r00, r10, r20, 0.0]);
        let col1 = f64x4::from([r01, r11, r21, 0.0]);
        let col2 = f64x4::from([r02, r12, r22, 0.0]);
        let col3 = f64x4::from([tx, ty, tz, 1.0]);

        let result = col0 * f64x4::splat(px)
            + col1 * f64x4::splat(py)
            + col2 * f64x4::splat(pz)
            + col3;

        let arr = result.to_array();
        [arr[0], arr[1], arr[2]]
    }
}

#[cfg(feature = "simd")]
impl Motor<f32> {
    /// SIMD batch apply: transform every PGA object in `xs` in place, eight
    /// objects per `f32x8` vector (structure-of-arrays). Behind the `simd`
    /// feature; bit-faithful to [`Motor::apply_each`] (tail uses scalar path),
    /// with ~2× the throughput of the `f64` path on 256-bit SIMD hardware.
    pub fn apply_each_simd(&self, xs: &mut [Pga<f32>]) {
        crate::simd::sandwich_each_pga_f32(&self.versor, xs);
    }
}

impl Motor<f64> {
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

    /// `√(norm_squared())`. A unit motor (every rotor, translator, and their
    /// composition) has `norm() ≈ 1.0`.
    ///
    /// ```
    /// use garust_geo::Motor;
    /// use garust_core::Pga3;
    /// use std::f64::consts::TAU;
    /// assert!((Motor::<f64>::identity().norm() - 1.0).abs() < 1e-12);
    /// assert!((Motor::rotor(TAU / 3.0, Pga3::basis(0b0110)).norm() - 1.0).abs() < 1e-12);
    /// ```
    pub fn norm(&self) -> T {
        self.versor.norm()
    }

    /// Return the unit motor geometrically nearest to `self`.
    ///
    /// Uses the first-order approximation `M / ‖M‖`, which is exact for
    /// rotors and translators and a good approximation for general motors
    /// that have drifted slightly from unit norm. For severely denormalised
    /// motors, iterate.
    ///
    /// ```
    /// use garust_geo::Motor;
    /// let id = Motor::<f64>::identity().renormalize();
    /// assert_eq!(id, Motor::identity());
    /// ```
    pub fn renormalize(&self) -> Self {
        Self {
            versor: self.versor.normalized(),
        }
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

    /// Exponential of a **screw bivector** — the inverse of
    /// [`log`](Motor::log), mapping Lie-algebra coordinates back to the
    /// motor: `Motor::exp(m.log())` reproduces `m` whenever `⟨m⟩₀ ≥ 0`,
    /// and `−m` — the same motion — otherwise (the log folds that sign
    /// away; see [`Multivector::log`](garust_core::Multivector::log)).
    ///
    /// # Generator contract
    ///
    /// `b` must be a PGA **bivector** (grade 2 only, debug-asserted): a
    /// Euclidean plane gives a rotor, a null (`e0`-containing) plane a
    /// translator, and a mixed screw bivector — anything
    /// [`log`](Motor::log) can return — a general screw motion. For such
    /// generators the result is always a unit motor. Anything else
    /// exponentiates fine in the kernel but is not a rigid motion, so it
    /// has no business being wrapped in a [`Motor`].
    ///
    /// This is [`Multivector::exp`](garust_core::Multivector::exp) +
    /// [`Motor::from_versor`] with the contract stated once, so
    /// log-space code (blending, filtering, integration) never has to
    /// leak the raw kernel type.
    ///
    /// ```
    /// use garust_geo::Motor;
    /// use garust_core::Pga3;
    /// let m = Motor::rotor(1.1, Pga3::basis(0b0110)) * Motor::translator(0.5, 0.0, -2.0);
    /// let back = Motor::exp(m.log());
    /// for i in 0..16 {
    ///     assert!((back.versor().coeffs[i] - m.versor().coeffs[i]).abs() < 1e-12);
    /// }
    /// ```
    pub fn exp(b: Pga<T>) -> Self {
        debug_assert!(
            (0..16).all(|i| grade_of(i) == 2 || b.coeffs[i] == T::ZERO),
            "Motor::exp: the generator must be a screw bivector (grade 2)",
        );
        Self { versor: b.exp() }
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
        self.slerp_from_generator(&self.screw_generator(other), t)
    }

    /// The **span generator** of the screw from `self` to `other`: the
    /// bivector `log(other ∘ self⁻¹)` that [`slerp`](Motor::slerp)
    /// exponentiates. Computing it is the expensive half of a slerp (a
    /// bivector split plus several dense 16×16 product traversals), and
    /// it is *constant over a keyframe span* — so an animation track can
    /// pay for it once per span and evaluate frames with
    /// [`slerp_from_generator`](Motor::slerp_from_generator), instead of
    /// re-deriving it every frame inside [`slerp`](Motor::slerp).
    ///
    /// `a.slerp_from_generator(&a.screw_generator(&b), t)` returns
    /// exactly — bit for bit — what `a.slerp(&b, t)` returns (`slerp` is
    /// literally that composition). The generator inherits `log`'s
    /// principal branch, so the span it encodes takes the short way and
    /// carries at most τ/2 of rotation; see [`slerp`](Motor::slerp).
    pub fn screw_generator(&self, other: &Self) -> Pga<T> {
        (other.versor * self.versor.versor_inverse()).log()
    }

    /// Evaluate the screw interpolation from `self` along a precomputed
    /// span generator: `exp(t·gen) ∘ self`, the cheap half of
    /// [`slerp`](Motor::slerp) with the
    /// [`screw_generator`](Motor::screw_generator) factored out.
    ///
    /// `gen` must be the generator of a span *starting at `self`*
    /// (normally `self.screw_generator(&other)`, cached while the span
    /// lasts); pairing it with any other start motor evaluates some
    /// other screw entirely. `t = 0` gives `self`, `t = 1` the span's
    /// end, and `t` outside `[0, 1]` extrapolates — exactly as in
    /// [`slerp`](Motor::slerp).
    pub fn slerp_from_generator(&self, gen: &Pga<T>, t: T) -> Self {
        Self {
            versor: (*gen * t).exp() * self.versor,
        }
    }

    /// [`slerp`](Motor::slerp) with `extra_turns` additional full turns
    /// wound about the span's screw axis — the escape hatch from the
    /// short-way fold for animation that must *sweep* a large rotation
    /// rather than merely arrive at one.
    ///
    /// The span generator is the same principal log that `slerp` uses;
    /// its elliptic (rotation) part is extended by `extra_turns · τ/2`
    /// of half-angle along its own unit bivector — the rotation about
    /// the screw axis, wherever that axis lies. A full turn about any
    /// line is the identity motion and the added term commutes with the
    /// generator, so the endpoints are untouched: `t = 0` still gives
    /// `self` and `t = 1` still gives `other` (as motions) while the
    /// path between them winds the extra turns. Positive `extra_turns`
    /// wind with the span's own rotation direction, negative against
    /// it, and `extra_turns = 0` is exactly — bit for bit —
    /// [`slerp`](Motor::slerp).
    ///
    /// # Degenerate spans
    ///
    /// A span with no rotation — a pure translation, identical
    /// rotations at both keys, or the τ-collapse where the fold leaves
    /// only rotation dust (numerically: a half-angle at or below
    /// `1e-9`) — has no canonical plane to wind around, so
    /// `extra_turns` is **ignored** and the result equals
    /// [`slerp`](Motor::slerp). Winding such a span is an authoring
    /// decision: put the plane into the keys (e.g. subdivided
    /// quarter-turn keyframes) instead.
    ///
    /// ```
    /// use garust_geo::Motor;
    /// use garust_core::Pga3;
    /// use std::f64::consts::TAU;
    /// let a = Motor::<f64>::identity();
    /// let b = Motor::rotor(TAU / 4.0, Pga3::basis(0b0011));
    /// // Half-way along the wound path: τ/8 plus half the extra turn.
    /// let mid = a.slerp_unwrapped(&b, 0.5, 1);
    /// let expect = Motor::rotor(TAU / 8.0 + TAU / 2.0, Pga3::basis(0b0011));
    /// let p = Pga3::point(1.0, 0.0, 0.0);
    /// let (got, want) = (mid.apply(&p), expect.apply(&p));
    /// for i in 0..16 {
    ///     assert!((got.coeffs[i] - want.coeffs[i]).abs() < 1e-9);
    /// }
    /// ```
    pub fn slerp_unwrapped(&self, other: &Self, t: T, extra_turns: i32) -> Self {
        let mut gen = self.screw_generator(other);
        if extra_turns != 0 {
            // A PGA screw bivector always splits into an elliptic part
            // (the rotation about the screw axis, squaring to −φ² with
            // φ the half-angle) plus a null translation part. `None`
            // (isoclinic) cannot happen in Cl(3,0,1): ⟨B²⟩₀ = −|E|² ≤ 0
            // and the grade-4 part of B² squares to 0, so the split's
            // discriminant only degenerates when the Euclidean part
            // vanishes — which it reports as "already simple" instead.
            // Fall back to the unmodified span defensively all the same.
            let (b1, b2) = gen
                .try_bivector_split()
                .unwrap_or((gen, Pga::<T>::zero()));
            let (sq1, sq2) = ((b1 * b1).scalar_part(), (b2 * b2).scalar_part());
            // The split's part order is arbitrary; the rotation is the
            // part with the strictly negative square.
            let (rot, sq) = if sq1 < sq2 { (b1, sq1) } else { (b2, sq2) };
            let phi = if sq < T::ZERO { (-sq).sqrt() } else { T::ZERO };
            if phi > T::from_f64(1e-9) {
                let winds = T::from_f64(0.5 * core::f64::consts::TAU * f64::from(extra_turns));
                gen += rot * (winds / phi);
            }
        }
        self.slerp_from_generator(&gen, t)
    }

    /// Evaluate the **Bézier curve on the motor manifold** defined by the
    /// control motors `ctrl` at parameter `t`, by de Casteljau over
    /// [`slerp`](Motor::slerp): each round replaces adjacent control pairs
    /// with their screw interpolant until one motor remains.
    ///
    /// Because every step is a geodesic blend of unit versors, the result
    /// is a **unit motor at every `t`** — no renormalization needed — and
    /// the endpoints are interpolated exactly (`t = 0` gives `ctrl[0]`,
    /// `t = 1` gives `ctrl[n-1]` as motions). Like `slerp`, `t` outside
    /// `[0, 1]` extrapolates. A single control motor is returned as-is;
    /// two controls reduce to plain `slerp` (RFC-013 R1).
    ///
    /// # Panics
    ///
    /// If `ctrl` is empty or holds more than 8 motors (fixed stack
    /// scratch; cubic — 4 controls — is the working case).
    pub fn bezier(ctrl: &[Self], t: T) -> Self {
        assert!(
            !ctrl.is_empty() && ctrl.len() <= 8,
            "Motor::bezier takes 1..=8 control motors"
        );
        let mut buf = [Self::identity(); 8];
        buf[..ctrl.len()].copy_from_slice(ctrl);
        let mut n = ctrl.len();
        while n > 1 {
            for i in 0..n - 1 {
                buf[i] = buf[i].slerp(&buf[i + 1], t);
            }
            n -= 1;
        }
        buf[0]
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

    /// Assert two motors are the same *motion* (their action on a probe
    /// point agrees) — the versor's overall sign is irrelevant.
    fn same_motion(a: &Motor<f64>, b: &Motor<f64>, tol: f64) {
        let p = Pga3::point(1.0, -0.5, 0.25);
        approx_eq(&a.apply(&p).coeffs, &b.apply(&p).coeffs, tol);
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

    // --- Slerp principal-branch semantics ------------------------------
    //
    // `log` folds the versor sign to ⟨R⟩₀ ≥ 0 before recovering the screw,
    // so a single slerp span always takes the *short way* and its rotation
    // is capped at a half turn (τ/2). These tests pin what that means at
    // the branch boundaries — none of it is an accident to be "fixed"
    // silently; see the `Motor::slerp` docs.

    #[test]
    fn slerp_just_under_half_turn_interpolates_forward() {
        // θ < τ/2: forward really is the short way — the span sweeps the
        // authored rotation, scaled by t.
        let theta = TAU / 2.0 - 1e-3;
        let plane = Pga3::basis(0b0011);
        let b = Motor::rotor(theta, plane);
        let id = Motor::identity();
        for &t in &[0.25, 0.5, 0.75, 1.0] {
            same_motion(&id.slerp(&b, t), &Motor::rotor(t * theta, plane), 1e-9);
        }
    }

    #[test]
    fn slerp_at_exactly_half_turn_direction_is_a_float_tie_break() {
        // rotor(τ/2, P) and rotor(−τ/2, P) are the SAME motion (a half
        // turn lands identically either way; the versors are ∓P). With
        // ⟨R⟩₀ = 0 there is no geometric short way: the branch is picked
        // by the float sign of the versor's coefficients — the scalar's
        // rounding dust (cos(τ/4) ≈ +6.1e-17 > 0, so no fold) and then
        // the bivector's sign. Net effect today: each versor slerps in
        // its *authored* direction, so identical end motions take
        // opposite midpoints.
        let plane = Pga3::basis(0b0011);
        let pos = Motor::rotor(TAU / 2.0, plane);
        let neg = Motor::rotor(-TAU / 2.0, plane);
        same_motion(&pos, &neg, 1e-12); // one motion...
        let id = Motor::identity();
        same_motion(&id.slerp(&pos, 0.5), &Motor::rotor(TAU / 4.0, plane), 1e-9);
        same_motion(&id.slerp(&neg, 0.5), &Motor::rotor(-TAU / 4.0, plane), 1e-9);
    }

    #[test]
    fn slerp_just_past_half_turn_flips_to_the_short_way() {
        // θ = τ/2 + ε: the short way is now *backwards* by τ − θ, so the
        // midpoint jumps to the far side of where a forward sweep would
        // sit — an ε-sized change of key crosses the branch cut. The
        // endpoint still agrees as a motion (θ − τ ≡ θ mod τ).
        let theta = TAU / 2.0 + 1e-3;
        let plane = Pga3::basis(0b0011);
        let b = Motor::rotor(theta, plane);
        let id = Motor::identity();
        same_motion(&id.slerp(&b, 0.5), &Motor::rotor((theta - TAU) / 2.0, plane), 1e-9);
        same_motion(&id.slerp(&b, 1.0), &b, 1e-9);
    }

    #[test]
    fn slerp_just_under_a_full_turn_plays_a_small_backwards_rotation() {
        // θ = τ − ε folds to a rotation by −ε: the pose at t = 1 is
        // right, but the sweep is a tiny reversal instead of an
        // almost-full turn. Subdivide keys to author the full sweep.
        let theta = TAU - 1e-3;
        let plane = Pga3::basis(0b0011);
        let b = Motor::rotor(theta, plane);
        let id = Motor::identity();
        for &t in &[0.5, 1.0] {
            same_motion(&id.slerp(&b, t), &Motor::rotor(t * (theta - TAU), plane), 1e-9);
        }
    }

    #[test]
    fn slerp_to_a_full_turn_collapses_to_the_identity() {
        // rotor(τ, P) is the identity motion carried by the versor −1;
        // the sign fold erases it, log ≈ 0, and the whole span
        // degenerates — every t returns (numerically) the identity.
        // This is why RFC-012 §3.5 must not author a full turn in one
        // span.
        let plane = Pga3::basis(0b0011);
        let full = Motor::rotor(TAU, plane);
        assert!((full.versor().coeffs[0] + 1.0).abs() < 1e-12, "versor is −1");
        let id = Motor::identity();
        for &t in &[0.25, 0.5, 0.75, 1.0] {
            same_motion(&id.slerp(&full, t), &id, 1e-9);
        }
    }

    #[test]
    fn slerp_span_rotation_never_exceeds_a_half_turn() {
        // The folded log caps the recovered half-angle at τ/4, i.e. a
        // single span sweeps at most τ/2 of rotation however much was
        // authored (|log| peaks at θ = τ/2, then *decreases* again).
        let plane = Pga3::basis(0b0011);
        for &theta in &[0.1, 1.0, 2.0, 3.0, TAU / 2.0, 4.0, 5.0, 6.0, TAU - 1e-6, TAU] {
            let n = Motor::rotor(theta, plane).log().norm();
            assert!(n <= TAU / 4.0 + 1e-9, "theta={theta}: |log| = {n}");
        }
    }

    // --- Motor::exp (the log's inverse) ---------------------------------

    #[test]
    fn exp_round_trips_translator_rotor_and_screw_logs() {
        let cases = [
            Motor::translator(1.5, -2.0, 0.5),
            Motor::rotor(2.0, Pga3::basis(0b0011)),
            Motor::translator(1.0, 2.0, 3.0) * Motor::rotor(1.1, Pga3::basis(0b0101)),
        ];
        for m in cases {
            approx_eq(&Motor::exp(m.log()).versor().coeffs, &m.versor().coeffs, 1e-12);
        }
    }

    #[test]
    fn exp_of_a_folded_log_flips_the_versor_sign_only() {
        // Past a half turn the log folds the versor sign, so exp returns
        // −versor: a different versor, the same motion.
        let m = Motor::rotor(TAU / 2.0 + 1.0, Pga3::basis(0b0011));
        let back = Motor::exp(m.log());
        approx_eq(&back.versor().coeffs, &(m.versor() * -1.0).coeffs, 1e-12);
        same_motion(&back, &m, 1e-12);
    }

    // --- Generator-cached slerp -----------------------------------------

    #[test]
    fn slerp_from_generator_replays_slerp_bit_for_bit() {
        let a = Motor::rotor(0.7, Pga3::basis(0b0011)) * Motor::translator(0.0, 2.0, -1.0);
        let b = Motor::rotor(-0.4, Pga3::basis(0b0110)) * Motor::translator(3.0, 0.0, 0.5);
        let gen = a.screw_generator(&b);
        for &t in &[-0.5, 0.0, 0.25, 0.5, 1.0, 1.5] {
            assert_eq!(a.slerp_from_generator(&gen, t), a.slerp(&b, t));
        }
    }

    // --- slerp_unwrapped (winding past the short-way fold) --------------

    #[test]
    fn slerp_unwrapped_zero_turns_is_slerp_bit_for_bit() {
        let a = Motor::rotor(0.7, Pga3::basis(0b0011)) * Motor::translator(0.0, 2.0, -1.0);
        let b = Motor::rotor(-0.4, Pga3::basis(0b0110)) * Motor::translator(3.0, 0.0, 0.5);
        for &t in &[0.0, 0.25, 0.5, 1.0] {
            assert_eq!(a.slerp_unwrapped(&b, t, 0), a.slerp(&b, t));
        }
    }

    #[test]
    fn slerp_unwrapped_sweeps_the_long_way() {
        // identity → just-under-half-turn with one extra turn: the path
        // sweeps θ + τ in the span's own direction and still lands on
        // the target.
        let theta = TAU / 2.0 - 1e-3;
        let plane = Pga3::basis(0b0011);
        let b = Motor::rotor(theta, plane);
        let id = Motor::identity();
        for &t in &[0.25, 0.5, 0.75] {
            same_motion(
                &id.slerp_unwrapped(&b, t, 1),
                &Motor::rotor(t * (theta + TAU), plane),
                1e-9,
            );
        }
        same_motion(&id.slerp_unwrapped(&b, 1.0, 1), &b, 1e-9);
    }

    #[test]
    fn slerp_unwrapped_negative_turns_wind_against_the_span() {
        let theta = 1.0;
        let plane = Pga3::basis(0b0011);
        let b = Motor::rotor(theta, plane);
        let id = Motor::identity();
        same_motion(
            &id.slerp_unwrapped(&b, 0.5, -1),
            &Motor::rotor((theta - TAU) / 2.0, plane),
            1e-9,
        );
        same_motion(&id.slerp_unwrapped(&b, 1.0, -1), &b, 1e-9);
    }

    #[test]
    fn slerp_unwrapped_winds_about_the_screw_axis_not_the_origin() {
        // An off-origin screw: the extra turn must wind about the span's
        // own axis (the vertical line through (1,0,0)) — winding about a
        // parallel origin plane would not commute with the generator and
        // would drag the midpoint somewhere else entirely.
        let axis = Pga3::point(1.0, 0.0, 0.0).line_through(&Pga3::point(1.0, 0.0, 1.0));
        let b = Motor::translator(0.0, 0.0, 0.7) * Motor::rotation_about(axis, 0.8);
        let id = Motor::identity();
        let mid = id.slerp_unwrapped(&b, 0.5, 1);
        let expected =
            Motor::translator(0.0, 0.0, 0.35) * Motor::rotation_about(axis, (0.8 + TAU) / 2.0);
        same_motion(&mid, &expected, 1e-9);
        same_motion(&id.slerp_unwrapped(&b, 1.0, 1), &b, 1e-9);
    }

    #[test]
    fn slerp_unwrapped_ignores_turns_on_a_rotation_free_span() {
        // No rotation, no canonical plane to wind around: extra turns are
        // ignored (the documented degenerate rule) rather than spun about
        // whatever numerical dust the log left behind.
        let id = Motor::identity();
        let tr = Motor::translator(1.0, 2.0, 3.0);
        assert_eq!(id.slerp_unwrapped(&tr, 0.5, 1), id.slerp(&tr, 0.5));
        // The τ-collapsed span leaves ~1e-16 of rotation dust — below
        // the threshold, so it must not wind either.
        let full = Motor::rotor(TAU, Pga3::basis(0b0011));
        assert_eq!(id.slerp_unwrapped(&full, 0.5, 1), id.slerp(&full, 0.5));
    }

    // --- Issue #32: Motor::from_unit_quaternion ---

    #[test]
    fn from_unit_quaternion_round_trip_via_matrix() {
        // Four distinct unit quaternions and expected 3×3 rotation blocks.
        // q = (cos θ/2, sin θ/2 · n̂) for rotations about each axis.
        let half = TAU / 8.0; // θ = 90°  →  cos(π/4), sin(π/4)
        let c = half.cos();
        let s = half.sin();
        let cases: &[(f64, f64, f64, f64, [f64; 3], [f64; 3])] = &[
            // rotation about x-axis 90°: e_y → e_z
            (c, s, 0.0, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            // rotation about y-axis 90°: e_z → e_x
            (c, 0.0, s, 0.0, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            // rotation about z-axis 90°: e_x → e_y
            (c, 0.0, 0.0, s, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            // identity (redundant sanity)
            (1.0, 0.0, 0.0, 0.0, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ];
        for &(w, x, y, z, inp, exp) in cases {
            let m = Motor::<f64>::from_unit_quaternion(w, x, y, z);
            let mat = m.to_matrix();
            // Apply the matrix to inp.
            let got = [
                mat[0][0] * inp[0] + mat[1][0] * inp[1] + mat[2][0] * inp[2],
                mat[0][1] * inp[0] + mat[1][1] * inp[1] + mat[2][1] * inp[2],
                mat[0][2] * inp[0] + mat[1][2] * inp[1] + mat[2][2] * inp[2],
            ];
            approx_eq(&got, &exp, 1e-12);
        }
    }

    // --- Issue #33: Motor::norm ---

    #[test]
    fn norm_of_identity_is_one() {
        assert!((Motor::<f64>::identity().norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn norm_of_any_rotor_is_one() {
        for &theta in &[0.0, 0.5, 1.0, TAU / 4.0] {
            let r = Motor::rotor(theta, Pga3::basis(0b0110));
            assert!((r.norm() - 1.0).abs() < 1e-12, "theta={theta} norm={}", r.norm());
        }
    }

    #[test]
    fn norm_consistency_with_norm_squared() {
        let m = Motor::translator(1.0, 2.0, 3.0) * Motor::rotor(0.7, Pga3::basis(0b0011));
        let n = m.norm();
        assert!((n * n - m.norm_squared()).abs() < 1e-12);
    }

    // --- Issue #35: Motor::renormalize ---

    #[test]
    fn renormalize_is_stable_fixed_point_for_identity() {
        let id = Motor::<f64>::identity().renormalize();
        assert_eq!(id, Motor::identity());
    }

    #[test]
    fn renormalize_restores_unit_norm_after_drift() {
        // Simulate drift by scaling the versor slightly.
        let m = Motor::translator(1.0, -2.0, 0.5) * Motor::rotor(1.1, Pga3::basis(0b0110));
        // Introduce artificial scale drift.
        let drifted = Motor::from_versor(m.versor() * 1.05);
        let fixed = drifted.renormalize();
        assert!((fixed.norm_squared() - 1.0).abs() < 1e-12, "norm² = {}", fixed.norm_squared());
    }

    #[test]
    fn renormalize_is_idempotent() {
        let m = Motor::translator(0.5, 1.5, -1.0) * Motor::rotor(0.3, Pga3::basis(0b0101));
        let drifted = Motor::from_versor(m.versor() * 1.03);
        let once = drifted.renormalize();
        let twice = once.renormalize();
        approx_eq(&once.versor().coeffs, &twice.versor().coeffs, 1e-12);
    }

    // --- Issue #34: Motor::frechet_mean ---

    #[test]
    fn frechet_mean_of_single_motor_is_identity_distance() {
        let m = Motor::translator(1.0, -1.0, 2.0);
        let mean = Motor::frechet_mean(&[m], 1e-8, 20);
        assert!(mean.geodesic_distance(&m) < 1e-8, "dist = {}", mean.geodesic_distance(&m));
    }

    #[test]
    fn frechet_mean_of_two_motors_matches_slerp_midpoint() {
        let a = Motor::identity();
        let b = Motor::rotor(TAU / 4.0, Pga3::basis(0b0110));
        let mean = Motor::frechet_mean(&[a, b], 1e-10, 30);
        let mid = a.slerp(&b, 0.5);
        let p = Pga3::point(0.0, 1.0, 0.0);
        approx_eq(&mean.apply(&p).coeffs, &mid.apply(&p).coeffs, 1e-8);
    }

    #[test]
    fn frechet_mean_in_one_parameter_subgroup() {
        // N rotors all at multiples of a unit step; mean ≈ the centroid angle.
        let angles = [0.1f64, 0.3, 0.5, 0.7, 0.9];
        let plane = Pga3::basis(0b0110);
        let motors: Vec<Motor<f64>> = angles.iter().map(|&a| Motor::rotor(a, plane)).collect();
        let mean = Motor::frechet_mean(&motors, 1e-10, 30);
        let avg_angle: f64 = angles.iter().sum::<f64>() / angles.len() as f64;
        let expected = Motor::rotor(avg_angle, plane);
        let p = Pga3::point(0.0, 1.0, 0.0);
        approx_eq(&mean.apply(&p).coeffs, &expected.apply(&p).coeffs, 1e-8);
    }

    // --- Issue #36: Motor::geodesic_distance ---

    #[test]
    fn geodesic_distance_self_is_zero() {
        let m = Motor::translator(2.0, -1.0, 3.0) * Motor::rotor(0.6, Pga3::basis(0b0110));
        assert!((m.geodesic_distance(&m)).abs() < 1e-12);
    }

    #[test]
    fn geodesic_distance_is_symmetric() {
        let a = Motor::translator(1.0, 0.0, 0.0) * Motor::rotor(0.4, Pga3::basis(0b0110));
        let b = Motor::translator(0.0, 2.0, -1.0) * Motor::rotor(0.8, Pga3::basis(0b0011));
        let da = a.geodesic_distance(&b);
        let db = b.geodesic_distance(&a);
        assert!((da - db).abs() < 1e-12, "a→b = {da}, b→a = {db}");
    }

    #[test]
    fn geodesic_distance_triangle_inequality() {
        let a = Motor::translator(1.0, 0.0, 0.0);
        let b = Motor::rotor(0.5, Pga3::basis(0b0110));
        let c = Motor::translator(0.0, 1.0, 0.0) * Motor::rotor(0.3, Pga3::basis(0b0011));
        let ab = a.geodesic_distance(&b);
        let bc = b.geodesic_distance(&c);
        let ac = a.geodesic_distance(&c);
        assert!(ac <= ab + bc + 1e-12, "triangle inequality: {ac} <= {ab} + {bc}");
    }
}

