//! # garust-geo — the typed geometry layer
//!
//! Where [`garust-core`](https://crates.io/crates/garust-core) provides the
//! raw [`Multivector`](garust_core::Multivector) and its constructors, this
//! crate gives the *transformations* a name and a small, total API:
//!
//! - [`Motor`] — a rigid-body motion in 3D PGA `Cl(3, 0, 1)` (rotors,
//!   translators, and their screw-motion compositions);
//! - [`Conformal`] — a conformal transformation in 3D CGA `Cl(4, 1, 0)`
//!   (the rigid motions plus uniform scaling about the origin);
//! - the [`pga`] module — typed PGA geometry ([`pga::Point`],
//!   [`pga::Line`], [`pga::Plane`]) with type-checked join/meet incidence;
//! - the [`cga`] module — typed CGA geometry ([`cga::Point`],
//!   [`cga::Sphere`], [`cga::Plane`]) with type-checked incidence tests.
//!
//! The transforms are thin newtypes over a core multivector: build the
//! generators, compose them with `*`, and apply them to geometry with the
//! sandwich product. Most users depend on the umbrella
//! [`garust`](https://crates.io/crates/garust) crate, which re-exports this
//! one alongside the core.

// `no_std` by default (test builds get `std`). The typed layer calls only
// `garust-core` and `core`, never the standard library directly, so the
// `std` / `libm` features just forward to the kernel's math backend.
#![cfg_attr(not(test), no_std)]
#![deny(missing_docs)]

pub mod cga;
pub mod chain;
pub mod conformal;
pub mod motor;
pub mod pga;

// SoA SIMD batch transforms, behind the `simd` feature. Private: it only
// backs the `apply_each_simd` methods on `Motor`/`Conformal`.
#[cfg(feature = "simd")]
mod simd;

pub use conformal::Conformal;
pub use motor::Motor;

/// A rigid-body [`Motor`] in 3D PGA over `f64`.
pub type Motor3 = Motor<f64>;
/// A rigid-body [`Motor`] in 3D PGA over `f32`.
pub type Motor3f = Motor<f32>;

/// A [`Conformal`] transformation in 3D CGA over `f64`.
pub type Conformal3 = Conformal<f64>;
/// A [`Conformal`] transformation in 3D CGA over `f32`.
pub type Conformal3f = Conformal<f32>;

// The `#[derive(Algebra)]` macro is exercised from *this* crate on purpose:
// garust-geo depends on garust-core by name, so the derive must resolve and
// emit a cross-crate `::garust_core` path (the realistic consumer case),
// which a test inside garust-core itself could not check.
#[cfg(all(test, feature = "derive"))]
mod derive_tests {
    use garust_core::{Algebra, Multivector};

    /// A marker the user writes by hand, with the `Algebra` impl derived.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Algebra)]
    #[algebra(p = 3, q = 0, r = 1)]
    struct DerivedPga;

    #[test]
    fn derive_yields_the_same_signature_as_define_algebra() {
        assert_eq!(<DerivedPga as Algebra>::P, 3);
        assert_eq!(<DerivedPga as Algebra>::Q, 0);
        assert_eq!(<DerivedPga as Algebra>::R, 1);
        assert_eq!(<DerivedPga as Algebra>::N, 4);
        assert_eq!(<DerivedPga as Algebra>::DIM, 16);

        // The derived storage has 2^4 = 16 slots and the kernel runs on it.
        let zero = Multivector::<DerivedPga, f64>::zero();
        assert_eq!(zero.coeffs.len(), 16);
        let e1 = Multivector::<DerivedPga, f64>::basis(1);
        assert_eq!((e1 * e1).scalar_part(), 1.0);
    }

    #[test]
    fn omitted_q_and_r_default_to_zero() {
        #[derive(Clone, Copy, Debug, Algebra)]
        #[algebra(p = 2)]
        struct Euclidean2;

        assert_eq!(<Euclidean2 as Algebra>::Q, 0);
        assert_eq!(<Euclidean2 as Algebra>::R, 0);
        assert_eq!(<Euclidean2 as Algebra>::DIM, 4);
    }
}

// The typed objects are `#[serde(transparent)]` newtypes, so each (de)
// serializes exactly as its underlying multivector — a bare coefficient
// array. These round-trips confirm the geometry layer rides on the kernel's
// `serde` impls cleanly across PGA and CGA.
#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use crate::{cga, pga, Conformal, Motor};
    use garust_core::Pga3;
    use std::f64::consts::TAU;

    #[test]
    fn motor_round_trips_through_json() {
        let m = Motor::translator(1.0, 2.0, 3.0) * Motor::rotor(TAU / 4.0, Pga3::basis(0b0110));
        let json = serde_json::to_string(&m).unwrap();
        // Transparent: a motor is its bare versor (a flat array, not an object).
        assert!(json.starts_with('['));
        let back: Motor<f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn conformal_round_trips_through_json() {
        let c = Conformal::translator(1.0, 0.0, -2.0) * Conformal::dilator(2.0);
        let back: Conformal<f64> =
            serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn typed_pga_and_cga_objects_round_trip() {
        let p: pga::Point = pga::Point::new(1.0, 2.0, 3.0);
        let back: pga::Point = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!(back, p);

        let s: cga::Sphere = cga::Sphere::new(0.0, 0.0, 0.0, 1.0);
        let back: cga::Sphere = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(back, s);
    }
}

// Each typed object is `#[repr(transparent)]` over its multivector, so under
// the `bytemuck` feature it is `Pod` and a slice of them reinterprets as a
// flat scalar buffer — the shape a GPU upload wants.
#[cfg(all(test, feature = "bytemuck"))]
mod bytemuck_tests {
    use crate::{cga, pga, Motor};
    use garust_core::Pga3;

    #[test]
    fn motor_round_trips_through_bytes() {
        let m = Motor::translator(1.0, 2.0, 3.0) * Motor::rotor(0.7, Pga3::basis(0b0110));
        let bytes = bytemuck::bytes_of(&m);
        let back: Motor<f64> = *bytemuck::from_bytes(bytes);
        assert_eq!(back, m);
    }

    #[test]
    fn object_buffers_cast_to_flat_scalars() {
        // A PGA point cloud views as a flat f64 buffer — 16 scalars per point.
        let points = [
            pga::Point::new(1.0, 2.0, 3.0),
            pga::Point::new(4.0, 5.0, 6.0),
        ];
        let flat: &[f64] = bytemuck::cast_slice(&points);
        assert_eq!(flat.len(), 32);

        // CGA objects are 32-wide; confirm the larger signature casts too.
        let spheres = [cga::Sphere::new(0.0, 0.0, 0.0, 1.0)];
        let flat: &[f64] = bytemuck::cast_slice(&spheres);
        assert_eq!(flat.len(), 32);
    }
}

// The SoA SIMD batch transforms must match the scalar batch (the slow,
// obviously-correct path) within floating-point tolerance — a non-multiple-
// of-4/8 batch also exercises the scalar tail.
#[cfg(all(test, feature = "simd"))]
mod simd_tests {
    use crate::{Conformal, Motor};
    use garust_core::{Cga3, Cga3f, Pga3, Pga3f};
    use std::f32::consts::TAU as TAU_F32;
    use std::f64::consts::TAU;

    #[test]
    fn motor_apply_each_simd_matches_scalar() {
        let m = Motor::translator(1.0, -2.0, 0.5) * Motor::rotor(TAU / 8.0, Pga3::basis(0b0110));
        // 10 points: two full SIMD chunks of 4 plus a remainder of 2.
        let pts: Vec<_> = (0..10)
            .map(|i| Pga3::point(i as f64, (i % 3) as f64 - 1.0, -(i as f64) * 0.5))
            .collect();
        let mut simd = pts.clone();
        let mut scalar = pts;
        m.apply_each_simd(&mut simd);
        m.apply_each(&mut scalar);
        for (a, b) in simd.iter().zip(scalar.iter()) {
            for k in 0..16 {
                assert!(
                    (a.coeffs[k] - b.coeffs[k]).abs() < 1e-12,
                    "coeff {k}: {} vs {}",
                    a.coeffs[k],
                    b.coeffs[k]
                );
            }
        }
    }

    #[test]
    fn conformal_apply_each_simd_matches_scalar() {
        let v = Conformal::translator(1.0, 0.0, -2.0)
            * Conformal::rotor(TAU / 8.0, Cga3::basis(0b0110))
            * Conformal::dilator(1.5);
        let pts: Vec<_> = (0..10)
            .map(|i| Cga3::cga_point(i as f64 * 0.3, (i % 4) as f64, 1.0 - i as f64))
            .collect();
        let mut simd = pts.clone();
        let mut scalar = pts;
        v.apply_each_simd(&mut simd);
        v.apply_each(&mut scalar);
        for (a, b) in simd.iter().zip(scalar.iter()) {
            for k in 0..32 {
                assert!((a.coeffs[k] - b.coeffs[k]).abs() < 1e-12, "coeff {k}");
            }
        }
    }

    // ── f32 paths (8-lane) ──────────────────────────────────────────────────

    #[test]
    fn motor_f32_apply_each_simd_matches_scalar() {
        // 17 points: two full f32x8 chunks of 8 plus a tail of 1.
        let m: Motor<f32> =
            Motor::translator(1.0, -2.0, 0.5) * Motor::rotor(TAU_F32 / 8.0, Pga3f::basis(0b0110));
        let pts: Vec<Pga3f> = (0..17)
            .map(|i| Pga3f::point(i as f32, (i % 3) as f32 - 1.0, -(i as f32) * 0.5))
            .collect();
        let mut simd = pts.clone();
        let mut scalar = pts;
        m.apply_each_simd(&mut simd);
        m.apply_each(&mut scalar);
        for (a, b) in simd.iter().zip(scalar.iter()) {
            for k in 0..16 {
                assert!(
                    (a.coeffs[k] - b.coeffs[k]).abs() < 1e-5,
                    "coeff {k}: {} vs {}",
                    a.coeffs[k],
                    b.coeffs[k]
                );
            }
        }
    }

    #[test]
    fn conformal_f32_apply_each_simd_matches_scalar() {
        // 17 points: two full f32x8 chunks of 8 plus a tail of 1.
        let v: Conformal<f32> = Conformal::translator(1.0, 0.0, -2.0)
            * Conformal::rotor(TAU_F32 / 8.0, Cga3f::basis(0b0110))
            * Conformal::dilator(1.5);
        let pts: Vec<Cga3f> = (0..17)
            .map(|i| Cga3f::cga_point(i as f32 * 0.3, (i % 4) as f32, 1.0 - i as f32))
            .collect();
        let mut simd = pts.clone();
        let mut scalar = pts;
        v.apply_each_simd(&mut simd);
        v.apply_each(&mut scalar);
        for (a, b) in simd.iter().zip(scalar.iter()) {
            for k in 0..32 {
                assert!(
                    (a.coeffs[k] - b.coeffs[k]).abs() < 1e-5,
                    "coeff {k}: {} vs {}",
                    a.coeffs[k],
                    b.coeffs[k]
                );
            }
        }
    }
}
