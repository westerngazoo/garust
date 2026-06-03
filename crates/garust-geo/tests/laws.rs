//! Property-based **law tests** for the typed geometry layer.
//!
//! Where the kernel's `tests/laws.rs` checks the algebra, these check the
//! *transformations*: that a [`Motor`] (PGA rigid motion) and a
//! [`Conformal`] versor (CGA conformal map) form well-behaved groups —
//! composition is associative and agrees with sequential application,
//! inverses round-trip a point back to itself, and every generated versor
//! stays unit-norm. Versors are built from random translators, rotors, and
//! (for CGA) dilators, then exercised against `proptest`.

use garust_core::{Algebra, Cga3, Multivector, Pga3};
use garust_geo::{cga, pga, Conformal, Conformal3, Motor, Motor3};
use proptest::prelude::*;
use std::f64::consts::PI;

/// Coefficient-wise approximate equality, generic over the signature.
fn close<A: Algebra>(a: &Multivector<A, f64>, b: &Multivector<A, f64>) -> bool {
    (0..A::DIM).all(|i| {
        let (x, y) = (a.coeffs[i], b.coeffs[i]);
        (x - y).abs() <= 1e-7 + 1e-6 * x.abs().max(y.abs())
    })
}

/// Approximate equality of Euclidean coordinate triples.
fn xyz_close(got: (f64, f64, f64), want: (f64, f64, f64)) -> bool {
    let d = |a: f64, b: f64| (a - b).abs() <= 1e-7 + 1e-6 * a.abs().max(b.abs());
    d(got.0, want.0) && d(got.1, want.1) && d(got.2, want.2)
}

/// One of the three Euclidean basis bivectors of PGA — a valid rotation
/// plane (it squares to −1): `e1e2`, `e1e3`, `e2e3`.
fn pga_axis(choice: usize) -> Pga3 {
    Pga3::basis([0b0110, 0b1010, 0b1100][choice % 3])
}

/// One of the three Euclidean basis bivectors of CGA — `e1e2`, `e1e3`,
/// `e2e3` at indices 3, 5, 6 (the conformal directions sit in higher bits).
fn cga_axis(choice: usize) -> Cga3 {
    Cga3::basis([0b00011, 0b00101, 0b00110][choice % 3])
}

/// A random Euclidean point coordinate.
fn coord() -> impl Strategy<Value = f64> {
    -2.0f64..2.0
}

prop_compose! {
    // A random rigid motion: a rotation about an origin axis, then a
    // translation (a general screw motion once composed).
    fn any_motor()(
        dx in -2.0f64..2.0, dy in -2.0f64..2.0, dz in -2.0f64..2.0,
        angle in -PI..PI, axis in 0usize..3,
    ) -> Motor3 {
        Motor::translator(dx, dy, dz) * Motor::rotor(angle, pga_axis(axis))
    }
}

prop_compose! {
    // A random conformal map: a dilation, then a rotation, then a
    // translation, all about the origin.
    fn any_conformal()(
        dx in -2.0f64..2.0, dy in -2.0f64..2.0, dz in -2.0f64..2.0,
        angle in -PI..PI, axis in 0usize..3, scale in 0.25f64..4.0,
    ) -> Conformal3 {
        Conformal::translator(dx, dy, dz)
            * Conformal::rotor(angle, cga_axis(axis))
            * Conformal::dilator(scale)
    }
}

proptest! {
    // --- Motor (PGA rigid motions) --------------------------------------

    // (m1 m2) m3 = m1 (m2 m3) on the underlying versor.
    #[test]
    fn motor_composition_is_associative(m1 in any_motor(), m2 in any_motor(), m3 in any_motor()) {
        prop_assert!(close(&((m1 * m2) * m3).versor(), &(m1 * (m2 * m3)).versor()));
    }

    // Composing then applying equals applying in sequence.
    #[test]
    fn motor_compose_equals_sequential_apply(
        m1 in any_motor(), m2 in any_motor(),
        x in coord(), y in coord(), z in coord(),
    ) {
        let p = Pga3::point(x, y, z);
        prop_assert!(close(&(m1 * m2).apply(&p), &m1.apply(&m2.apply(&p))));
    }

    // A motor and its inverse round-trip a point back to itself.
    #[test]
    fn motor_inverse_round_trips_a_point(
        m in any_motor(), x in coord(), y in coord(), z in coord(),
    ) {
        let p = pga::Point::new(x, y, z);
        let back = p.transform(&m).transform(&m.inverse());
        prop_assert!(xyz_close(back.to_euclidean(), (x, y, z)));
    }

    // Every generated motor is a unit versor.
    #[test]
    fn motor_is_unit_norm(m in any_motor()) {
        prop_assert!((m.norm_squared() - 1.0).abs() < 1e-9);
    }

    // --- Conformal (CGA conformal maps) ---------------------------------

    // (c1 c2) c3 = c1 (c2 c3) on the underlying versor.
    #[test]
    fn conformal_composition_is_associative(
        c1 in any_conformal(), c2 in any_conformal(), c3 in any_conformal(),
    ) {
        prop_assert!(close(&((c1 * c2) * c3).versor(), &(c1 * (c2 * c3)).versor()));
    }

    // Composing then applying equals applying in sequence.
    #[test]
    fn conformal_compose_equals_sequential_apply(
        c1 in any_conformal(), c2 in any_conformal(),
        x in coord(), y in coord(), z in coord(),
    ) {
        let p = Cga3::cga_point(x, y, z);
        prop_assert!(close(&(c1 * c2).apply(&p), &c1.apply(&c2.apply(&p))));
    }

    // A conformal versor and its inverse round-trip a point back to itself.
    #[test]
    fn conformal_inverse_round_trips_a_point(
        c in any_conformal(), x in coord(), y in coord(), z in coord(),
    ) {
        let p = cga::Point::new(x, y, z);
        let back = p.transform(&c).transform(&c.inverse());
        prop_assert!(xyz_close(back.to_euclidean(), (x, y, z)));
    }

    // Every generated conformal versor is unit-norm.
    #[test]
    fn conformal_is_unit_norm(c in any_conformal()) {
        prop_assert!((c.norm_squared() - 1.0).abs() < 1e-9);
    }
}
