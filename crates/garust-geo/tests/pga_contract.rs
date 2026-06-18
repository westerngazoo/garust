//! Conformance test for the **UFL ↔ garust PGA capability contract**
//! (`ufl/docs/garust-pga-contract.md`, UFL requirement R-0009).
//!
//! Each capability C1–C11 the contract pins is exercised here over real
//! `f64` `Pga3`, and the §3 invariants are asserted. If this test compiles
//! and passes, garust upholds the contract on the current commit — making the
//! promised surface a CI-enforced guarantee rather than a claim. Keep it green
//! across releases; a change that breaks it is a breaking change to the
//! contract and warrants a major-version bump.

use std::f64::consts::TAU;

use garust_core::{Algebra, Pga3, Pga3Sig, Real};
use garust_geo::{
    pga::{Plane, Point},
    Motor,
};

/// Coefficient-wise approximate equality (the products/exp are floating).
fn close(a: &Pga3, b: &Pga3) -> bool {
    (0..16).all(|i| (a.coeffs[i] - b.coeffs[i]).abs() < 1e-12)
}
fn close3(g: (f64, f64, f64), w: (f64, f64, f64)) -> bool {
    (g.0 - w.0).abs() < 1e-12 && (g.1 - w.1).abs() < 1e-12 && (g.2 - w.2).abs() < 1e-12
}

// C1 — Cl(3,0,1) PGA over f64, 16 blades. (§3.1)
#[test]
fn c1_pga3_is_cl301_16_blades() {
    assert_eq!(Pga3::zero().coeffs.len(), 16);
    assert_eq!((Pga3Sig::P, Pga3Sig::Q, Pga3Sig::R), (3, 0, 1));
    // e1,e2,e3 square to +1; the ideal generator e0 squares to 0 (degenerate).
    assert_eq!((Pga3::basis(1) * Pga3::basis(1)).scalar_part(), 1.0);
    assert_eq!((Pga3::basis(8) * Pga3::basis(8)).scalar_part(), 0.0);
}

// C2 — geometric product.
#[test]
fn c2_geometric_product() {
    // e1 e2 = e12; e2 e1 = −e12 (anticommute).
    assert_eq!(Pga3::basis(1) * Pga3::basis(2), Pga3::basis(3));
    assert_eq!(Pga3::basis(2) * Pga3::basis(1), -Pga3::basis(3));
}

// C3 — outer (wedge) and inner products.
#[test]
fn c3_wedge_and_inner() {
    let (e1, e2) = (Pga3::basis(1), Pga3::basis(2));
    assert_eq!(e1.wedge(&e2), Pga3::basis(3)); // e1 ∧ e2 = e12
    assert_eq!(e1.wedge(&e1), Pga3::zero()); // a ∧ a = 0
    assert_eq!(e1.inner(&e1).scalar_part(), 1.0); // e1 · e1 = 1
    assert_eq!(e1.inner(&e2), Pga3::zero()); // orthogonal
}

// C4 — grade projection ⟨·⟩ₖ. (§3.3)
#[test]
fn c4_grade_projection() {
    let m = Pga3::scalar(2.0) + Pga3::basis(1) + Pga3::basis(3); // grades 0,1,2
    assert_eq!(m.grade(0), Pga3::scalar(2.0));
    assert_eq!(m.grade(1), Pga3::basis(1));
    assert_eq!(m.grade(2), Pga3::basis(3));
}

// C5 — reverse, grade involution, conjugate.
#[test]
fn c5_involutions() {
    let e12 = Pga3::basis(3);
    assert_eq!(e12.reverse(), -e12); // ~ flips grade 2
    assert_eq!(Pga3::basis(1).grade_involution(), -Pga3::basis(1)); // ^ flips grade 1
    assert_eq!(e12.conjugate(), -e12); // conjugation of a bivector
}

// C6 — versor sandwich, and C7's exp: R = exp(−θ/2·B) rotates by θ. (§3.2)
#[test]
fn c6_c7_sandwich_rotates_by_theta() {
    let theta = TAU / 4.0; // quarter turn
    let r = (Pga3::basis(3) * (-theta / 2.0)).exp(); // exp(−θ/2 · e12), a unit rotor
    let rotated = r.sandwich(&Pga3::basis(1)); // rotate e1 in the e12 plane
    assert!(close(&rotated, &Pga3::basis(2))); // e1 → e2
}

// C7 — exp of a null bivector is a translator (truncates after one term).
#[test]
fn c7_exp_of_null_bivector() {
    let b = Pga3::basis(8) * Pga3::basis(1); // e0 e1, squares to 0
    assert_eq!((b * b).scalar_part(), 0.0);
    assert!(close(&(b * 0.5).exp(), &(Pga3::one() + b * 0.5)));
}

// C8 — norm, norm², normalize.
#[test]
fn c8_norms() {
    let v = Pga3::basis(1) * 3.0 + Pga3::basis(2) * 4.0; // 3e1 + 4e2
    assert_eq!(v.norm_squared(), 25.0);
    assert_eq!(v.norm(), 5.0);
    assert!(close(
        &v.normalized(),
        &(Pga3::basis(1) * 0.6 + Pga3::basis(2) * 0.8)
    ));
}

// C9 — rigid-body Motor: identity / translator / rotor / rotation_about,
// composition (`*`), and `apply`. (§3.2)
#[test]
fn c9_motor_rigid_motion() {
    // 90° about the x-axis (e23 plane): (0,1,0) → (0,0,1).
    let r = Motor::rotor(TAU / 4.0, Pga3::basis(0b0110));
    assert!(close3(
        Point::new(0.0, 1.0, 0.0).transform(&r).to_euclidean(),
        (0.0, 0.0, 1.0)
    ));

    // Compose translate-after-rotate, apply to a raw Pga point (C9's signature).
    let m = Motor::translator(3.0, 0.0, 0.0) * r;
    let moved = m.apply(&Pga3::point(0.0, 1.0, 0.0));
    assert!(close3(
        Point::from_multivector(moved).to_euclidean(),
        (3.0, 0.0, 1.0)
    ));

    // identity is a no-op; rotation_about a line exists and is a unit motor.
    assert_eq!(Motor::<f64>::identity(), Motor::<f64>::default());
    let axis: Pga3 = Point::new(1.0, 0.0, 0.0)
        .join(&Point::new(1.0, 0.0, 1.0))
        .multivector();
    let about = Motor::rotation_about(axis, 0.5);
    assert!((about.norm_squared() - 1.0).abs() < 1e-12);
}

// C10 — typed PGA geometry with join / meet incidence.
#[test]
fn c10_incidence() {
    // Three planes meet in their common point: x=1, y=2, z=3 → (1,2,3).
    let px = Plane::new(1.0, 0.0, 0.0, -1.0);
    let py = Plane::new(0.0, 1.0, 0.0, -2.0);
    let pz = Plane::new(0.0, 0.0, 1.0, -3.0);
    let point = px.meet(&py).meet(&pz); // (Plane ∨ Plane) ∨ Plane = Point
    assert!(close3(point.to_euclidean(), (1.0, 2.0, 3.0)));

    // Two points join into the line through them (the dual operation).
    let _line = Point::new(0.0, 0.0, 0.0).join(&Point::new(1.0, 0.0, 0.0));
}

// C11 — the real scalar trait with ordered ops; f64 implements it.
#[test]
fn c11_f64_is_real() {
    fn requires_real<T: Real>() {}
    requires_real::<f64>();
}
