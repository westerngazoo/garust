//! The Lie bridge, round-tripped: `exp` from bivectors to versors and
//! `log` back again, across every shipped signature.
//!
//! The contract under test: `exp` is total; for any bivector `B` (below
//! six dimensions) `R = exp(B)` is a normalized versor, `R.log()` is a
//! bivector, and `exp(R.log()) = ±R` — equality up to the sign
//! double-cover, since `log` folds to `⟨R⟩₀ ≥ 0` (both signs act
//! identically in the sandwich).

use garust_core::signature::grade_of;
use garust_core::{Algebra, BladeStore, Cga3, Multivector, Pga3, Sta, Vga3};
use proptest::prelude::*;

/// Largest absolute coefficient difference between two multivectors.
fn max_diff<A: Algebra>(a: &Multivector<A, f64>, b: &Multivector<A, f64>) -> f64 {
    a.coeffs
        .as_slice()
        .iter()
        .zip(b.coeffs.as_slice().iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// `exp(log R)` must reproduce `R` up to the ± double-cover.
fn roundtrip_error<A: Algebra>(r: &Multivector<A, f64>) -> f64 {
    let back = r.log().exp();
    max_diff(r, &back).min(max_diff(&-*r, &back))
}

/// A random bivector with coefficients in `[-lim, lim]`.
fn any_bivector<A: Algebra>(lim: f64) -> impl Strategy<Value = Multivector<A, f64>>
where
    A::Blades<f64>: Default,
{
    let lanes: Vec<_> = (0..A::DIM)
        .map(|i| {
            if grade_of(i) == 2 {
                (-lim..lim).boxed()
            } else {
                Just(0.0).boxed()
            }
        })
        .collect();
    lanes.prop_map(|coeffs| {
        let mut m = Multivector::<A, f64>::zero();
        for (i, c) in coeffs.into_iter().enumerate() {
            m.coeffs[i] = c;
        }
        m
    })
}

/// Stamp the exp/log law-suite out for one signature.
///
/// `$rt_tol` is the `exp(log R) = ±R` round-trip tolerance. Only the
/// Euclidean `Vga3` round-trips to near machine precision; the degenerate
/// (`Pga3`), indefinite (`Sta`, with its hyperbolic boosts), and 32-blade
/// conformal (`Cga3`) algebras accumulate materially more floating-point
/// error through the exp/log pair (measured worst cases of ~4e-6, ~1.4e-6,
/// and ~2e-5 respectively over 200k random bivectors), so they get looser
/// bounds. The tolerances still catch order-of-magnitude regressions.
macro_rules! lie_laws {
    ($name:ident, $alg:ty, $rt_tol:expr) => {
        mod $name {
            use super::*;

            proptest! {
                // exp of a bivector is a normalized versor: R ~R = 1.
                #[test]
                fn exp_of_bivector_is_normalized_versor(b in any_bivector::<$alg>(1.2)) {
                    let r = b.exp();
                    let n = r * r.reverse();
                    prop_assert!((n.scalar_part() - 1.0).abs() < 1e-9);
                    for i in 1..<$alg>::DIM {
                        prop_assert!(n.coeffs[i].abs() < 1e-9, "blade {}: {}", i, n.coeffs[i]);
                    }
                }

                // log lands in grade 2 and exp(log R) = ±R.
                #[test]
                fn exp_log_roundtrip(b in any_bivector::<$alg>(1.2)) {
                    let r = b.exp();
                    let l = r.log();
                    for i in 0..<$alg>::DIM {
                        if grade_of(i) != 2 {
                            prop_assert!(l.coeffs[i].abs() < 1e-9);
                        }
                    }
                    prop_assert!(roundtrip_error(&r) < $rt_tol, "error {}", roundtrip_error(&r));
                }

                // The invariant decomposition really is one: parts sum to B,
                // commute, and square to scalars.
                #[test]
                fn bivector_split_is_invariant_decomposition(b in any_bivector::<$alg>(1.2)) {
                    if let Some((b1, b2)) = b.try_bivector_split() {
                        prop_assert!(max_diff(&(b1 + b2), &b) < 1e-9);
                        prop_assert!(max_diff(&(b1 * b2), &(b2 * b1)) < 1e-8);
                        for part in [b1, b2] {
                            let sq = part * part;
                            for i in 1..<$alg>::DIM {
                                prop_assert!(sq.coeffs[i].abs() < 1e-8,
                                    "part² blade {}: {}", i, sq.coeffs[i]);
                            }
                        }
                    }
                }
            }
        }
    };
}

lie_laws!(vga3, garust_core::Vga3Sig, 1e-7);
lie_laws!(pga3, garust_core::Pga3Sig, 1e-4);
lie_laws!(sta, garust_core::StaSig, 1e-4);
lie_laws!(cga3, garust_core::Cga3Sig, 1e-4);

// --- Targeted edge cases ---------------------------------------------------

/// A PGA screw (rotation + translation along the axis) round-trips through
/// log exactly — the physics-engine case.
#[test]
fn pga_screw_roundtrips() {
    let mut b = Pga3::zero();
    b.coeffs[0b0011] = 0.7; // rotation plane e12
    b.coeffs[0b1001] = 0.4; // ideal (translation) parts e0-ish
    b.coeffs[0b1010] = -0.2;
    let r = b.exp();
    assert!(((r * r.reverse()).scalar_part() - 1.0).abs() < 1e-12);
    assert!(roundtrip_error(&r) < 1e-12, "err {}", roundtrip_error(&r));
    // For this principal-range screw, log recovers the generator itself.
    assert!(max_diff(&r.log(), &b) < 1e-12);
}

/// A pure translator's log is the translation bivector itself (parabolic).
#[test]
fn pga_translator_log_is_exact() {
    let mut b = Pga3::zero();
    b.coeffs[0b1001] = 1.5;
    b.coeffs[0b1010] = -2.0;
    b.coeffs[0b1100] = 0.25;
    let r = b.exp();
    assert!(max_diff(&r.log(), &b) < 1e-12);
}

/// A half-turn rotor (`R = e12`, zero scalar part) sits exactly on the
/// quarter-angle singularity and must still log/exp cleanly.
#[test]
fn half_turn_rotor_roundtrips() {
    let r = Vga3::basis(1) * Vga3::basis(2);
    let l = r.log();
    // log(e12) = τ/4 · e12 (quarter-turn bivector angle = half-turn rotation).
    let mut expected = Vga3::zero();
    expected.coeffs[0b011] = core::f64::consts::TAU / 4.0;
    assert!(max_diff(&l, &expected) < 1e-12);
    assert!(roundtrip_error(&r) < 1e-12);
}

/// An STA boost (hyperbolic part) round-trips: log recovers rapidity.
#[test]
fn sta_boost_roundtrips() {
    let mut b = Sta::zero();
    b.coeffs[0b0011] = 0.9; // e1e2 in Cl(1,3): mixed-signature plane, B² > 0
    let r = b.exp();
    assert!(max_diff(&r.log(), &b) < 1e-12);
}

/// Isoclinic bivectors (equal-weight `e12 + e34`) have no unique split, so
/// exp takes the series path — check it against the hand-expanded product
/// of commuting rotors.
#[test]
fn isoclinic_exp_matches_closed_form() {
    let t = 0.8_f64;
    // basis() takes blade *indexes*: e12 = 0b0011, e34 = 0b1100.
    let e12 = Cga3::basis(0b0011);
    let e34 = Cga3::basis(0b1100);
    let b = (e12 + e34) * t;
    assert!(
        b.try_bivector_split().is_none(),
        "isoclinic must refuse to split"
    );
    let r = b.exp();
    let expected =
        (Cga3::scalar(t.cos()) + e12 * t.sin()) * (Cga3::scalar(t.cos()) + e34 * t.sin());
    assert!(
        max_diff(&r, &expected) < 1e-12,
        "err {}",
        max_diff(&r, &expected)
    );
}

/// The series fallback also makes exp total beyond bivectors: a non-simple
/// even element exponentiates to exp(s)·exp(B)·… consistency check against
/// a scalar + commuting-bivector closed form.
#[test]
fn exp_is_total_beyond_bivectors() {
    // X = s + B with B simple: exp(X) = e^s · exp(B), since scalars commute.
    let s = 0.3_f64;
    let b = Vga3::basis(1) * Vga3::basis(2) * 0.6;
    let x = Vga3::scalar(s) + b;
    let expected = b.exp() * s.exp();
    assert!(max_diff(&x.exp(), &expected) < 1e-12);
}

/// Large-angle bivectors still round-trip (up to double cover): the fold
/// returns the short-way generator, whose exp is ±R.
#[test]
fn large_angle_folds_to_short_way() {
    let theta = 2.0; // bivector angle beyond τ/4 ⇒ ⟨R⟩₀ < 0
    let b = Vga3::basis(1) * Vga3::basis(2) * theta;
    let r = b.exp();
    assert!(r.scalar_part() < 0.0, "test premise: scalar part negative");
    assert!(roundtrip_error(&r) < 1e-12);
    // The recovered angle is the short way: τ/2 − θ on the reversed plane.
    let l = r.log();
    let expected = core::f64::consts::TAU / 2.0 - theta;
    assert!((l.coeffs[0b011].abs() - expected).abs() < 1e-12);
}
