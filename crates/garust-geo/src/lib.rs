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

pub mod cga;
pub mod conformal;
pub mod motor;
pub mod pga;

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
