//! # garust-geo — the typed geometry layer
//!
//! Where [`garust-core`](https://crates.io/crates/garust-core) provides the
//! raw [`Multivector`](garust_core::Multivector) and its constructors, this
//! crate gives the *transformations* a name and a small, total API:
//!
//! - [`Motor`] — a rigid-body motion in 3D PGA `Cl(3, 0, 1)` (rotors,
//!   translators, and their screw-motion compositions);
//! - [`Conformal`] — a conformal transformation in 3D CGA `Cl(4, 1, 0)`
//!   (the rigid motions plus uniform scaling about the origin).
//!
//! Both are thin newtypes over a core multivector: build the generators,
//! compose them with `*`, and apply them to geometry with the sandwich
//! product. Most users depend on the umbrella
//! [`garust`](https://crates.io/crates/garust) crate, which re-exports this
//! one alongside the core.

pub mod conformal;
pub mod motor;

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
