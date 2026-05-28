//! # garust — Geometric Algebra for Rust
//!
//! A learning-oriented implementation of Geometric Algebra
//! (a.k.a. Clifford Algebra) over `f64`, generic in the signature so
//! you pick your algebra at the type level.
//!
//! ## The signature `Cl(P, Q, R)` in one paragraph
//!
//! An algebra is fixed by three integers: how many basis vectors square
//! to `+1` (that's `P`), how many square to `-1` (`Q`), and how many
//! square to `0` (`R`, the degenerate / null ones). From those
//! `N = P + Q + R` generators you build `2^N` basis *blades* by taking
//! every subset and wedging it together. So:
//!
//! - `Cl(2, 0, 0)` — 2D Euclidean — has 4 blades: `1, e1, e2, e12`
//! - `Cl(3, 0, 0)` — 3D Euclidean — has 8 blades
//! - `Cl(3, 0, 1)` — 3D Projective GA — has 16 blades; uniform
//!   representation of points, lines, planes, and rigid motions
//! - `Cl(4, 1, 0)` — 3D Conformal GA — has 32 blades; adds circles,
//!   spheres, and conformal transformations to the mix
//! - `Cl(1, 3, 0)` — Spacetime Algebra — has 16 blades; signature
//!   `(+, −, −, −)`
//!
//! ## What's here so far
//!
//! Just the bones:
//! - [`Multivector`] — the dense `[f64; 2^N]` element type
//! - addition, subtraction, negation, and equality
//!
//! No products yet — geometric, wedge, inner, and friends land next.

pub mod multivector;
pub mod signature;

pub use multivector::Multivector;

/// 2D Euclidean Geometric Algebra `Cl(2, 0, 0)` — 4 basis blades.
pub type Vga2 = Multivector<2, 0, 0, 4>;

/// 3D Euclidean Geometric Algebra `Cl(3, 0, 0)` — 8 basis blades.
pub type Vga3 = Multivector<3, 0, 0, 8>;

/// 3D Projective Geometric Algebra `Cl(3, 0, 1)` — 16 basis blades.
pub type Pga3 = Multivector<3, 0, 1, 16>;

/// 3D Conformal Geometric Algebra `Cl(4, 1, 0)` — 32 basis blades.
pub type Cga3 = Multivector<4, 1, 0, 32>;

/// Spacetime Algebra `Cl(1, 3, 0)` — 16 basis blades, `(+, −, −, −)`.
pub type Sta = Multivector<1, 3, 0, 16>;
