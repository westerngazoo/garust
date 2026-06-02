//! # garust — Geometric Algebra for Rust
//!
//! A from-scratch, zero-dependency implementation of Geometric Algebra
//! (a.k.a. Clifford Algebra), generic in **both** the signature and the
//! scalar type so you pick your algebra and your numeric precision at
//! the type level.
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
//! ## Generic over the scalar type
//!
//! [`Multivector`] is parameterised by a coefficient type `T`
//! implementing [`Scalar`] (and [`Real`] for the exponential). `f32`
//! and `f64` are provided out of the box, and any ordered field — dual
//! numbers for autodiff, fixed-point types — can opt in by implementing
//! those traits.
//!
//! For ergonomics each algebra ships **two concrete aliases**: an `f64`
//! one ([`Vga3`], [`Pga3`], …) and an `f32` one with an `f` suffix
//! ([`Vga3f`], [`Pga3f`], …). Concrete aliases mean no turbofish is
//! needed at call sites:
//!
//! ```
//! use garust::{Vga3, Vga3f};
//!
//! // f64 — the everyday default
//! let v = Vga3::basis(1) + Vga3::basis(2);
//! assert_eq!((v * v).scalar_part(), 2.0_f64);
//!
//! // f32 — for graphics / GPU work
//! let w = Vga3f::basis(1) + Vga3f::basis(2);
//! assert_eq!((w * w).scalar_part(), 2.0_f32);
//! ```
//!
//! For a custom scalar type `S`, name the full generic form:
//! `Multivector::<S, 3, 0, 0, 8>`.
//!
//! ## What's implemented
//!
//! - [`Multivector`] — dense `[T; 2^N]` element type
//! - linear ops: add, sub, neg, scalar multiplication, equality
//! - the geometric product, plus wedge `∧`, inner `·`, scalar product
//! - grade projection, reverse, grade involution, Clifford conjugation
//! - `norm_squared`, the magnitude `norm`, and `normalized`
//! - versor inverse, the sandwich product, and a closed-form `exp`
//! - the pseudoscalar, metric-independent complements, and the
//!   regressive product `∨` (the *meet*, dual to the wedge's *join*)
//! - PGA geometric constructors for `Cl(3,0,1)`: `point`, `plane`,
//!   `line_through`, with meet/join doing real incidence geometry
//! - [`Motor`] — rigid-body motions in PGA (rotors, translators, and
//!   their screw-motion compositions)
//! - CGA geometric constructors for `Cl(4,1,0)`: `cga_point`, `sphere`,
//!   `cga_plane`, on the null-cone conformal model

pub mod cga;
pub mod dual;
pub mod involutions;
pub mod motor;
pub mod multivector;
pub mod pga;
pub mod products;
pub mod scalar;
pub mod signature;
pub mod transform;

pub use motor::Motor;
pub use multivector::Multivector;
pub use scalar::{Real, Scalar};

/// 2D Euclidean Geometric Algebra `Cl(2, 0, 0)` over `f64` — 4 blades.
pub type Vga2 = Multivector<f64, 2, 0, 0, 4>;
/// 3D Euclidean Geometric Algebra `Cl(3, 0, 0)` over `f64` — 8 blades.
pub type Vga3 = Multivector<f64, 3, 0, 0, 8>;
/// 3D Projective Geometric Algebra `Cl(3, 0, 1)` over `f64` — 16 blades.
pub type Pga3 = Multivector<f64, 3, 0, 1, 16>;
/// 3D Conformal Geometric Algebra `Cl(4, 1, 0)` over `f64` — 32 blades.
pub type Cga3 = Multivector<f64, 4, 1, 0, 32>;
/// Spacetime Algebra `Cl(1, 3, 0)` over `f64` — 16 blades, `(+, −, −, −)`.
pub type Sta = Multivector<f64, 1, 3, 0, 16>;

/// 2D Euclidean Geometric Algebra `Cl(2, 0, 0)` over `f32` — 4 blades.
pub type Vga2f = Multivector<f32, 2, 0, 0, 4>;
/// 3D Euclidean Geometric Algebra `Cl(3, 0, 0)` over `f32` — 8 blades.
pub type Vga3f = Multivector<f32, 3, 0, 0, 8>;
/// 3D Projective Geometric Algebra `Cl(3, 0, 1)` over `f32` — 16 blades.
pub type Pga3f = Multivector<f32, 3, 0, 1, 16>;
/// 3D Conformal Geometric Algebra `Cl(4, 1, 0)` over `f32` — 32 blades.
pub type Cga3f = Multivector<f32, 4, 1, 0, 32>;
/// Spacetime Algebra `Cl(1, 3, 0)` over `f32` — 16 blades, `(+, −, −, −)`.
pub type Staf = Multivector<f32, 1, 3, 0, 16>;

/// A rigid-body [`Motor`] in 3D PGA over `f64`.
pub type Motor3 = Motor<f64>;
/// A rigid-body [`Motor`] in 3D PGA over `f32`.
pub type Motor3f = Motor<f32>;
