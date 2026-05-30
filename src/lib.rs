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
//! and `f64` are provided. Every algebra alias defaults to `f64`, so
//! `Vga3` means `Vga3<f64>`, but you can write `Vga3<f32>` for graphics
//! work or plug in your own scalar type.
//!
//! ```
//! use garust::Vga3;
//! let v = Vga3::<f32>::basis(1) + Vga3::<f32>::basis(2);
//! assert_eq!((v * v).scalar_part(), 2.0_f32);
//! ```
//!
//! ## What's implemented
//!
//! - [`Multivector`] — dense `[T; 2^N]` element type
//! - linear ops: add, sub, neg, scalar multiplication, equality
//! - the geometric product, plus wedge `∧`, inner `·`, scalar product
//! - grade projection, reverse, grade involution, Clifford conjugation
//! - versor inverse, the sandwich product, and a closed-form `exp`

pub mod involutions;
pub mod multivector;
pub mod products;
pub mod scalar;
pub mod signature;
pub mod transform;

pub use multivector::Multivector;
pub use scalar::{Real, Scalar};

/// 2D Euclidean Geometric Algebra `Cl(2, 0, 0)` — 4 basis blades.
pub type Vga2<T = f64> = Multivector<T, 2, 0, 0, 4>;

/// 3D Euclidean Geometric Algebra `Cl(3, 0, 0)` — 8 basis blades.
pub type Vga3<T = f64> = Multivector<T, 3, 0, 0, 8>;

/// 3D Projective Geometric Algebra `Cl(3, 0, 1)` — 16 basis blades.
pub type Pga3<T = f64> = Multivector<T, 3, 0, 1, 16>;

/// 3D Conformal Geometric Algebra `Cl(4, 1, 0)` — 32 basis blades.
pub type Cga3<T = f64> = Multivector<T, 4, 1, 0, 32>;

/// Spacetime Algebra `Cl(1, 3, 0)` — 16 basis blades, `(+, −, −, −)`.
pub type Sta<T = f64> = Multivector<T, 1, 3, 0, 16>;
