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
//! For a custom scalar type `S`, name the full generic form with the
//! signature marker: `Multivector::<Vga3Sig, S>`. Mint a marker for a
//! brand-new signature with [`define_algebra!`].
//!
//! ## What's implemented
//!
//! - [`Algebra`] — signatures reified as zero-sized marker types, so a
//!   multivector is `Multivector<A, T>` (no redundant `DIM` parameter);
//!   mint your own with [`define_algebra!`]
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
//! - [`Conformal`] — conformal transformations in CGA (translators,
//!   rotors, and dilations about the origin)

pub mod algebra;
pub mod cga;
pub mod conformal;
pub mod dual;
pub mod involutions;
pub mod motor;
pub mod multivector;
pub mod pga;
pub mod products;
pub mod scalar;
pub mod signature;
pub mod transform;

pub use algebra::{Algebra, BladeStore};
pub use conformal::Conformal;
pub use motor::Motor;
pub use multivector::Multivector;
pub use scalar::{Real, Scalar};

// Standard signature markers. Each is a zero-sized type implementing
// [`Algebra`]; the type aliases below pair them with a scalar. Downstream
// crates can mint their own with [`define_algebra!`](crate::define_algebra).
define_algebra!(
    /// Signature of 2D Euclidean GA `Cl(2, 0, 0)`.
    pub Vga2Sig = Cl(2, 0, 0)
);
define_algebra!(
    /// Signature of 3D Euclidean GA `Cl(3, 0, 0)`.
    pub Vga3Sig = Cl(3, 0, 0)
);
define_algebra!(
    /// Signature of 3D Projective GA `Cl(3, 0, 1)`.
    pub Pga3Sig = Cl(3, 0, 1)
);
define_algebra!(
    /// Signature of 3D Conformal GA `Cl(4, 1, 0)`.
    pub Cga3Sig = Cl(4, 1, 0)
);
define_algebra!(
    /// Signature of Spacetime Algebra `Cl(1, 3, 0)`.
    pub StaSig = Cl(1, 3, 0)
);

/// 2D Euclidean Geometric Algebra `Cl(2, 0, 0)` over `f64` — 4 blades.
pub type Vga2 = Multivector<Vga2Sig, f64>;
/// 3D Euclidean Geometric Algebra `Cl(3, 0, 0)` over `f64` — 8 blades.
pub type Vga3 = Multivector<Vga3Sig, f64>;
/// 3D Projective Geometric Algebra `Cl(3, 0, 1)` over `f64` — 16 blades.
pub type Pga3 = Multivector<Pga3Sig, f64>;
/// 3D Conformal Geometric Algebra `Cl(4, 1, 0)` over `f64` — 32 blades.
pub type Cga3 = Multivector<Cga3Sig, f64>;
/// Spacetime Algebra `Cl(1, 3, 0)` over `f64` — 16 blades, `(+, −, −, −)`.
pub type Sta = Multivector<StaSig, f64>;

/// 2D Euclidean Geometric Algebra `Cl(2, 0, 0)` over `f32` — 4 blades.
pub type Vga2f = Multivector<Vga2Sig, f32>;
/// 3D Euclidean Geometric Algebra `Cl(3, 0, 0)` over `f32` — 8 blades.
pub type Vga3f = Multivector<Vga3Sig, f32>;
/// 3D Projective Geometric Algebra `Cl(3, 0, 1)` over `f32` — 16 blades.
pub type Pga3f = Multivector<Pga3Sig, f32>;
/// 3D Conformal Geometric Algebra `Cl(4, 1, 0)` over `f32` — 32 blades.
pub type Cga3f = Multivector<Cga3Sig, f32>;
/// Spacetime Algebra `Cl(1, 3, 0)` over `f32` — 16 blades, `(+, −, −, −)`.
pub type Staf = Multivector<StaSig, f32>;

/// A rigid-body [`Motor`] in 3D PGA over `f64`.
pub type Motor3 = Motor<f64>;
/// A rigid-body [`Motor`] in 3D PGA over `f32`.
pub type Motor3f = Motor<f32>;

/// A [`Conformal`] transformation in 3D CGA over `f64`.
pub type Conformal3 = Conformal<f64>;
/// A [`Conformal`] transformation in 3D CGA over `f32`.
pub type Conformal3f = Conformal<f32>;
