//! # garust-core — the signature-generic geometric-algebra kernel
//!
//! This is the foundation crate of [garust](https://crates.io/crates/garust):
//! the part that knows nothing about any *particular* geometry, only about
//! Clifford algebras `Cl(P, Q, R)` in the abstract. It provides
//!
//! - [`Algebra`] — a signature reified as a zero-sized marker type, with
//!   [`BladeStore`] as the coefficient-storage seam; mint your own with
//!   [`define_algebra!`];
//! - [`Multivector`] — the dense `[T; 2^N]` element, generic over the
//!   signature `A` and the scalar `T`;
//! - the product kernels (geometric, wedge, inner, scalar), the
//!   involutions, versor inversion / sandwich / `exp`, and the
//!   metric-independent duality (complements + regressive product);
//! - the **raw** geometric constructors for PGA `Cl(3,0,1)` and CGA
//!   `Cl(4,1,0)` ([`Multivector::point`], [`Multivector::cga_point`], …),
//!   which live here because Rust requires inherent methods to sit in the
//!   crate that defines the type.
//!
//! The *typed* geometry layer — `Motor`, `Conformal`, and the geometric
//! objects built on top — lives in the sibling `garust-geo` crate. Most
//! users depend on the umbrella [`garust`](https://crates.io/crates/garust)
//! crate, which re-exports both; reach for `garust-core` directly only when
//! you want the algebra without the geometry.

pub mod algebra;
pub mod cga;
pub mod dual;
pub mod involutions;
pub mod multivector;
pub mod pga;
pub mod products;
pub mod scalar;
pub mod signature;
pub mod transform;

pub use algebra::{Algebra, BladeStore};
pub use multivector::Multivector;
pub use scalar::{Real, Scalar};

// Standard signature markers. Each is a zero-sized type implementing
// [`Algebra`]; the type aliases below pair them with a scalar. Downstream
// crates can mint their own with [`define_algebra!`].
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
