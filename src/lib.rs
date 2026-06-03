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
//! brand-new signature with [`define_algebra!`] — or, under the optional
//! `derive` feature, by writing the marker yourself and adding
//! `#[derive(Algebra)]` with an `#[algebra(p = …, q = …, r = …)]` attribute.
//!
//! ## What's implemented
//!
//! - [`Algebra`] — signatures reified as zero-sized marker types, so a
//!   multivector is `Multivector<A, T>` (no redundant `DIM` parameter);
//!   mint your own with [`define_algebra!`] (or, under the optional
//!   `derive` feature, `#[derive(Algebra)]`)
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
//! - typed PGA geometry in [`pga`]: [`pga::Point`], [`pga::Line`],
//!   [`pga::Plane`] with type-checked join/meet incidence
//!   (`a.meet(&b).meet(&c)`, `a.join(&b).join(&c)`)
//! - CGA geometric constructors for `Cl(4,1,0)`: `cga_point`, `sphere`,
//!   `cga_plane`, on the null-cone conformal model
//! - [`Conformal`] — conformal transformations in CGA (translators,
//!   rotors, and dilations about the origin)
//! - typed CGA geometry in [`cga`]: [`cga::Point`], [`cga::Sphere`],
//!   [`cga::Plane`] with type-checked incidence (point-on-sphere/plane)
//!
//! ## Crate layout
//!
//! `garust` is a thin umbrella over a small workspace, so a larger
//! project — a graph database, a physics engine, an experimental-algebra
//! sandbox — can depend on exactly the layer it needs:
//!
//! - [`garust_core`] — the signature-generic kernel: [`Algebra`] and
//!   [`define_algebra!`], [`Multivector`], the products and involutions,
//!   duality, and the *raw* PGA/CGA constructors.
//! - [`garust_geo`] — the *typed* geometry layer built on the kernel:
//!   [`Motor`], [`Conformal`], and the typed [`pga`] and [`cga`] objects.
//!
//! This crate re-exports everything from both, so `use garust::…` is
//! unchanged; reach past it to a member crate only when you want the
//! kernel without the geometry (or vice versa).

// --- Modules (re-exported so `garust::signature`, … keep resolving) ------
#[doc(no_inline)]
pub use garust_core::{
    algebra, dual, involutions, multivector, products, scalar, signature, transform,
};
// `pga` and `cga` come from the geometry layer: each bundles that model's
// typed objects (Point/Line/Plane, Point/Sphere/Plane), and `pga` also
// re-exports the kernel's PGA-aware Display adapter.
#[doc(no_inline)]
pub use garust_geo::{cga, conformal, motor, pga};

// --- The kernel: traits, the multivector, signatures, and aliases --------
#[doc(inline)]
pub use garust_core::{
    Algebra, BladeStore, Cga3, Cga3Sig, Cga3f, Multivector, Pga3, Pga3Sig, Pga3f, Real, Scalar,
    Sta, StaSig, Staf, Vga2, Vga2Sig, Vga2f, Vga3, Vga3Sig, Vga3f,
};

// --- The typed geometry layer --------------------------------------------
#[doc(inline)]
pub use garust_geo::{Conformal, Conformal3, Conformal3f, Motor, Motor3, Motor3f};

// The signature-minting macro. `#[macro_export]` puts it at the
// `garust_core` root; re-export it so downstream crates can write
// `garust::define_algebra!` (the macro's `$crate` paths still resolve
// through the dependency graph).
#[doc(inline)]
pub use garust_core::define_algebra;
