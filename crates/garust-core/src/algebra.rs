//! The [`Algebra`] trait — a Clifford signature `Cl(P, Q, R)` reified as
//! a zero-sized **marker type**, plus the [`BladeStore`] abstraction over
//! its backing coefficient array.
//!
//! # Why a trait instead of const generics
//!
//! A multivector in `Cl(P, Q, R)` needs exactly `2^(P+Q+R)` coefficients.
//! We would love to write
//!
//! ```ignore
//! struct Multivector<T, const P: usize, const Q: usize, const R: usize> {
//!     coeffs: [T; 1 << (P + Q + R)], // ← not allowed on stable Rust
//! }
//! ```
//!
//! but evaluating `1 << (P + Q + R)` in an array-length position needs the
//! unstable `generic_const_exprs` feature. The stable workaround is to let
//! each *concrete* signature carry its own array type: a marker type `A`
//! implements [`Algebra`] and names `2^N` once, as the associated type
//! [`Algebra::Blades`]. [`Multivector`](crate::Multivector) then stores an
//! `A::Blades<T>` and never has to compute the length generically.
//!
//! This buys three things the old `Multivector<T, P, Q, R, DIM>` form
//! could not:
//!
//! - **No redundant `DIM`.** The length is derived from `P + Q + R`, so it
//!   can never be specified inconsistently — the old `_DIM_CHECK`
//!   const-assert is gone because the bug it guarded is now unrepresentable.
//! - **Named, self-documenting signatures.** `Multivector<Pga3Sig, f64>`
//!   reads better than `Multivector<f64, 3, 0, 1, 16>`, and downstream
//!   crates can mint their own signatures with [`define_algebra!`](crate::define_algebra).
//! - **A single place to hang signature-level data** (the metric, later a
//!   cached Cayley table, …) as more associated items.
//!
//! # Defining your own algebra
//!
//! Use [`define_algebra!`](crate::define_algebra) — it generates the marker type and its
//! [`Algebra`] impl from the three signature integers:
//!
//! ```
//! use garust_core::{define_algebra, Multivector};
//!
//! define_algebra!(
//!     /// 2D Euclidean plane, `Cl(2, 0, 0)`.
//!     pub MyPlane = Cl(2, 0, 0)
//! );
//!
//! let e1 = Multivector::<MyPlane, f64>::basis(1);
//! assert_eq!((e1 * e1).scalar_part(), 1.0);
//! ```

use core::fmt::Debug;
use core::ops::{Index, IndexMut};

use crate::scalar::Ring;

/// Backing storage for a multivector's `2^N` blade coefficients.
///
/// This is the seam that lets [`Multivector`](crate::Multivector) hold an
/// associated array type without naming its length generically. The only
/// implementor today is the fixed array `[T; D]`, but the trait is the
/// hook a future SoA / arena / SIMD layout would slot into without
/// touching the algebra kernels (see the crate roadmap's performance
/// phase).
///
/// The supertrait bounds are exactly what the kernels need: `Copy` to pass
/// multivectors by value, and `Index`/`IndexMut` so `coeffs[i]` works on
/// the projected associated type.
pub trait BladeStore<T>: Copy + Index<usize, Output = T> + IndexMut<usize> {
    /// A store with every coefficient set to `value`. Used to build the
    /// zero multivector (and, with `T::ONE`, basis blades).
    fn splat(value: T) -> Self;

    /// View the coefficients as a slice, in blade-index order.
    fn as_slice(&self) -> &[T];

    /// View the coefficients as a mutable slice, in blade-index order.
    fn as_mut_slice(&mut self) -> &mut [T];
}

impl<T: Copy, const D: usize> BladeStore<T> for [T; D] {
    #[inline]
    fn splat(value: T) -> Self {
        [value; D]
    }

    #[inline]
    fn as_slice(&self) -> &[T] {
        self
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

/// A Clifford signature `Cl(P, Q, R)`, reified as a marker type.
///
/// Implementors are zero-sized types (`Vga3Sig`, `Pga3Sig`, …) that exist
/// only to pin a [`Multivector`](crate::Multivector)'s signature at the
/// type level. The three required constants give the metric; the derived
/// [`N`](Algebra::N) and [`DIM`](Algebra::DIM) and the
/// [`Blades`](Algebra::Blades) storage type all follow from them.
///
/// Prefer [`define_algebra!`](crate::define_algebra) over a hand-written impl — it fills in the
/// associated type with the correct `[T; 2^N]` length and derives the
/// marker's `Copy`/`Debug`/… for you.
///
/// # Metric convention
///
/// Generators are partitioned by index (see [`crate::signature`]): the
/// first `P` square to `+1`, the next `Q` to `-1`, the last `R` to `0`.
pub trait Algebra: Copy + Debug {
    /// Number of generators squaring to `+1`.
    const P: usize;
    /// Number of generators squaring to `-1`.
    const Q: usize;
    /// Number of generators squaring to `0` (degenerate / null).
    const R: usize;

    /// Total generator count `N = P + Q + R`. Derived; do not override.
    const N: usize = Self::P + Self::Q + Self::R;

    /// Number of basis blades, `2^N`. Derived; do not override. This is
    /// the length of [`Blades`](Algebra::Blades).
    const DIM: usize = 1 << Self::N;

    /// The backing array for this signature's `2^N` coefficients, generic
    /// over the coefficient type `T`. For every concrete signature this
    /// resolves to a fixed-size `[T; 2^N]`. Bounded by [`Ring`] (not the
    /// stronger [`Scalar`](crate::Scalar)) so multivectors can hold exact,
    /// division-free coefficients such as the integers.
    type Blades<T: Ring>: BladeStore<T>;

    /// Precomputed geometric-product Cayley table: a flat, row-major
    /// `DIM × DIM` slice whose cell `[a * DIM + b]` is the
    /// `(target index, sign)` of the basis-blade product `e_a · e_b` (see
    /// [`CayleyEntry`](crate::signature::CayleyEntry)).
    ///
    /// [`Multivector`](crate::Multivector)'s geometric product indexes this
    /// once per blade pair instead of recomputing
    /// [`blade_product`](crate::signature::blade_product) in the hot loop.
    /// It is generated for you by [`define_algebra!`](crate::define_algebra)
    /// and the `Algebra` derive — a hand-written impl must supply it, e.g.
    /// with [`cayley_table`](crate::signature::cayley_table).
    const CAYLEY: &'static [crate::signature::CayleyEntry];
}

/// Define a Clifford-algebra marker type and its [`Algebra`] impl from a
/// signature `Cl(P, Q, R)`.
///
/// Generates a zero-sized `struct` (deriving the standard marker traits)
/// and an [`Algebra`] impl whose [`Blades`](Algebra::Blades) is
/// `[T; 2^(P+Q+R)]`. Optional leading attributes (doc comments, …) and a
/// visibility qualifier are passed through to the generated struct.
///
/// ```
/// use garust_core::{define_algebra, Multivector};
///
/// define_algebra!(pub Spacetime = Cl(1, 3, 0));
/// assert_eq!(Multivector::<Spacetime, f64>::zero().coeffs.len(), 16);
/// ```
#[macro_export]
macro_rules! define_algebra {
    ($(#[$meta:meta])* $vis:vis $name:ident = Cl($p:literal, $q:literal, $r:literal) $(;)?) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
        $vis struct $name;

        impl $crate::algebra::Algebra for $name {
            const P: usize = $p;
            const Q: usize = $q;
            const R: usize = $r;
            type Blades<T: $crate::scalar::Ring> = [T; 1 << ($p + $q + $r)];
            const CAYLEY: &'static [$crate::signature::CayleyEntry] =
                &$crate::signature::cayley_table::<
                    { (1usize << ($p + $q + $r)) * (1usize << ($p + $q + $r)) },
                >(1usize << ($p + $q + $r), $p, $q);
        }
    };
}
