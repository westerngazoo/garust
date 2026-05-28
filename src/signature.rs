//! Compile-time bookkeeping for the algebra's signature.
//!
//! A signature `Cl(P, Q, R)` is just three integers, so we don't need a
//! full type for it yet. This module documents the conventions garust
//! uses and provides a couple of helpers for indexing basis blades.
//!
//! # Basis-blade indexing convention
//!
//! With `N = P + Q + R` generators `e_1, e_2, ..., e_N`, there are
//! `2^N` basis blades. We index each blade by a `usize` whose bit
//! pattern is the *characteristic vector* of the subset of generators
//! that make it up:
//!
//! | index | bits  | blade        |
//! |-------|-------|--------------|
//! |   0   | `000` | `1`          |
//! |   1   | `001` | `e1`         |
//! |   2   | `010` | `e2`         |
//! |   3   | `011` | `e1 e2`      |
//! |   4   | `100` | `e3`         |
//! |   5   | `101` | `e1 e3`      |
//! |   6   | `110` | `e2 e3`      |
//! |   7   | `111` | `e1 e2 e3`   |
//!
//! Bit `k` set means generator `e_{k+1}` appears in the blade. We
//! always store the generators in ascending index order; any sign that
//! comes from reordering only matters when we multiply.

/// Number of basis vectors making up the blade with this index.
/// For example `grade_of(0b1011) == 3` (it's a trivector).
pub const fn grade_of(blade_index: usize) -> usize {
    blade_index.count_ones() as usize
}

/// `2^n`. A named helper so call sites read clearly.
pub const fn dim_for(n: usize) -> usize {
    1 << n
}
