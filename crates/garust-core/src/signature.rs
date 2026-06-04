//! Compile-time bookkeeping for the algebra's signature.
//!
//! A signature `Cl(P, Q, R)` is just three integers, so we don't need a
//! full type for it yet. This module documents the conventions garust
//! uses and provides a couple of helpers for indexing basis blades and
//! computing the product of two of them.
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
//!
//! # Signature → metric
//!
//! Generators are partitioned by index. With `P + Q + R = N`:
//!
//! - bit `k < P`         ⇒ `e_{k+1}` squares to `+1`
//! - bit `P ≤ k < P+Q`   ⇒ `e_{k+1}` squares to `-1`
//! - bit `P+Q ≤ k < N`   ⇒ `e_{k+1}` squares to `0` (null / degenerate)

/// Number of basis vectors making up the blade with this index.
/// For example `grade_of(0b1011) == 3` (it's a trivector).
pub const fn grade_of(blade_index: usize) -> usize {
    blade_index.count_ones() as usize
}

/// `2^n`. A named helper so call sites read clearly.
pub const fn dim_for(n: usize) -> usize {
    1 << n
}

/// Sign of reordering the concatenation `a · b` into sorted order.
///
/// Returns `+1` if the number of adjacent-generator swaps is even and
/// `-1` if it's odd. The count is the number of *inversions* between the
/// set bits of `a` and `b`, i.e. pairs `(i ∈ a, j ∈ b)` with `i > j`.
///
/// The bit trick: shifting `a` right and AND-ing with `b` exposes,
/// shift-by-shift, every pair `(i, j)` with `i - j` equal to that shift.
/// Pop-counting each and summing gives the total inversion count in
/// `O(N)` ops.
pub const fn swap_sign(a: usize, b: usize) -> i32 {
    let mut a = a >> 1;
    let mut inversions: u32 = 0;
    while a != 0 {
        inversions += (a & b).count_ones();
        a >>= 1;
    }
    if inversions & 1 == 0 {
        1
    } else {
        -1
    }
}

/// Geometric product of two basis blades in `Cl(P, Q, R)`.
///
/// Given the bitmask indices of two basis blades, returns
/// `(result_index, sign)` where `sign ∈ {-1, 0, +1}`. The result is
/// always a single signed basis blade — the bilinear lift to full
/// multivectors lives in [`crate::multivector`].
///
/// Cancellation rule: bit positions present in *both* `a` and `b`
/// contribute a metric factor (`+1`, `-1`, or `0` per the partition
/// in the module docs) and *disappear* from the result. Bit positions
/// present in exactly one survive. Hence `result_index = a ^ b`.
///
/// As soon as a shared bit lands in the `R` (null) group, the whole
/// product is zero and we short-circuit.
pub const fn blade_product(a: usize, b: usize, p: usize, q: usize) -> (usize, i32) {
    let result = a ^ b;
    let mut sign = swap_sign(a, b);
    let mut shared = a & b;
    while shared != 0 {
        let k = shared.trailing_zeros() as usize;
        if k < p {
            // generator squares to +1; no change
        } else if k < p + q {
            sign = -sign;
        } else {
            return (result, 0);
        }
        shared &= shared - 1; // clear the lowest set bit
    }
    (result, sign)
}

/// One cell of a precomputed Cayley table: the `(target blade index, sign)`
/// of a single basis-blade geometric product, with `sign ∈ {-1, 0, 1}`. The
/// index is `u16`, so every signature up to `N = 16` (65 536 blades) fits.
pub type CayleyEntry = (u16, i8);

/// Build the geometric-product Cayley table for `Cl(p, q, r)` at compile
/// time: a flat, row-major `dim × dim` array whose cell `[a * dim + b]` is
/// [`blade_product`]`(a, b, p, q)` packed into a [`CayleyEntry`].
///
/// `dim` must be `2^(p + q + r)` and the const parameter `NN` must be
/// `dim * dim`; [`define_algebra!`](crate::define_algebra) and the `Algebra`
/// derive supply both from the signature literals. Materializing this once
/// per algebra lets [`Multivector`](crate::Multivector)'s geometric product
/// replace the per-pair [`blade_product`] call with a single table lookup.
pub const fn cayley_table<const NN: usize>(dim: usize, p: usize, q: usize) -> [CayleyEntry; NN] {
    let mut table = [(0u16, 0i8); NN];
    let mut a = 0;
    while a < dim {
        let mut b = 0;
        while b < dim {
            let (idx, sign) = blade_product(a, b, p, q);
            table[a * dim + b] = (idx as u16, sign as i8);
            b += 1;
        }
        a += 1;
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- swap_sign --------------------------------------------------------

    #[test]
    fn swap_sign_disjoint_no_inversions() {
        // 0b01 then 0b10  →  e1 e2, already sorted, no swaps
        assert_eq!(swap_sign(0b01, 0b10), 1);
    }

    #[test]
    fn swap_sign_disjoint_one_inversion() {
        // 0b10 then 0b01  →  e2 e1, one swap to sort
        assert_eq!(swap_sign(0b10, 0b01), -1);
    }

    #[test]
    fn swap_sign_e12_times_e1_is_minus() {
        // e1 e2 · e1 — the only inversion is the trailing (a:e2, b:e1)
        assert_eq!(swap_sign(0b11, 0b01), -1);
    }

    // --- blade_product ----------------------------------------------------

    #[test]
    fn vector_squares_to_plus_one_in_p_group() {
        // Cl(2,0,0): e1 * e1 = 1
        assert_eq!(blade_product(0b01, 0b01, 2, 0), (0, 1));
    }

    #[test]
    fn vector_squares_to_minus_one_in_q_group() {
        // Cl(0,1,0): the only vector squares to -1
        assert_eq!(blade_product(0b01, 0b01, 0, 1), (0, -1));
    }

    #[test]
    fn vector_squares_to_zero_in_r_group() {
        // Cl(0,0,1): the only vector squares to 0
        assert_eq!(blade_product(0b01, 0b01, 0, 0), (0, 0));
    }

    #[test]
    fn vectors_anticommute() {
        // Cl(2,0,0): e1 * e2 = +e12,  e2 * e1 = -e12
        assert_eq!(blade_product(0b01, 0b10, 2, 0), (0b11, 1));
        assert_eq!(blade_product(0b10, 0b01, 2, 0), (0b11, -1));
    }

    #[test]
    fn bivector_squares_to_minus_one_in_euclidean() {
        // Cl(2,0,0): e12 * e12 = -1
        assert_eq!(blade_product(0b11, 0b11, 2, 0), (0, -1));
    }

    #[test]
    fn pga_null_generator_squares_to_zero() {
        // Cl(3,0,1): generator at bit 3 is in the R group ⇒ squares to 0
        assert_eq!(blade_product(0b1000, 0b1000, 3, 0), (0, 0));
    }

    #[test]
    fn pseudoscalar_squared_in_3d_euclidean() {
        // Cl(3,0,0): I = e1 e2 e3, I*I should be -1
        assert_eq!(blade_product(0b111, 0b111, 3, 0), (0, -1));
    }
}
