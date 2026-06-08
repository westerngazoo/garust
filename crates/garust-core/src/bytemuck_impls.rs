//! `bytemuck` support for [`Multivector`], behind the `bytemuck` feature.
//!
//! A multivector is `#[repr(transparent)]` over its `[T; 2^N]` coefficient
//! array, so when the scalar `T` is itself plain-old-data the whole
//! multivector is too. That makes a `&[Multivector<A, T>]` reinterpretable
//! as a flat `&[T]` (or `&[u8]`) with zero copying — exactly the layout a
//! GPU vertex / storage buffer wants.
//!
//! Both impls are bounded on the backing store (`A::Blades<T>: Zeroable` /
//! `: Pod`) rather than on `T` directly, so they stay sound for any custom
//! [`BladeStore`](crate::BladeStore): a padded or otherwise non-`Pod` store
//! simply doesn't receive the impl.

use bytemuck::{Pod, Zeroable};

use crate::algebra::Algebra;
use crate::multivector::Multivector;
use crate::scalar::Ring;

// SAFETY: `Multivector` is `#[repr(transparent)]` over its sole field
// `coeffs: A::Blades<T>`. When that field is `Zeroable`, an all-zero bit
// pattern is the valid zero multivector, so the transparent wrapper is
// `Zeroable` too.
unsafe impl<A: Algebra, T: Ring> Zeroable for Multivector<A, T> where A::Blades<T>: Zeroable {}

// SAFETY: with `#[repr(transparent)]`, `Multivector<A, T>` has the same
// size, alignment, and (absence of) padding as its `A::Blades<T>` field.
// When that field is `Pod` — every bit pattern valid, no padding — so is
// the wrapper. `Copy` holds unconditionally; the `'static` bounds satisfy
// `Pod`'s `'static` supertrait (every `Algebra` marker and every numeric
// scalar is `'static` in practice).
unsafe impl<A: Algebra + 'static, T: Ring + 'static> Pod for Multivector<A, T> where
    A::Blades<T>: Pod
{
}

#[cfg(test)]
mod tests {
    use crate::{Cga3, Pga3, Vga3};
    use core::mem::size_of;

    #[test]
    fn multivector_round_trips_through_bytes() {
        let m = Pga3::point(1.0, 2.0, 3.0);
        let bytes = bytemuck::bytes_of(&m);
        assert_eq!(bytes.len(), 16 * size_of::<f64>());
        let back: Pga3 = *bytemuck::from_bytes(bytes);
        assert_eq!(back, m);
    }

    #[test]
    fn slice_of_multivectors_casts_to_flat_scalars() {
        // Two Vga3 (8 coeffs each) view as 16 contiguous f64 — the GPU case.
        let a = Vga3 {
            coeffs: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        };
        let b = Vga3 {
            coeffs: [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        };
        let pair = [a, b];
        let flat: &[f64] = bytemuck::cast_slice(&pair);
        assert_eq!(flat.len(), 16);
        assert_eq!(flat[0], 1.0);
        assert_eq!(flat[8], 9.0);
        assert_eq!(flat[15], 16.0);
    }

    #[test]
    fn zeroed_is_the_zero_multivector() {
        // Exercises the largest standard signature (Cga3, 32 blades).
        let z: Cga3 = bytemuck::Zeroable::zeroed();
        assert_eq!(z, Cga3::zero());
    }
}
