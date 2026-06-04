//! SIMD batch transforms behind the `simd` feature.
//!
//! The batch sandwich is embarrassingly data-parallel: every object is
//! transformed by the *same* versor with the *same* control flow, and each
//! object is independent. So we lay the batch out structure-of-arrays —
//! coefficient `j` of `LANES` consecutive objects packed into one
//! [`wide::f64x4`] — and run the two sandwich products with the versor's
//! coefficients as broadcast scalars. No cross-lane shuffles: each lane is
//! its own object.
//!
//! The result is bit-faithful to the scalar
//! [`sandwich_each`](garust_core::Multivector::sandwich_each) (same table,
//! same accumulation order, just `LANES` objects at a time); the tail that
//! doesn't fill a vector falls back to the scalar path.

use garust_core::{Algebra, Multivector};
use wide::f64x4;

/// Objects transformed per SIMD vector (`f64x4`).
const LANES: usize = 4;

/// Generate a SoA SIMD batch sandwich for one concrete signature. `$dim`
/// must be that signature's blade count (it sizes the per-lane accumulators,
/// which can't be sized generically — the same `2^N` limit behind
/// `BladeStore`).
macro_rules! simd_sandwich_each {
    ($name:ident, $sig:ty, $dim:literal) => {
        /// Sandwich `versor` over every element of `xs` in place, `LANES`
        /// objects per SIMD vector; identical result to the scalar batch.
        pub(crate) fn $name(versor: &Multivector<$sig, f64>, xs: &mut [Multivector<$sig, f64>]) {
            const DIM: usize = $dim;
            let table = <$sig as Algebra>::CAYLEY;
            let v = versor.coeffs;
            let rev = versor.reverse().coeffs;

            let mut chunks = xs.chunks_exact_mut(LANES);
            for chunk in &mut chunks {
                // Gather: x[j] holds coefficient j of all LANES objects.
                let mut x = [f64x4::splat(0.0); DIM];
                for (j, xj) in x.iter_mut().enumerate() {
                    *xj = f64x4::from([
                        chunk[0].coeffs[j],
                        chunk[1].coeffs[j],
                        chunk[2].coeffs[j],
                        chunk[3].coeffs[j],
                    ]);
                }

                // t = versor * x  (versor scalar broadcast, x vectorized).
                let mut t = [f64x4::splat(0.0); DIM];
                for i in 0..DIM {
                    let vi = v[i];
                    if vi == 0.0 {
                        continue;
                    }
                    let row = i * DIM;
                    for j in 0..DIM {
                        let (idx, sign) = table[row + j];
                        if sign != 0 {
                            let s = f64x4::splat(vi * sign as f64);
                            t[idx as usize] += x[j] * s;
                        }
                    }
                }

                // r = t * ~versor.
                let mut r = [f64x4::splat(0.0); DIM];
                for i in 0..DIM {
                    let ti = t[i];
                    let row = i * DIM;
                    for j in 0..DIM {
                        let rj = rev[j];
                        if rj == 0.0 {
                            continue;
                        }
                        let (idx, sign) = table[row + j];
                        if sign != 0 {
                            let s = f64x4::splat(rj * sign as f64);
                            r[idx as usize] += ti * s;
                        }
                    }
                }

                // Scatter back: lane `l` of r[k] is object `l`'s coefficient k.
                let lanes = r.map(|vec| vec.to_array());
                for (l, obj) in chunk.iter_mut().enumerate() {
                    for k in 0..DIM {
                        obj.coeffs[k] = lanes[k][l];
                    }
                }
            }

            // Tail (< LANES objects): scalar fallback.
            let rem = chunks.into_remainder();
            if !rem.is_empty() {
                versor.sandwich_each(rem);
            }
        }
    };
}

simd_sandwich_each!(sandwich_each_pga, garust_core::Pga3Sig, 16);
simd_sandwich_each!(sandwich_each_cga, garust_core::Cga3Sig, 32);
