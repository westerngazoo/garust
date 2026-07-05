//! SIMD batch transforms behind the `simd` feature.
//!
//! The batch sandwich is embarrassingly data-parallel: every object is
//! transformed by the *same* versor with the *same* control flow, and each
//! object is independent. So we lay the batch out structure-of-arrays —
//! coefficient `j` of `LANES` consecutive objects packed into one SIMD vector
//! — and run the two sandwich products with the versor's coefficients as
//! broadcast scalars. No cross-lane shuffles: each lane is its own object.
//!
//! Two widths are provided:
//! - **`f64x4`** — 4 objects/vector, `f64` precision.  Used by `Motor<f64>`
//!   and `Conformal<f64>` via `apply_each_simd`.
//! - **`f32x8`** — 8 objects/vector, `f32` precision (~2× throughput on
//!   256-bit AVX2 / NEON; see RFC-001 Appendix E).  Used by `Motor<f32>` and
//!   `Conformal<f32>` via `apply_each_simd`.
//!
//! Both are bit-faithful to the scalar
//! [`sandwich_each`](garust_core::Multivector::sandwich_each) (same table,
//! same accumulation order, just `LANES` objects at a time); the tail that
//! doesn't fill a full vector falls back to the scalar path.

use garust_core::{Algebra, Multivector};
use wide::{f32x8, f64x4};

// ── f64 path (4 lanes) ───────────────────────────────────────────────────────

const F64_LANES: usize = 4;

/// Generate a SoA SIMD batch sandwich for one concrete signature using
/// `f64x4`. `$dim` must be that signature's blade count.
macro_rules! simd_sandwich_each_f64 {
    ($name:ident, $sig:ty, $dim:literal) => {
        /// Sandwich `versor` over every element of `xs` in place, 4 objects
        /// per `f64x4`; identical result to the scalar batch.
        pub(crate) fn $name(versor: &Multivector<$sig, f64>, xs: &mut [Multivector<$sig, f64>]) {
            const DIM: usize = $dim;
            let table = <$sig as Algebra>::CAYLEY;
            let v = versor.coeffs;
            let rev = versor.reverse().coeffs;

            let mut chunks = xs.chunks_exact_mut(F64_LANES);
            for chunk in &mut chunks {
                let mut x = [f64x4::splat(0.0); DIM];
                for (j, xj) in x.iter_mut().enumerate() {
                    *xj = f64x4::from([
                        chunk[0].coeffs[j],
                        chunk[1].coeffs[j],
                        chunk[2].coeffs[j],
                        chunk[3].coeffs[j],
                    ]);
                }
                let mut t = [f64x4::splat(0.0); DIM];
                for i in 0..DIM {
                    let vi = v[i];
                    if vi == 0.0 { continue; }
                    let row = i * DIM;
                    for j in 0..DIM {
                        let (idx, sign) = table[row + j];
                        if sign != 0 {
                            t[idx as usize] += x[j] * f64x4::splat(vi * sign as f64);
                        }
                    }
                }
                let mut r = [f64x4::splat(0.0); DIM];
                for i in 0..DIM {
                    let ti = t[i];
                    let row = i * DIM;
                    for j in 0..DIM {
                        let rj = rev[j];
                        if rj == 0.0 { continue; }
                        let (idx, sign) = table[row + j];
                        if sign != 0 {
                            r[idx as usize] += ti * f64x4::splat(rj * sign as f64);
                        }
                    }
                }
                let lanes = r.map(|vec| vec.to_array());
                for (l, obj) in chunk.iter_mut().enumerate() {
                    for k in 0..DIM {
                        obj.coeffs[k] = lanes[k][l];
                    }
                }
            }
            let rem = chunks.into_remainder();
            if !rem.is_empty() {
                versor.sandwich_each(rem);
            }
        }
    };
}

simd_sandwich_each_f64!(sandwich_each_pga, garust_core::Pga3Sig, 16);
simd_sandwich_each_f64!(sandwich_each_cga, garust_core::Cga3Sig, 32);

// ── f32 path (8 lanes) ───────────────────────────────────────────────────────

const F32_LANES: usize = 8;

/// Generate a SoA SIMD batch sandwich for one concrete signature using
/// `f32x8`. `$dim` must be that signature's blade count.
///
/// Uses 8-wide lanes instead of 4-wide, doubling throughput on 256-bit
/// vectors (AVX2 / NEON). Bit-faithful to the scalar path.
macro_rules! simd_sandwich_each_f32 {
    ($name:ident, $sig:ty, $dim:literal) => {
        /// Sandwich `versor` over every element of `xs` in place, 8 objects
        /// per `f32x8`; identical result to the scalar batch.
        pub(crate) fn $name(versor: &Multivector<$sig, f32>, xs: &mut [Multivector<$sig, f32>]) {
            const DIM: usize = $dim;
            let table = <$sig as Algebra>::CAYLEY;
            let v = versor.coeffs;
            let rev = versor.reverse().coeffs;

            let mut chunks = xs.chunks_exact_mut(F32_LANES);
            for chunk in &mut chunks {
                let mut x = [f32x8::splat(0.0); DIM];
                for (j, xj) in x.iter_mut().enumerate() {
                    *xj = f32x8::from([
                        chunk[0].coeffs[j],
                        chunk[1].coeffs[j],
                        chunk[2].coeffs[j],
                        chunk[3].coeffs[j],
                        chunk[4].coeffs[j],
                        chunk[5].coeffs[j],
                        chunk[6].coeffs[j],
                        chunk[7].coeffs[j],
                    ]);
                }
                let mut t = [f32x8::splat(0.0); DIM];
                for i in 0..DIM {
                    let vi = v[i];
                    if vi == 0.0 { continue; }
                    let row = i * DIM;
                    for j in 0..DIM {
                        let (idx, sign) = table[row + j];
                        if sign != 0 {
                            t[idx as usize] += x[j] * f32x8::splat(vi * sign as f32);
                        }
                    }
                }
                let mut r = [f32x8::splat(0.0); DIM];
                for i in 0..DIM {
                    let ti = t[i];
                    let row = i * DIM;
                    for j in 0..DIM {
                        let rj = rev[j];
                        if rj == 0.0 { continue; }
                        let (idx, sign) = table[row + j];
                        if sign != 0 {
                            r[idx as usize] += ti * f32x8::splat(rj * sign as f32);
                        }
                    }
                }
                let lanes = r.map(|vec| vec.to_array());
                for (l, obj) in chunk.iter_mut().enumerate() {
                    for k in 0..DIM {
                        obj.coeffs[k] = lanes[k][l];
                    }
                }
            }
            let rem = chunks.into_remainder();
            if !rem.is_empty() {
                versor.sandwich_each(rem);
            }
        }
    };
}

simd_sandwich_each_f32!(sandwich_each_pga_f32, garust_core::Pga3Sig, 16);
simd_sandwich_each_f32!(sandwich_each_cga_f32, garust_core::Cga3Sig, 32);
