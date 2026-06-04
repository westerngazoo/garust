//! Baseline micro-benchmarks for the product kernels.
//!
//! These establish the "before" numbers for the performance work in
//! `rfcs/RFC-001-garust-performance.md`: the dense `O(DIM²)` geometric
//! product across the three standard signatures (Vga3 = 8 blades, Pga3 =
//! 16, Cga3 = 32), plus the derived wedge and regressive products. Operands
//! are fully dense (every coefficient non-zero) to measure the worst case —
//! the path the RFC's const-Cayley-table and SIMD work targets.
//!
//! Run with `cargo bench -p garust-core`.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use garust_core::{Algebra, BladeStore, Cga3Sig, Multivector, Pga3Sig, Vga3Sig};

/// A dense multivector with a deterministic, non-trivial spread of nonzero
/// coefficients (no transcendentals, so the fill itself costs nothing).
fn dense<A: Algebra>() -> Multivector<A, f64> {
    let mut m = Multivector::<A, f64>::zero();
    for (i, slot) in m.coeffs.as_mut_slice().iter_mut().enumerate() {
        *slot = ((i * 7 + 3) % 13) as f64 * 0.25 - 1.5;
    }
    m
}

fn geometric_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("geometric_product");

    let (a, b) = (dense::<Vga3Sig>(), dense::<Vga3Sig>());
    group.bench_function("vga3_dim8", |bn| bn.iter(|| black_box(a) * black_box(b)));

    let (a, b) = (dense::<Pga3Sig>(), dense::<Pga3Sig>());
    group.bench_function("pga3_dim16", |bn| bn.iter(|| black_box(a) * black_box(b)));

    let (a, b) = (dense::<Cga3Sig>(), dense::<Cga3Sig>());
    group.bench_function("cga3_dim32", |bn| bn.iter(|| black_box(a) * black_box(b)));

    group.finish();
}

fn derived_products(c: &mut Criterion) {
    let mut group = c.benchmark_group("derived_products");

    let (a, b) = (dense::<Cga3Sig>(), dense::<Cga3Sig>());
    group.bench_function("wedge_cga3", |bn| {
        bn.iter(|| black_box(a).wedge(black_box(&b)))
    });
    group.bench_function("regressive_cga3", |bn| {
        bn.iter(|| black_box(a).regressive(black_box(&b)))
    });

    group.finish();
}

criterion_group!(benches, geometric_product, derived_products);
criterion_main!(benches);
