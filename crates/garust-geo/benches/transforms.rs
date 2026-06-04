//! Baseline micro-benchmarks for the typed transforms.
//!
//! The VIGA hot path in `rfcs/RFC-001-garust-performance.md` is *transform a
//! point with a versor*. Done generically that is two full dense sandwich
//! products over mostly-zero operands — the case the RFC's specialized
//! `apply` fast-paths target. These benches measure that hot path (PGA motor
//! and CGA conformal) plus versor composition, so the optimization work has
//! a "before" to beat.
//!
//! Run with `cargo bench -p garust-geo`.

use std::f64::consts::TAU;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use garust_core::{Cga3, Pga3};
use garust_geo::{Conformal, Motor};

fn motor_apply(c: &mut Criterion) {
    // A general screw motion (rotation about an origin axis + translation).
    let m = Motor::translator(1.0, 2.0, 3.0) * Motor::rotor(TAU / 8.0, Pga3::basis(0b0110));
    let p = Pga3::point(0.3, -1.2, 2.5);
    c.bench_function("motor_apply_point", |bn| {
        bn.iter(|| black_box(&m).apply(black_box(&p)))
    });
}

fn conformal_apply(c: &mut Criterion) {
    let v =
        Conformal::translator(1.0, 0.0, -2.0) * Conformal::rotor(TAU / 8.0, Cga3::basis(0b0110));
    let p = Cga3::cga_point(0.3, -1.2, 2.5);
    c.bench_function("conformal_apply_point", |bn| {
        bn.iter(|| black_box(&v).apply(black_box(&p)))
    });
}

fn versor_compose(c: &mut Criterion) {
    let a = Motor::translator(1.0, 2.0, 3.0) * Motor::rotor(TAU / 8.0, Pga3::basis(0b0110));
    let b = Motor::translator(-1.0, 0.5, 2.0) * Motor::rotor(TAU / 5.0, Pga3::basis(0b1010));
    c.bench_function("motor_compose", |bn| {
        bn.iter(|| black_box(a) * black_box(b))
    });
}

fn batch_apply(c: &mut Criterion) {
    // Transform a 1024-point cloud by one motor: batch vs a per-point loop.
    let m = Motor::translator(1.0, 2.0, 3.0) * Motor::rotor(TAU / 8.0, Pga3::basis(0b0110));
    let points: Vec<_> = (0..1024)
        .map(|i| Pga3::point(i as f64 * 0.1, (i % 7) as f64, -(i as f64) * 0.05))
        .collect();

    let mut group = c.benchmark_group("motor_apply_batch_1024");
    group.bench_function("apply_each", |bn| {
        bn.iter_batched(
            || points.clone(),
            |mut buf| {
                m.apply_each(&mut buf);
                buf
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("loop", |bn| {
        bn.iter_batched(
            || points.clone(),
            |mut buf| {
                for x in buf.iter_mut() {
                    *x = m.apply(x);
                }
                buf
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(
    benches,
    motor_apply,
    conformal_apply,
    versor_compose,
    batch_apply
);
criterion_main!(benches);
