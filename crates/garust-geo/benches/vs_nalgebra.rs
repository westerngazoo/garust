//! garust vs nalgebra — apples-to-apples micro-benchmarks.
//!
//! The same rigid-motion operations in each library, built and run under one
//! release profile, so "GA motor" and "quaternion / 4×4 matrix" compare
//! directly. A [`Motor`] is the geometric-algebra analogue of nalgebra's
//! [`Isometry3`] (a unit quaternion plus a translation); [`Matrix4`] is the
//! homogeneous-matrix form.
//!
//! This exists to settle benchmark confusion at the source: measure *one*
//! transform, not a whole engine update loop. Run it with the optimizations
//! that matter:
//!
//! ```sh
//! CARGO_PROFILE_BENCH_LTO=fat CARGO_PROFILE_BENCH_CODEGEN_UNITS=1 \
//!   cargo bench -p garust-geo --features simd --bench vs_nalgebra
//! ```

use std::f64::consts::TAU;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use garust_core::Pga3;
use garust_geo::Motor;
use nalgebra::{Isometry3, Matrix4, Point3, Translation3, UnitQuaternion, Vector3};

// Equivalent rigid motions: rotate τ/8 about x, then translate by (1, 2, 3).
fn garust_motor() -> Motor<f64> {
    Motor::translator(1.0, 2.0, 3.0) * Motor::rotor(TAU / 8.0, Pga3::basis(0b0110))
}
fn na_isometry() -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::new(1.0, 2.0, 3.0),
        UnitQuaternion::from_axis_angle(&Vector3::x_axis(), TAU / 8.0),
    )
}

/// Transform a single point.
fn transform_point(c: &mut Criterion) {
    let mut g = c.benchmark_group("transform_point");

    let m = garust_motor();
    let gp = Pga3::point(0.3, -1.2, 2.5);
    g.bench_function("garust_motor", |b| {
        b.iter(|| black_box(&m).apply(black_box(&gp)))
    });

    let iso = na_isometry();
    let np = Point3::new(0.3, -1.2, 2.5);
    g.bench_function("nalgebra_isometry3", |b| {
        b.iter(|| black_box(&iso).transform_point(black_box(&np)))
    });

    let mat: Matrix4<f64> = iso.to_homogeneous();
    g.bench_function("nalgebra_matrix4", |b| {
        b.iter(|| black_box(&mat).transform_point(black_box(&np)))
    });

    g.finish();
}

/// Compose two rigid motions.
fn compose(c: &mut Criterion) {
    let mut g = c.benchmark_group("compose");

    let a = garust_motor();
    let b2 = Motor::translator(-1.0, 0.5, 2.0) * Motor::rotor(TAU / 5.0, Pga3::basis(0b1010));
    g.bench_function("garust_motor", |bn| {
        bn.iter(|| black_box(a) * black_box(b2))
    });

    let ia = na_isometry();
    let ib = Isometry3::from_parts(
        Translation3::new(-1.0, 0.5, 2.0),
        UnitQuaternion::from_axis_angle(&Vector3::y_axis(), TAU / 5.0),
    );
    g.bench_function("nalgebra_isometry3", |bn| {
        bn.iter(|| black_box(&ia) * black_box(&ib))
    });

    let ma = ia.to_homogeneous();
    let mb = ib.to_homogeneous();
    g.bench_function("nalgebra_matrix4", |bn| {
        bn.iter(|| black_box(&ma) * black_box(&mb))
    });

    g.finish();
}

/// Transform a 1024-point cloud by one motion (garust's batch/SIMD strength).
fn transform_cloud(c: &mut Criterion) {
    let n = 1024usize;
    let mut g = c.benchmark_group("transform_1024_points");

    let m = garust_motor();
    let gpts: Vec<Pga3> = (0..n)
        .map(|i| Pga3::point(i as f64 * 0.1, (i % 7) as f64, -(i as f64) * 0.05))
        .collect();
    g.bench_function("garust_apply_each", |bn| {
        bn.iter_batched(
            || gpts.clone(),
            |mut buf| {
                m.apply_each(&mut buf);
                buf
            },
            BatchSize::SmallInput,
        )
    });
    #[cfg(feature = "simd")]
    g.bench_function("garust_apply_each_simd", |bn| {
        bn.iter_batched(
            || gpts.clone(),
            |mut buf| {
                m.apply_each_simd(&mut buf);
                buf
            },
            BatchSize::SmallInput,
        )
    });

    let iso = na_isometry();
    let npts: Vec<Point3<f64>> = (0..n)
        .map(|i| Point3::new(i as f64 * 0.1, (i % 7) as f64, -(i as f64) * 0.05))
        .collect();
    g.bench_function("nalgebra_isometry3_loop", |bn| {
        bn.iter_batched(
            || npts.clone(),
            |mut buf| {
                for p in buf.iter_mut() {
                    *p = iso.transform_point(p);
                }
                buf
            },
            BatchSize::SmallInput,
        )
    });

    g.finish();
}

criterion_group!(benches, transform_point, compose, transform_cloud);
criterion_main!(benches);
