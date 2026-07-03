//! Smoke-test for the five Motor sensor-bridge methods added in PR #37.
//! Run: cargo run --example motor_sensor_bridge

use garust_core::Pga3;
use garust_geo::{
    pga::Point,
    Motor,
};
use std::f64::consts::TAU;

fn main() {
    println!("=== Motor sensor bridge smoke test ===\n");

    // ── #32  from_unit_quaternion ─────────────────────────────────────────────
    println!("--- from_unit_quaternion ---");

    let id = Motor::<f64>::from_unit_quaternion(1.0, 0.0, 0.0, 0.0);
    assert_eq!(id, Motor::identity(), "identity quaternion must give identity motor");
    println!("  (1,0,0,0) → identity  ✓");

    // 90° about x-axis: (0,1,0) → (0,0,1)
    let half = TAU / 8.0;
    let c = half.cos();
    let s = half.sin();
    let rx = Motor::<f64>::from_unit_quaternion(c, s, 0.0, 0.0);
    let (_, y, z) = Point::from_multivector(rx.apply(&Pga3::point(0.0, 1.0, 0.0))).to_euclidean();
    assert!(y.abs() < 1e-10 && (z - 1.0).abs() < 1e-10,
        "x-axis 90° should send (0,1,0)→(0,0,1), got y={y}, z={z}");
    println!("  x-axis 90°: (0,1,0) → (0,{y:.3},{z:.3})  ✓");

    // 90° about y-axis: (0,0,1) → (1,0,0)
    let ry = Motor::<f64>::from_unit_quaternion(c, 0.0, s, 0.0);
    let (x, _, z) = Point::from_multivector(ry.apply(&Pga3::point(0.0, 0.0, 1.0))).to_euclidean();
    assert!((x - 1.0).abs() < 1e-10 && z.abs() < 1e-10,
        "y-axis 90° should send (0,0,1)→(1,0,0), got x={x}, z={z}");
    println!("  y-axis 90°: (0,0,1) → ({x:.3},0,{z:.3})  ✓");

    // 90° about z-axis: (1,0,0) → (0,1,0)
    let rz = Motor::<f64>::from_unit_quaternion(c, 0.0, 0.0, s);
    let (x, y, _) = Point::from_multivector(rz.apply(&Pga3::point(1.0, 0.0, 0.0))).to_euclidean();
    assert!(x.abs() < 1e-10 && (y - 1.0).abs() < 1e-10,
        "z-axis 90° should send (1,0,0)→(0,1,0), got x={x}, y={y}");
    println!("  z-axis 90°: (1,0,0) → ({x:.3},{y:.3},0)  ✓");

    // ── #33  norm ──────────────────────────────────────────────────────────────
    println!("\n--- norm ---");
    let n_id = Motor::<f64>::identity().norm();
    assert!((n_id - 1.0).abs() < 1e-12, "identity norm should be 1.0, got {n_id}");
    println!("  identity.norm() = {n_id:.15}  ✓");

    let m = Motor::translator(3.0, -1.0, 2.0) * Motor::rotor(0.7, Pga3::basis(0b0110));
    let n = m.norm();
    assert!((n - 1.0).abs() < 1e-12, "composed motor norm should be 1.0, got {n}");
    println!("  (translate * rotor).norm() = {n:.15}  ✓");

    // ── #35  renormalize ──────────────────────────────────────────────────────
    println!("\n--- renormalize ---");
    let base = Motor::rotor(1.2, Pga3::basis(0b0011)) * Motor::translator(0.5, -0.5, 1.0);
    // Introduce drift (isotropic scale)
    let drifted = Motor::from_versor(base.versor() * 1.08);
    println!("  before renormalize: norm² = {:.8}", drifted.norm_squared());
    let fixed = drifted.renormalize();
    let n2 = fixed.norm_squared();
    assert!((n2 - 1.0).abs() < 1e-12, "after renormalize norm² should be 1.0, got {n2}");
    println!("  after  renormalize: norm² = {n2:.15}  ✓");

    // idempotent
    let n2_twice = fixed.renormalize().norm_squared();
    assert!((n2_twice - 1.0).abs() < 1e-12, "renormalize idempotent failed, got {n2_twice}");
    println!("  idempotent:         norm² = {n2_twice:.15}  ✓");

    // ── #36  geodesic_distance ────────────────────────────────────────────────
    println!("\n--- geodesic_distance ---");
    let a = Motor::rotor(0.4, Pga3::basis(0b0110)) * Motor::translator(1.0, 0.0, 0.0);
    let b = Motor::rotor(0.8, Pga3::basis(0b0011)) * Motor::translator(0.0, 2.0, -1.0);

    let d_self = a.geodesic_distance(&a);
    assert!(d_self < 1e-12, "self-distance should be 0, got {d_self}");
    println!("  d(a,a) = {d_self:.2e}  ✓");

    let dab = a.geodesic_distance(&b);
    let dba = b.geodesic_distance(&a);
    assert!((dab - dba).abs() < 1e-12, "symmetry broken: d(a,b)={dab} d(b,a)={dba}");
    println!("  d(a,b) = {dab:.8},  d(b,a) = {dba:.8}  symmetric ✓");

    // triangle inequality
    let mid = a.slerp(&b, 0.5);
    let d_a_mid = a.geodesic_distance(&mid);
    let d_mid_b = mid.geodesic_distance(&b);
    assert!(dab <= d_a_mid + d_mid_b + 1e-12,
        "triangle inequality: {dab} > {d_a_mid} + {d_mid_b}");
    println!("  triangle: {dab:.6} ≤ {d_a_mid:.6} + {d_mid_b:.6}  ✓");

    // ── #34  frechet_mean ─────────────────────────────────────────────────────
    println!("\n--- frechet_mean ---");

    // Single motor → returns that motor
    let m1 = Motor::translator(1.0, -1.0, 2.0);
    let mean1 = Motor::frechet_mean(&[m1], 1e-8, 20);
    let dist1 = mean1.geodesic_distance(&m1);
    assert!(dist1 < 1e-8, "single-motor mean dist = {dist1}");
    println!("  mean([m]) dist from m = {dist1:.2e}  ✓");

    // Two motors: mean ≈ slerp midpoint
    let ma = Motor::identity();
    let mb = Motor::rotor(TAU / 4.0, Pga3::basis(0b0110));
    let mean2 = Motor::frechet_mean(&[ma, mb], 1e-10, 30);
    let slerp_mid = ma.slerp(&mb, 0.5);
    let p = Pga3::point(0.0, 1.0, 0.0);
    let got = mean2.apply(&p).coeffs;
    let exp = slerp_mid.apply(&p).coeffs;
    let err: f64 = got.iter().zip(exp.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(err < 1e-7, "two-motor mean vs slerp midpoint err = {err}");
    println!("  mean([id, rotor90]) vs slerp(0.5) err = {err:.2e}  ✓");

    // One-parameter subgroup: angles 0.1..0.9, mean ≈ 0.5
    let plane = Pga3::basis(0b0110);
    let angles = [0.1f64, 0.3, 0.5, 0.7, 0.9];
    let motors: Vec<Motor<f64>> = angles.iter().map(|&a| Motor::rotor(a, plane)).collect();
    let mean_m = Motor::frechet_mean(&motors, 1e-10, 30);
    let avg = angles.iter().sum::<f64>() / angles.len() as f64;
    let expected = Motor::rotor(avg, plane);
    let p2 = Pga3::point(0.0, 1.0, 0.0);
    let err2: f64 = mean_m.apply(&p2).coeffs.iter()
        .zip(expected.apply(&p2).coeffs.iter())
        .map(|(a, b)| (a - b).abs()).sum();
    assert!(err2 < 1e-7, "subgroup mean err = {err2}, avg_angle = {avg}");
    println!("  subgroup mean (avg θ={avg:.1}) err = {err2:.2e}  ✓");

    // Empty slice panics
    let result = std::panic::catch_unwind(|| Motor::frechet_mean(&[], 1e-8, 20));
    assert!(result.is_err(), "expected panic on empty slice");
    println!("  frechet_mean([]) panics  ✓");

    println!("\n=== all checks passed ===");
}
