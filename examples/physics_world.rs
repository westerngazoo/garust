//! physics_world — a runnable rigid-body simulation on `garust-physics`.
//!
//! Two vignettes: a couple of balls bouncing on a floor (the `World` drive
//! loop — integrate, detect, resolve), and the Dzhanibekov / tennis-racket
//! effect (a free body flipping about its intermediate axis, the dramatic
//! known-answer test for the symplectic integrator).
//!
//! Run it:
//!
//! ```sh
//! cargo run --release --example physics_world --features physics
//! ```

use garust::physics::world::{Body, World};
use garust::physics::{Inertia, RigidBody};
use garust::Pga3;

fn main() {
    bouncing_balls();
    println!();
    tumbling_body();
}

/// Drop two balls into a gravity well with a floor; one `world.step` per frame
/// advances gravity, collisions, and the ground bounce together.
fn bouncing_balls() {
    println!("== Bouncing balls on a floor (World drive loop) ==");
    let world = World::new().with_ground(0.0); // earth gravity, floor at y = 0

    let mut balls = [
        {
            let mut b = Body::ball(1.0, 0.5);
            b.rigid.position = [0.0, 4.0, 0.0];
            b.restitution = 0.9; // lively
            b
        },
        {
            let mut b = Body::ball(2.0, 0.5);
            b.rigid.position = [0.4, 6.0, 0.0]; // offset so they also collide
            b.restitution = 0.6; // dead-er
            b
        },
    ];

    let dt = 1.0 / 120.0;
    println!("  t (s)   ball0.y   ball1.y");
    for frame in 0..=720 {
        world.step(&mut balls, dt);
        if frame % 120 == 0 {
            println!(
                "  {:4.1}    {:6.3}    {:6.3}",
                frame as f64 * dt,
                balls[0].rigid.position[1],
                balls[1].rigid.position[1]
            );
        }
    }
    println!("  → each loses height per bounce and settles toward y = radius = 0.5");
}

/// A free asymmetric body spun about its (unstable) intermediate axis: the
/// spin axis periodically flips, while energy and ‖Π‖ stay put.
fn tumbling_body() {
    println!("== Dzhanibekov effect: a free body flips about its middle axis ==");
    let inertia = Inertia::principal([2.0, 3.0, 4.0]); // Ix < Iy < Iz
    let planes = Inertia::principal_planes();

    let mut body = RigidBody::new(1.0);
    body.angular_momentum = planes[1] * 5.0   // big spin about the intermediate (y) axis
        + planes[0] * 0.01; // a tiny nudge to seed the instability

    let momentum_y = |b: &RigidBody| -b.angular_momentum.scalar_product(&planes[1]);
    let energy0 = body.kinetic_energy(&inertia);
    let norm0 = body.angular_momentum_squared();

    let dt = 0.005;
    let mut prev = momentum_y(&body).signum();
    let mut flips = 0;
    for step in 0..20_000 {
        body = body.step(dt, &inertia, [0.0; 3], Pga3::zero());
        let sign = momentum_y(&body).signum();
        if sign != prev {
            flips += 1;
            prev = sign;
            if flips <= 5 {
                println!(
                    "  flip {} at t = {:5.2}s   Π_y now {:+.2}",
                    flips,
                    step as f64 * dt,
                    momentum_y(&body)
                );
            }
        }
    }
    println!("  total intermediate-axis flips: {flips}");
    println!(
        "  energy drift {:.1e}, ‖Π‖² drift {:.1e}  (conserved — symplectic)",
        (body.kinetic_energy(&inertia) - energy0).abs(),
        (body.angular_momentum_squared() - norm0).abs()
    );
}
