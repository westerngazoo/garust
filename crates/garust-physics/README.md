# garust-physics

**Rigid-body physics** on the [garust](https://crates.io/crates/garust) PGA
kernel.

- `RigidBody` — full 6-DOF state: orientation as a `Motor`, angular momentum as
  a grade-2 bivector, plus centre-of-mass position / linear momentum / mass.
- `Inertia` — a principal-moment inertia operator (grade-2 → grade-2).
- a **symplectic Lie-group integrator** — momentum advanced by an exact
  per-principal-axis splitting (conserves `‖Π‖` and energy without drift),
  orientation transported on the group by `exp` (stays a unit versor); it
  reproduces the Dzhanibekov / tennis-racket effect.
- `contact` — sphere collision detection and frictionless impulse response.
- `World` — the `integrate → detect → resolve` drive loop over a caller-owned
  `&mut [Body]` (so the crate stays allocation-free).

See RFC-010 in the [repository](https://github.com/westerngazoo/garust) for the
design. Available through the umbrella's `physics` feature
(`garust = { version = "…", features = ["physics"] }`), re-exported as
`garust::physics`.

`#![no_std]`; depends only on `garust-core` and `garust-geo`.

## License

MIT OR Apache-2.0.
