# garust-geo

The **typed geometry layer** of [garust](https://crates.io/crates/garust),
built on the [`garust-core`](https://crates.io/crates/garust-core) kernel.

- `Motor` — rigid-body motions in 3D PGA `Cl(3, 0, 1)`: rotors, translators,
  and their screw-motion compositions; `compose`, `apply`, `log`, `slerp`, and
  `to_matrix` (the bridge to matrix-speed bulk transforms).
- `Conformal` — conformal transformations in 3D CGA `Cl(4, 1, 0)`: translators,
  rotors, and dilations.
- typed `pga` and `cga` objects (`Point`, `Line`, `Plane`, `Sphere`) with
  type-checked join / meet incidence.

Most users want the umbrella crate **[`garust`](https://crates.io/crates/garust)**,
which re-exports both this layer and the kernel.

`#![no_std]`; zero dependencies on the default feature set (optional `serde`,
`bytemuck`, and stable-SIMD `simd` features).

## License

MIT OR Apache-2.0.
