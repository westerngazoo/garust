# garust-core

The signature-generic **geometric-algebra kernel** of
[garust](https://crates.io/crates/garust): Clifford algebras `Cl(P, Q, R)` in
the abstract, knowing nothing about any particular geometry.

- `Algebra` — a signature reified as a zero-sized marker type; mint your own
  with `define_algebra!`.
- `Multivector<A, T>` — the dense `[T; 2^N]` element, generic over the
  signature `A` and the scalar `T`.
- the product kernels (geometric, wedge, inner, scalar), the involutions,
  versor inversion / sandwich / `exp` / `log`, metric-independent duality,
  forward-mode AD (`Dual`), `∇_X` calculus, and symplectic dynamics.

Most users want the umbrella crate **[`garust`](https://crates.io/crates/garust)**,
which re-exports this kernel together with the typed geometry layer
`garust-geo`. Reach for `garust-core` directly only when you want the algebra
without the geometry.

`#![no_std]` and allocation-free; zero dependencies on the default feature set.

## License

MIT OR Apache-2.0.
