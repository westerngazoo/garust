# garust-derive

The optional `#[derive(Algebra)]` proc-macro for
[garust](https://crates.io/crates/garust) — the derive path to a Clifford
signature `Cl(P, Q, R)`.

The default way to mint a signature is the zero-dependency declarative macro
`define_algebra!` in
[`garust-core`](https://crates.io/crates/garust-core). This derive is for when
you'd rather write the marker type yourself — so it can carry your own derives,
attributes, and docs — and have only the `Algebra` impl generated:

```rust
use garust::{Algebra, Multivector};

/// 3D Projective GA, `Cl(3, 0, 1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Algebra)]
#[algebra(p = 3, q = 0, r = 1)]
pub struct Pga3Sig;
```

Don't depend on this crate directly — enable it through the umbrella's feature:
`garust = { version = "…", features = ["derive"] }`.

## License

MIT OR Apache-2.0.
