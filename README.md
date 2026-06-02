# garust

[![CI](https://github.com/westerngazoo/garust/actions/workflows/ci.yml/badge.svg)](https://github.com/westerngazoo/garust/actions/workflows/ci.yml)

**Geometric Algebra in Rust — generic over the Clifford signature `Cl(P, Q, R)` and the scalar type.**

A from-scratch, **zero-dependency** implementation of Geometric Algebra
(a.k.a. Clifford Algebra). You pick your algebra *and* your numeric
precision at the type level: 2D/3D Euclidean, Projective (PGA), Conformal
(CGA), Spacetime (STA), over `f64`, `f32`, or any scalar type you supply.

```toml
[dependencies]
garust = "0.1"
```

## The signature `Cl(P, Q, R)` in one paragraph

An algebra is fixed by three integers: how many basis vectors square to
`+1` (`P`), how many square to `−1` (`Q`), and how many square to `0`
(`R`, the degenerate / null ones). From those `N = P + Q + R` generators
you build `2^N` basis *blades* by wedging together every subset:

| signature   | name              | blades | what it buys you |
|-------------|-------------------|--------|------------------|
| `Cl(2,0,0)` | 2D Euclidean      | 4      | `1, e1, e2, e12` |
| `Cl(3,0,0)` | 3D Euclidean      | 8      | rotations as rotors |
| `Cl(3,0,1)` | 3D Projective     | 16     | points, lines, planes, rigid motions |
| `Cl(4,1,0)` | 3D Conformal      | 32     | adds circles, spheres, conformal maps |
| `Cl(1,3,0)` | Spacetime         | 16     | signature `(+, −, −, −)` |

Each ships two ready-made aliases — an `f64` one and an `f32` one with an
`f` suffix — so you rarely type the generic form:

```rust
use garust::{Vga2, Vga3, Pga3, Cga3, Sta};   // f64
use garust::{Vga2f, Vga3f, Pga3f, Cga3f, Staf}; // f32
```

## Quick start

```rust
use garust::Vga3;

// Vectors add, scale, and multiply with the geometric product.
let a = Vga3::basis(1) + Vga3::basis(2);     // e1 + e2
let b = Vga3::basis(2) + Vga3::basis(3);     // e2 + e3

// a*b splits into the symmetric (inner) and antisymmetric (wedge) parts:
//   a*b = a·b + a∧b
assert_eq!((a * b), a.inner(&b) + a.wedge(&b));

// |a|² for a Euclidean vector is its squared length.
assert_eq!(a.norm_squared(), 2.0);
```

### Rotations as rotors

```rust
use garust::Vga3;
use std::f64::consts::FRAC_PI_2;

// A unit rotor for 90° in the e23 plane (i.e. about the x-axis):
//   R = exp(−θ/2 · e23)
let r = (Vga3::basis(0b110) * (-FRAC_PI_2 / 2.0)).exp();

// Apply it with the sandwich product R x ~R: it sends e2 → e3.
let rotated = r.sandwich(&Vga3::basis(2)).cleaned(1e-10);
assert!((rotated.coeffs[4] - 1.0).abs() < 1e-10); // the e3 coefficient ≈ 1
```

### PGA geometry: points, planes, and the lines that meet & join them

```rust
use garust::Pga3;

// Three planes meet at a point (the wedge ∧):
let px = Pga3::plane(1.0, 0.0, 0.0, -1.0); // x = 1
let py = Pga3::plane(0.0, 1.0, 0.0, -2.0); // y = 2
let pz = Pga3::plane(0.0, 0.0, 1.0, -3.0); // z = 3
assert_eq!(px.wedge(&py).wedge(&pz), Pga3::point(1.0, 2.0, 3.0));

// Two points join into the line through them (the regressive product ∨):
let line = Pga3::point(0.0, 0.0, 0.0).line_through(&Pga3::point(1.0, 0.0, 0.0));
```

### Motors: rigid-body motions

```rust
use garust::{Motor3, Pga3};
use std::f64::consts::FRAC_PI_2;

// Rotate 90° about the x-axis, then translate +3 along x.
let r = Motor3::rotor(FRAC_PI_2, Pga3::basis(0b0110)); // e23 plane
let t = Motor3::translator(3.0, 0.0, 0.0);
let m = t * r;                       // compose: `*` applies r first

let moved = m.apply(&Pga3::point(0.0, 1.0, 0.0)); // → point(3, 0, 1)
let back  = m.inverse().apply(&moved);            // → point(0, 1, 0)
```

### Conformal transforms: CGA adds scaling

Where a `Motor` covers rigid motions, a `Conformal` versor in CGA
`Cl(4,1,0)` also gives you uniform **scaling** about the origin — and
it acts on spheres and planes just as readily as on points.

```rust
use garust::{Cga3, Conformal3};

let scale = Conformal3::dilator(2.0);          // ×2 about the origin
let shift = Conformal3::translator(1.0, 0.0, 0.0);

// Order matters, just like matrices:
let p = Cga3::cga_point(1.0, 0.0, 0.0);
let a = (shift * scale).apply(&p).to_euclidean(); // ×2 then +1 → (3,0,0)
let b = (scale * shift).apply(&p).to_euclidean(); // +1 then ×2 → (4,0,0)

// A dilation grows the whole sphere, not just a point:
let unit = Cga3::sphere(0.0, 0.0, 0.0, 1.0);
let big  = scale.apply(&unit);                    // now radius 2
```

## What's implemented

- `Multivector<T, P, Q, R, DIM>` — a dense `[T; 2^N]` element, generic
  over signature and scalar
- linear ops: `+`, `−`, negation, scalar multiplication (both sides),
  equality, `Display`
- the **geometric product**, plus wedge `∧`, inner `·`, and scalar
  product `⟨ab⟩₀`
- grade projection, reverse, grade involution, Clifford conjugation,
  `norm_squared`
- versor inverse, the **sandwich product**, and a closed-form `exp`
  (the bivector → rotor bridge)
- duality: the **pseudoscalar**, metric-independent left/right
  complements, and the **regressive product** `∨` (the *meet*)
- **PGA constructors** for `Cl(3,0,1)`: `point`, `plane`, `line_through`
- **`Motor`** — rotors, translators, and their screw-motion compositions
- **CGA constructors** for `Cl(4,1,0)`: `cga_point`, `sphere`, `cga_plane`
- **`Conformal`** — translators, rotors, and origin dilations on the
  conformal model

## Generic over the scalar type

`Multivector` is parameterised by a coefficient type `T` implementing
[`Scalar`] (and [`Real`] for the exponential). `f32` and `f64` are
provided; any ordered field — dual numbers for autodiff, fixed-point
types — can opt in by implementing those traits. For a custom scalar `S`,
name the full form: `Multivector::<S, 3, 0, 0, 8>`.

## Design notes

- **Zero dependencies.** Just the standard library.
- **Metric-independent duality.** The complements are combinatorial, not
  `M·I⁻¹`, so they stay well-defined in degenerate algebras like PGA
  where the pseudoscalar is null.
- **`O(DIM²)` products.** Fine for the algebras a human writes by hand
  (≤ 1024 ops for `Cga3`); not tuned for large-`N` work.
- **One indexing convention.** Blade index = bitmask of its generators;
  `coeffs[0]` is the scalar. Generators partition by index into the `+1`,
  `−1`, then `0` groups. See the `signature` module docs for details.

## License

Licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE) at your option.
