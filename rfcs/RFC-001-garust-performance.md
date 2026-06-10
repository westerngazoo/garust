# RFC 001: Edge Performance Optimizations for `garust`

**Author:** VIGA Team  
**Status:** Draft / Request for Comments  
**Target:** `garust` (Geometric Algebra in Rust)  

## 1. Context and Motivation
In recent neck-to-neck benchmarking for the VIGA (Visual Intelligence with Geometric Algebra) edge project, we identified a critical performance gap. While `garust` provides elegant, mathematically sound spatial tracking without interpreter overhead, its current raw throughput (~180,000 FPS) trails behind a standard Python baseline utilizing `numpy` (~327,000 FPS).

The bottleneck stems from `garust-core`'s reliance on dense, scalar-by-scalar arithmetic loops for products (e.g., the `O(DIM²)` geometric product in `products.rs`). In contrast, `numpy` relies on highly optimized BLAS/LAPACK C/Fortran backends that heavily exploit hardware SIMD (Single Instruction, Multiple Data).

To establish `garust` as the premiere geometric algebra framework for edge computing, we must close this gap.

## 2. Proposed Architecture

We propose two primary architectural changes to `garust`: **Portable SIMD Vectorization** and **Grade-Aware Sparse Operations**.

### 2.1 Portable SIMD Vectorization
The current implementation of the geometric product in `crates/garust-core/src/multivector.rs` looks like this:
```rust
for a in 0..A::DIM {
    for b in 0..A::DIM {
        let (idx, sign) = blade_product(a, b, A::P, A::Q);
        if sign != 0 {
            let term = self.coeffs[a] * rhs.coeffs[b];
            // Add or subtract term from out.coeffs[idx]
        }
    }
}
```
**Proposal**: Introduce Rust's `core::simd` (Portable SIMD) feature to vectorize these loops.
- **Data Layout**: Since `coeffs` is exactly sized as `[T; 4]`, `[T; 8]`, `[T; 16]`, or `[T; 32]`, these map perfectly to SIMD vectors like `f32x4`, `f32x8`, and `f32x16`.
- **Precomputed Masks**: The signs and target indices (`idx`, `sign`) from `blade_product` are strictly a function of the bit-index. We can precompute these routing masks as `const` arrays. The product then becomes a series of SIMD shuffles, fused multiply-adds (FMA), and sign flips.

### 2.2 Grade-Aware Sparse Operations
In edge tracking pipelines (like VIGA), we rarely multiply two dense multivectors where every coefficient is non-zero.
- When casting a 3D ray in PGA (`Pga3`), we only populate grades 1 and 2 (vectors and bivectors).
- When applying a Motor (`Motor3`), we are applying an even subalgebra.

**Proposal**: Introduce strongly-typed, sparse representations for specific grades, or fast paths inside `Mul`.
- For specific operations like the Sandwich Product (`R * v * ~R`), we can write hardcoded, unrolled math routines for `Motor3 * Point` instead of leaning entirely on the generic dense `Multivector` product loop. 
- A specialized `impl Mul<PgaPoint> for Motor3` would skip 90% of the arithmetic because we mathematically guarantee most operands are zero.

## 3. Implementation Plan

If approved, the implementation will proceed in three phases:

1. **Phase 1: Const-Time Precomputations**
   - Refactor `blade_product` logic to generate `const` arrays mapping indices and signs at compile-time, eliminating the branch overhead `if sign != 0` inside the hot loops.
2. **Phase 2: Unrolled Sandwich Fast-Paths**
   - Implement hand-unrolled logic for `Motor::apply` in `garust-geo` to immediately benchmark the gains for VIGA's specific use case.
3. **Phase 3: SIMD Intrinsics**
   - Gate `core::simd` usage behind a `garust` cargo feature (e.g., `feature = "simd"`), allowing the core library to retain standard compatibility where SIMD isn't supported. 
   - Implement `f32x16` multiplication for `Pga3f` and `Sta3f`.

## 4. Open Questions

1. **Compiler Channel**: `core::simd` currently requires the Nightly Rust compiler. Should we use `core::simd` (Nightly only for now), or drop down to a third-party crate like `wide` or `safe_arch` which you are already fetching as dependencies elsewhere, to stay on Stable Rust?
2. **Code Generation**: Should we use the `garust-derive` procedural macro to auto-generate the unrolled sandwich products, or hand-write them for the primary algebras (`Pga3`, `Cga3`)?
3. **Approval**: Does this RFC align with your vision for `garust`?

---

## Appendix A — Implementation decisions (resolved 2026-06-03)

The RFC was accepted with these adjustments, reflected in the agreed plan below:

- **Measure first.** A `criterion` harness (dev-dependency only — out of the
  normal dependency graph, like `proptest`) lands before any optimization, so
  every change is quantified. Baselines are in Appendix B.
- **Sparsity before SIMD.** The hot path (transform a point with a versor) is
  a sandwich = *two* dense `O(DIM²)` products over mostly-zero operands (the
  baseline confirms it). Specialized closed-form `apply` is the largest single
  win and needs no SIMD.
- **Const Cayley table via the `Algebra` seam.** A `2^N × 2^N` table can't be
  sized generically (the `generic_const_exprs` limit that gave us
  `BladeStore`), so the table is exposed through a new `Algebra` associated
  item, generated by `define_algebra!` from a `const fn` `blade_product`. This
  is the "cached Cayley table" hook `algebra.rs` was always designed for.
- **SIMD on stable only.** `core::simd` (nightly) is rejected; an explicit
  backend uses the stable [`wide`](https://crates.io/crates/wide) crate behind
  an optional `simd` feature (a *new* optional dep — the crate is otherwise
  zero-dependency, contrary to the §4 note).
- **Correctness net.** Every fast path (table-driven product, unrolled
  sandwich, SIMD) is `proptest`-verified to be bit-faithful to the generic
  reference; the existing law-tests guard associativity etc.

### Agreed phase order (by ROI × risk)

0. **Benchmarks** — criterion baselines (done; Appendix B).
1. **Const Cayley tables** — branchless general product via the `Algebra` seam.
2. **Specialized `Motor`/`Conformal::apply`** — closed-form sandwich fast paths.
3. **Batch / SoA `apply`** over slices, leaning on the `bytemuck` layout (the
   fair comparison to batched `numpy`).
4. **Explicit SIMD** — stable `wide` behind a `simd` feature.

## Appendix B — Measured baseline (2026-06-03)

Criterion, release build, `aarch64-apple-darwin`. Lower is better.

| Benchmark                       | Time      |
|---------------------------------|-----------|
| geometric product, Vga3 (8)     | ~52 ns    |
| geometric product, Pga3 (16)    | ~324 ns   |
| geometric product, Cga3 (32)    | ~1.99 µs  |
| wedge, Cga3                     | ~535 ns   |
| regressive (meet), Cga3         | ~685 ns   |
| `Motor::apply` point (PGA)      | ~648 ns   |
| `Conformal::apply` point (CGA)  | ~5.0 µs   |
| motor compose (PGA product)     | ~337 ns   |

Note `Motor::apply` ≈ 2 × the Pga3 product and `Conformal::apply` ≈ 2 × the
Cga3 product: the sandwich is exactly two dense products, confirming the
sparsity opportunity that phases 1–2 target.

## Appendix C — Phase 1 results: const Cayley tables

Replacing the per-pair `blade_product` call in the geometric product with a
compile-time Cayley table (`Algebra::CAYLEY`, generated by `define_algebra!`)
gives, on the same machine:

| Benchmark                       | Baseline | Phase 1  | Speed-up |
|---------------------------------|----------|----------|----------|
| geometric product, Vga3 (8)     | ~52 ns   | ~6.6 ns  | 7.8×     |
| geometric product, Pga3 (16)    | ~324 ns  | ~139 ns  | 2.3×     |
| geometric product, Cga3 (32)    | ~1.99 µs | ~517 ns  | 3.9×     |
| `Motor::apply` point (PGA)      | ~648 ns  | ~125 ns  | 5.2×     |
| `Conformal::apply` point (CGA)  | ~5.0 µs  | ~478 ns  | 10.5×    |
| motor compose (PGA product)     | ~337 ns  | ~61 ns   | 5.5×     |

The sandwich-based `apply` paths inherit the product speed-up (they call it
twice), so the headline VIGA hot path is already ~5–10× faster *before* the
phase-2 closed-form `apply`. `wedge`/`regressive` are unchanged — they use
their own loops, not the geometric-product table, and get their own tables in
a later step if benchmarks warrant. The table-driven product is proptest-
verified bit-identical to the `blade_product` reference.

## Appendix D — Phase 2 results: sparse sandwich

`sandwich` (and therefore `Motor`/`Conformal::apply` and the typed
`transform`s) now runs as two *sparse* geometric products that skip blade
pairs with a zero coefficient on either side — exactly the all-zero blades
that dominate a sandwich's operands (an even-grade versor is half zeros; a
point/line/plane is a single grade). It is correct by construction (a zero
coefficient contributes a `0` term either way) and proptest-verified equal to
the `self * x * ~self` product form.

| Benchmark                      | Baseline | Phase 1  | Phase 2  | Total   |
|--------------------------------|----------|----------|----------|---------|
| `Motor::apply` point (PGA)     | ~648 ns  | ~125 ns  | ~78 ns   | **8.3×**  |
| `Conformal::apply` point (CGA) | ~5.0 µs  | ~478 ns  | ~375 ns  | **13.3×** |

The plain product `*` stays dense (its per-pair zero test would only slow the
dense workloads it is tuned for); only the sandwich opts into sparsity. The
hot path is now ~8–13× over the original baseline — comfortably past the
numpy figure that motivated this RFC.

## Appendix E — Phases 3 & 4: batch API and SoA SIMD

**Phase 3** adds `Multivector::sandwich_each` and `Motor`/`Conformal::apply_each`,
transforming a whole slice in place with the versor reversed once. Scalar, so
the win is small (~4%), but it is the data-parallel shape phase 4 vectorizes.

**Phase 4** adds `Motor`/`Conformal::apply_each_simd` behind a `simd` feature
(the stable [`wide`](https://crates.io/crates/wide) crate — *not* nightly
`core::simd`). The batch is laid out structure-of-arrays — coefficient `j` of
four consecutive objects in one `f64x4` — and the sandwich runs with the
versor's coefficients as broadcast scalars. No cross-lane shuffles: each lane
is an independent object. The tail (< 4 objects) falls back to the scalar
path, and a test checks the SIMD result matches the scalar batch.

Batch of 1024 PGA points (criterion):

| Variant                       | Time / 1024 pts | Per point |
|-------------------------------|-----------------|-----------|
| per-point `apply` loop        | ~81 µs          | ~79 ns    |
| `apply_each` (scalar batch)   | ~80 µs          | ~78 ns    |
| `apply_each_simd`             | ~47 µs          | ~46 ns    |

~1.7× over the scalar batch on this aarch64/NEON machine (where `f64x4` is two
128-bit halves); on x86 AVX, `f64x4` is native 256-bit, so closer to ~3–4×.
Combined with phases 1–2, point-cloud throughput is ~14× the original
baseline. `wide` stays out of the default dependency graph (verified with
`cargo tree -e normal`), and `simd` composes with `no_std` + `libm`.

### Status: phases 0–4 shipped. Phase order followed ROI × risk; remaining
ideas (own tables for `wedge`/`regressive`, `f32` SIMD lanes, wider vectors)
are left to a future round, driven by an end-to-end VIGA re-measurement.

## Appendix F — Phase 5 (continued): sparse wedge table

The wedge was the one product the const-table work hadn't touched. The
phase-1 recipe — a dense `DIM × DIM` table indexed in the hot loop — turned
out to **pessimize** it: the wedge of two basis blades vanishes whenever they
share a generator, so only `3^N` of the `4^N` pairs survive (76% are zero at
`N = 5`), and the dense table traded the old loop's cheap `a & b != 0` skip
for an unconditional multiply-accumulate on every pair. Measured on
`Cl(4,1,0)`: wedge ~507 ns → ~590 ns (+15%). The dense layout was right for
the geometric product only because its pairs are nearly all live.

The fix is a **sparse, CSR-style table** (`Algebra::WEDGE`, a
`signature::WedgeTable`): per left blade `a`, only the surviving
`(b, a | b, swap_sign)` cells, built at compile time by const fns
`wedge_rows` / `wedge_pairs`. The hot loop visits exactly the `3^N` live
pairs — no overlap test, no `swap_sign` — and still skips whole rows when the
left coefficient is zero. It is also *smaller* than the dense table (`3^N × 6 B`
vs `4^N × 4 B`; 1.5 KB vs 4 KB at `N = 5`). The wedge is metric-independent,
so the table never sees `(p, q)`.

`Cl(4,1,0)` dense operands (criterion, aarch64):

| Op                      | Before   | After    | Speedup |
|-------------------------|----------|----------|---------|
| `wedge` (a ∧ b)         | ~507 ns  | ~150 ns  | **3.4×** |
| `regressive` (a ∨ b)    | ~674 ns  | ~283 ns  | **2.4×** |

(`regressive` rides the same table through its complement-wedge-complement
form.) A `wedge_matches_reference` proptest pins the table path bit-faithful
to the definitional `swap_sign` loop, mirroring the geometric product's
reference law. Still open: `f32` SIMD lanes / wider vectors, pending an
end-to-end VIGA re-measurement.
