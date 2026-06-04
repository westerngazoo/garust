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
