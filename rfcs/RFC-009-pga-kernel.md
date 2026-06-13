# RFC 009: PGA Kernel — `Cl(3,0,1)` as the Geometric Alphabet

**Author:** garust maintainers
**Status:** Accepted — Implemented
**Target:** `garust` (Geometric Algebra in Rust)
**Supersedes:** R-0002 (`G(3,0,0)` Euclidean GA — *accepted but unbuilt*)
**Program:** neuroevolution of geometric ASTs (R-0008 → R-0011); garust owns R-0009.

## 1. Context

The program sequences four capabilities, ordered so the cheap thing is proven
before the expensive thing:

```
R-0008  ENGINE          genetic search + predicate-discharge fitness + certificates
 (separate flow)        → proven on a KNOWN-ANSWER problem (rediscover Strassen)
   │  reuse the engine machinery wholesale
   ▼
R-0009  PGA KERNEL      Cl(3,0,1) multivectors, geo/outer/inner products, rotors/motors
 (THIS RFC — garust)    → the geometric alphabet
   │
   ▼
R-0010  GEOMETRIC LISP  GA operations as s-expr forms + GRADE INFERENCE (decidable
 (separate flow)        dimensional type system)
   │
   ▼
R-0011  NEUROEVOLUTION  AST = chromosome; node-mutation / dimensional-shift /
 (separate flow)        subtree-crossover; fitness = accuracy − parsimony −
                        grade-entropy. Minimal gate: rediscover the sandwich
                        R x R̃ for a known rotation, unaided.
```

`garust` owns **R-0009** only: the kernel and the alphabet the layers above
consume. R-0008's engine and R-0010/R-0011 live in the separate GA flow.

## 2. Decision

### 2.1 Signature: `Cl(3,0,1)`, superseding `G(3,0,0)`

R-0002 proposed `G(3,0,0)` — 3D Euclidean GA, 8 blades. It was accepted but
never built. We supersede it with **`Cl(3,0,1)`** (3D Projective GA, 16
blades, one null generator `e0` with `e0² = 0`).

The null generator is the whole point: it is what lets the even subalgebra
represent **translations**, and therefore the full group of rigid motions
(a *motor* = a screw, rotation composed with translation along its axis).
`G(3,0,0)` has no null direction, so it can express rotations but not
translations — it cannot represent the rigid-motion group the program's
geometric programs must search over. `Cl(3,0,1)` is the signature the GATr
(Geometric Algebra Transformer) lineage uses, for exactly this reason.

`G(3,0,0)` is **not removed** — it still ships as `Vga3Sig` for callers who
want Euclidean-only GA. The program simply standardizes its alphabet on
`Pga3Sig = Cl(3,0,1)`.

### 2.2 Coefficient field: real `f64` (not `Complex`)

- `f64` satisfies `Scalar`'s `PartialOrd + abs`; `Complex` does not. Real
  `Cl(3,0,1)` is therefore buildable today, with no dependency on the
  Complex-coefficient `Scalar` split that belongs to the separate EML pillar.
- It matches the GA-ML literature this program draws on (CliffordNet, GATr
  are real-GA).
- The Complex / EML line stays a **separate pillar**; this RFC does not couple
  to it.

### 2.3 The alphabet as data

R-0010/R-0011 need the kernel's operations as *enumerable, grade-typed
values* — building blocks for an AST or chromosome — not only as Rust
methods. This RFC includes a value-level operation catalog (`Op`, `GradeSet`)
to that end, with a clean ownership boundary (§5).

## 3. Status: Implemented

Every R-0009 requirement is on `main`. The kernel and typed geometry were
built over the project's early phases; the Lie bridge and the operation
catalog are the most recent additions.

| R-0009 requirement            | garust API (on `main`)                                                        |
|-------------------------------|-------------------------------------------------------------------------------|
| `Cl(3,0,1)` multivectors      | `Pga3` / `Pga3Sig` — 16 blades, null `e0`                                     |
| geometric product             | `Multivector::mul` (`*`), const Cayley table                                  |
| outer (wedge) product         | `Multivector::wedge`, sparse table                                            |
| inner product                 | `Multivector::inner`                                                          |
| regressive product (meet)     | `Multivector::regressive`                                                     |
| rotors                        | bivector `Multivector::exp`; `Motor::rotor`                                   |
| motors (rigid motions)        | `Motor`: translator / rotor / rotation_about / compose / apply / log / slerp  |
| Lie bridge (algebra ↔ group)  | total `exp`, `try_bivector_split`, principal `log` (PR #12)                   |
| the alphabet as data          | `catalog::{Op, GradeSet}` (PR #15)                                            |
| validation-gate target `R x R̃`| `Multivector::sandwich`, `Motor::apply`                                       |

The crate is real-`f64`-first (`Pga3 = Multivector<Pga3Sig, f64>`),
`#![no_std]`, zero-dependency by default, and `#![deny(missing_docs)]`.

## 4. Validation gate

The smallest known-answer geometric program — the geometric analogue of
Strassen for R-0008 — is to **rediscover the sandwich `R x R̃`**: evolve an
expression that rotates a vector by a known rotor without being handed the
sandwich form. Success is checkable by a predicate discharge, so a failure is
diagnosable.

The alphabet already expresses and evaluates the target, and can grade-type
it without evaluation:

```rust
use garust::{GradeSet, Op, Vga3};
use std::f64::consts::TAU;

let r = (Vga3::basis(0b011) * (-TAU / 8.0)).exp(); // quarter-turn rotor in e12
let x = Vga3::basis(1);                            // the vector e1

// Sandwich R x ~R, built from catalog ops — what R-0011 must rediscover:
let rx  = Op::Geometric.apply(&[r, x]);
let rxr = Op::Geometric.apply(&[rx, Op::Reverse.apply(&[r])]);
assert!((rxr - Vga3::basis(2)).norm() < 1e-12);    // e1 → e2 ✓

// Grade-type the same expression statically (rotor is {0,2}, vector {1}):
let rotor = GradeSet::EMPTY.with(0).with(2);
let vec = GradeSet::singleton(1);
let t = Op::Geometric.output_grades(
    &[Op::Geometric.output_grades(&[rotor, vec], 3), rotor],
    3,
);
// t = {1, 3}: a vector, plus the trivector the conservative type keeps
// (a *unit* rotor annihilates it).
```

## 5. Handoff boundary

This is the contract with the separate GA flow.

**garust (R-0009) provides:**
- The kernel: `Cl(3,0,1)` multivectors and all products / involutions /
  duality / versor algebra, over real `f64`.
- The alphabet: `Op` — a `Copy + Eq + Hash` symbol set of every operation,
  with `arity`, `name`, `apply`, and enumeration.
- The **per-operation grade signature**: `Op::output_grades(inputs, n)` — the
  dimensional type system's per-node rule.

**The separate GA flow owns:**
- **R-0010** — the s-expression / AST type, and grade **inference**: composing
  `output_grades` over a whole tree. (Decidable, per Haynes; garust supplies
  the per-node rule, the Lisp runs the propagation.)
- **R-0011** — genotype = AST; node-mutation / dimensional-shift /
  subtree-crossover; fitness = accuracy − parsimony − grade-entropy; reuse
  R-0008's engine (seeded PRNG, population, certificates) unchanged.

Explicitly **out of garust scope:** the AST type, the search loop, grade
inference across a tree, and value-carrying random constants (the engine
supplies these via the existing `Multivector::scalar` / `basis`; they are kept
out of `Op` so the alphabet stays `Copy + Eq + Hash`).

## 6. Numbering note

`garust`'s `rfcs/` directory carries **RFC-001** (edge performance), which is
garust-local. This document adopts the **program-wide R-0009** identifier to
coordinate with the cross-project tracker; the `002`–`008` range belongs to
sibling projects in the program (notably R-0008, the discovery engine). The
two numbering schemes coexist: garust-local RFCs and program RFCs that garust
happens to own.

## Appendix A — what R-0002 was, and why `Cl(3,0,1)` replaces it

R-0002 specified `G(3,0,0)`: the 8-blade Euclidean geometric algebra of 3D
space, with generators `e1, e2, e3` all squaring to `+1`. It captures
rotations (via the even subalgebra, the quaternions) and reflections, and is
the right tool for pure-rotation problems.

It is insufficient for this program because rigid motions include
**translation**, and a Euclidean GA cannot represent a translation as a
versor — there is no null direction to build the parabolic (translation)
rotor from. `Cl(3,0,1)` adds exactly one null generator `e0`, and with it:

- points, lines, and planes get a uniform grade-based representation;
- translators `exp(½ t·e0eᵢ)` are versors (the generating bivector is null,
  so the exponential truncates after one term);
- motors — rotation ∘ translation screws — are the even-subalgebra versors,
  covering every rigid motion.

**Migration:** none. `Vga3Sig` (`G(3,0,0)`) still ships for Euclidean-only
callers; nothing depending on it changes. New program work targets
`Pga3Sig` (`Cl(3,0,1)`).
