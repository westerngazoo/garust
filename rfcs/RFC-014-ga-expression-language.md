# RFC 014: A GA Expression Language on garust

**Author:** garust maintainers (drafted with the owner, 2026-08-24)
**Status:** Draft / Request for Comments — nothing is built until reviewed
**Target:** `garust` (the kernel is the point: this language is a thin
skin over it, never a re-implementation)
**Relates to:** physics-lab RFC-001 §6 (the lesson-language trigger);
the `clifford`/`cliffordc` project, deliberately (see §6)

## 1. Context and motivation

The owner's directive, verbatim in spirit: *build a Clifford-flavored
language on top of garust — garust is the most important piece we have —
and it may well be Rust itself.* This RFC resolves that into two layers,
because "a language" has two honest readings and they want different
machinery:

1. **Notation for humans writing Rust** — make GA-on-garust read like GA
   on paper inside ordinary Rust programs (motoreel scenes, physics-lab
   lessons, tests).
2. **Notation for humans who are not writing Rust** — a tiny interpreted
   expression language, so a *student in a browser lesson* or the owner
   at a REPL can type `R = exp(-a/2 * e12)` and watch garust evaluate it.

The second is the killer application. physics-lab's pedagogy is
claim-first and interactive; an **interactive multivector console** —
type an expression, garust computes it live via the already-proven
wasm architecture (38 KB, hand C-ABI, one fixed runtime) — turns the
algebra itself into a lesson. Nothing like it exists in the ecosystem,
and every piece it needs is already built and tested except the parser.

## 2. What this is NOT

- **Not cliffordc.** The `clifford` language continues its own 2026-05
  pivot as an effect-system/separation-logic *systems* language; its GA
  framing was retired by its own audit. This RFC is a pure *expression
  and notation* layer — no effects, no automata, no compiler backend, no
  concurrency claims. Two projects, two jobs; the name overlap is a real
  hazard and naming is therefore an owner decision (§7 Q1).
- **Not a physics engine or a re-implementation of anything.** Every
  operation bottoms out in a garust call. If the interpreter can compute
  something garust cannot, that is a bug in this RFC.
- **Not a general programming language.** Expressions, bindings, and a
  handful of functions. Loops, conditionals, I/O, effects: out of scope
  permanently — that road leads to badly rebuilding cliffordc.

## 3. Layer A — the embedded notation (Rust is the language)

A `garust::prelude`-style module plus operator sugar, addressing the
"maybe it has to be Rust as well" reading at near-zero cost:

- Const/fn shorthands: `e1(), e2(), e12(), e0()…` per algebra; `tau()`;
  `scalar(x)`.
- Operator traits garust already has (`*`, `+`, `-`) documented as the
  notation, plus named methods where Rust has no operator: `a.wedge(&b)`
  stays, but gains doc-aliases so `∧` is searchable.
- A `mv!` macro for literals: `mv!(1 + 2 e1 - 0.5 e12)` — compile-time
  parsed, expands to constructor calls, zero runtime cost.

Layer A is a small additive garust PR (feature `notation`), useful on
its own to motoreel and the lessons crates immediately.

## 4. Layer B — the expression language (the console)

A deliberately tiny interpreted language. Working name in this draft:
**`wedge`** (§7 Q1).

### 4.1 Surface

```
// bindings and expressions; newline- or ;-terminated
a  = e1 + 2*e2
B  = e1 ^ e2                  // wedge; ASCII ^ is the operator
R  = exp(-tau/8 * e12)        // rotors the house way: tau, not pi
v2 = R >> a                   // sandwich: R a ~R  (>> is "apply")
s  = a | b                    // inner (left contraction: §7 Q3)
rev(R); norm(a); grade(B, 2); dual(B)
```

- **Literals:** f64, `tau` as a keyword constant. Angles are τ-measured;
  `pi` does not exist in the language (τ/2 is spellable).
- **Basis names:** `e0…e3`, `e01, e12, e123 …` resolved against the
  *active algebra*, declared per session: `algebra vga2 | vga3 | pga3`.
  Unknown blades for the active algebra are errors, not zeros.
- **Operators** (precedence high→low): unary `-` and `~` (reverse);
  `*` geometric; `^` wedge and `|` inner (equal, left-assoc,
  parenthesize mixed uses — a *lint, not a footgun*: mixing `^` and `|`
  unparenthesized is a parse error by design); `>>` sandwich; `+ -`.
- **Functions:** `exp, log, rev, norm, norm2, grade(x, k), dual, inv`,
  and per-algebra constructors (`point(x,y,z)`, `plane(a,b,c,d)`,
  `translator(…)`, `rotor(angle, plane)`, `motor(…)`) — exactly the
  garust public surface, nothing else.
- **Bindings** are single-assignment per name per session (shadowing
  allowed, mutation not). The environment is a `Vec<(name, Multivector)>`.

### 4.2 Implementation shape

One new crate, `garust-lang` (workspace member, off-by-default feature
like `physics`):

- Hand-written lexer + Pratt parser (~600 lines; no parser deps — the
  zero-dependency discipline holds).
- Evaluator over an enum `Value { Mv(Multivector<A, f64>), Scalar(f64) }`
  generic in the algebra via the same signature markers garust already
  uses; `no_std + alloc` core so it drops into the physics-lab wasm
  architecture unchanged.
- Every eval step calls garust; the crate contains **zero arithmetic**.
- Errors are typed and positioned (`UnknownBlade { name, algebra }`,
  `MixedWedgeInner { span }`, …) — they will be shown to students, so
  they are part of the pedagogy and get written with care.

### 4.3 First consumers, in order

1. **REPL** (`garust-lang` example binary): the owner's own study tool.
2. **physics-lab lesson: "The Multivector Console"** — the console as a
   lesson, with `tryThis` prompts driving it ("compute e1^e2 ^ e1 — why
   zero?"). Rides the existing runtime + wasm contract; the lesson crate
   embeds `garust-lang` and exposes `eval_line()` over the C-ABI.
3. **Authoring experiments** — whether lesson scenes read better in
   `wedge` than in Rust is a question to *answer with these two
   consumers*, not to presuppose (§7 Q4).

## 5. Verification

The house rules, applied to a language:

- Property tests: parse∘print round-trips; evaluator vs hand-written
  garust calls agree bit-for-bit on generated expression trees.
- The console lesson's claims are cargo-tested like any lesson's.
- A golden corpus of expressions with expected multivector outputs —
  diffable, like the SVG goldens.

## 6. Relationship to cliffordc (explicit, to prevent drift)

cliffordc keeps: sigils, automata, effects, SRP, codegen, its spec and
its own roadmap. `wedge` keeps: expressions over garust, full stop.
If `wedge` ever wants statements, control flow, or effects, that is the
signal to stop and reread this section. The two share an owner and a
taste for τ; they must not grow toward each other.

## 7. Open questions for the owner

1. **Name.** `wedge` (this draft's placeholder; note the owner's shell
   logs already use wedge-*.log names), `garust-lang`, `mv`, or a
   reclaimed `clifford-notation`? Collision with cliffordc argues for
   distance.
2. **Home.** garust workspace member behind a feature (this draft's
   assumption — lockstep with the kernel it skins), or standalone repo?
3. **Inner product choice.** garust exposes `inner` and scalar product;
   the language's `|` must pin ONE convention (left contraction
   recommended) and name it in the docs — the classic GA-notation trap,
   decided once here rather than per-user.
4. **Layer A macro scope.** Is `mv!` worth proc-macro machinery (garust
   currently gates proc-macros behind the `derive` feature), or is the
   fn-call prelude enough for v1?
5. **Sequencing.** This is program-stage-1-adjacent (it exists FOR the
   lessons). Proposed: Layer B core + REPL as the next garust milestone
   after motoreel R-0005 lands; the console lesson follows as
   physics-lab S3 material. Approve or reorder.
