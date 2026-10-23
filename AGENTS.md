# garust

## Meta
- Context: personal            <!-- open source in ~/projects; not HappyDev: no company process, tooling or customers here -->
- Status: active
- Vault note: /Users/goose/vaults/goosecode/Projects/garust.md
- Owner: Gustavo Delgadillo (westerngazoo)

## What this is
Geometric Algebra in Rust, generic over the Clifford signature `Cl(P, Q, R)` and the
scalar type: a zero-dependency, `no_std` workspace (`garust-core`, `garust-geo`,
`garust-physics`, `garust-anim`, `garust-derive`, re-exported by the umbrella crate).
It is the one engine behind motoreel, physics-lab, guion and now the Tracking Ghost
Pepper display (`/Users/goose/code/personal/tracking-ghost-pepper`, library side in
`rfcs/RFC-014-cone-warp.md`). "Done" is never a feature count: it is that every
public item has a contract, a doctest and a rejection test, and the gates below are green.

## Commands
The merge gates are exactly what `.github/workflows/ci.yml` runs. Run them under CI's
toolchain, not the machine default (`rustup run stable ...`); a machine a few releases
behind cannot see the lints stable has since added.
```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets
cargo test --workspace --all-targets --all-features
cargo test --workspace --doc
cargo doc --workspace --no-deps --all-features
cargo hack check --no-dev-deps --workspace --exclude garust-derive --lib   # feature powerset
cargo build -p garust-core --no-default-features --features libm --target thumbv7em-none-eabi
cargo build -p garust-geo  --no-default-features --features libm --target thumbv7em-none-eabi
```

## Architecture
`crates/garust-core` (multivector, compile-time Cayley table, signatures, scalar trait),
`crates/garust-geo` (PGA/CGA primitives, `Motor`, kinematic chains, SIMD paths behind
`simd`), `crates/garust-physics` (RFC-010), `crates/garust-anim` (RFC-012),
`crates/garust-derive` (`#[derive(Algebra)]`). Design history is in `rfcs/`; the
generated tour of the code is in `docs/`. The umbrella crate at the root re-exports the
members and owns the feature flags (`std`, `libm`, `derive`, `serde`, `bytemuck`, `simd`).

## Co-architect protocol
Carried from Wari (`/Users/goose/projects/Wari/CLAUDE.md`); the owner is the architect,
the agent is a technical collaborator.
1. The owner has final authority on every technical decision: algebra API shape,
   crate boundaries, dependencies, feature flags, naming.
2. Every engineering decision is discussed before execution: options with trade-offs,
   the owner picks, then the agent executes.
3. No silent structural changes. Refactors, renames, new crates, new deps, feature-flag
   changes and module moves get the same discussion as a new feature. Tactical cleanup
   inside a task (a local rename, splitting an over-long function, a missing doc
   comment) is fine.
4. Ambiguity resolves by asking, not assuming. A wrong guess that compiles is still a
   wrong guess.
5. Disagreement is surfaced in writing, with reasoning, before executing. The owner may
   overrule; then comply.

## Design rules (Wari's absolute rules, adapted to a library)
- **G1 `unsafe` carries a proof.** Every `unsafe` block **and every `unsafe impl`** has a `// SAFETY:` comment
  citing an invariant in `docs/invariants.md` (create the catalog with the first entry).
  New invariants land in the same PR as the block that needs them.
- **G2 No allocation in the kernel.** `garust-core` and `garust-geo` stay `no_std` and
  allocation-free; anything that needs `alloc` or `std` goes behind a feature or into
  a higher crate (`garust-anim` is the precedent).
- **G3 No panics on library paths.** `unwrap()`/`expect()`/`panic!` are banned in
  `crates/*/src`. Fallible operations return `Option` or a typed error; degenerate
  inputs are decided by a documented threshold (see the `exp`/`log` threshold fixes in
  #70 and #71), never by a crash. Last-resort assertions need a justifying comment.
- **G4 Every public item has a contract.** Preconditions, postconditions, errors and
  panics in the doc comment, plus a doctest that shows correct use and one that shows
  a rejected or degenerate case.
- **G5 Numerics are explicit.** Every tolerance is a named constant with a comment
  saying what it protects against. Golden fixtures are trig-free so CI agrees across
  hosts (motoreel lesson, 2026-08-20: `sin`/`cos` differ in the last ulp between libms).
- **G6 Pure before impure.** Scalar reference code lives in files that compile on the
  host with no features; SIMD and feature-gated paths live in adjacent files and ship
  with a scalar-vs-fast equivalence test.
- **G7 Cite prior art.** Dorst/Fontijne/Mann, Gunn, De Keninck (bivector.net),
  the RFC that introduced the construction. When departing from a norm, say why in the
  doc comment. "We invented this" is rarely the right sentence.
- **G8 Zero dependencies by default.** A new dependency is feature-gated, off by default,
  and justified in an RFC with the alternative rejected. `Cargo.lock` is committed.

## Change workflow (Wari's PR loop + the kit)
- **Design first.** Anything that adds a type, a crate, a feature flag or a public
  surface starts as an RFC in `rfcs/` (next free number; see the numbering note in
  RFC-013). Small fixes need an issue or a sentence in the PR, not an RFC.
- **One branch per PR**, named `<kebab-topic>` or `claude/<kebab-topic>` as in the git
  history. Never stack unrelated work.
- **Size discipline.** Preferred 100–400 changed lines, one conceptual change; up to
  ~800 if genuinely atomic; larger needs the split proposed first.
- **PR body**, no exceptions:
  ```markdown
  ## What            one paragraph, skimmable from the PR list
  ## Why             the RFC / issue / consumer need (motoreel, physics-lab, ghost-pepper)
  ## How             2–5 bullets, modules touched, trade-offs and why
  ## Numerics & safety   thresholds added or changed, every new `unsafe` with its INV-N,
                     SIMD lane behaviour, `no_std` impact, any new panic path (should be none)
  ## Tests           unit · doctest (correct use + rejection) · equivalence (scalar vs simd) · regression
  ## Local verification  the exact commands and expected output the reviewer runs
  ## Out of scope    what this PR deliberately does not do
  ## Rollback        usually `git revert`; call out any serialized-format or API break
  ```
- **Review checklist** (the owner runs it before merging): G1–G8 hold · contracts on new
  public items · tests at the right layer · Why/How give the reasoning, not just the
  pick · size within discipline · all gates green under CI's toolchain · local
  verification reproduces the claimed output.
- **Squash merge.** The PR body becomes the commit on `main`. The owner merges; the
  agent never merges. Delete the branch after.
- Commits follow Conventional Commits in English; messages say why.

## Testing standard
Not a coverage percentage. The standard is:
- Every public API has a doctest for correct use and a test for a rejected or
  degenerate input.
- Every bug fix ships the test that reproduces it (#70 and #71 are the pattern).
- Every algebraic identity the docs claim (`a*b = a·b + a∧b`, rotor sandwich preserves
  norm, motor compose/inverse round-trips) has a test that would fail if it broke.
- Every SIMD or feature-gated path has a scalar equivalence test.
- Every trust-like boundary (parsing, `bytemuck`, `serde`, derive input) has an
  adversarial test. "We'll add it in a follow-up" is not acceptable there.

## Boundaries
- Don't touch: `target/`; `docs/architecture-overview.md`, `docs/code-walkthrough.md`,
  `docs/flows.md`, `docs/design-philosophy.md`, `docs/diagrams/` (generated; regenerate
  rather than hand-edit); `rfcs/` that are `Accepted` (amend by a new RFC).
- Ask first: a new crate, a new dependency, a public API change, a feature flag, an MSRV
  bump, anything that changes `no_std` builds.

## Decisions
RFCs in `rfcs/` are the design record. Non-design decisions (tooling, process, releases)
go to `docs/decisions/` as ADRs (`adr` skill; create the folder on first use).

## Handoff
The latest session handoff is in the vault note's `## Log` section, or in
`.agents/HANDOFF.md` if the vault isn't reachable. Start a session with the
`project-context` skill and end it with `session-handoff`.

## Related projects
motoreel (consumer; adds to garust "as needed" through this RFC/PR process), physics-lab,
guion, tracking-ghost-pepper (consumer of RFC-014). Who does what across agents:
`~/.agents/AGENTS.md`, "Agent roles".
