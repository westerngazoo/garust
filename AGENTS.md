# garust

*The maintainer also follows an internal process (tickets, branch names, review, mirror
sync) kept in an untracked `CLAUDE.local.md`, exactly as Wari does. It is deliberately not
part of this repository: everything here is public.*

## Meta
- Context: happydev            <!-- source of truth is the maintainer's private remote; github.com/westerngazoo/garust is the public read-only mirror -->
- Status: active
- Vault note: see `CLAUDE.local.md` (untracked); fallback handoff in `.agents/HANDOFF.md`
- Owner: Gustavo Delgadillo (westerngazoo)

## What this is
Geometric Algebra in Rust, generic over the Clifford signature `Cl(P, Q, R)` and the
scalar type: a zero-dependency, `no_std` workspace (`garust-core`, `garust-geo`,
`garust-physics`, `garust-anim`, `garust-derive`, re-exported by the umbrella crate).
It is the one engine behind motoreel, physics-lab, guion, the Tracking Ghost Pepper
display (library side in `rfcs/RFC-014-cone-warp.md`) and goose-rover. "Done" is never a
feature count: it is that every public item has a contract, a doctest and a rejection
test, and the gates below are green.

## Commands
These are the merge gates. Run them under the CI toolchain, not the machine default
(`rustup run stable ...`); a machine a few releases behind cannot see the lints stable
has since added. `.github/workflows/ci.yml` is the historical record of the same list.
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
scripts/coverage-new.sh   # >= 99 % of the .rs lines the branch adds; needs cargo-llvm-cov 0.9.1
```

## Architecture
`crates/garust-core` (multivector, compile-time Cayley table, signatures, scalar trait),
`crates/garust-geo` (PGA/CGA primitives, `Motor`, kinematic chains, SIMD paths behind
`simd`), `crates/garust-physics` (RFC-010), `crates/garust-anim` (RFC-012),
`crates/garust-derive` (`#[derive(Algebra)]`). Design history is in `rfcs/`; the
generated tour of the code is in `docs/`. The umbrella crate at the root re-exports the
members and owns the feature flags (`std`, `libm`, `derive`, `serde`, `bytemuck`, `simd`).

## Engineering method
Two layers, both mandatory. Technical rules win on anything technical; the maintainer's
process (untracked) wins on process.
- **Technical rules**: `docs/GARUST-RULES.md`. Co-architect protocol, the absolute rules
  G1–G8 (unsafe with a proof, no allocation in the kernel, no panics on library paths,
  contracts everywhere, explicit numerics, pure before impure, cite prior art, zero
  dependencies by default), the PR body, the testing standard and the boundaries.
  Read it at the start of every session.
- **Project facts** (single source): `project-specifics.md`.
- **Constitution**: `ENGINEERING.md` (think before coding, simplicity, surgical changes,
  goal-driven execution, test-first). Where it names a generic requirement loop, the
  maintainer's process supplies the concrete one.
- **Roadmap**: `ROADMAP.md` records milestone themes; the design record is `rfcs/`
  (RFC-015 is the current roadmap RFC) and `docs/decisions/` holds the ADRs.
- **Who does what across agents**: `~/.agents/AGENTS.md`, "Agent roles". The owner
  reviews and merges every change; agents never merge.

## Handoff
Start a session with the `project-context` skill and end it with `session-handoff`. The
latest handoff is in the vault note's `## Log` section (path in `CLAUDE.local.md`), or in
`.agents/HANDOFF.md` if the vault isn't reachable.

## Related projects
motoreel (consumer; adds to garust "as needed" through the RFC/PR process), physics-lab,
guion, tracking-ghost-pepper (consumer of RFC-014), goose-rover (pins a `rev`).
