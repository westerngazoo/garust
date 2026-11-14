# Project Specifics

This is the **single per-project file**. Every other document in this
methodology is generic and identical across all projects — only this file
changes. Fill it in when the project starts; keep it current as these facts
change.

`AGENTS.md` points to this file, so every agent reads it at session start.

## Identity

- **Project name:** garust (Goose Garust)
- **One-line description:** Geometric Algebra in Rust, generic over the Clifford signature
  `Cl(P,Q,R)` and the scalar type: a zero-dependency, `no_std` workspace.
- **Owner / final decision authority:** Gustavo Delgadillo
- **Repository URL:** public mirror `https://github.com/westerngazoo/garust` (read-only:
  `main` and tags). The source of truth is the maintainer's private remote, recorded in the
  untracked `CLAUDE.local.md`.
- **Context / flavor:** happydev
- **Vault note:** path in `CLAUDE.local.md` (untracked)

## Language & toolchain

The concrete commands referenced by `ENGINEERING.md` §6 and by the `architect` and
`qa` agents as merge gates.

- **Primary language / version:** Rust stable, edition 2021, MSRV 1.79 (`rust-version` in
  `Cargo.toml`); run every gate under `rustup run stable`. Zero runtime dependencies; the
  feature flags (`std`, `libm`, `derive`, `serde`, `bytemuck`, `simd`) are owned by the
  umbrella crate.
- **Build command:** `cargo build --workspace`
- **Test command:** `cargo test --workspace --all-targets` and the same with
  `--all-features`, plus `cargo test --workspace --doc`
- **Lint command:** `cargo clippy --workspace --all-targets -- -D warnings` and the same
  with `--all-features`
- **Format-check command:** `cargo fmt --all --check`
- **Feature powerset:** `cargo hack check --no-dev-deps --workspace --exclude garust-derive --lib`
- **Embedded build:** `cargo build -p garust-core --no-default-features --features libm --target thumbv7em-none-eabi`
  and the same for `garust-geo`
- **Docs:** `cargo doc --workspace --no-deps --all-features` (warnings are errors)
- **New-code coverage command:** `scripts/coverage-new.sh` (needs
  `cargo install cargo-llvm-cov --version 0.9.1 --locked` once). It runs the workspace
  tests under `cargo llvm-cov`, crosses the report with the `*.rs` lines of
  `git diff -U0 origin/main...HEAD` and fails below 99 % of the added executable lines,
  listing the uncovered ones. It reads commits only, so it refuses uncommitted `.rs`
  changes. `--base <ref>` compares with another ref; `COVERAGE_NEW_MIN` changes the
  threshold. Its own tests: `scripts/coverage-new.test.sh`.
- **Quality gate** (2026-11-02): the full list in `AGENTS.md` "Commands", all green, before
  the code review and in the test run that backs the PR. No external code-quality service:
  the maintainer's instance does not analyze Rust workspaces.

## Domain notes

- Plane-based PGA (`Cl(3,0,1)`): the regressive product `∨` is the **join**, the outer
  product `∧` is the meet (PR #76 fixed the README; `dual.rs` and `catalog.rs` still carry
  the point-based reading, tracked in RFC-015).
- Numerics: every production tolerance is a named constant (G5, PR #74); golden fixtures
  are trig-free so hosts with different libms agree.
- `docs/invariants.md` is the catalog every `// SAFETY:` comment cites (G1, PR #73).
- Consumers: motoreel, physics-lab, guion, tracking-ghost-pepper (RFC-014) and goose-rover
  (pins a `rev` of the public mirror).

## Milestone themes

Mirrored into `ROADMAP.md`; the detailed plan is `rfcs/RFC-015-garust-dyn-roadmap.md`.

- M0 — Methodology and migration (this scaffold; G3/G4/G5 gaps listed in RFC-015).
- M1 — `garust-physics` as the dynamics backend (single-motor step, then RNEA/CRBA/ABA).
- M2 — `garust-anim` sinks and derived shapes (the former GitHub issues #53–#56).

## Process

Internal: tracker name, ids, inbox ticket and the mirror procedure are in the untracked
`CLAUDE.local.md`. Nothing from it goes into tracked files.
