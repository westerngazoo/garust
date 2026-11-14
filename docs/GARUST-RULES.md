# garust technical rules

> The technical layer of garust. `AGENTS.md` is the public entry point; the maintainer's
> process (tickets, branch names, review chain, mirror sync) lives in an untracked
> `CLAUDE.local.md` and wins on process; this file wins on everything technical.
>
> Source: Wari's `CLAUDE.md` (co-architect protocol and absolute rules R1–R8) adapted to a
> library, first adopted as garust's `AGENTS.md` in PR #72 (2026-10-23) and moved here when
> the repository joined the maintainer's process (ADR 0001). What changed in the move is
> listed at the end.

**Mission:** a Geometric Algebra kernel that is correct on every signature, allocation-free
on the smallest target, and explicit about every numeric decision. Every choice optimizes
for **correctness, then size, then speed**, before convenience.

---

## 1. Co-architect protocol

**The owner is the architect. The agents are technical collaborators.**

1. The owner has final authority on every technical decision: algebra API shape, crate
   boundaries, dependencies, feature flags, naming.
2. Every engineering decision is discussed before execution: options with trade-offs, the
   owner picks, then the agent executes. Decisions that matter become RFCs in `rfcs/`
   (design) or ADRs in `docs/decisions/` (tooling, process, releases).
3. No silent structural changes. Refactors, renames, new crates, new deps, feature-flag
   changes and module moves get the same discussion as a new feature. Tactical cleanup
   inside a task (a local rename, splitting an over-long function, a missing doc comment)
   is fine.
4. Ambiguity resolves by asking, not assuming. A wrong guess that compiles is still a wrong
   guess.
5. Disagreement is surfaced in writing, with reasoning, before executing. The owner may
   overrule; then comply.

## 2. Design rules (Wari's absolute rules, adapted to a library)

- **G1 `unsafe` carries a proof.** Every `unsafe` block **and every `unsafe impl`** has a
  `// SAFETY:` comment citing an invariant in `docs/invariants.md`. New invariants land in
  the same PR as the block that needs them.
- **G2 No allocation in the kernel.** `garust-core` and `garust-geo` stay `no_std` and
  allocation-free; anything that needs `alloc` or `std` goes behind a feature or into a
  higher crate (`garust-anim` is the precedent).
- **G3 No panics on library paths.** `unwrap()`/`expect()`/`panic!` are banned in
  `crates/*/src`. Fallible operations return `Option` or a typed error; degenerate inputs
  are decided by a documented threshold (see the `exp`/`log` threshold fixes in #70 and
  #71), never by a crash. Last-resort assertions need a justifying comment.
- **G4 Every public item has a contract.** Preconditions, postconditions, errors and panics
  in the doc comment, plus a doctest that shows correct use and one that shows a rejected
  or degenerate case.
- **G5 Numerics are explicit.** Every tolerance is a named constant with a comment saying
  what it protects against (#74 is the pattern). Golden fixtures are trig-free so CI agrees
  across hosts (motoreel lesson, 2026-08-20: `sin`/`cos` differ in the last ulp between
  libms).
- **G6 Pure before impure.** Scalar reference code lives in files that compile on the host
  with no features; SIMD and feature-gated paths live in adjacent files and ship with a
  scalar-vs-fast equivalence test.
- **G7 Cite prior art.** Dorst/Fontijne/Mann, Gunn, De Keninck (bivector.net), the RFC that
  introduced the construction. When departing from a norm, say why in the doc comment. "We
  invented this" is rarely the right sentence.
- **G8 Zero dependencies by default.** A new dependency is feature-gated, off by default,
  and justified in an RFC with the alternative rejected. `Cargo.lock` is committed.

## 3. Change workflow

- **Design first.** Anything that adds a type, a crate, a feature flag or a public surface
  starts as an RFC in `rfcs/` (next free number; see the numbering note in RFC-013). Small
  fixes need a ticket or a sentence in the PR, not an RFC.
- **One branch per PR.** The branch name comes from the maintainer's process; never stack
  unrelated work.
- **Size discipline.** Preferred 100–400 changed lines, one conceptual change; up to ~800
  if genuinely atomic; larger needs the split proposed first. Lockfiles and generated files
  do not count.
- **PR body**, no exceptions:
  ```markdown
  ## What            one paragraph, skimmable from the PR list
  ## Why             the RFC / ticket / consumer need (motoreel, physics-lab, ghost-pepper, goose-rover)
  ## How             2–5 bullets, modules touched, trade-offs and why
  ## Numerics & safety   thresholds added or changed, every new `unsafe` with its INV-N,
                     SIMD lane behaviour, `no_std` impact, any new panic path (should be none)
  ## Tests           unit · doctest (correct use + rejection) · equivalence (scalar vs simd) · regression
  ## Local verification  the exact commands and expected output the reviewer runs
  ## Out of scope    what this PR deliberately does not do
  ## Rollback        usually `git revert`; call out any serialized-format or API break
  ```
- **Review checklist** (the owner runs it before merging): G1–G8 hold · contracts on new
  public items · tests at the right layer · Why/How give the reasoning, not just the pick ·
  size within discipline · all gates green under the CI toolchain · local verification
  reproduces the claimed output.
- **The owner merges; the agent never merges.** The merge style is the reviewer's call;
  the PR body is the record either way. Delete the branch after.
- Commits follow Conventional Commits in English; messages say why.

## 4. Testing standard

Not a coverage percentage. The standard is:
- Every public API has a doctest for correct use and a test for a rejected or degenerate
  input.
- Every bug fix ships the test that reproduces it (#70 and #71 are the pattern).
- Every algebraic identity the docs claim (`a*b = a·b + a∧b`, rotor sandwich preserves
  norm, motor compose/inverse round-trips) has a test that would fail if it broke.
- Every SIMD or feature-gated path has a scalar equivalence test.
- Every trust-like boundary (parsing, `bytemuck`, `serde`, derive input) has an adversarial
  test. "We'll add it in a follow-up" is not acceptable there.
- Line coverage is a side effect, not the target. The new-code gate
  (`scripts/coverage-new.sh`, ≥ 99 % of the `.rs` lines a branch adds) still applies;
  meeting it with tests that assert nothing does not count.

## 5. Boundaries

- Don't touch: `target/`; `docs/architecture-overview.md`, `docs/code-walkthrough.md`,
  `docs/flows.md`, `docs/design-philosophy.md`, `docs/diagrams/` (generated; regenerate
  rather than hand-edit); `rfcs/` that are `Accepted` (amend by a new RFC).
- Ask first: a new crate, a new dependency, a public API change, a feature flag, an MSRV
  bump, anything that changes `no_std` builds.

## 6. Decisions

RFCs in `rfcs/` are the design record. Non-design decisions (tooling, process, releases)
go to `docs/decisions/` as ADRs (`adr` skill).

---

## What changed from the GitHub-era `AGENTS.md`, and why

| Before (PR #72) | Now | Why |
|---|---|---|
| Branches `<kebab-topic>` or `claude/<kebab-topic>` | Named by the maintainer's process | One naming scheme per host; the public mirror only receives `main` and tags. |
| "Squash merge" as the rule | The reviewer's call | The merge is done by a human on the private host; the PR body stays the record. |
| No coverage number | New-code gate ≥ 99 % (`scripts/coverage-new.sh`) | Same gate as the maintainer's other Rust workspaces; the qualitative standard above is unchanged. |
| GitHub issues as the small-fix record | Tickets in the maintainer's tracker | GitHub is a read-only mirror; open issues were recreated as tickets. |
