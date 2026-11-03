# Roadmap

Milestone themes for garust. The detailed, claim-by-claim plan is
[`rfcs/RFC-015-garust-dyn-roadmap.md`](rfcs/RFC-015-garust-dyn-roadmap.md); the day-to-day
board is the maintainer's tracker (see the untracked `CLAUDE.local.md`). This file only
records what each milestone is for and where it stands.

## Status legend

`Backlog` → `Discussing` → `Spec'd` → `In progress` → `In review` → `Done`

## Milestones

### M0 — Methodology and migration  ·  *in progress*

| Item | Status |
|------|--------|
| Wari-derived rules adopted (`AGENTS.md`, PR #72) | Done |
| G1 invariant catalog (`docs/invariants.md`, PR #73) | Done |
| G5 named tolerances in core and geo (PR #74) | Done |
| Repository joined the maintainer's process; public mirror on GitHub (ADR 0001) | In review |
| G5 in `garust-physics` (nine literals in `world.rs` and `contact.rs`) | Backlog |
| G3 `versor_inverse` on `transform.rs` | Backlog |
| G4 doctests (correct use + rejection) per crate | Backlog |
| CI on the private host (port of `.github/workflows/ci.yml`) | Backlog |

### M1 — `garust-physics` as the dynamics backend

RFC-015: `Bivector` newtype, `Motor::sqrt`, motor·line and motor·motor kernels, then the
single-motor step and RNEA/CRBA/ABA over `T: Real` so dual numbers run without a rewrite.

| Item | Status |
|------|--------|
| RFC-015 M0 kernels (`Bivector`, `Motor::sqrt`, motor·line) | Discussing |
| Single-motor integration step | Backlog |
| RNEA / CRBA / ABA | Backlog |

### M2 — `garust-anim` sinks and derived shapes

The former GitHub issues #53–#56: `SvgSink` end-to-end render, `PpmSink` with a
zero-dependency stroke rasterizer, `JoinLine`/`MeetPoint` from live incidence, SIMD fast
path in the render loop.

| Item | Status |
|------|--------|
| A3 `SvgSink` frame writer + demo render | Backlog |
| A4 `PpmSink` rasterizer | Backlog |
| A5 derived shapes from incidence | Backlog |
| A6 SIMD fast path in the render loop | Backlog |

## Sequencing rules

- Anything that adds a type, crate, feature flag or public surface starts as an RFC
  (`docs/GARUST-RULES.md` §3).
- A milestone item enters `In progress` only with a ticket and a branch of its own.
- This file is updated when a milestone theme changes, not per ticket.

## Current focus

Finish M0 (merge the migration PR, then the G5/G3/G4 gaps), then open RFC-015 M0.
