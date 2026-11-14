# 0001. The maintainer's private remote is the source of truth; GitHub is a read-only mirror

- Date: 2026-11-02
- Status: accepted
- Deciders: Gustavo Delgadillo

## Context
garust started as a public GitHub repository with GitHub Actions as CI and GitHub PRs as
the review record. The maintainer's other systems projects (Wari, the baremetal framework)
run on a private host with a ticket tracker that links commits and pull requests to
tickets, a per-ticket review chain and a human merge. GitHub Actions no longer runs for
this account, so the public repository had lost its CI, and the library is a dependency of
projects already on the private host (goose-rover pins a `rev`). The repository is public
and must stay public: consumers and the crate metadata point at GitHub.

## Decision
- The maintainer's private remote holds the full history (imported with `git clone
  --mirror` / `git push --mirror` on 2026-11-03: every branch and the `v0.1.0` tag; history
  not rewritten) and is where tickets, branches, PRs and the human merge happen.
- `github.com/westerngazoo/garust` is a **read-only public mirror**: after each merge the
  maintainer pushes `main` and tags by hand (the same arrangement Wari uses). Working
  branches never reach the mirror. New issues or PRs opened there are recreated as tickets.
- Public metadata keeps pointing at GitHub: `repository` in `Cargo.toml`, README links.
  The dead Actions badge was removed.
- The process itself (tracker names, ids, URLs, the inbox ticket, the mirror commands)
  lives in an untracked `CLAUDE.local.md`. Tracked files carry only the technical layer:
  `AGENTS.md`, `docs/GARUST-RULES.md` (the former `AGENTS.md` rules), `ENGINEERING.md`,
  `project-specifics.md`, `ROADMAP.md`, `docs/decisions/`.
- Quality gate: the ten commands in `AGENTS.md` plus `scripts/coverage-new.sh` (≥ 99 % of
  the `.rs` lines a branch adds). No external code-quality service: it does not analyze
  Rust workspaces.

## Alternatives considered
- **Server-side push mirror from the private host to GitHub:** automatic, but it pushes
  every branch, including working ones named after internal tickets, and needs a GitHub
  token stored on the host. Rejected: the mirror should carry only `main` and tags.
- **Bidirectional (keep accepting GitHub PRs):** doubles the review surface and the
  tracker never sees the GitHub side; Wari did this once as a one-off catch-up, not as a
  rule. Rejected.
- **Process in a tracked `AGENTS.md`** (as the private baremetal framework does): that
  repository is private; here it would publish internal names and URLs. Rejected.
- **Stay on GitHub only:** no CI, no commit-to-ticket linking, a second process to
  maintain for one library. Rejected.

## Consequences
- Easier: one process for every systems project; commits and PRs linked to tickets; the
  new-code coverage gate shared with the other Rust workspaces.
- Harder: the mirror push is a manual step after every merge (documented in
  `CLAUDE.local.md`); contributors on GitHub have no PR path and must be pointed at the
  maintainer.
- Revisit: CI on the private host (a port of `.github/workflows/ci.yml`) is a follow-up
  ticket; until then the gates run locally and the test run in the tracker is the record.
