# Releasing garust

garust is a Cargo workspace of five crates with internal dependencies, so they
publish to [crates.io](https://crates.io) in **dependency order**. A crate
cannot be published — or even fully `cargo publish --dry-run`'d — until the
crates it depends on are already on the index. For example
`cargo publish --dry-run -p garust-core` fails with

```
no matching package named `garust-derive` found
```

until `garust-derive` is published. That is expected, not a defect: only the
leaf (`garust-derive`) verifies in isolation; the rest verify at real-publish
time, in order.

## Publish order

| # | crate | depends on |
|---|-------|------------|
| 1 | `garust-derive`  | crates.io only (`syn`, `quote`, …) |
| 2 | `garust-core`    | `garust-derive` (optional, `derive` feature) |
| 3 | `garust-geo`     | `garust-core` |
| 4 | `garust-physics` | `garust-core`, `garust-geo` |
| 5 | `garust`         | all of the above (umbrella) |

## Steps

1. **Bump the version.** Edit `[workspace.package].version` in the root
   `Cargo.toml` (every crate inherits `version.workspace = true`) **and** the
   matching `version = "X.Y.Z"` on each path dependency in
   `[workspace.dependencies]` — they must agree, or the dependent crates fail
   to publish.
2. **Green gates.** `cargo test --workspace --all-features`, plus the CI gates
   (fmt, clippy, doc, `no_std`, MSRV via `cargo hack check --no-dev-deps`).
3. **`cargo login`** with a crates.io token (one-time).
4. **Publish in order**, letting each crate land on the index before the next:
   ```sh
   cargo publish -p garust-derive
   cargo publish -p garust-core
   cargo publish -p garust-geo
   cargo publish -p garust-physics
   cargo publish -p garust
   ```
5. **Tag**: `git tag vX.Y.Z && git push --tags`.

## Notes

- Every crate carries `name` / `description` / `license` (`MIT OR Apache-2.0`)
  / `repository` / `keywords` / `categories` / `readme`, so each crates.io page
  renders with a description and README.
- The default build is **zero-dependency**; optional deps (`serde`, `bytemuck`,
  `wide`, `libm`, and the `syn`/`quote` derive stack) only enter through
  features.
- Dev-dependencies (`proptest`, `criterion`, `nalgebra`, `serde_json`) never
  reach consumers — they are stripped from the published manifests' dependency
  resolution.
- MSRV is **1.79**, enforced by the CI `cargo hack check --no-dev-deps` job.
