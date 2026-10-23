# RFC 014: Cone Warp — the library side of Tracking Ghost Pepper

**Author:** garust maintainers
**Status:** Draft / Request for Comments
**Target:** `garust` (Geometric Algebra in Rust)
**Builds on:** RFC-009 (PGA kernel), RFC-012 (anim, for the emission precedent)
**Consumer:** `tracking-ghost-pepper` (`/Users/goose/code/personal/tracking-ghost-pepper`),
whose founding document `docs/overview-and-platform.md` defines the product, the
platform decision and the milestone interlock this RFC mirrors.

## 1. Context and motivation

A Pepper's Cone is a thin transparent cone standing on a flat panel. The panel
shows a pre-distorted image; the cone reflects it so that, from the viewer's
eye, the reflection looks like an object floating inside the cone. The 2017
system fixed the viewing spot and calibrated once. The product recomputes the
distortion **per frame from the tracked eye position**, so the viewer can walk
around the cone.

The product owner isolated the two sides on 2026-10-23: tracking, camera,
display and calibration are the product; **turning an eye point into a
distortion map is the library**. This RFC is the library half. It carries the
four milestones the product's roadmap depends on (L1–L4) and nothing else from
garust's general roadmap: the grade-typed API, the dualize macro, compile-time
folding and the ML layers are explicitly not on the path to a shipping unit.

## 2. What garust owns and what it does not

| garust owns | garust does **not** own |
|---|---|
| Cone and panel geometry model | Camera capture, exposure, timestamps |
| Ray → cone intersection and the reflection at the hit | Face / eye detection, the 3D eye estimate |
| The eye point → distortion map function (CPU) | `EyeSample`, time, confidence, the tracker loop |
| Emission of the same function as a GPU shader | The wgpu surface, DRM/KMS, frame pacing |
| Multi-viewer maps | Calibration procedure and on-device storage |
| Tests, numerics, `no_std` discipline | Latency measurement, enclosure, BOM |

**Interface (proposed, decision D1 below):** garust takes a plain eye point
`[S; 3]` in display coordinates plus a `ConeDisplay<S>` geometry, and writes a
map into a caller-provided buffer. `EyeSample` (point + capture `Instant` +
confidence) stays in the product crate. The library never learns about time.

## 3. Milestones

### L1 — Core geometry

A cone is not a PGA flat and not a CGA round: it is a quadric, so it does not
become a single blade. What *is* a blade is the **tangent plane at a hit
point**, and reflection in a plane is the PGA sandwich garust already has. The
L1 construction is therefore:

1. `Cone<S>`: apex, axis (a PGA line), half-angle (45°–51° for the 0.5 mm PETG
   sheet the paper used). Stored in `garust-geo`, `no_std`, no allocation.
2. `Cone::intersect(&Ray) -> Option<Hit>`: the quadratic in the ray parameter,
   with a documented threshold for grazing rays (G5), returning the nearest
   hit on the physical sheet (height bounds, not the full double cone).
3. `Hit::tangent_plane() -> Plane<S>`: a PGA plane; `Plane::reflect(&Ray)` via
   the existing sandwich.
4. `Panel<S>`: the display plane with its pixel grid (width, height, pitch) and
   the display coordinate frame the product fixes in R-0001 (open question 1
   there; this RFC inherits the answer).

Tests: analytic hits (axis-aligned rays, rays through the apex), reflection law
(angle of incidence equals angle of reflection, as a property over sampled
rays), symmetry (a mirrored eye point gives a mirrored hit).

### L2 — CPU distortion-map generator

`warp_map(eye: [S; 3], geo: &ConeDisplay<S>, out: &mut [[S; 2]])`: for each
panel pixel, trace eye → virtual image through the cone and write the UV the
panel must show at that pixel. Caller-provided buffer (G2), optional
row-parallel chunking left to the caller. This is what the product's R-0002
(first head-tracked image on the laptop, CPU warp) consumes.

Tests: a trig-free fixture (G5) for one eye position; symmetry under eye
mirroring; every output UV inside `[0, 1]` or flagged outside the sheet.

### L3 — Shader emission

The same per-pixel function emitted as **WGSL** text, with cone and panel
constants baked and the eye point as a uniform, so the product's R-0005 runs
the warp on the SBC's GPU under wgpu. Emission is string building, so it needs
`alloc`; it lives behind a feature or in a higher crate (decision D2). RFC-012's
`garust-anim` is the precedent for std-dependent layers above the kernel.

Tests: the emitted source parses (validation needs `naga`, a dev-dependency —
decision D3); a CPU interpreter of the emitted expression matches the L2 map
within a stated tolerance on the same fixture.

### L4 — Multi-viewer

Two or more eye points per frame: either a stereo pair (one viewer, two eyes)
or several viewers with spatially split or time-multiplexed maps. Out of scope
until the product's T4; recorded so L2's signature does not paint us in (the
map function takes a slice of eye points from L2 onward, even if length one).

## 4. Interlock with the product roadmap

| Product milestone | Needs | Why |
|---|---|---|
| T1 webcam on the laptop (R-0002) | L1 + L2 | first head-tracked image, speed irrelevant |
| T3 embedded unit (R-0005) | L3 | the warp must run on the GPU to hit frame rate |
| T4 multi-viewer (R-0008) | L4 | two or more eye points per frame |

A library gap found in the product is fixed **here**, through this RFC's PR
sequence, never patched downstream.

## 5. Decisions to make with the owner

- **D1 — Where the shared type lives.** Proposal: `EyeSample` in the product;
  garust takes `[S; 3]` + geometry. Alternative: a tiny shared crate. The
  proposal keeps garust free of `Instant` and usable from motoreel/physics-lab.
- **D2 — Where `Cone` and emission live.** Proposal: `Cone`/`Panel`/`warp_map`
  in `garust-geo` (no_std, zero-dep, like the chains in RFC-013); WGSL emission
  behind a `wgsl` feature of the umbrella crate or in `garust-anim`.
- **D3 — Dev-dependencies for shader validation.** `naga` as a dev-dependency
  only (G8), or validate in the product's integration tests instead.
- **D4 — Scalar generic or `f32` only.** The map is consumed by a GPU; `f32`
  may be the only instantiation that matters, but the kernel convention is
  generic over `S: Scalar`. Default: generic, with `f32` tests.

## 6. Non-goals

No camera, tracking, time, wgpu or windowing code in garust. No calibration
solver (the product measures; garust consumes the measured geometry). No
tablet-specific paths.

## 7. Prior art

- Pepper's Cone (Luo, Lawrence, Seitz; UIST 2017): cone reflector on a tablet,
  fixed viewpoint, one-off calibration. The 0.5 mm sheet and the 45°–51° range
  come from that paper's findings.
- PGA reflection as the sandwich product: Gunn; De Keninck, bivector.net.
- Ray–quadric intersection: standard; the only non-GA step, confined to L1.
