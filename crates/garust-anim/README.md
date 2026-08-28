# garust-anim

The **animation layer** of [garust](https://crates.io/crates/garust), built
on the [`garust-geo`](https://crates.io/crates/garust-geo) motor manifold
(RFC-012).

A keyframe [`Track`] holds `(time, Motor)` pairs; sampling is screw
interpolation with each span's generator computed once and cached (`log`
per span, not per frame), and easing is pure time remapping — the screw
path and its schedule stay decoupled. Every sampled pose is a unit motor
by construction, so there is no renormalization step and no gimbal
handling anywhere.

This crate is milestone A1 of RFC-012; the scene graph, camera, and frame
sinks (A2–A6) build on it.
