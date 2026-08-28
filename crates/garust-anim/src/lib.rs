//! # garust-anim — the animation layer
//!
//! Motor-native keyframe animation on the garust PGA kernel (RFC-012,
//! milestone A1). A [`Track`] is a sequence of `(time, `[`Motor3`]`)`
//! keys; sampling it is **screw interpolation** between the surrounding
//! keys, with an [`Ease`] remapping the local time first. Because the
//! interpolant is [`Motor::slerp`] under the hood, every sampled pose is
//! a unit motor by construction — no renormalization, no gimbal handling,
//! no separate position/rotation channels.
//!
//! Two properties of the interpolant shape the design (RFC-012 §3.2):
//!
//! - **Per-span generator caching.** A span's screw generator
//!   `log(m1 ∘ m0⁻¹)` is constant, so [`Track::keys`] computes it once
//!   per span ([`Motor::screw_generator`]) and [`Track::sample`] pays
//!   only the cheap exponential per frame
//!   ([`Motor::slerp_from_generator`]).
//! - **The short-way fold.** A span can sweep at most a half turn (τ/2)
//!   of rotation; larger sweeps must be subdivided across keys.
//!   [`Track::spin`] does that subdivision automatically.
//!
//! ```
//! use garust_anim::{Ease, Track};
//! use garust_geo::Motor;
//! use garust_core::Pga3;
//! use core::f64::consts::TAU;
//!
//! // Rise 2 units while turning a quarter turn, eased, over 2 seconds.
//! let lift = Motor::translator(0.0, 0.0, 2.0)
//!     * Motor::rotor(TAU / 4.0, Pga3::basis(0b0011));
//! let track = Track::keys([(0.0, Motor::identity()), (2.0, lift)])
//!     .ease(Ease::SmootherStep);
//!
//! let mid = track.sample(1.0);           // halfway along the screw
//! assert!((mid.norm() - 1.0).abs() < 1e-12);
//! ```

#![deny(missing_docs)]

pub mod scene;
pub use scene::{Camera, Object, ObjectId, Prim2, Projection, Scene, Shape, Style};

use garust_core::Pga3;
use garust_geo::{Motor, Motor3};

use core::f64::consts::TAU;

/// A time-remapping curve applied to each span's local parameter before
/// the screw interpolation. Easing changes the *schedule* of a span, never
/// its path: the pose still travels the same geodesic screw, just not at
/// constant speed.
///
/// Every variant maps `0 → 0` and `1 → 1`, so key poses are always hit
/// exactly regardless of easing.
#[derive(Clone, Copy, Debug)]
pub enum Ease {
    /// Constant-speed screw motion: `s ↦ s`.
    Linear,
    /// Hermite ease-in-out: `s ↦ 3s² − 2s³`.
    SmoothStep,
    /// Perlin's C² ease-in-out: `s ↦ 6s⁵ − 15s⁴ + 10s³`.
    SmootherStep,
    /// Any user curve. It should map `0 → 0` and `1 → 1` if key poses are
    /// to be hit exactly; values outside `[0, 1]` overshoot along the
    /// span's screw (sometimes wanted — e.g. anticipation/follow-through).
    Custom(fn(f64) -> f64),
}

impl Ease {
    /// Remap a span-local parameter `s` (0 at the span's start key, 1 at
    /// its end key).
    pub fn apply(&self, s: f64) -> f64 {
        match *self {
            Ease::Linear => s,
            Ease::SmoothStep => s * s * (3.0 - 2.0 * s),
            Ease::SmootherStep => s * s * s * (s * (s * 6.0 - 15.0) + 10.0),
            Ease::Custom(f) => f(s),
        }
    }
}

/// A motor keyframe track: strictly time-ascending `(time, pose)` keys,
/// an [`Ease`] shared by every span, and the per-span screw generators
/// cached at construction.
///
/// [`sample`](Track::sample) clamps outside the keyed range (holds the
/// first pose before the first key and the last pose after the last key),
/// and between keys evaluates the cached generator — the logarithm is
/// paid once per span when the track is built, never per frame.
#[derive(Clone, Debug)]
pub struct Track {
    keys: Vec<(f64, Motor3)>,
    /// `gens[i]` generates the span from `keys[i]` to `keys[i + 1]`.
    gens: Vec<Pga3>,
    ease: Ease,
}

impl Track {
    /// Build a track from `(time, pose)` keys with [`Ease::Linear`];
    /// chain [`ease`](Track::ease) to change the schedule.
    ///
    /// Each span's screw generator is computed here, once. Note the
    /// short-way fold: a span sweeps at most τ/2 of rotation, so author
    /// larger sweeps as subdivided keys (or use [`Track::spin`]).
    ///
    /// # Panics
    ///
    /// If `keys` is empty, or the times are not strictly ascending.
    pub fn keys<I: IntoIterator<Item = (f64, Motor3)>>(keys: I) -> Self {
        let keys: Vec<_> = keys.into_iter().collect();
        assert!(!keys.is_empty(), "Track needs at least one key");
        assert!(
            keys.windows(2).all(|w| w[0].0 < w[1].0),
            "Track key times must be strictly ascending"
        );
        let gens = keys
            .windows(2)
            .map(|w| w[0].1.screw_generator(&w[1].1))
            .collect();
        Self { keys, gens, ease: Ease::Linear }
    }

    /// Set the easing curve shared by every span (builder-style).
    pub fn ease(mut self, ease: Ease) -> Self {
        self.ease = ease;
        self
    }

    /// A track that holds one `pose` forever.
    pub fn still(pose: Motor3) -> Self {
        Self::keys([(0.0, pose)])
    }

    /// A constant-rate rotation sweeping `angle` radians in the plane of
    /// the unit Euclidean bivector `plane` over `duration` seconds.
    ///
    /// Sweeps beyond the τ/2 per-span ceiling are subdivided into
    /// quarter-turn (or smaller) spans automatically, so a full turn —
    /// or several — is one call: `Track::spin(3.0 * TAU, plane, 6.0)`.
    ///
    /// # Panics
    ///
    /// If `duration` is not strictly positive.
    pub fn spin(angle: f64, plane: Pga3, duration: f64) -> Self {
        assert!(duration > 0.0, "Track::spin needs a positive duration");
        // Subdivide so each span sweeps ≤ τ/4 — half the short-way limit.
        let spans = ((angle.abs() / (TAU / 4.0)).ceil() as usize).max(1);
        Self::keys((0..=spans).map(|i| {
            let f = i as f64 / spans as f64;
            (f * duration, Motor::rotor(angle * f, plane))
        }))
    }

    /// The time of the first key.
    pub fn start_time(&self) -> f64 {
        self.keys[0].0
    }

    /// The time of the last key.
    pub fn end_time(&self) -> f64 {
        self.keys[self.keys.len() - 1].0
    }

    /// Evaluate the track at time `t`.
    ///
    /// Before the first key the first pose is held; after the last key,
    /// the last pose. Between two keys, the span-local parameter is
    /// eased and fed to the cached-generator screw evaluation — exactly
    /// [`Motor::slerp`] between the surrounding keys, bit for bit, minus
    /// the per-frame logarithm.
    pub fn sample(&self, t: f64) -> Motor3 {
        let (first, last) = (&self.keys[0], &self.keys[self.keys.len() - 1]);
        if t <= first.0 {
            return first.1;
        }
        if t >= last.0 {
            return last.1;
        }
        // Strictly inside the range: find the span containing `t`.
        let idx = self.keys.partition_point(|k| k.0 <= t);
        let (t0, m0) = self.keys[idx - 1];
        let t1 = self.keys[idx].0;
        let s = self.ease.apply((t - t0) / (t1 - t0));
        m0.slerp_from_generator(&self.gens[idx - 1], s)
    }
}

#[cfg(test)]
mod tests {
    use super::{Ease, Track};
    use garust_core::Pga3;
    use garust_geo::{Motor, Motor3};
    use proptest::prelude::*;
    use std::f64::consts::TAU;

    /// A valid rotation plane (squares to −1): e1e2, e1e3, e2e3.
    fn plane(choice: usize) -> Pga3 {
        Pga3::basis([0b0011, 0b0101, 0b0110][choice % 3])
    }

    prop_compose! {
        fn any_motor()(
            dx in -2.0f64..2.0, dy in -2.0f64..2.0, dz in -2.0f64..2.0,
            angle in 0.0f64..TAU, axis in 0usize..3,
        ) -> Motor3 {
            Motor::translator(dx, dy, dz) * Motor::rotor(angle, plane(axis))
        }
    }

    proptest! {
        // Sampling at a key time returns that key's pose — regardless of
        // easing, because every ease maps 0 → 0 and 1 → 1.
        #[test]
        fn sample_at_key_times_returns_the_keys(
            m1 in any_motor(), m2 in any_motor(), m3 in any_motor(),
        ) {
            let track = Track::keys([(0.0, m1), (1.0, m2), (2.5, m3)])
                .ease(Ease::SmootherStep);
            prop_assert!(track.sample(0.0).geodesic_distance(&m1) < 1e-9);
            prop_assert!(track.sample(1.0).geodesic_distance(&m2) < 1e-9);
            prop_assert!(track.sample(2.5).geodesic_distance(&m3) < 1e-9);
        }

        // Between two keys with Linear ease, sampling IS slerp — bit for
        // bit, since the cached-generator path replays it exactly.
        #[test]
        fn sample_between_keys_is_slerp(
            m1 in any_motor(), m2 in any_motor(), t in 0.001f64..0.999,
        ) {
            let track = Track::keys([(0.0, m1), (1.0, m2)]);
            prop_assert_eq!(track.sample(t), m1.slerp(&m2, t));
        }

        // Outside the keyed range the track holds its end poses.
        #[test]
        fn sample_clamps_outside_the_range(m1 in any_motor(), m2 in any_motor()) {
            let track = Track::keys([(1.0, m1), (2.0, m2)]);
            prop_assert_eq!(track.sample(-100.0), m1);
            prop_assert_eq!(track.sample(0.999), m1);
            prop_assert_eq!(track.sample(2.001), m2);
            prop_assert_eq!(track.sample(100.0), m2);
        }
    }

    #[test]
    fn translation_midpoint_is_the_half_translation() {
        let track = Track::keys([
            (0.0, Motor::identity()),
            (1.0, Motor::translator(2.0, -4.0, 6.0)),
        ]);
        let want = Motor::translator(1.0, -2.0, 3.0);
        assert!(track.sample(0.5).geodesic_distance(&want) < 1e-12);
    }

    #[test]
    fn every_ease_hits_its_endpoints() {
        fn quad(s: f64) -> f64 {
            s * s
        }
        for ease in [
            Ease::Linear,
            Ease::SmoothStep,
            Ease::SmootherStep,
            Ease::Custom(quad),
        ] {
            assert_eq!(ease.apply(0.0), 0.0);
            assert_eq!(ease.apply(1.0), 1.0);
        }
    }

    #[test]
    fn smoother_step_slows_the_start_without_moving_the_path() {
        // Eased sampling stays ON the screw (it equals slerp at the eased
        // parameter) — easing changes schedule, never geometry.
        let (m1, m2) = (
            Motor::identity(),
            Motor::translator(1.0, 0.0, 0.0) * Motor::rotor(TAU / 6.0, Pga3::basis(0b0011)),
        );
        let track = Track::keys([(0.0, m1), (1.0, m2)]).ease(Ease::SmootherStep);
        let s = Ease::SmootherStep.apply(0.25);
        assert!(s < 0.25, "smootherstep must ease in");
        assert_eq!(track.sample(0.25), m1.slerp(&m2, s));
    }

    #[test]
    fn spin_subdivides_a_full_turn_past_the_short_way_fold() {
        let track = Track::spin(TAU, plane(0), 4.0);
        // A τ sweep in one span would fold to the identity; subdivided,
        // the quarter- and half-turn marks land where a real sweep does.
        let p = Pga3::point(1.0, 0.0, 0.0);
        let quarter = track.sample(1.0).apply(&p);
        let want_q = Motor::rotor(TAU / 4.0, plane(0)).apply(&p);
        let half = track.sample(2.0).apply(&p);
        let want_h = Motor::rotor(TAU / 2.0, plane(0)).apply(&p);
        for i in 0..16 {
            assert!((quarter.coeffs[i] - want_q.coeffs[i]).abs() < 1e-9, "quarter, coeff {i}");
            assert!((half.coeffs[i] - want_h.coeffs[i]).abs() < 1e-9, "half, coeff {i}");
        }
        // And the full turn arrives back at the identity *motion*.
        assert!(track.sample(4.0).geodesic_distance(&Motor::identity()) < 1e-9);
    }

    #[test]
    fn still_holds_its_pose_forever() {
        let m = Motor::translator(1.0, 2.0, 3.0) * Motor::rotor(TAU / 5.0, plane(1));
        let track = Track::still(m);
        for t in [-10.0, 0.0, 0.5, 1e6] {
            assert_eq!(track.sample(t), m);
        }
    }

    #[test]
    #[should_panic(expected = "at least one key")]
    fn empty_track_panics() {
        let _ = Track::keys([]);
    }

    #[test]
    #[should_panic(expected = "strictly ascending")]
    fn unsorted_keys_panic() {
        let _ = Track::keys([(1.0, Motor::identity()), (1.0, Motor::identity())]);
    }
}
