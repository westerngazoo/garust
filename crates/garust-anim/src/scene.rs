//! Scene graph and camera projection (RFC-012, milestone A2).
//!
//! A [`Scene`] owns a list of [`Object`]s — typed PGA geometry, a
//! [`Style`], and the [`Track`] that moves it — plus one [`Camera`]
//! whose pose is a motor like everything else. Evaluating a frame
//! ([`Scene::frame_at`]) samples every track, applies the resulting
//! motors to the geometry, transforms into camera space with the
//! *inverse* of the camera's pose, projects, and returns flat 2D
//! primitives ([`Prim2`]) ready for a frame sink (milestone A3).
//!
//! The pipeline is deterministic: the same scene at the same time
//! yields the same primitives, bit for bit — pinned by a golden test.

use crate::Track;
use garust_geo::pga;

/// Camera-frame convention: the camera looks down **−z**, with x to the
/// right and y up (the GL convention). A camera-space point is visible
/// when its z is below this epsilon — everything at or behind the
/// camera plane is culled.
const NEAR_EPSILON: f64 = 1e-9;

/// Handle to an [`Object`] in a [`Scene`], returned by [`Scene::add`].
///
/// The derived shapes ([`Shape::JoinLine`], [`Shape::MeetPoint`]) refer
/// to other objects through these handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObjectId(pub(crate) usize);

/// A drawable: typed PGA geometry, moved by the owning object's track.
#[derive(Clone, Debug)]
pub enum Shape {
    /// A single point.
    Point(pga::Point),
    /// A straight segment between two points.
    Segment(pga::Point, pga::Point),
    /// An open polyline through the points, in order.
    Polyline(Vec<pga::Point>),
    /// **Derived** (milestone A5): the live line joining two objects'
    /// anchors, recomputed from transformed geometry each frame.
    /// Plumbed here; [`Scene::frame_at`] emits nothing for it yet.
    JoinLine(ObjectId, ObjectId),
    /// **Derived** (milestone A5): the live intersection point of two
    /// planes. Plumbed here; [`Scene::frame_at`] emits nothing for it
    /// yet.
    MeetPoint(ObjectId, ObjectId),
}

/// Stroke styling for a drawable — deliberately minimal (RFC-012 open
/// question 3): color, width, opacity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Style {
    /// Stroke color as 8-bit RGB.
    pub stroke: [u8; 3],
    /// Stroke width in output pixels.
    pub width: f64,
    /// Opacity in `[0, 1]`.
    pub alpha: f64,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            stroke: [0, 0, 0],
            width: 2.0,
            alpha: 1.0,
        }
    }
}

/// A scene element: a [`Shape`], its [`Style`], and the [`Track`] that
/// poses it over time. Build with the named constructors, then chain
/// [`track`](Object::track) and [`style`](Object::style).
#[derive(Clone, Debug)]
pub struct Object {
    /// What is drawn.
    pub shape: Shape,
    /// How it is stroked.
    pub style: Style,
    /// Where it is over time; defaults to holding the identity pose.
    pub track: Track,
}

impl Object {
    fn new(shape: Shape) -> Self {
        Self {
            shape,
            style: Style::default(),
            track: Track::still(garust_geo::Motor::identity()),
        }
    }

    /// A point at `p`.
    pub fn point(p: pga::Point) -> Self {
        Self::new(Shape::Point(p))
    }

    /// A segment from `a` to `b`.
    pub fn segment(a: pga::Point, b: pga::Point) -> Self {
        Self::new(Shape::Segment(a, b))
    }

    /// An open polyline through `pts`, in order.
    pub fn polyline(pts: Vec<pga::Point>) -> Self {
        Self::new(Shape::Polyline(pts))
    }

    /// The live line joining objects `a` and `b` (derived each frame —
    /// evaluated from milestone A5 on).
    pub fn join_line(a: ObjectId, b: ObjectId) -> Self {
        Self::new(Shape::JoinLine(a, b))
    }

    /// The live intersection point of plane objects `a` and `b`
    /// (derived each frame — evaluated from milestone A5 on).
    pub fn meet_point(a: ObjectId, b: ObjectId) -> Self {
        Self::new(Shape::MeetPoint(a, b))
    }

    /// Attach the motion track (builder-style).
    pub fn track(mut self, track: Track) -> Self {
        self.track = track;
        self
    }

    /// Attach the stroke style (builder-style).
    pub fn style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }
}

/// How camera-space geometry lands on the image plane — one enum, two
/// divide rules (RFC-012 open question 2, resolved as *both*).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Projection {
    /// Perspective pinhole: `(x, y, z) ↦ focal · (x, y) / −z`.
    Pinhole {
        /// Focal length: image-plane units per unit of tangent.
        focal: f64,
    },
    /// Orthographic: `(x, y, z) ↦ scale · (x, y)` — depth only culls,
    /// it never forshortens. The classic math-diagram look.
    Orthographic {
        /// Uniform image-plane scale.
        scale: f64,
    },
}

/// The scene's viewpoint: a motor pose (like every other moving thing)
/// and a [`Projection`].
///
/// The camera sits at its pose looking down its local **−z** axis,
/// x right, y up. World geometry enters camera space through
/// `pose.inverse()`.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// The camera's rigid pose in the world.
    pub pose: garust_geo::Motor3,
    /// The divide rule onto the image plane.
    pub projection: Projection,
}

impl Default for Camera {
    /// Identity pose at the origin, pinhole with focal length 1.
    fn default() -> Self {
        Self {
            pose: garust_geo::Motor::identity(),
            projection: Projection::Pinhole { focal: 1.0 },
        }
    }
}

impl Camera {
    /// Project a **camera-space** Euclidean point onto the image plane;
    /// `None` when it is at or behind the camera (z ≥ −ε).
    fn project(&self, (x, y, z): (f64, f64, f64)) -> Option<[f64; 2]> {
        if z >= -NEAR_EPSILON {
            return None;
        }
        Some(match self.projection {
            Projection::Pinhole { focal } => [focal * x / -z, focal * y / -z],
            Projection::Orthographic { scale } => [scale * x, scale * y],
        })
    }
}

/// A flat 2D primitive, the output of [`Scene::frame_at`] and the input
/// of every frame sink (milestone A3).
#[derive(Clone, Debug, PartialEq)]
pub enum Prim2 {
    /// A dot at image coordinates, with the source object's style.
    Dot([f64; 2], Style),
    /// An open stroke through image coordinates, in order, with the
    /// source object's style. Segments arrive as two-point strokes.
    Stroke(Vec<[f64; 2]>, Style),
}

/// A timed collection of [`Object`]s seen through one [`Camera`].
#[derive(Clone, Debug)]
pub struct Scene {
    objects: Vec<Object>,
    /// The viewpoint; move it by giving `pose` a motor (or, later, a
    /// track of its own — a follow-shot is A3+ material).
    pub camera: Camera,
    /// Total running time in seconds — consumed by the render loop
    /// (milestone A3); [`frame_at`](Scene::frame_at) itself accepts any
    /// `t`.
    pub duration: f64,
}

impl Scene {
    /// An empty scene of the given duration with the default camera.
    pub fn new(duration: f64) -> Self {
        Self {
            objects: Vec::new(),
            camera: Camera::default(),
            duration,
        }
    }

    /// Add an object; the returned handle names it for derived shapes.
    pub fn add(&mut self, object: Object) -> ObjectId {
        self.objects.push(object);
        ObjectId(self.objects.len() - 1)
    }

    /// Borrow an object by handle.
    pub fn object(&self, id: ObjectId) -> &Object {
        &self.objects[id.0]
    }

    /// Number of objects in the scene.
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Whether the scene holds no objects.
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// Evaluate one frame at time `t`: sample every track, pose the
    /// geometry, transform into camera space, project, and return the
    /// visible primitives in object order.
    ///
    /// A primitive with **any** vertex at or behind the camera plane is
    /// culled whole (no partial clipping in A2 — subdividing at the
    /// plane is sink-independent work that can layer on later). Derived
    /// shapes emit nothing until milestone A5.
    pub fn frame_at(&self, t: f64) -> Vec<Prim2> {
        let view = self.camera.pose.inverse();
        let mut prims = Vec::new();
        for obj in &self.objects {
            // One combined world→camera motor per object per frame:
            // camera⁻¹ ∘ track pose, applied in a single sandwich.
            let m = view * obj.track.sample(t);
            let project = |p: &pga::Point| self.camera.project(p.transform(&m).to_euclidean());
            match &obj.shape {
                Shape::Point(p) => {
                    if let Some(xy) = project(p) {
                        prims.push(Prim2::Dot(xy, obj.style));
                    }
                }
                Shape::Segment(a, b) => {
                    if let (Some(a2), Some(b2)) = (project(a), project(b)) {
                        prims.push(Prim2::Stroke(vec![a2, b2], obj.style));
                    }
                }
                Shape::Polyline(pts) => {
                    let flat: Option<Vec<[f64; 2]>> = pts.iter().map(project).collect();
                    if let Some(flat) = flat {
                        if flat.len() >= 2 {
                            prims.push(Prim2::Stroke(flat, obj.style));
                        }
                    }
                }
                // Derived shapes are resolved from *transformed* peer
                // geometry — that resolution is milestone A5.
                Shape::JoinLine(..) | Shape::MeetPoint(..) => {}
            }
        }
        prims
    }
}

#[cfg(test)]
mod tests {
    use super::{Camera, Object, Prim2, Projection, Scene, Style};
    use crate::{Ease, Track};
    use garust_core::Pga3;
    use garust_geo::{pga, Motor};
    use std::f64::consts::TAU;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn axis_point_projects_to_center_under_both_models() {
        for projection in [
            Projection::Pinhole { focal: 2.0 },
            Projection::Orthographic { scale: 2.0 },
        ] {
            let mut scene = Scene::new(1.0);
            scene.camera = Camera {
                pose: Motor::identity(),
                projection,
            };
            scene.add(Object::point(pga::Point::new(0.0, 0.0, -3.0)));
            let prims = scene.frame_at(0.0);
            assert_eq!(prims.len(), 1);
            let Prim2::Dot(xy, _) = &prims[0] else {
                panic!("expected a dot")
            };
            assert!(
                close(xy[0], 0.0) && close(xy[1], 0.0),
                "{projection:?}: {xy:?}"
            );
        }
    }

    #[test]
    fn pinhole_foreshortens_with_depth_orthographic_does_not() {
        let at = |proj: Projection, z: f64| {
            let mut scene = Scene::new(1.0);
            scene.camera = Camera {
                pose: Motor::identity(),
                projection: proj,
            };
            scene.add(Object::point(pga::Point::new(1.0, 0.0, z)));
            let prims = scene.frame_at(0.0);
            let Prim2::Dot(xy, _) = prims[0].clone() else {
                panic!()
            };
            xy[0]
        };
        let pin = Projection::Pinhole { focal: 1.0 };
        assert!(close(at(pin, -1.0), 1.0));
        assert!(close(at(pin, -2.0), 0.5), "pinhole must foreshorten");
        let ortho = Projection::Orthographic { scale: 1.0 };
        assert!(close(at(ortho, -1.0), 1.0));
        assert!(close(at(ortho, -9.0), 1.0), "orthographic must not");
    }

    #[test]
    fn geometry_behind_the_camera_is_culled() {
        let mut scene = Scene::new(1.0);
        scene.add(Object::point(pga::Point::new(0.0, 0.0, 1.0))); // behind
        scene.add(Object::segment(
            pga::Point::new(0.0, 0.0, -2.0),
            pga::Point::new(0.0, 0.0, 2.0), // one endpoint behind: cull whole
        ));
        scene.add(Object::point(pga::Point::new(0.5, 0.0, -2.0))); // visible
        assert_eq!(scene.frame_at(0.0).len(), 1);
    }

    #[test]
    fn camera_pose_is_applied_inverse() {
        // Camera shifted to (1, 0, 5) looking down −z sees the world
        // origin at camera-space (−1, 0, −5): left of center.
        let mut scene = Scene::new(1.0);
        scene.camera = Camera {
            pose: Motor::translator(1.0, 0.0, 5.0),
            projection: Projection::Pinhole { focal: 1.0 },
        };
        scene.add(Object::point(pga::Point::new(0.0, 0.0, 0.0)));
        let prims = scene.frame_at(0.0);
        let Prim2::Dot(xy, _) = prims[0].clone() else {
            panic!()
        };
        assert!(close(xy[0], -0.2) && close(xy[1], 0.0), "{xy:?}");
    }

    #[test]
    fn tracks_move_objects_between_frames() {
        let mut scene = Scene::new(2.0);
        scene.camera.projection = Projection::Orthographic { scale: 1.0 };
        scene.add(
            Object::point(pga::Point::new(0.0, 0.0, -5.0)).track(Track::keys([
                (0.0, Motor::identity()),
                (2.0, Motor::translator(4.0, 0.0, 0.0)),
            ])),
        );
        let x_at = |t: f64| {
            let Prim2::Dot(xy, _) = scene.frame_at(t)[0].clone() else {
                panic!()
            };
            xy[0]
        };
        assert!(close(x_at(0.0), 0.0));
        assert!(close(x_at(1.0), 2.0), "linear midpoint of the translation");
        assert!(close(x_at(2.0), 4.0));
    }

    #[test]
    fn derived_shapes_are_plumbed_but_silent_until_a5() {
        let mut scene = Scene::new(1.0);
        let a = scene.add(Object::point(pga::Point::new(0.0, 0.0, -1.0)));
        let b = scene.add(Object::point(pga::Point::new(1.0, 0.0, -1.0)));
        scene.add(Object::join_line(a, b));
        scene.add(Object::meet_point(a, b));
        // The two real points render; the two derived shapes do not yet.
        assert_eq!(scene.frame_at(0.0).len(), 2);
    }

    /// Serialize primitives with fixed formatting for the golden test.
    /// `+ 0.0` folds negative zero so the golden reads naturally.
    fn dump(prims: &[Prim2]) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let n = |v: f64| v + 0.0;
        for p in prims {
            match p {
                Prim2::Dot(xy, st) => {
                    writeln!(
                        s,
                        "dot {:.9} {:.9} | {:?} {} {}",
                        n(xy[0]),
                        n(xy[1]),
                        st.stroke,
                        st.width,
                        st.alpha
                    )
                    .unwrap();
                }
                Prim2::Stroke(pts, st) => {
                    write!(s, "stroke").unwrap();
                    for q in pts {
                        write!(s, " {:.9} {:.9}", n(q[0]), n(q[1])).unwrap();
                    }
                    writeln!(s, " | {:?} {} {}", st.stroke, st.width, st.alpha).unwrap();
                }
            }
        }
        s
    }

    #[test]
    fn golden_frame_is_bit_stable() {
        // A known scene: an eased spinning segment and a styled dot, seen
        // by an offset pinhole camera, sampled mid-animation.
        let mut scene = Scene::new(4.0);
        scene.camera = Camera {
            pose: Motor::translator(0.0, 0.0, 6.0),
            projection: Projection::Pinhole { focal: 2.0 },
        };
        scene.add(
            Object::segment(
                pga::Point::new(-1.0, 0.0, 0.0),
                pga::Point::new(1.0, 0.0, 0.0),
            )
            .track(Track::spin(TAU / 2.0, Pga3::basis(0b0011), 4.0).ease(Ease::SmootherStep)),
        );
        scene.add(Object::point(pga::Point::new(0.0, 1.0, 0.0)).style(Style {
            stroke: [200, 40, 40],
            width: 3.0,
            alpha: 0.5,
        }));

        let got = dump(&scene.frame_at(1.0));
        // Hand-checkable: at t = 1 the eased spin has swept τ/8, so the
        // segment ends sit at (∓√2/2, ∓√2/2, 0); through the z = 6
        // pinhole with focal 2 that is ∓2·(√2/2)/6 = ∓0.235702260. The
        // dot at (0, 1, 0) lands at v = 2·1/6 = ⅓.
        let want = "\
stroke -0.235702260 -0.235702260 0.235702260 0.235702260 | [0, 0, 0] 2 1
dot 0.000000000 0.333333333 | [200, 40, 40] 3 0.5
";
        assert_eq!(got, want, "golden frame drifted:\n{got}");
    }
}
