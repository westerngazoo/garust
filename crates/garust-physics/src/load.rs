//! External loads on a posed skeleton (R-0001 §6.1).
//!
//! A load answers one question: **where does it push, and which way**.
//! That is all inverse dynamics needs, and keeping it to that is what
//! makes the catalogue open — a band, a cam, a chain and a person leaning
//! on you are all the same shape of thing.
//!
//! Purely mechanical on purpose. Nothing here knows about a gym, a
//! barbell or a hip: that vocabulary belongs upstream, in the model that
//! *names* the joints. A kernel that knows what a squat is has the wrong
//! boundary.

use garust_core::Pga3;
use garust_geo::Motor;

use crate::world::pga_point_xyz;

/// El punto de un marco, en mundo: `pose · (offset)`.
///
/// Se usa `apply` y no `apply_point_fast` porque ésa vive detrás de la
/// feature `simd`, y una carga no debería existir sólo a veces.
fn en_mundo(pose: &Motor<f64>, offset: [f64; 3]) -> [f64; 3] {
    pga_point_xyz(&pose.apply(&Pga3::point(offset[0], offset[1], offset[2])))
}

/// One external force, in world frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Applied {
    /// Point of application.
    pub at: [f64; 3],
    /// The force vector — direction **and** magnitude, newtons.
    pub force: [f64; 3],
}

/// Something that applies a force to a posed skeleton.
pub trait Load {
    /// Which link the force is attached to.
    ///
    /// Inverse dynamics needs it to decide whether this load is distal to
    /// a given joint, and therefore whether that joint carries it.
    fn link(&self) -> usize;

    /// The force, given every link's world pose.
    ///
    /// It takes **all** the poses, not just its own link's, because a
    /// load's direction can depend on somewhere else entirely: a cable
    /// pulls toward its pulley, and a band's tension depends on both of
    /// its ends.
    fn applied(&self, poses: &[Motor<f64>]) -> Applied;
}

/// A dead weight: mass at a fixed point of a link, pulled by gravity.
///
/// `gravity` is a parameter and not a constant so that the same type
/// covers a body segment, a barbell, and a load on an incline without a
/// second implementation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weight {
    /// The link the mass rides on.
    pub link: usize,
    /// Where the mass sits, in the link's own frame.
    pub offset: [f64; 3],
    /// Kilograms.
    pub mass: f64,
    /// Gravity vector, m/s². A parameter and not a constant: the same
    /// type then covers a segment, a barbell and a load on an incline.
    pub gravity: [f64; 3],
}

impl Load for Weight {
    fn link(&self) -> usize {
        self.link
    }
    fn applied(&self, poses: &[Motor<f64>]) -> Applied {
        Applied {
            at: en_mundo(&poses[self.link], self.offset),
            force: [
                self.gravity[0] * self.mass,
                self.gravity[1] * self.mass,
                self.gravity[2] * self.mass,
            ],
        }
    }
}

/// A cable under tension, pulling toward a fixed pulley.
///
/// The direction is **not** vertical, and that is the entire point: a
/// cable's line of action moves with the posture, so its moment arm does
/// too. Treating a cable like a weight is the most common way to get a
/// machine exercise wrong.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cable {
    /// The link the cable attaches to.
    pub link: usize,
    /// Where the cable attaches, in the link's own frame.
    pub offset: [f64; 3],
    /// Tension, newtons. Positive pulls toward the pulley.
    pub tension: f64,
    /// The pulley, in world frame.
    pub pulley: [f64; 3],
}

impl Load for Cable {
    fn link(&self) -> usize {
        self.link
    }
    fn applied(&self, poses: &[Motor<f64>]) -> Applied {
        let at = en_mundo(&poses[self.link], self.offset);
        let d = [
            self.pulley[0] - at[0],
            self.pulley[1] - at[1],
            self.pulley[2] - at[2],
        ];
        let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        // At the pulley itself the direction is undefined. Zero is the
        // honest answer: a cable of zero length exerts no line of action,
        // and guessing one would put a torque where there is none.
        let k = if n > f64::EPSILON {
            self.tension / n
        } else {
            0.0
        };
        Applied {
            at,
            force: [d[0] * k, d[1] * k, d[2] * k],
        }
    }
}
