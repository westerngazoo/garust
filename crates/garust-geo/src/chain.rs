//! Open kinematic chains: screw-axis joints, forward kinematics, and
//! damped-least-squares inverse kinematics (RFC-013).
//!
//! The GA advantage is a single representation end to end: a joint is a
//! screw axis (a PGA line or ideal direction), a link offset is a
//! [`Motor`], forward kinematics is a motor *product*, and the pose error
//! IK drives to zero is the `log` of a motor — a twist bivector. There is
//! no quaternion/translation split anywhere, so there are no
//! frame-convention seams for sign errors to hide in.
//!
//! Everything is allocation-free: a [`Chain`] borrows its links, and all
//! solver scratch lives on the stack (chains are capped at
//! [`MAX_LINKS`] joints; splines and typical arms sit far below it).
//!
//! ```
//! use garust_geo::chain::{Chain, ChainJoint, IkParams, Link};
//! use garust_geo::Motor;
//! use garust_core::Pga3;
//! use core::f64::consts::TAU;
//!
//! // A 2-link planar arm: both joints revolve about the z-axis.
//! let z = Pga3::point(0.0, 0.0, 0.0).line_through(&Pga3::point(0.0, 0.0, 1.0));
//! let links = [
//!     Link { offset: Motor::identity(), joint: ChainJoint::Revolute(z) },
//!     Link { offset: Motor::translator(1.0, 0.0, 0.0), joint: ChainJoint::Revolute(z) },
//! ];
//! let arm = Chain::new(&links);
//!
//! // Reach for the pose the arm holds at q = (τ/8, −τ/8).
//! let target = arm.fk(&[TAU / 8.0, -TAU / 8.0]);
//! let mut q = [0.0_f64; 2];
//! let result = arm.ik_dls(&target, &[0.3, -0.1], &mut q, IkParams::default());
//! assert!(result.converged);
//! ```

use crate::Motor;
use garust_core::{Pga3, Real};

/// Maximum number of links a [`Chain`] may hold (stack-scratch bound).
pub const MAX_LINKS: usize = 16;

/// Bivector blade indices `[e12, e13, e23, e01, e02, e03]` — the fixed
/// coordinate order used for error twists.
const BIV: [usize; 6] = [3, 5, 6, 9, 10, 12];

/// Central-difference step for the numeric Jacobian.
const FD_STEP: f64 = 1e-6;

/// One degree of freedom: how a joint variable `q` becomes a motor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChainJoint {
    /// Rotation by `q` radians about a fixed line in the parent frame —
    /// any PGA 2-blade with non-zero Euclidean weight (normalized
    /// internally, as [`Motor::rotation_about`]).
    Revolute(Pga3),
    /// Translation by `q` along a direction in the parent frame. The
    /// direction sets the slide axis; its length scales `q`.
    Prismatic([f64; 3]),
}

impl ChainJoint {
    /// The joint's motion at variable value `q`.
    fn motor(&self, q: f64) -> Motor<f64> {
        match *self {
            ChainJoint::Revolute(line) => Motor::rotation_about(line, q),
            ChainJoint::Prismatic(d) => Motor::translator(d[0] * q, d[1] * q, d[2] * q),
        }
    }
}

/// One link: a rigid `offset` from the parent frame, then the joint's
/// motion. A link's frame is `offset · joint(q)` relative to its parent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Link {
    /// Rigid transform from the parent frame to this joint's frame.
    pub offset: Motor<f64>,
    /// The degree of freedom that follows the offset.
    pub joint: ChainJoint,
}

/// An open kinematic chain over borrowed links — the chain itself owns
/// nothing and allocates nothing.
///
/// The end-effector pose at joint vector `q` is the motor product
/// `Π offsetᵢ · jointᵢ(qᵢ)`. A fixed tool transform is *not* stored:
/// fold it into the IK target instead (`target · tool⁻¹`).
#[derive(Clone, Copy, Debug)]
pub struct Chain<'a> {
    links: &'a [Link],
}

/// Damped-least-squares IK settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IkParams {
    /// Damping `λ` added as `λ²I` to `J Jᵀ` — keeps the solve full-rank
    /// at singular poses (e.g. a straight arm).
    pub damping: f64,
    /// Convergence threshold on the error-twist norm.
    pub tol: f64,
    /// Iteration cap.
    pub max_iter: usize,
}

impl Default for IkParams {
    fn default() -> Self {
        Self { damping: 0.1, tol: 1e-10, max_iter: 200 }
    }
}

/// What [`Chain::ik_dls`] achieved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IkResult {
    /// Whether the error-twist norm fell below `tol`.
    pub converged: bool,
    /// Iterations actually run.
    pub iters: usize,
    /// Final error-twist norm (best effort when not converged).
    pub err: f64,
}

impl<'a> Chain<'a> {
    /// Wrap a slice of links as a chain.
    ///
    /// # Panics
    ///
    /// If `links` is empty or longer than [`MAX_LINKS`].
    pub fn new(links: &'a [Link]) -> Self {
        assert!(
            !links.is_empty() && links.len() <= MAX_LINKS,
            "Chain takes 1..=MAX_LINKS links"
        );
        Self { links }
    }

    /// Number of degrees of freedom (one per link).
    pub fn dof(&self) -> usize {
        self.links.len()
    }

    /// Forward kinematics: the base-to-end-effector pose at joint vector
    /// `q` — one motor product per link.
    ///
    /// # Panics
    ///
    /// If `q.len() != self.dof()`.
    pub fn fk(&self, q: &[f64]) -> Motor<f64> {
        assert_eq!(q.len(), self.links.len(), "joint vector length != dof");
        let mut pose = Motor::identity();
        for (link, &qi) in self.links.iter().zip(q.iter()) {
            pose = pose * link.offset * link.joint.motor(qi);
        }
        pose
    }

    /// The body-frame error twist carrying `fk(q)` to `target`, as the six
    /// bivector coordinates of `log(fk(q)⁻¹ · target)`.
    fn error_twist(&self, q: &[f64], target: &Motor<f64>) -> [f64; 6] {
        let delta = (self.fk(q).inverse() * *target).log();
        let mut e = [0.0_f64; 6];
        for (k, &idx) in BIV.iter().enumerate() {
            e[k] = delta.coeffs[idx];
        }
        e
    }

    /// Damped-least-squares inverse kinematics: iterate
    /// `Δq = −Jᵀ (J Jᵀ + λ²I)⁻¹ e` from seed `q0` until the error twist
    /// `e = log(fk(q)⁻¹ · target)` is below `tol`, writing the best joint
    /// vector found into `out`.
    ///
    /// The Jacobian is taken by central finite differences of the error
    /// coordinates — deliberately convention-proof (RFC-013 §3.3); the
    /// damping keeps singular poses (straight arm) solvable. On a
    /// non-converged return, `out` still holds the best-effort solution
    /// and [`IkResult::err`] the residual.
    ///
    /// # Panics
    ///
    /// If `q0.len()` or `out.len()` differ from `self.dof()`.
    pub fn ik_dls(
        &self,
        target: &Motor<f64>,
        q0: &[f64],
        out: &mut [f64],
        params: IkParams,
    ) -> IkResult {
        let n = self.links.len();
        assert_eq!(q0.len(), n, "seed length != dof");
        assert_eq!(out.len(), n, "output length != dof");

        let mut q = [0.0_f64; MAX_LINKS];
        q[..n].copy_from_slice(q0);

        let mut err = 0.0_f64;
        for iter in 0..params.max_iter {
            let e = self.error_twist(&q[..n], target);
            err = e.iter().map(|x| x * x).sum::<f64>().sqrt();
            if err < params.tol {
                out.copy_from_slice(&q[..n]);
                return IkResult { converged: true, iters: iter, err };
            }

            // J[i][j] = ∂eᵢ/∂qⱼ by central differences.
            let mut jac = [[0.0_f64; MAX_LINKS]; 6];
            for j in 0..n {
                let (mut qp, mut qm) = (q, q);
                qp[j] += FD_STEP;
                qm[j] -= FD_STEP;
                let ep = self.error_twist(&qp[..n], target);
                let em = self.error_twist(&qm[..n], target);
                for i in 0..6 {
                    jac[i][j] = (ep[i] - em[i]) / (2.0 * FD_STEP);
                }
            }

            // A = J Jᵀ + λ²I (6×6), solve A·w = e.
            let lambda2 = params.damping * params.damping;
            let mut a = [[0.0_f64; 6]; 6];
            for i in 0..6 {
                for k in 0..6 {
                    let mut s = 0.0;
                    for j in 0..n {
                        s += jac[i][j] * jac[k][j];
                    }
                    a[i][k] = s;
                }
                a[i][i] += lambda2;
            }
            let w = solve6(a, e);

            // Δq = −Jᵀ w.
            for j in 0..n {
                let mut dq = 0.0;
                for i in 0..6 {
                    dq += jac[i][j] * w[i];
                }
                q[j] -= dq;
            }
        }

        out.copy_from_slice(&q[..n]);
        IkResult { converged: err < params.tol, iters: params.max_iter, err }
    }
}

/// Solve the 6×6 system `A·x = b` by Gaussian elimination with partial
/// pivoting. `A` is `J Jᵀ + λ²I`, symmetric positive definite for any
/// `λ > 0`, so the pivot never vanishes.
fn solve6(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> [f64; 6] {
    for col in 0..6 {
        let mut pivot = col;
        for row in col + 1..6 {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        a.swap(col, pivot);
        b.swap(col, pivot);

        let inv = 1.0 / a[col][col];
        for row in col + 1..6 {
            let f = a[row][col] * inv;
            if f == 0.0 {
                continue;
            }
            for k in col..6 {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0_f64; 6];
    for col in (0..6).rev() {
        let mut s = b[col];
        for k in col + 1..6 {
            s -= a[col][k] * x[k];
        }
        x[col] = s / a[col][col];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::{Chain, ChainJoint, IkParams, Link, MAX_LINKS};
    use crate::{pga, Motor};
    use garust_core::Pga3;
    use std::f64::consts::TAU;

    fn z_axis() -> Pga3 {
        Pga3::point(0.0, 0.0, 0.0).line_through(&Pga3::point(0.0, 0.0, 1.0))
    }

    /// End-effector position: chain pose + tool offset applied to origin.
    fn ee(chain: &Chain, q: &[f64], tool_x: f64) -> (f64, f64, f64) {
        let pose = chain.fk(q) * Motor::translator(tool_x, 0.0, 0.0);
        pga::Point::new(0.0, 0.0, 0.0).transform(&pose).to_euclidean()
    }

    fn two_link() -> [Link; 2] {
        [
            Link { offset: Motor::identity(), joint: ChainJoint::Revolute(z_axis()) },
            Link { offset: Motor::translator(1.0, 0.0, 0.0), joint: ChainJoint::Revolute(z_axis()) },
        ]
    }

    #[test]
    fn fk_of_zero_q_is_the_product_of_offsets() {
        let links = two_link();
        let arm = Chain::new(&links);
        let (x, y, z) = ee(&arm, &[0.0, 0.0], 1.0);
        assert!((x - 2.0).abs() < 1e-12 && y.abs() < 1e-12 && z.abs() < 1e-12);
    }

    #[test]
    fn fk_quarter_turn_at_the_base_swings_the_whole_arm() {
        let links = two_link();
        let arm = Chain::new(&links);
        let (x, y, z) = ee(&arm, &[TAU / 4.0, 0.0], 1.0);
        // 90° about z: the straight arm leaves the x-axis for the y-axis
        // (orientation of the line fixes which way; the length is exact).
        assert!(x.abs() < 1e-12, "x = {x}");
        assert!((y.abs() - 2.0).abs() < 1e-12, "y = {y}");
        assert!(z.abs() < 1e-12);
    }

    #[test]
    fn fk_elbow_bend_shortens_the_reach() {
        let links = two_link();
        let arm = Chain::new(&links);
        let (x, y, _z) = ee(&arm, &[0.0, TAU / 4.0], 1.0);
        // Elbow at 90°: |ee| = √(1² + 1²).
        let r = (x * x + y * y).sqrt();
        assert!((r - 2.0_f64.sqrt()).abs() < 1e-12, "r = {r}");
    }

    #[test]
    fn fk_mixed_revolute_prismatic_chain() {
        // A rotary base plus a vertical lift.
        let links = [
            Link { offset: Motor::identity(), joint: ChainJoint::Revolute(z_axis()) },
            Link {
                offset: Motor::translator(1.0, 0.0, 0.0),
                joint: ChainJoint::Prismatic([0.0, 0.0, 1.0]),
            },
        ];
        let arm = Chain::new(&links);
        let (x, y, z) = ee(&arm, &[0.0, 0.7], 0.0);
        assert!((x - 1.0).abs() < 1e-12 && y.abs() < 1e-12);
        assert!((z - 0.7).abs() < 1e-12, "lift z = {z}");
    }

    #[test]
    fn ik_reaches_a_pose_the_arm_can_hold() {
        let links = two_link();
        let arm = Chain::new(&links);
        let target = arm.fk(&[TAU / 8.0, -TAU / 6.0]);
        let mut q = [0.0_f64; 2];
        let r = arm.ik_dls(&target, &[0.3, -0.1], &mut q, IkParams::default());
        assert!(r.converged, "no convergence: err = {}", r.err);
        assert!(arm.fk(&q).geodesic_distance(&target) < 1e-8);
    }

    #[test]
    fn ik_converges_from_the_singular_straight_arm_seed() {
        let links = two_link();
        let arm = Chain::new(&links);
        let target = arm.fk(&[0.6, 0.9]);
        let mut q = [0.0_f64; 2];
        // Seed [0, 0] is the fully-stretched (singular) configuration —
        // the damping term must carry the solve through it.
        let r = arm.ik_dls(&target, &[0.0, 0.0], &mut q, IkParams::default());
        assert!(r.converged, "stuck at the singularity: err = {}", r.err);
        assert!(arm.fk(&q).geodesic_distance(&target) < 1e-8);
    }

    #[test]
    fn ik_reports_failure_on_an_unreachable_target() {
        let links = two_link();
        let arm = Chain::new(&links);
        // Identity orientation 3 units out: past the 2-unit reach.
        let target = Motor::translator(3.0, 0.0, 0.0);
        let mut q = [0.0_f64; 2];
        let r = arm.ik_dls(&target, &[0.3, -0.1], &mut q, IkParams::default());
        assert!(!r.converged);
        assert!(r.err > 0.1, "err = {} suspiciously small", r.err);
    }

    #[test]
    #[should_panic(expected = "1..=MAX_LINKS")]
    fn chain_rejects_too_many_links() {
        let links = [Link {
            offset: Motor::identity(),
            joint: ChainJoint::Prismatic([1.0, 0.0, 0.0]),
        }; MAX_LINKS + 1];
        let _ = Chain::new(&links);
    }
}
