//! Inverse dynamics on a kinematic tree: what each joint has to hold.
//!
//! The rule is one line — **the moment at a joint is the moment of
//! everything distal to it** — and the tree is what makes "distal"
//! answerable ([`Tree::is_distal_to`](garust_geo::tree::Tree::is_distal_to)).
//!
//! Quasi-static: this is the moment the joint must *balance*, not the one
//! that accelerates it. For slow movement the difference is small and the
//! statement is exact; for a throw it is neither, and the caller is the
//! one who knows which it has.

use garust_core::Pga3;
use garust_geo::tree::Tree;
use garust_geo::Motor;

use crate::load::Load;
use crate::Inertia;

/// `a × b`.
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The world-frame position of a joint's own frame origin.
#[must_use]
pub fn joint_point(poses: &[Motor<f64>], joint: usize) -> [f64; 3] {
    crate::world::pga_point_xyz(&poses[joint].apply(&Pga3::point(0.0, 0.0, 0.0)))
}

/// The moment about `joint` of every load distal to it, world frame.
///
/// Loads that are **not** distal are skipped rather than cancelled: a
/// weight hanging off the other arm is carried by the torso, not by this
/// shoulder, and including it with a lever arm would be a fabrication,
/// not a rounding error.
#[must_use]
pub fn joint_torque(
    tree: &Tree<'_>,
    poses: &[Motor<f64>],
    loads: &[&dyn Load],
    joint: usize,
) -> [f64; 3] {
    let pivot = joint_point(poses, joint);
    let mut tau = [0.0_f64; 3];
    for l in loads {
        if !tree.is_distal_to(l.link(), joint) {
            continue;
        }
        let a = l.applied(poses);
        let r = [a.at[0] - pivot[0], a.at[1] - pivot[1], a.at[2] - pivot[2]];
        let m = cross(r, a.force);
        for k in 0..3 {
            tau[k] += m[k];
        }
    }
    tau
}

/// The same moment as a PGA bivector, in the crate's existing convention.
///
/// `world.rs` already turns a world-frame torque into a bivector with
/// [`Inertia::principal_planes`]; this reuses it rather than introducing a
/// second convention, because two conventions for the same quantity is
/// how sign errors get a place to hide.
#[must_use]
pub fn joint_torque_bivector(
    tree: &Tree<'_>,
    poses: &[Motor<f64>],
    loads: &[&dyn Load],
    joint: usize,
) -> Pga3 {
    let t = joint_torque(tree, poses, loads, joint);
    let planes = Inertia::principal_planes();
    planes[0] * t[0] + planes[1] * t[1] + planes[2] * t[2]
}

/// Magnitude of [`joint_torque`], N·m — what goes on screen.
#[must_use]
pub fn joint_torque_magnitude(
    tree: &Tree<'_>,
    poses: &[Motor<f64>],
    loads: &[&dyn Load],
    joint: usize,
) -> f64 {
    let t = joint_torque(tree, poses, loads, joint);
    (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::{joint_torque, joint_torque_magnitude};
    use crate::load::{Cable, Load, Weight};
    use garust_core::Pga3;
    use garust_geo::chain::ChainJoint;
    use garust_geo::tree::{Tree, TreeLink};
    use garust_geo::Motor;

    const G: f64 = 9.81;

    fn z() -> Pga3 {
        Pga3::point(0.0, 0.0, 0.0).line_through(&Pga3::point(0.0, 0.0, 1.0))
    }

    /// One link that rotates about z: a torso hinging at the hip.
    fn un_eslabon() -> [TreeLink; 1] {
        [TreeLink {
            parent: None,
            offset: Motor::identity(),
            joint: ChainJoint::Revolute(z()),
        }]
    }

    /// A load straight above the pivot has no lever, so no moment.
    #[test]
    fn no_lever_no_moment() {
        let links = un_eslabon();
        let tree = Tree::new(&links);
        let mut poses = [Motor::identity(); 1];
        tree.fk(&[0.0], &mut poses);
        let w = Weight {
            link: 0,
            offset: [0.0, 0.53, 0.0], // justo encima del pivote
            mass: 100.0,
            gravity: [0.0, -G, 0.0],
        };
        let loads: [&dyn Load; 1] = [&w];
        assert!(joint_torque_magnitude(&tree, &poses, &loads, 0) < 1e-9);
    }

    /// The general machinery reproduces the closed form `τ = m·g·L·sin φ`.
    ///
    /// This is the one that matters. `mecanica::gluteo::Rumano` states that
    /// formula and has it under claim; here a tree, a weight and a cross
    /// product arrive at the same number without being told the formula.
    /// If these two ever disagree, one of them is lying to an audience.
    #[test]
    fn reproduces_the_hip_hinge_closed_form() {
        let links = un_eslabon();
        let tree = Tree::new(&links);
        let (m, l) = (100.0_f64, 0.53_f64);
        let w = Weight {
            link: 0,
            offset: [0.0, l, 0.0],
            mass: m,
            gravity: [0.0, -G, 0.0],
        };
        let loads: [&dyn Load; 1] = [&w];
        for deg in [0.0, 15.0, 30.0, 45.0, 72.0, 90.0] {
            let phi = f64::to_radians(deg);
            let mut poses = [Motor::identity(); 1];
            tree.fk(&[phi], &mut poses);
            let tau = joint_torque_magnitude(&tree, &poses, &loads, 0);
            let esperado = m * G * l * phi.sin();
            assert!(
                (tau - esperado).abs() < 1e-9,
                "a {deg}°: el árbol da {tau:.6}, la forma cerrada {esperado:.6}"
            );
        }
    }

    /// Moments add: two loads are the sum of each on its own.
    #[test]
    fn moments_superpose() {
        let links = un_eslabon();
        let tree = Tree::new(&links);
        let a = Weight {
            link: 0,
            offset: [0.3, 0.4, 0.0],
            mass: 40.0,
            gravity: [0.0, -G, 0.0],
        };
        let b = Weight {
            link: 0,
            offset: [-0.2, 0.5, 0.0],
            mass: 25.0,
            gravity: [0.0, -G, 0.0],
        };
        let mut poses = [Motor::identity(); 1];
        tree.fk(&[0.4], &mut poses);
        let sola_a: [&dyn Load; 1] = [&a];
        let sola_b: [&dyn Load; 1] = [&b];
        let juntas: [&dyn Load; 2] = [&a, &b];
        let ta = joint_torque(&tree, &poses, &sola_a, 0);
        let tb = joint_torque(&tree, &poses, &sola_b, 0);
        let tj = joint_torque(&tree, &poses, &juntas, 0);
        for k in 0..3 {
            assert!((tj[k] - (ta[k] + tb[k])).abs() < 1e-12);
        }
    }

    /// A load on a sibling branch is carried by the root, not by us.
    #[test]
    fn a_sibling_branch_is_not_ours_to_carry() {
        let links = [
            TreeLink { parent: None, offset: Motor::identity(), joint: ChainJoint::Revolute(z()) },
            TreeLink { parent: Some(0), offset: Motor::translator(0.4, 0.0, 0.0), joint: ChainJoint::Revolute(z()) },
            TreeLink { parent: Some(0), offset: Motor::translator(-0.4, 0.0, 0.0), joint: ChainJoint::Revolute(z()) },
        ];
        let tree = Tree::new(&links);
        let mut poses = [Motor::identity(); 3];
        tree.fk(&[0.0, 0.0, 0.0], &mut poses);
        let en_la_otra_rama = Weight {
            link: 2,
            offset: [0.3, 0.0, 0.0],
            mass: 50.0,
            gravity: [0.0, -G, 0.0],
        };
        let loads: [&dyn Load; 1] = [&en_la_otra_rama];
        assert!(
            joint_torque_magnitude(&tree, &poses, &loads, 1) < 1e-12,
            "la articulación 1 no carga la rama 2"
        );
        assert!(
            joint_torque_magnitude(&tree, &poses, &loads, 0) > 1.0,
            "pero la raíz sí"
        );
    }

    /// A cable whose line passes through the pivot exerts no moment,
    /// however hard it pulls. A weight in that position would not: this is
    /// the difference a line of action makes.
    #[test]
    fn a_cable_through_the_pivot_has_no_moment() {
        let links = un_eslabon();
        let tree = Tree::new(&links);
        let mut poses = [Motor::identity(); 1];
        tree.fk(&[0.0], &mut poses);
        let c = Cable {
            link: 0,
            offset: [0.0, 0.5, 0.0],
            tension: 900.0,
            pulley: [0.0, -2.0, 0.0], // la línea pasa por el pivote
        };
        let loads: [&dyn Load; 1] = [&c];
        assert!(joint_torque_magnitude(&tree, &poses, &loads, 0) < 1e-9);
    }
}
