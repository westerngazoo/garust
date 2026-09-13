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

/// The **generalized** force at joint `j`: what that joint's own degree of
/// freedom feels, by virtual work.
///
/// `joint_torque` answers geometrically — the moment of everything distal.
/// This answers variationally — how much work the loads do when only `q[j]`
/// moves, `τ_j = Σ F · ∂x/∂q_j`. The two are independent derivations of the
/// same quantity, and the claim that they agree is worth more than either:
/// a cross product and a gradient do not make the same mistake.
///
/// It is also the general one. It needs no joint axis, so it works for a
/// prismatic joint as it does for a revolute one, and it is the same
/// principle of virtual work that a variational formulation is built on —
/// which is the reason this exists now rather than the day it is needed.
///
/// Central differences with step `h`. `h` is a knob because the right one
/// depends on the scale of the mechanism, and a hidden default would be
/// wrong for somebody.
#[must_use]
pub fn generalized_torque(
    tree: &Tree<'_>,
    loads: &[&dyn Load],
    q: &[f64],
    j: usize,
    h: f64,
    poses: &mut [Motor<f64>],
    scratch: &mut [f64],
) -> f64 {
    debug_assert!(h > 0.0, "the step must be positive");
    scratch.copy_from_slice(q);

    let mut trabajo = |dq: f64, poses: &mut [Motor<f64>], scratch: &mut [f64]| -> f64 {
        scratch[j] = q[j] + dq;
        tree.fk(scratch, poses);
        // Potential-like term: −F·x summed over loads. Only its change
        // matters, so any origin does.
        let mut w = 0.0;
        for l in loads {
            let a = l.applied(poses);
            w -= a.force[0] * a.at[0] + a.force[1] * a.at[1] + a.force[2] * a.at[2];
        }
        w
    };

    let mas = trabajo(h, poses, scratch);
    let menos = trabajo(-h, poses, scratch);
    scratch[j] = q[j];
    tree.fk(scratch, poses);
    (mas - menos) / (2.0 * h)
}

/// Work done **by** the external loads along a trajectory, joules.
///
/// The trajectory is a function of progress `u ∈ [0, 1]` writing the joint
/// vector. Trapezoid over `n` steps of each load's `F · dx`, which is the
/// definition — force through the distance it actually travels, along its
/// own line of action — and not `F` times how far something looked like it
/// moved.
pub fn load_work<F>(
    tree: &Tree<'_>,
    loads: &[&dyn Load],
    mut trajectory: F,
    n: usize,
    poses: &mut [Motor<f64>],
    scratch: &mut [f64],
) -> f64
where
    F: FnMut(f64, &mut [f64]),
{
    let mut total = 0.0;
    let mut previo: Option<[f64; 3]> = None;
    let mut anteriores: [[f64; 3]; 16] = [[0.0; 3]; 16];
    for i in 0..=n {
        let u = i as f64 / n as f64;
        trajectory(u, scratch);
        tree.fk(scratch, poses);
        for (k, l) in loads.iter().enumerate() {
            let a = l.applied(poses);
            if previo.is_some() {
                let dx = [
                    a.at[0] - anteriores[k][0],
                    a.at[1] - anteriores[k][1],
                    a.at[2] - anteriores[k][2],
                ];
                total += a.force[0] * dx[0] + a.force[1] * dx[1] + a.force[2] * dx[2];
            }
            anteriores[k] = a.at;
        }
        previo = Some([0.0; 3]);
    }
    total
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

    /// Un caso a mano, para fijar el signo antes de confiar en nada.
    #[test]
    fn el_signo_a_mano() {
        let links = [TreeLink {
            parent: None,
            offset: Motor::identity(),
            joint: ChainJoint::Revolute(z()),
        }];
        let tree = Tree::new(&links);
        let w = Weight { link: 0, offset: [0.71, 0.0, 0.0], mass: 60.0, gravity: [0.0, -G, 0.0] };
        let loads: [&dyn Load; 1] = [&w];
        let mut poses = [Motor::identity(); 1];
        let mut scratch = [0.0_f64; 1];
        tree.fk(&[0.0], &mut poses);
        // 60 kg a 0.71 m del eje, brazo horizontal: 60·9.81·0.71 = 417.906,
        // y negativo porque el peso hace girar hacia −z. Aritmética de
        // mano, que es el único árbitro cuando dos derivaciones no
        // coinciden en el signo — y no coincidían.
        let a = super::joint_torque(&tree, &poses, &loads, 0);
        let v = super::generalized_torque(&tree, &loads, &[0.0], 0, 1e-6, &mut poses, &mut scratch);
        assert!((a[2] + 417.906).abs() < 1e-6, "geométrico {a:?}");
        assert!((v + 417.906).abs() < 1e-4, "variacional {v}");
    }

    /// **The two derivations agree.** A cross product and a gradient do
    /// not make the same mistake, so when the geometric route (moments of
    /// everything distal) and the variational route (work under a virtual
    /// displacement) land on the same number, both are right.
    ///
    /// The joints here turn about `z`, so the generalized force is the `z`
    /// component of the moment. For a joint about some other axis it is
    /// the projection onto that axis, and the variational route is the one
    /// that does not need to know it.
    #[test]
    fn the_geometric_and_variational_torques_agree() {
        let links = [
            TreeLink { parent: None, offset: Motor::identity(), joint: ChainJoint::Revolute(z()) },
            TreeLink { parent: Some(0), offset: Motor::translator(0.40, 0.0, 0.0), joint: ChainJoint::Revolute(z()) },
        ];
        let tree = Tree::new(&links);
        let w = Weight {
            link: 1,
            offset: [0.31, 0.0, 0.0],
            mass: 60.0,
            gravity: [0.0, -G, 0.0],
        };
        let loads: [&dyn Load; 1] = [&w];
        let mut poses = [Motor::identity(); 2];
        let mut scratch = [0.0_f64; 2];
        for (a, b) in [(0.0, 0.0), (0.4, -0.7), (-1.1, 0.35), (1.3, 1.0)] {
            let q = [a, b];
            tree.fk(&q, &mut poses);
            for j in 0..2 {
                let geom = super::joint_torque(&tree, &poses, &loads, j)[2];
                let var = super::generalized_torque(
                    &tree, &loads, &q, j, 1e-6, &mut poses, &mut scratch,
                );
                assert!(
                    (geom - var).abs() < 1e-4,
                    "q={q:?} j={j}: geométrico {geom:.6}, variacional {var:.6}"
                );
            }
        }
    }

    /// Work is force through the distance its point of application really
    /// travels: lifting `m` by `h` costs `m·g·h`, whatever the mechanism
    /// did on the way.
    #[test]
    fn load_work_is_force_through_distance() {
        let links = [TreeLink {
            parent: None,
            offset: Motor::identity(),
            joint: ChainJoint::Prismatic([0.0, 1.0, 0.0]),
        }];
        let tree = Tree::new(&links);
        let w = Weight {
            link: 0,
            offset: [0.0, 0.0, 0.0],
            mass: 100.0,
            gravity: [0.0, -G, 0.0],
        };
        let loads: [&dyn Load; 1] = [&w];
        let mut poses = [Motor::identity(); 1];
        let mut scratch = [0.0_f64; 1];
        let subida = 0.40;
        let w_carga = super::load_work(
            &tree,
            &loads,
            |u, q| q[0] = u * subida,
            400,
            &mut poses,
            &mut scratch,
        );
        // La carga hace trabajo NEGATIVO al subir: se lo hacen a ella.
        assert!(
            (w_carga + 100.0 * G * subida).abs() < 1e-9,
            "trabajo de la carga {w_carga:.4}, esperado {:.4}",
            -100.0 * G * subida
        );
    }

    /// **Balance de energía.** Lo que hacen las articulaciones es lo que
    /// se le hace a la carga. No se puede fingir: cierra sólo si la
    /// cinemática, los torques y el recorrido cuentan la misma historia,
    /// y ninguna de las tres pasa sola.
    #[test]
    fn joint_work_matches_load_work() {
        let links = [
            TreeLink { parent: None, offset: Motor::identity(), joint: ChainJoint::Revolute(z()) },
            TreeLink { parent: Some(0), offset: Motor::translator(0.40, 0.0, 0.0), joint: ChainJoint::Revolute(z()) },
        ];
        let tree = Tree::new(&links);
        let w = Weight {
            link: 1,
            offset: [0.31, 0.0, 0.0],
            mass: 60.0,
            gravity: [0.0, -G, 0.0],
        };
        let loads: [&dyn Load; 1] = [&w];
        let mut poses = [Motor::identity(); 2];
        let mut scratch = [0.0_f64; 2];
        // una trayectoria cualquiera: las dos articulaciones se mueven
        let camino = |u: f64| [0.9 * u, -0.6 * u + 0.3 * u * u];

        let n = 2000;
        let mut w_art = 0.0;
        let mut previo = camino(0.0);
        for i in 1..=n {
            let u = i as f64 / n as f64;
            let q = camino(u);
            let medio = [(q[0] + previo[0]) / 2.0, (q[1] + previo[1]) / 2.0];
            for j in 0..2 {
                let tau = super::generalized_torque(
                    &tree, &loads, &medio, j, 1e-6, &mut poses, &mut scratch,
                );
                w_art += tau * (q[j] - previo[j]);
            }
            previo = q;
        }
        let w_carga = super::load_work(
            &tree,
            &loads,
            |u, q| q.copy_from_slice(&camino(u)),
            n,
            &mut poses,
            &mut scratch,
        );
        // Las articulaciones hacen sobre la carga lo contrario de lo que la
        // carga hace sobre ellas.
        assert!(
            (w_art + w_carga).abs() < 1e-4 * w_art.abs().max(1.0),
            "articulaciones {w_art:.6} J contra carga {w_carga:.6} J"
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
