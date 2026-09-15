//! Kinematic **trees**: branching skeletons by parent array (R-0001 §6.1).
//!
//! [`Chain`](crate::chain::Chain) is one strand of joints. A skeleton is
//! not: a torso carries two arms, a pelvis carries two legs, and the two
//! branches move independently. Modelling that as several chains means
//! re-deriving the shared root for each one and hoping the copies agree.
//!
//! The representation is a **parent array** — each link stores the index
//! of the link it hangs from. It is the standard multibody encoding, and
//! it subsumes the open chain exactly: with `parent[i] = i - 1` the tree
//! *is* a chain, and [`Tree::fk`] returns what [`Chain::fk`] returns. That
//! equivalence is a test, not a claim.
//!
//! Same discipline as `chain.rs`: the tree borrows its links, owns
//! nothing, allocates nothing, and forward kinematics writes into a
//! caller-provided slice.
//!
//! ## Links are stored parent-before-child
//!
//! `parent[i] < i` is enforced at construction. That is what lets forward
//! kinematics be a single forward pass with no recursion, no visited set
//! and no stack — a parent's pose is always already computed when its
//! child is reached. The cost is that the caller orders the links, which
//! is how skeletons are written anyway.

use crate::chain::ChainJoint;
use crate::Motor;

/// Cap on links in one tree. A full human skeleton at the granularity
/// this is for — the segments a lift actually loads — sits far below it.
pub const MAX_LINKS: usize = 64;

/// One link: where it hangs from, how it is offset, and its freedom.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TreeLink {
    /// Index of the parent link, or `None` for a root.
    ///
    /// Must be strictly less than this link's own index.
    pub parent: Option<usize>,
    /// Rigid transform from the parent's frame to this joint's frame.
    pub offset: Motor<f64>,
    /// The degree of freedom that follows the offset.
    pub joint: ChainJoint,
}

/// A kinematic tree over borrowed links.
#[derive(Clone, Copy, Debug)]
pub struct Tree<'a> {
    links: &'a [TreeLink],
}

impl<'a> Tree<'a> {
    /// Build a tree.
    ///
    /// # Panics
    ///
    /// If `links` is empty, longer than [`MAX_LINKS`], or if any link's
    /// parent index is not strictly less than its own. The ordering
    /// requirement is checked rather than sorted for: silently reordering
    /// would invalidate every index the caller holds, including the ones
    /// in `q`.
    #[must_use]
    pub fn new(links: &'a [TreeLink]) -> Self {
        assert!(!links.is_empty(), "a tree needs at least one link");
        assert!(
            links.len() <= MAX_LINKS,
            "tree has {} links, cap is {MAX_LINKS}",
            links.len()
        );
        for (i, l) in links.iter().enumerate() {
            if let Some(p) = l.parent {
                assert!(
                    p < i,
                    "link {i} hangs from {p}: parents must come first, so \
                     forward kinematics stays a single pass"
                );
            }
        }
        Self { links }
    }

    /// How many joint variables the tree takes — one per link.
    #[must_use]
    pub fn dof(&self) -> usize {
        self.links.len()
    }

    /// Forward kinematics for **every** link, into `poses`.
    ///
    /// `poses[i]` is the world pose of link `i`'s frame:
    /// `poses[parent] · offsetᵢ · jointᵢ(qᵢ)`, with the identity standing
    /// in for a root's parent.
    ///
    /// Every link at once, rather than one end effector, because that is
    /// what a skeleton is asked for: a torque needs the position of every
    /// joint distal to it, and a drawing needs all of them.
    ///
    /// # Panics
    ///
    /// If `q` or `poses` is not exactly [`Tree::dof`] long.
    pub fn fk(&self, q: &[f64], poses: &mut [Motor<f64>]) {
        assert_eq!(q.len(), self.links.len(), "joint vector length != dof");
        assert_eq!(poses.len(), self.links.len(), "poses length != dof");
        for (i, (link, &qi)) in self.links.iter().zip(q.iter()).enumerate() {
            let base = match link.parent {
                Some(p) => poses[p],
                None => Motor::identity(),
            };
            poses[i] = base * link.offset * link.joint.motor(qi);
        }
    }

    /// The chain of link indices from `i` up to its root, nearest first.
    ///
    /// This is what an inverse-dynamics pass walks: the torque at a joint
    /// is the moment of everything **distal** to it, and distal means
    /// "has this link on its way to the root".
    pub fn ancestors(&self, mut i: usize) -> impl Iterator<Item = usize> + '_ {
        core::iter::from_fn(move || {
            let p = self.links.get(i)?.parent?;
            i = p;
            Some(p)
        })
    }

    /// Whether `maybe_distal` hangs anywhere below `i` (or is `i`).
    #[must_use]
    pub fn is_distal_to(&self, maybe_distal: usize, i: usize) -> bool {
        maybe_distal == i || self.ancestors(maybe_distal).any(|a| a == i)
    }
}

#[cfg(test)]
mod tests {
    use super::{Tree, TreeLink};
    use crate::chain::{Chain, ChainJoint, Link};
    use crate::Motor;
    use garust_core::Pga3;

    fn z_axis() -> Pga3 {
        Pga3::point(0.0, 0.0, 0.0).line_through(&Pga3::point(0.0, 0.0, 1.0))
    }

    /// A tree whose every parent is the previous link **is** a chain.
    ///
    /// This is the whole justification for choosing the parent array over
    /// a separate chain type (R-0001 Q2): the general representation has
    /// to give the special one back, exactly, or it is a second
    /// implementation pretending to be a generalisation.
    #[test]
    fn a_degenerate_tree_is_a_chain() {
        let z = z_axis();
        let links = [
            Link {
                offset: Motor::identity(),
                joint: ChainJoint::Revolute(z),
            },
            Link {
                offset: Motor::translator(1.0, 0.0, 0.0),
                joint: ChainJoint::Revolute(z),
            },
            Link {
                offset: Motor::translator(0.7, 0.0, 0.0),
                joint: ChainJoint::Revolute(z),
            },
        ];
        let tree_links = [
            TreeLink {
                parent: None,
                offset: links[0].offset,
                joint: links[0].joint,
            },
            TreeLink {
                parent: Some(0),
                offset: links[1].offset,
                joint: links[1].joint,
            },
            TreeLink {
                parent: Some(1),
                offset: links[2].offset,
                joint: links[2].joint,
            },
        ];
        let chain = Chain::new(&links);
        let tree = Tree::new(&tree_links);
        let q = [0.3, -0.8, 1.1];
        let mut poses = [Motor::identity(); 3];
        tree.fk(&q, &mut poses);

        let a = chain.fk(&q).apply(&Pga3::point(0.0, 0.0, 0.0));
        let b = poses[2].apply(&Pga3::point(0.0, 0.0, 0.0));
        for k in 0..16 {
            assert!(
                (a.coeffs[k] - b.coeffs[k]).abs() < 1e-12,
                "tree tip != chain tip at coeff {k}"
            );
        }
    }

    /// Two branches off one root move independently.
    #[test]
    fn branches_do_not_drag_each_other() {
        let z = z_axis();
        let links = [
            TreeLink {
                parent: None,
                offset: Motor::identity(),
                joint: ChainJoint::Revolute(z),
            },
            TreeLink {
                parent: Some(0),
                offset: Motor::translator(1.0, 0.0, 0.0),
                joint: ChainJoint::Revolute(z),
            },
            TreeLink {
                parent: Some(0),
                offset: Motor::translator(-1.0, 0.0, 0.0),
                joint: ChainJoint::Revolute(z),
            },
        ];
        let tree = Tree::new(&links);
        let mut a = [Motor::identity(); 3];
        let mut b = [Motor::identity(); 3];
        tree.fk(&[0.0, 0.5, 0.2], &mut a);
        tree.fk(&[0.0, 1.9, 0.2], &mut b); // sólo se mueve la rama 1
                                           // Se mide un punto FUERA del eje. Una revoluta no mueve el origen
                                           // de su propio marco -- mueve lo que cuelga de él -- así que
                                           // probar en el origen habría pasado sin probar nada.
        let sonda = Pga3::point(0.5, 0.0, 0.0);
        let pa = a[2].apply(&sonda);
        let pb = b[2].apply(&sonda);
        for k in 0..16 {
            assert!(
                (pa.coeffs[k] - pb.coeffs[k]).abs() < 1e-12,
                "branch 2 moved"
            );
        }
        let qa = a[1].apply(&sonda);
        let qb = b[1].apply(&sonda);
        assert!(
            (0..16).any(|k| (qa.coeffs[k] - qb.coeffs[k]).abs() > 1e-9),
            "branch 1 did not move"
        );
    }

    /// Moving the root moves everything below it.
    #[test]
    fn the_root_carries_its_branches() {
        let z = z_axis();
        let links = [
            TreeLink {
                parent: None,
                offset: Motor::identity(),
                joint: ChainJoint::Revolute(z),
            },
            TreeLink {
                parent: Some(0),
                offset: Motor::translator(1.0, 0.0, 0.0),
                joint: ChainJoint::Revolute(z),
            },
            TreeLink {
                parent: Some(0),
                offset: Motor::translator(-1.0, 0.0, 0.0),
                joint: ChainJoint::Revolute(z),
            },
        ];
        let tree = Tree::new(&links);
        let mut a = [Motor::identity(); 3];
        let mut b = [Motor::identity(); 3];
        tree.fk(&[0.0, 0.0, 0.0], &mut a);
        tree.fk(&[0.6, 0.0, 0.0], &mut b);
        let sonda = Pga3::point(0.5, 0.0, 0.0);
        for i in 1..3 {
            let pa = a[i].apply(&sonda);
            let pb = b[i].apply(&sonda);
            assert!(
                (0..16).any(|k| (pa.coeffs[k] - pb.coeffs[k]).abs() > 1e-9),
                "link {i} ignored the root"
            );
        }
    }

    /// What inverse dynamics needs: who hangs below whom.
    #[test]
    fn distal_is_everything_below() {
        let z = z_axis();
        let links = [
            TreeLink {
                parent: None,
                offset: Motor::identity(),
                joint: ChainJoint::Revolute(z),
            },
            TreeLink {
                parent: Some(0),
                offset: Motor::identity(),
                joint: ChainJoint::Revolute(z),
            },
            TreeLink {
                parent: Some(1),
                offset: Motor::identity(),
                joint: ChainJoint::Revolute(z),
            },
            TreeLink {
                parent: Some(0),
                offset: Motor::identity(),
                joint: ChainJoint::Revolute(z),
            },
        ];
        let tree = Tree::new(&links);
        assert!(tree.is_distal_to(2, 0), "the whole tree hangs off the root");
        assert!(tree.is_distal_to(2, 1));
        assert!(!tree.is_distal_to(3, 1), "a sibling is not distal");
        assert!(
            !tree.is_distal_to(0, 2),
            "a parent is not distal to its child"
        );
        assert!(tree.is_distal_to(1, 1), "a link is distal to itself");
    }

    /// A child before its parent is rejected, not quietly reordered.
    #[test]
    #[should_panic(expected = "parents must come first")]
    fn out_of_order_links_are_refused() {
        let z = z_axis();
        let links = [
            TreeLink {
                parent: Some(1),
                offset: Motor::identity(),
                joint: ChainJoint::Revolute(z),
            },
            TreeLink {
                parent: None,
                offset: Motor::identity(),
                joint: ChainJoint::Revolute(z),
            },
        ];
        let _ = Tree::new(&links);
    }
}
