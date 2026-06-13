//! The geometric **operation catalog** — the GA alphabet as *data*.
//!
//! Every kernel operation is also a Rust method (`*`, [`wedge`], …), which
//! is all a hand-written program needs. A program that *evolves* or
//! *synthesizes* geometric expressions needs more: to enumerate the
//! available operations, pick one, ask its arity, apply it, and reason about
//! the grades it can produce — all at the value level. That is what this
//! module provides, so an AST / s-expression layer (a "geometric Lisp") and
//! a search engine over it can treat the algebra as an alphabet.
//!
//! - [`Op`] — a `Copy + Eq + Hash` enum of every operation: the binary
//!   products and sums, the unary involutions / duality / grade projection,
//!   and the nullary basis-blade terminals. Cheap to clone, compare, and
//!   hash, so it drops straight into an AST node or a chromosome.
//! - [`Op::arity`], [`Op::name`] — the structural metadata an AST builder
//!   needs.
//! - [`Op::apply`] — evaluate an op against a slice of operand multivectors.
//! - [`Op::output_grades`] — the op's **grade signature**: given the grades
//!   its operands may carry, which grades can the result carry. This is the
//!   per-node rule of the dimensional type system; composing it across a
//!   tree (grade *inference*) is the consuming layer's job.
//! - [`Op::BINARY`], [`Op::UNARY`], [`Op::grade_projections`],
//!   [`Op::basis_blades`] — enumeration, so a generator can sample the
//!   alphabet.
//!
//! The catalog is generic over every signature (the grade rules take the
//! generator count `n`); [`Op`] itself is not generic, so the alphabet is a
//! single static set of symbols regardless of the algebra it is later
//! applied in.
//!
//! Value-carrying leaves (an arbitrary scalar weight, a specific input
//! multivector) are the *engine's* terminals — it supplies them with
//! [`Multivector::scalar`] / [`Multivector::basis`] — so they stay out of
//! this symbol set, which is kept value-free to remain `Copy + Eq + Hash`.
//! [`Op::Basis`] covers the structural unit-blade terminals (index `0` is
//! the scalar unit `1`).
//!
//! ```
//! use garust_core::catalog::{GradeSet, Op};
//! use garust_core::Vga3;
//!
//! // The alphabet is data: enumerate, inspect arity, apply.
//! let op = Op::Geometric;
//! assert_eq!(op.arity(), 2);
//! let r = op.apply(&[Vga3::basis(1), Vga3::basis(1)]); // e1 * e1 = 1
//! assert_eq!(r.scalar_part(), 1.0);
//!
//! // Grade signatures drive the (downstream) dimensional type system: the
//! // geometric product of two vectors is scalar-or-bivector.
//! let vector = GradeSet::singleton(1);
//! let out = Op::Geometric.output_grades(&[vector, vector], 3);
//! assert_eq!(out, GradeSet::EMPTY.with(0).with(2));
//! ```
//!
//! [`wedge`]: crate::Multivector::wedge

use crate::algebra::Algebra;
use crate::multivector::Multivector;
use crate::scalar::Real;
use crate::signature::grade_of;

/// A set of grades, as a bitmask (bit `k` set ⇔ grade `k` is present).
///
/// The currency of the grade type system: a multivector expression is
/// "typed" by the set of grades it may carry. Supports grades `0..=31`,
/// comfortably past the kernel's maximum `N = 16`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct GradeSet(u32);

impl GradeSet {
    /// The empty grade set — the type of the identically-zero multivector.
    pub const EMPTY: Self = GradeSet(0);

    /// The set containing exactly grade `k`.
    pub fn singleton(k: usize) -> Self {
        GradeSet(1 << k)
    }

    /// Every grade an `n`-generator algebra has: `0..=n`.
    pub fn full(n: usize) -> Self {
        GradeSet((1u32 << (n + 1)) - 1)
    }

    /// Is grade `k` in the set?
    pub fn contains(self, k: usize) -> bool {
        self.0 & (1 << k) != 0
    }

    /// The set with grade `k` added (`self` unchanged).
    pub fn with(self, k: usize) -> Self {
        GradeSet(self.0 | (1 << k))
    }

    /// Grades in either set.
    pub fn union(self, other: Self) -> Self {
        GradeSet(self.0 | other.0)
    }

    /// Grades in both sets.
    pub fn intersection(self, other: Self) -> Self {
        GradeSet(self.0 & other.0)
    }

    /// No grades present (the zero type).
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// How many distinct grades are present.
    pub fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    /// The grades present, in ascending order.
    pub fn iter(self) -> GradeIter {
        GradeIter(self.0)
    }
}

impl core::fmt::Debug for GradeSet {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{{")?;
        for (i, k) in self.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{k}")?;
        }
        write!(f, "}}")
    }
}

/// Iterator over the grades in a [`GradeSet`], ascending. Created by
/// [`GradeSet::iter`].
#[derive(Clone, Copy, Debug)]
pub struct GradeIter(u32);

impl Iterator for GradeIter {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            return None;
        }
        let k = self.0.trailing_zeros() as usize;
        self.0 &= self.0 - 1; // clear the lowest set bit
        Some(k)
    }
}

/// One operation in the geometric alphabet.
///
/// A value-level symbol for a kernel operation — the building block of an
/// AST or chromosome. `Copy + Eq + Hash`, so it is cheap to store, compare,
/// and use as a cache key. Apply it with [`Op::apply`], ask its shape with
/// [`Op::arity`] / [`Op::name`], and reason about its output with
/// [`Op::output_grades`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    // --- Binary (arity 2) ---
    /// Geometric product `a * b`.
    Geometric,
    /// Outer / wedge product `a ∧ b`.
    Wedge,
    /// Inner product `a · b` (Hestenes; scalar operands contribute nothing).
    Inner,
    /// Addition `a + b`.
    Add,
    /// Subtraction `a − b`.
    Sub,
    /// Regressive product `a ∨ b` — the *meet*.
    Regressive,

    // --- Unary (arity 1) ---
    /// Reverse `~a`.
    Reverse,
    /// Grade involution `â`.
    GradeInvolution,
    /// Clifford conjugation.
    Conjugate,
    /// Duality — the metric-independent right complement.
    Dual,
    /// Negation `−a`.
    Negate,
    /// Normalization `a / ‖a‖`.
    Normalize,
    /// Projection onto a single grade `⟨a⟩_k`.
    GradeProject(u8),

    // --- Nullary (arity 0) ---
    /// The unit basis blade with this index (index `0` is the scalar `1`).
    Basis(u16),
}

impl Op {
    /// The binary operations, as a fixed set for enumeration.
    pub const BINARY: [Op; 6] = [
        Op::Geometric,
        Op::Wedge,
        Op::Inner,
        Op::Add,
        Op::Sub,
        Op::Regressive,
    ];

    /// The non-parameterized unary operations, as a fixed set for
    /// enumeration. (The parameterized [`Op::GradeProject`] is enumerated
    /// separately by [`Op::grade_projections`].)
    pub const UNARY: [Op; 6] = [
        Op::Reverse,
        Op::GradeInvolution,
        Op::Conjugate,
        Op::Dual,
        Op::Negate,
        Op::Normalize,
    ];

    /// The grade-projection operations for an `n`-generator algebra,
    /// `⟨·⟩_0 ..= ⟨·⟩_n`.
    pub fn grade_projections(n: usize) -> impl Iterator<Item = Op> {
        (0..=n).map(|k| Op::GradeProject(k as u8))
    }

    /// The basis-blade terminals for a `dim`-blade algebra (`dim = 2^n`),
    /// `Basis(0) ..= Basis(dim − 1)`.
    pub fn basis_blades(dim: usize) -> impl Iterator<Item = Op> {
        (0..dim).map(|i| Op::Basis(i as u16))
    }

    /// How many operands the op consumes: 2 (binary), 1 (unary), or 0
    /// (a basis-blade terminal).
    pub fn arity(&self) -> usize {
        match self {
            Op::Geometric | Op::Wedge | Op::Inner | Op::Add | Op::Sub | Op::Regressive => 2,
            Op::Reverse
            | Op::GradeInvolution
            | Op::Conjugate
            | Op::Dual
            | Op::Negate
            | Op::Normalize
            | Op::GradeProject(_) => 1,
            Op::Basis(_) => 0,
        }
    }

    /// A short symbol for the op, for printing s-expressions. Parameterized
    /// ops give their base name (`"grade"`, `"basis"`); read the parameter
    /// off the variant.
    pub fn name(&self) -> &'static str {
        match self {
            Op::Geometric => "geo",
            Op::Wedge => "wedge",
            Op::Inner => "inner",
            Op::Add => "add",
            Op::Sub => "sub",
            Op::Regressive => "meet",
            Op::Reverse => "rev",
            Op::GradeInvolution => "ginv",
            Op::Conjugate => "conj",
            Op::Dual => "dual",
            Op::Negate => "neg",
            Op::Normalize => "norm",
            Op::GradeProject(_) => "grade",
            Op::Basis(_) => "basis",
        }
    }

    /// Evaluate the op against `args`, which must hold exactly
    /// [`arity`](Op::arity) operands (debug-asserted). A `Basis(i)` ignores
    /// `args` and returns the unit blade `i` (which must be in range).
    ///
    /// Bound on [`Real`] because [`Op::Normalize`] needs a square root; the
    /// chosen coefficient field for evolved geometric programs is real
    /// `f64`, so this is the natural bound.
    pub fn apply<A: Algebra, T: Real>(&self, args: &[Multivector<A, T>]) -> Multivector<A, T> {
        debug_assert_eq!(args.len(), self.arity(), "Op::apply: arity mismatch");
        match self {
            Op::Geometric => args[0] * args[1],
            Op::Wedge => args[0].wedge(&args[1]),
            Op::Inner => args[0].inner(&args[1]),
            Op::Add => args[0] + args[1],
            Op::Sub => args[0] - args[1],
            Op::Regressive => args[0].regressive(&args[1]),
            Op::Reverse => args[0].reverse(),
            Op::GradeInvolution => args[0].grade_involution(),
            Op::Conjugate => args[0].conjugate(),
            Op::Dual => args[0].right_complement(),
            Op::Negate => -args[0],
            Op::Normalize => args[0].normalized(),
            Op::GradeProject(k) => args[0].grade(*k as usize),
            Op::Basis(i) => Multivector::basis(*i as usize),
        }
    }

    /// The op's **grade signature**: given the grade sets its operands may
    /// carry, the set of grades the result may carry, in an `n`-generator
    /// algebra. The per-node rule of the dimensional type system — compose
    /// it over an AST to *infer* the grade of a whole expression.
    ///
    /// `inputs` must hold [`arity`](Op::arity) grade sets (ignored for a
    /// nullary `Basis`). The rules:
    ///
    /// - `Geometric`: `e_r e_s` spans `|r−s|, |r−s|+2, …` up to
    ///   `min(r+s, 2n−r−s)`.
    /// - `Wedge`: `{r+s}` when `r+s ≤ n`, else empty (the wedge vanishes).
    /// - `Inner`: `{|r−s|}` for `r, s ≥ 1`, else empty.
    /// - `Regressive`: `{r+s−n}` when `r+s ≥ n`, else empty.
    /// - `Add`/`Sub`: the union of the operand grades.
    /// - `Reverse`/`GradeInvolution`/`Conjugate`/`Negate`/`Normalize`:
    ///   grade-preserving — the operand's grades unchanged.
    /// - `Dual`: `r ↦ n − r`.
    /// - `GradeProject(k)`: the operand grades intersected with `{k}`.
    /// - `Basis(i)`: `{grade_of(i)}`.
    pub fn output_grades(&self, inputs: &[GradeSet], n: usize) -> GradeSet {
        debug_assert_eq!(
            inputs.len(),
            self.arity(),
            "Op::output_grades: arity mismatch"
        );
        match self {
            Op::Basis(i) => GradeSet::singleton(grade_of(*i as usize)),
            Op::Reverse | Op::GradeInvolution | Op::Conjugate | Op::Negate | Op::Normalize => {
                inputs[0]
            }
            Op::Dual => {
                let mut g = GradeSet::EMPTY;
                for r in inputs[0].iter() {
                    g = g.with(n - r);
                }
                g
            }
            Op::GradeProject(k) => inputs[0].intersection(GradeSet::singleton(*k as usize)),
            Op::Add | Op::Sub => inputs[0].union(inputs[1]),
            Op::Geometric => {
                let mut g = GradeSet::EMPTY;
                for r in inputs[0].iter() {
                    for s in inputs[1].iter() {
                        g = g.union(geometric_pair(r, s, n));
                    }
                }
                g
            }
            Op::Wedge => {
                let mut g = GradeSet::EMPTY;
                for r in inputs[0].iter() {
                    for s in inputs[1].iter() {
                        if r + s <= n {
                            g = g.with(r + s);
                        }
                    }
                }
                g
            }
            Op::Inner => {
                let mut g = GradeSet::EMPTY;
                for r in inputs[0].iter() {
                    for s in inputs[1].iter() {
                        if r >= 1 && s >= 1 {
                            g = g.with(r.abs_diff(s));
                        }
                    }
                }
                g
            }
            Op::Regressive => {
                let mut g = GradeSet::EMPTY;
                for r in inputs[0].iter() {
                    for s in inputs[1].iter() {
                        if r + s >= n {
                            g = g.with(r + s - n);
                        }
                    }
                }
                g
            }
        }
    }
}

/// Grades a single-blade geometric product `e_r e_s` can land in: every
/// `r + s − 2t` for `t` from `max(0, r+s−n)` to `min(r, s)` — i.e.
/// `|r−s|, |r−s|+2, …, min(r+s, 2n−r−s)`.
fn geometric_pair(r: usize, s: usize, n: usize) -> GradeSet {
    let lo = (r + s).saturating_sub(n);
    let hi = if r < s { r } else { s };
    let mut g = GradeSet::EMPTY;
    let mut t = lo;
    while t <= hi {
        g = g.with(r + s - 2 * t);
        t += 1;
    }
    g
}

#[cfg(test)]
mod tests {
    use super::{GradeSet, Op};
    use crate::{Pga3, Vga3};

    // --- GradeSet ----------------------------------------------------------

    #[test]
    fn grade_set_basics() {
        let g = GradeSet::EMPTY.with(0).with(2).with(2);
        assert!(g.contains(0) && g.contains(2) && !g.contains(1));
        assert_eq!(g.len(), 2);
        assert!(!g.is_empty() && GradeSet::EMPTY.is_empty());
        // iter yields the grades ascending, once each.
        let mut it = g.iter();
        assert_eq!(it.next(), Some(0));
        assert_eq!(it.next(), Some(2));
        assert_eq!(it.next(), None);
        assert_eq!(
            GradeSet::full(3),
            GradeSet::EMPTY.with(0).with(1).with(2).with(3)
        );
        assert_eq!(
            GradeSet::singleton(1)
                .union(GradeSet::singleton(2))
                .intersection(GradeSet::singleton(2)),
            GradeSet::singleton(2)
        );
    }

    // --- Structure ---------------------------------------------------------

    #[test]
    fn arities_and_enumeration() {
        assert!(Op::BINARY.iter().all(|o| o.arity() == 2));
        assert!(Op::UNARY.iter().all(|o| o.arity() == 1));
        assert_eq!(Op::GradeProject(2).arity(), 1);
        assert_eq!(Op::Basis(3).arity(), 0);
        assert_eq!(Op::grade_projections(3).count(), 4); // grades 0..=3
        assert_eq!(Op::basis_blades(8).count(), 8);
        assert_eq!(Op::Geometric.name(), "geo");
        assert_eq!(Op::Regressive.name(), "meet");
    }

    // --- apply -------------------------------------------------------------

    #[test]
    fn apply_evaluates_each_op() {
        let e1 = Vga3::basis(1);
        let e2 = Vga3::basis(2);
        // geometric: e1 e1 = 1
        assert_eq!(Op::Geometric.apply(&[e1, e1]).scalar_part(), 1.0);
        // wedge: e1 ∧ e2 = e12
        assert_eq!(Op::Wedge.apply(&[e1, e2]), e1 * e2);
        // inner: e1 · e2 = 0 (orthogonal)
        assert_eq!(Op::Inner.apply(&[e1, e2]), Vga3::zero());
        // add / sub / negate
        assert_eq!(Op::Add.apply(&[e1, e2]), e1 + e2);
        assert_eq!(Op::Sub.apply(&[e1, e2]), e1 - e2);
        assert_eq!(Op::Negate.apply(&[e1]), -e1);
        // reverse of a bivector flips sign
        let e12 = e1 * e2;
        assert_eq!(Op::Reverse.apply(&[e12]), -e12);
        // grade projection picks one grade
        assert_eq!(Op::GradeProject(1).apply(&[e1 + e12]), e1);
        // basis terminal builds a blade, ignoring args
        assert_eq!(Op::Basis(3).apply::<_, f64>(&[]), e12);
        // normalize makes a unit
        let n = Op::Normalize.apply(&[e1 * 5.0]);
        assert_eq!(n, e1);
    }

    #[test]
    fn apply_meet_of_points_is_a_line() {
        // PGA: two points (grade 3) meet (regressive) in the line through
        // them (grade 2). Exercises the alphabet on the target algebra.
        let p = Pga3::point(0.0, 0.0, 0.0);
        let q = Pga3::point(1.0, 0.0, 0.0);
        let line = Op::Regressive.apply(&[p, q]);
        assert_eq!(line, p.regressive(&q));
        assert_eq!(line.grade(2), line); // purely grade 2
    }

    // --- Grade signatures --------------------------------------------------

    #[test]
    fn geometric_signature_of_two_vectors() {
        let v = GradeSet::singleton(1);
        // e_r e_s for vectors in 3D: scalar + bivector.
        assert_eq!(
            Op::Geometric.output_grades(&[v, v], 3),
            GradeSet::EMPTY.with(0).with(2)
        );
    }

    #[test]
    fn wedge_signature_caps_and_vanishes() {
        let v = GradeSet::singleton(1);
        let bi = GradeSet::singleton(2);
        assert_eq!(Op::Wedge.output_grades(&[v, v], 3), GradeSet::singleton(2));
        // grade 2 ∧ grade 2 = grade 4 > 3 ⇒ vanishes
        assert!(Op::Wedge.output_grades(&[bi, bi], 3).is_empty());
    }

    #[test]
    fn inner_signature_drops_scalar_operands() {
        let scal = GradeSet::singleton(0);
        let v = GradeSet::singleton(1);
        let bi = GradeSet::singleton(2);
        assert_eq!(Op::Inner.output_grades(&[v, bi], 3), GradeSet::singleton(1));
        assert!(Op::Inner.output_grades(&[scal, bi], 3).is_empty());
    }

    #[test]
    fn regressive_and_dual_signatures() {
        // Meet of two PGA points (grade 3) in n=4 ⇒ grade 2 (a line).
        let pt = GradeSet::singleton(3);
        assert_eq!(
            Op::Regressive.output_grades(&[pt, pt], 4),
            GradeSet::singleton(2)
        );
        // Dual maps grade r ↦ n − r.
        assert_eq!(
            Op::Dual.output_grades(&[GradeSet::singleton(1)], 3),
            GradeSet::singleton(2)
        );
    }

    #[test]
    fn grade_preserving_and_projection_signatures() {
        let mixed = GradeSet::EMPTY.with(0).with(2);
        assert_eq!(Op::Reverse.output_grades(&[mixed], 3), mixed);
        assert_eq!(
            Op::Add.output_grades(&[GradeSet::singleton(0), GradeSet::singleton(2)], 3),
            mixed
        );
        assert_eq!(
            Op::GradeProject(2).output_grades(&[mixed], 3),
            GradeSet::singleton(2)
        );
    }
}
