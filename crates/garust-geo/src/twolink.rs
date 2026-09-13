//! Closed-form inverse kinematics for a two-link limb.
//!
//! Given both ends and the two segment lengths, where is the joint in the
//! middle? It is the intersection of two spheres, which in general is a
//! **circle**: an elbow can swing all the way around the shoulder–hand
//! axis. Picking a point on that circle needs one more input, and this is
//! the whole reason the module exists as its own thing.
//!
//! # The branch must be given, never inferred
//!
//! A published reel once drew a rower's elbow bent the wrong way. The
//! cause was a rule that sounded reasonable — *take the higher solution* —
//! and is undecidable exactly where a row lives: with the hand hanging
//! straight below the shoulder, the two solutions are mirror images at the
//! **same height**, so the one that came out was whichever way floating
//! point happened to round. Nothing looked wrong in the numbers.
//!
//! So [`two_link_joint`] takes a `hint` direction and the answer is on
//! that side, always, including in the degenerate case where every
//! implicit rule is a coin flip. That is a claim below, not a comment.
//!
//! # Precision near full extension
//!
//! The joint's offset from the axis is `h = √(l1² − a²)`, and at full
//! extension that is the square root of a difference going to zero: the
//! usual catastrophic cancellation, so `h` carries roughly half the
//! significant digits there. It is a property of the problem and not of
//! this implementation — the same ill-conditioning that inflated a knee
//! figure once — and the honest consequence is that **a straight limb's
//! bend direction is not meaningful**. The position still is: `h` is
//! near zero, so the joint lands on the axis either way.
//!
//! [`Chain::ik_dls`](crate::chain::Chain::ik_dls) solves the general
//! problem iteratively. This is the closed form for the case a limb
//! actually is: exact, allocation-free, no seed, no convergence to fail.

/// What a two-link solve can refuse to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwoLinkError {
    /// A segment length was not positive.
    BadLength,
    /// The hint is parallel to the root→tip axis, so it names no side of
    /// the solution circle. Undecidable by construction, and reported
    /// rather than guessed: guessing is the defect this module exists for.
    HintAlongAxis,
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// The middle joint of a two-link limb, on the side `hint` points to.
///
/// `root` is the proximal end (shoulder, hip), `tip` the distal one
/// (hand, foot). `hint` is a direction in world frame; the joint lands on
/// the side of the root→tip axis that `hint` points toward. Only its
/// component perpendicular to that axis is used, so it does not need to be
/// perpendicular or normalized — "backwards" or "downward" is enough.
///
/// **Unreachable targets are clamped, not refused.** When the ends are
/// farther apart than `l1 + l2` the limb comes out straight and pointing
/// at the tip, which is what a real limb does at full extension, and the
/// caller gets a usable pose instead of a hole. Closer than `|l1 − l2|`
/// clamps the same way. If reach matters to the caller it is one
/// subtraction to check, and hiding it here would take the choice away.
///
/// # Errors
///
/// [`TwoLinkError::BadLength`] for non-positive lengths, and
/// [`TwoLinkError::HintAlongAxis`] when the hint names no side.
pub fn two_link_joint(
    root: [f64; 3],
    tip: [f64; 3],
    l1: f64,
    l2: f64,
    hint: [f64; 3],
) -> Result<[f64; 3], TwoLinkError> {
    // `is_sign_negative` no sirve aquí: hay que rechazar también el cero
    // y el NaN, y un NaN pasa cualquier comparación directa.
    if !(l1.is_finite() && l2.is_finite()) || l1 <= 0.0 || l2 <= 0.0 {
        return Err(TwoLinkError::BadLength);
    }
    let axis = sub(tip, root);
    let d_raw = norm(axis);
    // Ends on top of each other: no axis at all. The hint alone decides
    // the direction, and the joint sits a link away along it.
    let u = if d_raw > f64::EPSILON {
        [axis[0] / d_raw, axis[1] / d_raw, axis[2] / d_raw]
    } else {
        let hn = norm(hint);
        if hn <= f64::EPSILON {
            return Err(TwoLinkError::HintAlongAxis);
        }
        [hint[0] / hn, hint[1] / hn, hint[2] / hn]
    };
    let d = d_raw.clamp((l1 - l2).abs(), l1 + l2);

    // Perpendicular component of the hint: the side of the circle.
    let along = dot(hint, u);
    let w = [
        hint[0] - along * u[0],
        hint[1] - along * u[1],
        hint[2] - along * u[2],
    ];
    let wn = norm(w);
    if wn <= 1e-12 * norm(hint).max(1.0) {
        return Err(TwoLinkError::HintAlongAxis);
    }
    let w = [w[0] / wn, w[1] / wn, w[2] / wn];

    let a = (d * d + l1 * l1 - l2 * l2) / (2.0 * d);
    let h = (l1 * l1 - a * a).max(0.0).sqrt();
    Ok([
        root[0] + a * u[0] + h * w[0],
        root[1] + a * u[1] + h * w[1],
        root[2] + a * u[2] + h * w[2],
    ])
}

#[cfg(test)]
mod tests {
    use super::{two_link_joint, TwoLinkError};

    fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    /// The segments come out the length they were given.
    #[test]
    fn the_links_keep_their_lengths() {
        let root = [0.0, 1.2, 0.0];
        let tip = [0.35, 0.7, 0.0];
        let (l1, l2) = (0.34, 0.31);
        let j = two_link_joint(root, tip, l1, l2, [-1.0, 0.0, 0.0]).unwrap();
        assert!((dist(root, j) - l1).abs() < 1e-12, "el primer segmento");
        assert!((dist(j, tip) - l2).abs() < 1e-12, "el segundo");
    }

    /// **The defect, as a claim.** Hand straight below the shoulder: the
    /// two solutions are mirror images at the same height, so "take the
    /// higher one" is a coin flip. The hint decides, and it decides both
    /// ways.
    #[test]
    fn the_degenerate_case_obeys_the_hint() {
        let root = [0.0, 1.2, 0.0];
        // Exactamente debajo, y con el brazo DOBLADO: a 0.65 el brazo
        // queda recto (0.34 + 0.31) y entonces no hay dos ramas que
        // elegir. La primera versión de este test cayó justo ahí y el
        // fallo fue el que lo enseñó.
        let tip = [0.0, 0.70, 0.0];
        let (l1, l2) = (0.34, 0.31);
        let atras = two_link_joint(root, tip, l1, l2, [-1.0, 0.0, 0.0]).unwrap();
        let frente = two_link_joint(root, tip, l1, l2, [1.0, 0.0, 0.0]).unwrap();
        assert!(atras[0] < -1e-3, "el codo va atrás, quedó en x={}", atras[0]);
        assert!(frente[0] > 1e-3, "y adelante cuando se pide, x={}", frente[0]);
        // y son espejo: la misma altura, que es justo por lo que la regla
        // implícita no podía decidir
        assert!((atras[1] - frente[1]).abs() < 1e-12);
    }

    /// A hint that names no side is refused, not guessed.
    #[test]
    fn a_hint_along_the_axis_is_refused() {
        let e = two_link_joint([0.0, 1.0, 0.0], [0.0, 0.4, 0.0], 0.34, 0.31, [0.0, -1.0, 0.0]);
        assert_eq!(e, Err(TwoLinkError::HintAlongAxis));
    }

    /// Out of reach: the limb straightens instead of returning NaN.
    #[test]
    fn out_of_reach_straightens() {
        let root = [0.0, 0.0, 0.0];
        let tip = [10.0, 0.0, 0.0];
        let (l1, l2) = (0.34, 0.31);
        let j = two_link_joint(root, tip, l1, l2, [0.0, 1.0, 0.0]).unwrap();
        assert!(j.iter().all(|c| c.is_finite()), "nada de NaN");
        assert!((dist(root, j) - l1).abs() < 1e-12);
        // Recto y apuntando al objetivo. La tolerancia es 1e-6 y no 1e-12
        // a propósito: `h` es la raíz de una diferencia que aquí vale
        // cero, así que el desplazamiento fuera del eje trae ruido de
        // media precisión. Exigir 1e-12 sería exigirle a la aritmética
        // algo que no puede dar.
        assert!(j[1].abs() < 1e-6 && (j[0] - l1).abs() < 1e-6, "{j:?}");
    }

    /// Folded past the inner limit clamps the same way.
    #[test]
    fn too_close_also_clamps() {
        let j = two_link_joint([0.0; 3], [0.001, 0.0, 0.0], 0.34, 0.10, [0.0, 1.0, 0.0]).unwrap();
        assert!(j.iter().all(|c| c.is_finite()));
    }

    /// Non-positive lengths are an error, not a silent zero.
    #[test]
    fn bad_lengths_are_refused() {
        let e = two_link_joint([0.0; 3], [0.5, 0.0, 0.0], 0.0, 0.3, [0.0, 1.0, 0.0]);
        assert_eq!(e, Err(TwoLinkError::BadLength));
    }

    /// The hint need not be perpendicular: only its side matters.
    #[test]
    fn only_the_side_of_the_hint_matters() {
        let root = [0.0, 1.2, 0.0];
        let tip = [0.3, 0.8, 0.0];
        let a = two_link_joint(root, tip, 0.34, 0.31, [-1.0, 0.0, 0.0]).unwrap();
        let b = two_link_joint(root, tip, 0.34, 0.31, [-3.0, -7.0, 0.0]).unwrap();
        // la misma rama: el sesgo a lo largo del eje no la cambia
        let c = two_link_joint(root, tip, 0.34, 0.31, [-1.0, 0.0, 0.0]).unwrap();
        assert!((a[0] - c[0]).abs() < 1e-12);
        let lado_a = a[0] * 0.4 - a[1] * 0.3;
        let lado_b = b[0] * 0.4 - b[1] * 0.3;
        assert_eq!(lado_a.is_sign_negative(), lado_b.is_sign_negative());
    }
}
