//! wizzielyn — round 4 brings the algebra to its punch-line uses:
//! reflections, rotations, and the bivector-to-rotor bridge.
//! Round 5 adds duality: the pseudoscalar, the complements, and the
//! regressive product that *meets* subspaces where the wedge *joins*.
//! Round 6 cashes that out as PGA geometry: points, planes, and the
//! lines that meet and join them.
//! Round 7 wraps rigid motions in a Motor type — rotors and translators
//! composed with `*` and applied with one call.

use garust::{Vga2, Vga3};
use std::f64::consts::FRAC_PI_2;

fn main() {
    println!("== Rotors in 2D ==");
    // Build the unit rotor for a 90° rotation in the e12 plane:
    //   R = exp(-θ/2 · e12) = cos(θ/2) - sin(θ/2) e12
    let e12 = Vga2::basis(3);
    let r90 = (e12 * (-FRAC_PI_2 / 2.0)).exp();
    println!("R = exp(-π/4 · e12)  = {r90}");
    println!(
        "|R|²                  = {}   (unit norm by construction)",
        r90.norm_squared()
    );
    println!(
        "R · e1 · ~R           = {}   (rotate e1 ⇒ e2)",
        r90.sandwich(&Vga2::basis(1)).cleaned(1e-10)
    );
    println!(
        "R · e2 · ~R           = {}   (rotate e2 ⇒ -e1)",
        r90.sandwich(&Vga2::basis(2)).cleaned(1e-10)
    );

    // R⁻¹ · R = 1
    let r_inv = r90.versor_inverse();
    println!("R · R⁻¹               = {}", (r90 * r_inv).cleaned(1e-10));

    println!();
    println!("== Rotors in 3D — the axis is fixed ==");
    // 90° rotation in the e23 plane (i.e. about the e1 axis).
    let e23 = Vga3::basis(6); // 0b110
    let r3d = (e23 * (-FRAC_PI_2 / 2.0)).exp();
    println!("R = exp(-π/4 · e23)  = {r3d}");
    println!(
        "R · e1 · ~R           = {}   (axis ⇒ unchanged)",
        r3d.sandwich(&Vga3::basis(1)).cleaned(1e-10)
    );
    println!(
        "R · e2 · ~R           = {}   (in plane ⇒ rotates to e3)",
        r3d.sandwich(&Vga3::basis(2)).cleaned(1e-10)
    );
    println!(
        "R · e3 · ~R           = {}   (in plane ⇒ rotates to -e2)",
        r3d.sandwich(&Vga3::basis(4)).cleaned(1e-10)
    );

    println!();
    println!("== Reflection: sandwich with a unit vector ==");
    // n.sandwich(v) preserves the n-component and flips the perpendicular.
    let n = Vga3::basis(1); // unit x
    let v = Vga3::basis(1) + Vga3::basis(2); // e1 + e2
    println!("n = e1,  v = e1 + e2");
    println!(
        "n · v · ~n            = {}   (perpendicular e2 component flipped)",
        n.sandwich(&v)
    );

    println!();
    println!("== Two reflections compose into a rotation ==");
    // Mirror lines at 0° and 45° in the e12 plane:
    //   n1 = e1,  n2 = (e1+e2)/√2.
    // Reflecting through each in turn rotates by 2·45° = 90°.
    let n1 = Vga2::basis(1);
    let n2 = (Vga2::basis(1) + Vga2::basis(2)) * (1.0 / 2.0_f64.sqrt());
    let v = Vga2::basis(2);
    let after_first = n1.sandwich(&v);
    let after_both = n2.sandwich(&after_first);
    println!("v = e2");
    println!("after n1·v·~n1        = {after_first}   (e2 ⊥ x-axis ⇒ flips)");
    println!(
        "then n2·(...)·~n2     = {}   (net: e2 ⇒ -e1, a 90° rotation)",
        after_both.cleaned(1e-10)
    );

    println!();
    println!("== Bonus: exp on a vector in Cl(1,0,0) — hyperbolic Euler ==");
    // In a P-group-only 1D algebra, e1² = +1 makes exp behave with
    // cosh/sinh rather than cos/sin.
    use garust::Multivector;
    type Cl10 = Multivector<f64, 1, 0, 0, 2>;
    let v = Cl10 { coeffs: [0.0, 0.5] }; // 0.5 · e1
    println!(
        "exp(0.5 e1) in Cl(1,0,0) = {}   (= cosh(0.5) + sinh(0.5)·e1)",
        v.exp()
    );

    println!();
    println!("== PGA Cl(3,0,1): translation falls out of the same exp ==");
    // In Pga3 our bit-mask convention puts the null generator at bit 3.
    // Conventionally that's called e0 (PGA literature) — bit-index e4
    // when written by Display.
    use garust::Pga3;
    let e0 = Pga3::basis(8); // the null generator (R-group)
    let e1 = Pga3::basis(1);
    let e2 = Pga3::basis(2);
    let e3 = Pga3::basis(4);

    // The null bivector e0·e1 squares to zero — that's the whole trick.
    let b = e0 * e1;
    println!("e0·e1                 = {b}");
    println!(
        "(e0·e1)²              = {}   (null bivector!)",
        (b * b).cleaned(1e-10)
    );

    // Build a translator for displacement d = 3 in the e1 direction.
    // Because B² = 0 the exp series collapses to T = 1 + (-d/2)·B.
    let d = 3.0;
    let t = ((e0 * e1) * (-d / 2.0)).exp();
    println!("T = exp(-3/2 · e0·e1) = {t}");

    // Modern PGA point convention: a point at (x, y, z) is the trivector
    //   P = e1·e2·e3 − x·e0·e2·e3 + y·e0·e1·e3 − z·e0·e1·e2
    let origin = e1 * e2 * e3;
    let translated = t.sandwich(&origin).cleaned(1e-10);
    println!("T · (e1·e2·e3) · ~T   = {translated}");
    println!("                        (origin → point at (3, 0, 0):");
    println!("                         note the new e2e3e0 trivector with coefficient 3)");

    // Compose a rotor with a translator → a motor (rigid-body transform).
    // R rotates 90° about the e1 axis (in the e2-e3 plane).
    let r = ((e2 * e3) * (-FRAC_PI_2 / 2.0)).exp();
    let motor = t * r;
    println!();
    println!("Motor M = T · R       = {}", motor.cleaned(1e-10));
    println!(
        "|M|²                  = {}   (still unit-norm — motors are versors)",
        motor.norm_squared()
    );

    // Apply the motor to a point at (0, 1, 0). The rotation should
    // send (0, 1, 0) → (0, 0, 1), then translation by 3 in e1 gives
    // (3, 0, 1). In PGA point form: e123 − 3·e0e2e3 + 0·e0e1e3 − 1·e0e1e2
    // which Display shows as  e123 − e124 − 3·e234.
    let p010 = e1 * e2 * e3 + e0 * e1 * e3;
    let moved = motor.sandwich(&p010).cleaned(1e-10);
    println!("M · point(0,1,0) · ~M = {moved}");
    println!("                        (rotate (0,1,0) → (0,0,1), translate → (3,0,1))");

    println!();
    println!("== Round 5: duality — join with ∧, meet with ∨ ==");
    // The pseudoscalar I is the top blade and the unit of the meet.
    println!("I = pseudoscalar      = {}", Vga3::pseudoscalar());

    // Wedge JOINS: two distinct points/directions span the plane between.
    let ex = Vga3::basis(1);
    let ey = Vga3::basis(2);
    println!(
        "e1 ∧ e2  (join)       = {}   (the xy-plane bivector)",
        ex.wedge(&ey)
    );

    // The complement is the metric-independent dual: rc(e1) = e23, the
    // plane perpendicular to e1.
    println!(
        "rc(e1)                = {}   (plane ⟂ e1, no metric needed)",
        ex.right_complement()
    );

    // Regressive product MEETS: the xy-plane and xz-plane intersect in
    // the x-axis.
    let xy = Vga3::basis(3); // e12
    let xz = Vga3::basis(5); // e13
    println!(
        "e12 ∨ e13  (meet)     = {}   (xy ∩ xz = the x-axis)",
        xy.regressive(&xz)
    );
    println!(
        "e12 ∨ e23  (meet)     = {}   (xy ∩ yz = the y-axis)",
        xy.regressive(&Vga3::basis(6))
    );

    // And the meet works unchanged in degenerate PGA, where the
    // pseudoscalar is null and a metric dual would not exist.
    let ip = Pga3::pseudoscalar();
    println!(
        "PGA I² (null!)        = {}   (so the meet can't use a metric dual)",
        (ip * ip).cleaned(1e-10)
    );

    println!();
    println!("== Round 6: PGA geometry — points, planes, lines ==");
    // A point (grade-3 trivector) and a plane (grade-1 vector).
    println!("point(1,2,3)          = {}", Pga3::point(1.0, 2.0, 3.0));
    println!(
        "plane x=1  (x-1=0)    = {}",
        Pga3::plane(1.0, 0.0, 0.0, -1.0)
    );

    // Three planes MEET (wedge) at their common point.
    let px = Pga3::plane(1.0, 0.0, 0.0, -1.0); // x = 1
    let py = Pga3::plane(0.0, 1.0, 0.0, -2.0); // y = 2
    let pz = Pga3::plane(0.0, 0.0, 1.0, -3.0); // z = 3
    let meet = px.wedge(&py).wedge(&pz);
    println!("(x=1) ∧ (y=2) ∧ (z=3) = {meet}");
    println!("                        (= point(1,2,3), the three planes' intersection)");

    // Two points JOIN (regressive) into the line through them.
    let a = Pga3::point(0.0, 0.0, 0.0);
    let b = Pga3::point(1.0, 0.0, 0.0);
    let line = a.line_through(&b);
    println!("point(0,0,0) ∨ (1,0,0)= {line}   (the x-axis line)");

    // Collinearity is an incidence test: a third point on that line
    // makes the triple join vanish.
    let c = Pga3::point(2.0, 0.0, 0.0);
    let off = Pga3::point(0.0, 1.0, 0.0);
    println!(
        "join of 3 collinear   = {}   (all on the x-axis ⇒ 0)",
        line.regressive(&c).cleaned(1e-10)
    );
    println!(
        "join of 3 non-collinear = {}",
        line.regressive(&off).cleaned(1e-10)
    );

    println!();
    println!("== Round 7: Motors — rigid motions with a clean API ==");
    use garust::Motor3;
    // Rotate 90° about the x-axis (e23 plane), then translate +3 in x.
    let r = Motor3::rotor(FRAC_PI_2, Pga3::basis(0b0110));
    let t = Motor3::translator(3.0, 0.0, 0.0);
    let m = t * r; // compose: `*` applies r first, then t
    println!(
        "|M|²                  = {}   (a motor is a unit versor)",
        m.norm_squared()
    );

    // Apply the screw motion to the point (0, 1, 0).
    let moved = m.apply(&Pga3::point(0.0, 1.0, 0.0)).cleaned(1e-10);
    println!("M · point(0,1,0)      = {moved}");
    println!("                        (rotate (0,1,0)→(0,0,1), translate → (3,0,1))");

    // The inverse undoes the motion exactly.
    let back = m.inverse().apply(&moved).cleaned(1e-10);
    println!("M⁻¹ · (that)          = {back}   (back to point(0,1,0))");

    println!();
    println!("== Round 8: Conformal transforms — CGA adds scaling ==");
    use garust::{Cga3, Conformal3};
    // Translators and rotors behave like a motor, but CGA also has a
    // *dilator* — a uniform scaling about the origin — which no rigid
    // motion can express.
    let scale = Conformal3::dilator(2.0);
    let shift = Conformal3::translator(1.0, 0.0, 0.0);

    // Order matters: scale-then-shift ≠ shift-then-scale.
    let p = Cga3::cga_point(1.0, 0.0, 0.0);
    let st = (shift * scale).apply(&p); // ×2 then +1  ⇒ (3,0,0)
    let ts = (scale * shift).apply(&p); // +1 then ×2  ⇒ (4,0,0)
    println!("scale-then-shift (1,0,0) = {:?}", st.to_euclidean());
    println!("shift-then-scale (1,0,0) = {:?}", ts.to_euclidean());

    // A dilation grows a sphere: the unit sphere at the origin becomes
    // radius 2, so the point (2,0,0) now lies on it.
    let unit_sphere = Cga3::sphere(0.0, 0.0, 0.0, 1.0);
    let grown = scale.apply(&unit_sphere);
    let on = Cga3::cga_point(2.0, 0.0, 0.0).scalar_product(&grown);
    println!("(2,0,0) · scaled sphere  = {on:.3}   (≈0 ⇒ on the radius-2 sphere)");
}
