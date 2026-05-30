//! wizzielyn — round 4 brings the algebra to its punch-line uses:
//! reflections, rotations, and the bivector-to-rotor bridge.

use garust::{Vga2, Vga3};
use std::f64::consts::FRAC_PI_2;

fn main() {
    println!("== Rotors in 2D ==");
    // Build the unit rotor for a 90° rotation in the e12 plane:
    //   R = exp(-θ/2 · e12) = cos(θ/2) - sin(θ/2) e12
    let e12 = Vga2::basis(3);
    let r90 = (e12 * (-FRAC_PI_2 / 2.0)).exp();
    println!("R = exp(-π/4 · e12)  = {r90}");
    println!("|R|²                  = {}   (unit norm by construction)", r90.norm_squared());
    println!("R · e1 · ~R           = {}   (rotate e1 ⇒ e2)", r90.sandwich(&Vga2::basis(1)).cleaned(1e-10));
    println!("R · e2 · ~R           = {}   (rotate e2 ⇒ -e1)", r90.sandwich(&Vga2::basis(2)).cleaned(1e-10));

    // R⁻¹ · R = 1
    let r_inv = r90.versor_inverse();
    println!("R · R⁻¹               = {}", (r90 * r_inv).cleaned(1e-10));

    println!();
    println!("== Rotors in 3D — the axis is fixed ==");
    // 90° rotation in the e23 plane (i.e. about the e1 axis).
    let e23 = Vga3::basis(6); // 0b110
    let r3d = (e23 * (-FRAC_PI_2 / 2.0)).exp();
    println!("R = exp(-π/4 · e23)  = {r3d}");
    println!("R · e1 · ~R           = {}   (axis ⇒ unchanged)", r3d.sandwich(&Vga3::basis(1)).cleaned(1e-10));
    println!("R · e2 · ~R           = {}   (in plane ⇒ rotates to e3)", r3d.sandwich(&Vga3::basis(2)).cleaned(1e-10));
    println!("R · e3 · ~R           = {}   (in plane ⇒ rotates to -e2)", r3d.sandwich(&Vga3::basis(4)).cleaned(1e-10));

    println!();
    println!("== Reflection: sandwich with a unit vector ==");
    // n.sandwich(v) preserves the n-component and flips the perpendicular.
    let n = Vga3::basis(1);                          // unit x
    let v = Vga3::basis(1) + Vga3::basis(2);          // e1 + e2
    println!("n = e1,  v = e1 + e2");
    println!("n · v · ~n            = {}   (perpendicular e2 component flipped)", n.sandwich(&v));

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
    println!("then n2·(...)·~n2     = {}   (net: e2 ⇒ -e1, a 90° rotation)", after_both.cleaned(1e-10));

    println!();
    println!("== Bonus: exp on a vector in Cl(1,0,0) — hyperbolic Euler ==");
    // In a P-group-only 1D algebra, e1² = +1 makes exp behave with
    // cosh/sinh rather than cos/sin.
    use garust::Multivector;
    type Cl10 = Multivector<f64, 1, 0, 0, 2>;
    let v = Cl10 { coeffs: [0.0, 0.5] };  // 0.5 · e1
    println!("exp(0.5 e1) in Cl(1,0,0) = {}   (= cosh(0.5) + sinh(0.5)·e1)", v.exp());

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
    println!("(e0·e1)²              = {}   (null bivector!)", (b * b).cleaned(1e-10));

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
    println!("|M|²                  = {}   (still unit-norm — motors are versors)", motor.norm_squared());

    // Apply the motor to a point at (0, 1, 0). The rotation should
    // send (0, 1, 0) → (0, 0, 1), then translation by 3 in e1 gives
    // (3, 0, 1). In PGA point form: e123 − 3·e0e2e3 + 0·e0e1e3 − 1·e0e1e2
    // which Display shows as  e123 − e124 − 3·e234.
    let p010 = e1 * e2 * e3 + e0 * e1 * e3;
    let moved = motor.sandwich(&p010).cleaned(1e-10);
    println!("M · point(0,1,0) · ~M = {moved}");
    println!("                        (rotate (0,1,0) → (0,0,1), translate → (3,0,1))");
}
