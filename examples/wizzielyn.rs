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
    println!("R · e1 · ~R           = {}   (rotate e1 ⇒ e2)", r90.sandwich(&Vga2::basis(1)));
    println!("R · e2 · ~R           = {}   (rotate e2 ⇒ -e1)", r90.sandwich(&Vga2::basis(2)));

    // R⁻¹ · R = 1
    let r_inv = r90.versor_inverse();
    println!("R · R⁻¹               = {}", r90 * r_inv);

    println!();
    println!("== Rotors in 3D — the axis is fixed ==");
    // 90° rotation in the e23 plane (i.e. about the e1 axis).
    let e23 = Vga3::basis(6); // 0b110
    let r3d = (e23 * (-FRAC_PI_2 / 2.0)).exp();
    println!("R = exp(-π/4 · e23)  = {r3d}");
    println!("R · e1 · ~R           = {}   (axis ⇒ unchanged)", r3d.sandwich(&Vga3::basis(1)));
    println!("R · e2 · ~R           = {}   (in plane ⇒ rotates to e3)", r3d.sandwich(&Vga3::basis(2)));
    println!("R · e3 · ~R           = {}   (in plane ⇒ rotates to -e2)", r3d.sandwich(&Vga3::basis(4)));

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
    println!("then n2·(...)·~n2     = {after_both}   (net: e2 ⇒ -e1, a 90° rotation)");

    println!();
    println!("== Bonus: exp on a vector in Cl(1,0,0) — hyperbolic Euler ==");
    // In a P-group-only 1D algebra, e1² = +1 makes exp behave with
    // cosh/sinh rather than cos/sin.
    use garust::Multivector;
    type Cl10 = Multivector<1, 0, 0, 2>;
    let v = Cl10 { coeffs: [0.0, 0.5] };  // 0.5 · e1
    println!("exp(0.5 e1) in Cl(1,0,0) = {}   (= cosh(0.5) + sinh(0.5)·e1)", v.exp());
}
