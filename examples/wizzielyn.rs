//! wizzielyn — the running learning playground for `garust`.
//!
//! Each round of garust adds a new section here. Today: the derived
//! products (wedge, inner, scalar), the three involutions (reverse,
//! grade involution, Clifford conjugation), and the squared norm.

use garust::{Vga2, Vga3};

fn main() {
    println!("== Cl(2,0,0): basic geometric product (recap) ==");
    let e1 = Vga2::basis(1);
    let e2 = Vga2::basis(2);
    let e12 = Vga2::basis(3);
    println!("e1 * e2       = {}", e1 * e2);
    println!("e1 * e1       = {}", e1 * e1);
    println!("e12 * e12     = {}", e12 * e12);

    println!();
    println!("== Wedge, inner, and the iconic identity ==");
    // Two non-parallel, non-perpendicular vectors in 3D Euclidean.
    let a = Vga3 { coeffs: [0.0, 2.0, 3.0, 0.0, -1.0, 0.0, 0.0, 0.0] };
    let b = Vga3 { coeffs: [0.0, -1.0, 4.0, 0.0, 2.0, 0.0, 0.0, 0.0] };
    println!("a              = {a}");
    println!("b              = {b}");
    println!("a ∧ b          = {}   (oriented plane spanned by a, b)", a.wedge(&b));
    println!("a · b          = {}   (Euclidean dot product of a, b)", a.inner(&b));
    println!("a * b          = {}", a * b);
    println!("a·b + a∧b      = {}", a.inner(&b) + a.wedge(&b));
    let identity_holds = (a * b).coeffs == (a.inner(&b) + a.wedge(&b)).coeffs;
    println!("identity holds: {identity_holds}");

    println!();
    println!("== Grade projection: dissect a mixed multivector ==");
    let m = Vga3 { coeffs: [1.0, 2.0, 3.0, 7.0, 5.0, 11.0, 13.0, 17.0] };
    println!("m              = {m}");
    println!("⟨m⟩_0          = {}", m.grade(0));
    println!("⟨m⟩_1          = {}", m.grade(1));
    println!("⟨m⟩_2          = {}", m.grade(2));
    println!("⟨m⟩_3          = {}", m.grade(3));

    println!();
    println!("== Involutions ==");
    println!("m              = {m}");
    println!("~m  (reverse)  = {}", m.reverse());
    println!("m̂  (grade inv) = {}", m.grade_involution());
    println!("m̄  (conjugate) = {}", m.conjugate());

    println!();
    println!("== Norms and a sneak preview of rotors ==");
    let v = Vga2 { coeffs: [0.0, 3.0, 4.0, 0.0] };
    println!("v = 3·e1 + 4·e2");
    println!("|v|² = ⟨v ~v⟩_0 = {}   (= 9 + 16, the Pythagorean theorem)", v.norm_squared());

    // A unit rotor in Vga2: R = cos(θ/2) - sin(θ/2)·e12 rotates vectors
    // through angle θ when sandwiched as R·v·~R. We can't actually do the
    // sandwich until we have division (round 4), but we *can* verify the
    // rotor is unit-norm right now.
    let theta = std::f64::consts::FRAC_PI_3; // 60°
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    let r = Vga2 { coeffs: [c, 0.0, 0.0, -s] };
    println!("rotor R for 60°  = {r}");
    println!("R * ~R           = {}   (should be 1)", r * r.reverse());
    println!("|R|²             = {}   (also should be 1)", r.norm_squared());
}
