//! wizzielyn — the running learning playground for `garust`.
//!
//! Each round of garust adds a new section here. Today: the geometric
//! product shows up, and the algebra finally feels like an algebra.

use garust::{Pga3, Vga2};

fn main() {
    let e1 = Vga2::basis(1);
    let e2 = Vga2::basis(2);
    let e12 = Vga2::basis(3); // 0b11

    println!("== Cl(2,0,0): 2D Euclidean VGA ==");
    println!("e1            = {e1}");
    println!("e2            = {e2}");
    println!("e1 * e2       = {}    (= e12, the unit bivector)", e1 * e2);
    println!("e2 * e1       = {}   (anticommutes with e1*e2)", e2 * e1);
    println!("e1 * e1       = {}    (vectors in P-group square to +1)", e1 * e1);
    println!("e12 * e12     = {}   (bivector squares to -1 — like i)", e12 * e12);

    // A non-trivial mixed-grade product. Worked out by hand:
    //   (2 + e12) * (1 + e1)
    //     = 2*1 + 2*e1 + e12*1 + e12*e1
    //     = 2 + 2 e1 + e12 + (e1 e2)(e1)
    //     = 2 + 2 e1 + e12 - e2
    let lhs = Vga2::scalar(2.0) + e12;
    let rhs = Vga2::scalar(1.0) + e1;
    println!("(2 + e12)*(1 + e1) = {}", lhs * rhs);

    // Scalar scaling, both sides.
    let v = 3.0 * e1 + 4.0 * e2;
    println!("v = 3·e1 + 4·e2     = {v}");
    // Squaring a Euclidean vector returns its squared length as a scalar.
    println!("v * v               = {}   (= |v|² = 25)", v * v);

    println!();
    println!("== Cl(3,0,1): 3D PGA — the null generator ==");
    // In PGA the R-group generator (bit 3) is conventionally called e0;
    // garust's signature-agnostic Display labels it as e4 for now.
    let e4 = Pga3::basis(0b1000);
    println!("e4 (the null e0)    = {e4}");
    println!("e4 * e4             = {}   (null squares to 0 → whole product vanishes)", e4 * e4);
}
