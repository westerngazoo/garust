//! wizzielyn — the running learning playground for `garust`.
//!
//! This file grows alongside the library. Today it just exercises what
//! the library actually has: constructing multivectors in 2D VGA
//! `Cl(2, 0, 0)`, adding them, and printing them in human-readable form.

use garust::Vga2;

fn main() {
    // In Cl(2,0,0) the blade layout is: [1, e1, e2, e12].
    // So a generic 2D multivector looks like:  s + a·e1 + b·e2 + p·e12

    let e1 = Vga2::basis(1);
    let e2 = Vga2::basis(2);

    // The vector v = 3 e1 + 4 e2 (Euclidean length 5)
    let v = Vga2 { coeffs: [0.0, 3.0, 4.0, 0.0] };

    let two = Vga2::scalar(2.0);

    // Sums and negations are just coefficient arithmetic. None of the
    // *geometric* magic has kicked in yet — that arrives when we
    // implement the geometric product.
    let sum = v + two + e1;        // expect 2 + 4 e1 + 4 e2
    let diff = v - e1 - e2;        // expect 0 + 2 e1 + 3 e2

    println!("e1            = {}", show(&e1));
    println!("e2            = {}", show(&e2));
    println!("v             = {}", show(&v));
    println!("v + 2 + e1    = {}", show(&sum));
    println!("v - e1 - e2   = {}", show(&diff));
    println!("-v            = {}", show(&-v));
}

fn show(m: &Vga2) -> String {
    let labels = ["1", "e1", "e2", "e12"];
    let mut parts: Vec<String> = Vec::new();
    for (i, &c) in m.coeffs.iter().enumerate() {
        if c == 0.0 {
            continue;
        }
        parts.push(if i == 0 {
            format!("{c}")
        } else {
            format!("{c}·{}", labels[i])
        });
    }
    if parts.is_empty() {
        "0".into()
    } else {
        parts.join(" + ")
    }
}
