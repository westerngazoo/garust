//! Geometric calculus: derivatives of scalar- and multivector-valued
//! functions of a multivector variable, by forward-mode AD.
//!
//! For a function `f(X)` of a multivector variable `X = Σ_J x_J e_J`, the
//! multivector derivative (Hestenes; Doran & Lasenby, ch. 6) is
//!
//! ```text
//! ∂_X f  =  Σ_J  e^J · ∂f/∂x_J
//! ```
//!
//! where `{e_J}` is the basis of blades and `{e^J}` its reciprocal frame
//! (`e^J = e_J / (e_J · e_J)`). The partials `∂f/∂x_J` are computed *exactly*
//! with the [`Dual`] AD scalar — forward passes seeding `ε` on coefficients
//! and reading the result's `deriv`. No symbolic work, no finite differences.
//!
//! For **scalar-valued** `f` (energies, losses, Hamiltonians), layered:
//!
//! - [`partials`] — the raw gradient packed into a multivector
//!   (`coeff J = ∂f/∂x_J`), treating the coefficients as independent
//!   parameters. This is the gradient for coefficient-space optimization
//!   (gradient descent on an energy landscape).
//! - [`multivector_derivative`] — the coordinate-free `∂_X f`, i.e.
//!   [`partials`] with each coefficient scaled by its reciprocal-frame metric
//!   sign (`e_J · e_J ∈ {−1, +1}`). This is what makes the geometric
//!   identities — and Hamilton's equations `Q̇ = ∂_P H`, `Ṗ = −∂_Q H` —
//!   covariant.
//!
//! For **multivector-valued** maps `F: G → G` (versor-valued networks,
//! geometric message passing, physical fields):
//!
//! - [`differential`] — the pushforward `dF_x(h)`, exact in **one** forward
//!   pass (the JVP of ML autodiff). The Jacobian, served one column at a
//!   time, without ever materializing a `DIM × DIM` matrix.
//! - [`field_derivative`] — `∂_X F = Σ_J e^J (∂F/∂x_J)`, the
//!   multivector derivative of a field: each partial (itself a multivector)
//!   is carried back through the reciprocal frame by the **geometric
//!   product**. Reduces to [`multivector_derivative`] when `F` is scalar.
//! - [`vector_derivative`] — the same sum restricted to the grade-1
//!   directions: the `∇` of vector calculus, packing divergence and curl
//!   into one object (`∇F = ∇·F + ∇∧F`).
//!
//! ## Calculus on (flat) manifolds
//!
//! Splitting `∇` into its two halves gives the operators of differential-forms
//! / geometric calculus, here on the flat manifold of the algebra's space:
//!
//! - [`exterior_derivative`] — `dF = ∇∧F`, the grade-raising half: the
//!   gradient of a scalar, the curl of a vector, nilpotent (`d² = 0`).
//! - [`divergence`] — `∇·F`, the grade-lowering half.
//! - [`differential`] is the directional derivative `(a·∇)F`.
//!
//! These are forward-mode (so `no_std`-friendly). Curved manifolds — a
//! covariant derivative with a connection, the shape/curvature operator
//! (Hestenes–Sobczyk *vector manifolds*) — would build on these but need the
//! manifold itself represented (a metric/embedding + tangent projection); that
//! is a separate, larger effort, not provided here.
//!
//! **Cost:** one evaluation of `f` per seeded direction — `DIM` for the
//! gradients and [`field_derivative`], `N` for [`vector_derivative`] /
//! [`exterior_derivative`] / [`divergence`], and a single one for
//! [`differential`]. **Degenerate metrics:** a *null* blade
//! has `e_J · e_J = 0`, so its reciprocal does not exist; the
//! reciprocal-frame operators drop those directions (set them to zero). In a
//! non-degenerate algebra (e.g. spacetime `G(3,1,0)`) every blade squares to
//! `±1`, so the full derivative is recovered.

use crate::algebra::Algebra;
use crate::autodiff::Dual;
use crate::multivector::Multivector;
use crate::scalar::Real;
use crate::signature::{blade_product, grade_of};

/// The raw coefficient gradient of a scalar function `f`: the multivector
/// whose blade-`J` coefficient is `∂f/∂x_J`, computed by forward-mode AD.
///
/// `f` is evaluated over [`Dual`] coefficients; this seeds `ε` on each blade
/// coefficient in turn (`DIM` forward passes) and collects the derivative
/// parts. Treats the coefficients as independent real parameters — the right
/// gradient for coefficient-space optimization. For the coordinate-free
/// geometric derivative `∂_X f`, use [`multivector_derivative`].
pub fn partials<A, T, F>(x: &Multivector<A, T>, f: F) -> Multivector<A, T>
where
    A: Algebra,
    T: Real,
    F: Fn(&Multivector<A, Dual<T>>) -> Dual<T>,
{
    let mut grad = Multivector::<A, T>::zero();
    // Lift x to dual coefficients, all constant to begin with.
    let mut xd = Multivector::<A, Dual<T>>::zero();
    for i in 0..A::DIM {
        xd.coeffs[i] = Dual::constant(x.coeffs[i]);
    }
    // Seed ε on one coefficient at a time; the dual part is ∂f/∂x_J.
    for j in 0..A::DIM {
        xd.coeffs[j] = Dual::variable(x.coeffs[j]);
        grad.coeffs[j] = f(&xd).deriv;
        xd.coeffs[j] = Dual::constant(x.coeffs[j]);
    }
    grad
}

/// The multivector derivative `∂_X f = Σ_J e^J · ∂f/∂x_J` of a scalar
/// function — [`partials`] mapped through the reciprocal frame.
///
/// Each coefficient is scaled by its blade's metric sign `e_J · e_J ∈
/// {−1, +1}` (so `e^J = (e_J · e_J)·e_J`). This is the coordinate-free
/// gradient that keeps the geometric identities covariant — and the right
/// hand side of Hamilton's equations. Null blades (degenerate metrics) have
/// no reciprocal and are dropped to zero; in a non-degenerate algebra every
/// blade contributes.
pub fn multivector_derivative<A, T, F>(x: &Multivector<A, T>, f: F) -> Multivector<A, T>
where
    A: Algebra,
    T: Real,
    F: Fn(&Multivector<A, Dual<T>>) -> Dual<T>,
{
    let mut grad = partials(x, f);
    for j in 0..A::DIM {
        // e_J · e_J is the diagonal metric of the blade (the geometric
        // product of a blade with itself is a scalar).
        let (_idx, sign) = blade_product(j, j, A::P, A::Q);
        if sign == 0 {
            grad.coeffs[j] = T::ZERO; // null direction: no reciprocal
        } else if sign < 0 {
            grad.coeffs[j] = -grad.coeffs[j];
        }
        // sign > 0: e^J = e_J, coefficient unchanged.
    }
    grad
}

/// The differential (pushforward) of a multivector-valued map `F` at `x`
/// along the direction `h`:
///
/// ```text
/// dF_x(h) = d/dε F(x + ε·h) |_{ε=0}
/// ```
///
/// Computed exactly in **one** forward pass by seeding every coefficient's
/// dual part from `h` — the Jacobian-vector product (JVP) of ML autodiff.
/// Linear in `h`; column `J` of the Jacobian is
/// `differential(x, &basis(J), f)`, so the full matrix is available on
/// demand without ever being stored.
///
/// ```
/// use garust_core::{calculus::differential, Dual, Vga2};
///
/// // F(X) = X² — the differential is the (non-commutative!) product rule:
/// // dF_x(h) = x·h + h·x.
/// let x = Vga2::basis(1) + Vga2::basis(3) * 2.0; // e1 + 2·e12
/// let h = Vga2::basis(2) - Vga2::scalar(0.5); //    e2 − ½
/// let d = differential(&x, &h, |m| *m * *m);
/// assert_eq!(d, x * h + h * x);
/// ```
pub fn differential<A, T, F>(
    x: &Multivector<A, T>,
    h: &Multivector<A, T>,
    f: F,
) -> Multivector<A, T>
where
    A: Algebra,
    T: Real,
    F: Fn(&Multivector<A, Dual<T>>) -> Multivector<A, Dual<T>>,
{
    let mut xd = Multivector::<A, Dual<T>>::zero();
    for i in 0..A::DIM {
        xd.coeffs[i] = Dual::new(x.coeffs[i], h.coeffs[i]);
    }
    let fd = f(&xd);
    let mut out = Multivector::<A, T>::zero();
    for i in 0..A::DIM {
        out.coeffs[i] = fd.coeffs[i].deriv;
    }
    out
}

/// The multivector derivative of a multivector-valued **field**:
///
/// ```text
/// ∂_X F = Σ_J e^J (∂F/∂x_J)
/// ```
///
/// [`multivector_derivative`] generalized from scalar to multivector
/// outputs: each partial `∂F/∂x_J` (a full multivector, one
/// [`differential`] pass per blade) is carried back through the reciprocal
/// frame by the **geometric product** `e^J (∂F/∂x_J)`, so the result mixes
/// grades exactly as Hestenes' `∂_X` does. When `F` is scalar-valued this
/// collapses to [`multivector_derivative`]; when only the grade-1
/// directions matter, use the cheaper [`vector_derivative`].
///
/// Null blades (degenerate metrics) have no reciprocal and contribute
/// nothing. Cost: `DIM` evaluations of `f`.
pub fn field_derivative<A, T, F>(x: &Multivector<A, T>, f: F) -> Multivector<A, T>
where
    A: Algebra,
    T: Real,
    F: Fn(&Multivector<A, Dual<T>>) -> Multivector<A, Dual<T>>,
{
    let mut acc = Multivector::<A, T>::zero();
    for j in 0..A::DIM {
        let (_idx, sign) = blade_product(j, j, A::P, A::Q);
        if sign == 0 {
            continue; // null direction: no reciprocal
        }
        let column = differential(x, &Multivector::<A, T>::basis(j), &f);
        // e^J = (e_J · e_J)·e_J, so the reciprocal is the blade itself up
        // to the metric sign.
        let term = Multivector::<A, T>::basis(j) * column;
        if sign > 0 {
            acc += term;
        } else {
            acc -= term;
        }
    }
    acc
}

/// The **vector derivative** `∇F = Σ_k e^k (∂F/∂x_k)` — the grade-1
/// restriction of [`field_derivative`], differentiating only along the
/// vector directions: the `∇` of vector calculus, lifted to multivector
/// fields.
///
/// For a vector field in a Euclidean algebra the single result packs both
/// classical first-order operators: `∇F = ∇·F + ∇∧F` — divergence in the
/// scalar part, curl in the bivector part.
///
/// Null vectors (e.g. PGA's `e0`) contribute nothing. Cost: `N`
/// evaluations of `f`.
pub fn vector_derivative<A, T, F>(x: &Multivector<A, T>, f: F) -> Multivector<A, T>
where
    A: Algebra,
    T: Real,
    F: Fn(&Multivector<A, Dual<T>>) -> Multivector<A, Dual<T>>,
{
    let mut acc = Multivector::<A, T>::zero();
    for j in 0..A::DIM {
        if grade_of(j) != 1 {
            continue;
        }
        let (_idx, sign) = blade_product(j, j, A::P, A::Q);
        if sign == 0 {
            continue; // null vector: no reciprocal
        }
        let column = differential(x, &Multivector::<A, T>::basis(j), &f);
        let term = Multivector::<A, T>::basis(j) * column;
        if sign > 0 {
            acc += term;
        } else {
            acc -= term;
        }
    }
    acc
}

/// The **exterior derivative** `dF = ∇∧F = Σ_k e^k ∧ (∂F/∂x_k)` — the
/// grade-*raising* half of the [`vector_derivative`], i.e. the exterior
/// derivative of differential-forms calculus written in geometric algebra.
///
/// On a scalar field it is the gradient `∇φ` (a 1-form/vector); on a vector
/// field, the curl `∇∧V` (a bivector). It is **nilpotent** — `d(dF) = 0`
/// (Poincaré) — so `∇∧(∇φ) = 0` (curl of a gradient vanishes) and
/// `∇∧(∇∧A) = 0`. Together with [`divergence`], `∇F = ∇·F + ∇∧F`.
///
/// Forward-mode, so it is `no_std`-friendly. Null vectors contribute nothing;
/// cost: `N` evaluations of `f`.
pub fn exterior_derivative<A, T, F>(x: &Multivector<A, T>, f: F) -> Multivector<A, T>
where
    A: Algebra,
    T: Real,
    F: Fn(&Multivector<A, Dual<T>>) -> Multivector<A, Dual<T>>,
{
    let mut acc = Multivector::<A, T>::zero();
    for j in 0..A::DIM {
        if grade_of(j) != 1 {
            continue;
        }
        let (_idx, sign) = blade_product(j, j, A::P, A::Q);
        if sign == 0 {
            continue; // null vector: no reciprocal
        }
        let column = differential(x, &Multivector::<A, T>::basis(j), &f);
        let term = Multivector::<A, T>::basis(j).wedge(&column);
        if sign > 0 {
            acc += term;
        } else {
            acc -= term;
        }
    }
    acc
}

/// The **divergence** `∇·F = Σ_k e^k · (∂F/∂x_k)` — the grade-*lowering* half
/// of the [`vector_derivative`] (Hestenes inner product, so scalar field parts
/// contribute nothing). On a vector field it is the classical scalar
/// divergence. Together with [`exterior_derivative`], `∇F = ∇·F + ∇∧F`.
///
/// Forward-mode and `no_std`-friendly. Null vectors contribute nothing; cost:
/// `N` evaluations of `f`.
pub fn divergence<A, T, F>(x: &Multivector<A, T>, f: F) -> Multivector<A, T>
where
    A: Algebra,
    T: Real,
    F: Fn(&Multivector<A, Dual<T>>) -> Multivector<A, Dual<T>>,
{
    let mut acc = Multivector::<A, T>::zero();
    for j in 0..A::DIM {
        if grade_of(j) != 1 {
            continue;
        }
        let (_idx, sign) = blade_product(j, j, A::P, A::Q);
        if sign == 0 {
            continue; // null vector: no reciprocal
        }
        let column = differential(x, &Multivector::<A, T>::basis(j), &f);
        let term = Multivector::<A, T>::basis(j).inner(&column);
        if sign > 0 {
            acc += term;
        } else {
            acc -= term;
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::{
        differential, divergence, exterior_derivative, field_derivative, multivector_derivative,
        partials, vector_derivative,
    };
    use crate::autodiff::Dual;
    use crate::{Multivector, Vga2Sig, Vga3, Vga3Sig};

    #[test]
    fn partials_of_a_bilinear_form() {
        // f(X) = x0·x1 (scalar coefficient times the e1 coefficient).
        let mut x = Multivector::<Vga3Sig, f64>::zero();
        x.coeffs[0] = 5.0;
        x.coeffs[1] = 7.0;
        let g = partials(&x, |m: &Multivector<Vga3Sig, Dual<f64>>| {
            m.coeffs[0] * m.coeffs[1]
        });
        assert_eq!(g.coeffs[0], 7.0); // ∂/∂x0 = x1
        assert_eq!(g.coeffs[1], 5.0); // ∂/∂x1 = x0
        for j in 2..8 {
            assert_eq!(g.coeffs[j], 0.0);
        }
    }

    #[test]
    fn multivector_derivative_applies_reciprocal_frame_signs() {
        // f(X) = x1² + x3² in Vga2: e1² = +1, e12² = −1.
        let mut x = Multivector::<Vga2Sig, f64>::zero();
        x.coeffs[1] = 2.0; // e1
        x.coeffs[3] = 4.0; // e12
        let f = |m: &Multivector<Vga2Sig, Dual<f64>>| {
            m.coeffs[1] * m.coeffs[1] + m.coeffs[3] * m.coeffs[3]
        };

        // Raw partials: 2·x, no frame sign.
        let p = partials(&x, f);
        assert_eq!(p.coeffs[1], 4.0);
        assert_eq!(p.coeffs[3], 8.0);

        // Geometric derivative: e12's reciprocal flips its sign.
        let d = multivector_derivative(&x, f);
        assert_eq!(d.coeffs[1], 4.0); // e1²=+1  →  +2·x1
        assert_eq!(d.coeffs[3], -8.0); // e12²=−1 →  −2·x3
    }

    #[test]
    fn gradient_of_a_linear_form_is_constant() {
        // f(X) = ⟨c, X⟩ packed as Σ c_J x_J  ⇒  partials = c.
        let mut x = Multivector::<Vga3Sig, f64>::zero();
        x.coeffs[2] = 9.0; // arbitrary point; a linear form's gradient ignores it
        let g = partials(&x, |m: &Multivector<Vga3Sig, Dual<f64>>| {
            m.coeffs[1] * Dual::constant(3.0) + m.coeffs[4] * Dual::constant(-2.0)
        });
        assert_eq!(g.coeffs[1], 3.0);
        assert_eq!(g.coeffs[4], -2.0);
        assert_eq!(g.coeffs[2], 0.0);
    }

    // --- Multivector-valued maps -------------------------------------------

    #[test]
    fn differential_obeys_the_noncommutative_product_rule() {
        // F(X) = X²: dF_x(h) = x·h + h·x — order matters and both terms show.
        let x = Vga3::basis(1) + Vga3::basis(0b011) * 2.0 - Vga3::scalar(0.5);
        let h = Vga3::basis(2) + Vga3::basis(0b110) * 3.0;
        let d = differential(&x, &h, |m| *m * *m);
        assert_eq!(d, x * h + h * x);
    }

    #[test]
    fn differential_is_linear_in_the_direction() {
        let x = Vga3::basis(1) * 1.5 + Vga3::basis(0b101);
        let (a, b) = (Vga3::basis(2) * 2.0, Vga3::basis(0b011) - Vga3::scalar(1.0));
        let f = |m: &Multivector<Vga3Sig, Dual<f64>>| *m * *m * *m;
        let lhs = differential(&x, &(a + b), f);
        let rhs = differential(&x, &a, f) + differential(&x, &b, f);
        for i in 0..8 {
            assert!((lhs.coeffs[i] - rhs.coeffs[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn field_derivative_collapses_to_the_scalar_one() {
        // Embed a scalar function as a scalar-valued field: ∂_X agrees.
        let mut x = Multivector::<Vga2Sig, f64>::zero();
        x.coeffs[1] = 2.0;
        x.coeffs[3] = 4.0;
        let scalar_f = |m: &Multivector<Vga2Sig, Dual<f64>>| {
            m.coeffs[1] * m.coeffs[1] + m.coeffs[3] * m.coeffs[3]
        };
        let field_f = |m: &Multivector<Vga2Sig, Dual<f64>>| {
            Multivector::<Vga2Sig, Dual<f64>>::scalar(scalar_f(m))
        };
        assert_eq!(
            field_derivative(&x, field_f),
            multivector_derivative(&x, scalar_f)
        );
    }

    #[test]
    fn vector_derivative_of_the_identity_field_is_the_dimension() {
        // ∇x = Σ_k e^k e_k = n: the divergence of the position field.
        let x = Vga3::basis(1) * 0.3 + Vga3::basis(2) * 7.0; // any point
        let d = vector_derivative(&x, |m| m.grade(1));
        assert_eq!(d, Vga3::scalar(3.0));
    }

    #[test]
    fn vector_derivative_packs_divergence_and_curl() {
        // The plane rotation field F = x₁e2 − x₂e1: divergence-free, curl 2e12.
        let x = Vga3::basis(1) * 0.4 - Vga3::basis(2) * 1.1;
        let f = |m: &Multivector<Vga3Sig, Dual<f64>>| {
            let mut out = Multivector::<Vga3Sig, Dual<f64>>::zero();
            out.coeffs[2] = m.coeffs[1]; //  x₁ e2
            out.coeffs[1] = -m.coeffs[2]; // −x₂ e1
            out
        };
        let d = vector_derivative(&x, f);
        let mut expected = Vga3::zero();
        expected.coeffs[0b011] = 2.0; // pure curl: 2·e12, zero divergence
        assert_eq!(d, expected);
    }

    // --- Calculus on (flat) manifolds: ∇∧, ∇·, and the laws ----------------

    #[test]
    fn exterior_derivative_of_a_scalar_field_is_the_gradient() {
        // φ(X) = ½(x₁² + x₂²) ⇒ ∇φ = x₁ e1 + x₂ e2.
        let x = Vga3::basis(1) * 3.0 + Vga3::basis(2) * 4.0;
        let phi = |m: &Multivector<Vga3Sig, Dual<f64>>| {
            Multivector::<Vga3Sig, Dual<f64>>::scalar(
                (m.coeffs[1] * m.coeffs[1] + m.coeffs[2] * m.coeffs[2]) * Dual::constant(0.5),
            )
        };
        let grad = exterior_derivative(&x, phi);
        assert_eq!(grad, Vga3::basis(1) * 3.0 + Vga3::basis(2) * 4.0);
    }

    #[test]
    fn divergence_of_the_position_field_is_the_dimension() {
        // ∇·X = n (= 3 in Vga3), and its curl is zero.
        let x = Vga3::basis(1) * 0.3 + Vga3::basis(2) * 7.0;
        let id = |m: &Multivector<Vga3Sig, Dual<f64>>| m.grade(1);
        assert_eq!(divergence(&x, id), Vga3::scalar(3.0));
        assert_eq!(exterior_derivative(&x, id), Vga3::zero()); // curl-free
    }

    #[test]
    fn exterior_and_divergence_split_the_vector_derivative() {
        // ∇F = ∇·F + ∇∧F for the rotation field x₁e2 − x₂e1.
        let x = Vga3::basis(1) * 0.4 - Vga3::basis(2) * 1.1;
        let f = |m: &Multivector<Vga3Sig, Dual<f64>>| {
            let mut out = Multivector::<Vga3Sig, Dual<f64>>::zero();
            out.coeffs[2] = m.coeffs[1];
            out.coeffs[1] = -m.coeffs[2];
            out
        };
        let full = vector_derivative(&x, f);
        let parts = divergence(&x, f) + exterior_derivative(&x, f);
        assert_eq!(full, parts);
        // Pure rotation: zero divergence, curl 2·e12.
        assert_eq!(divergence(&x, f), Vga3::zero());
        assert_eq!(exterior_derivative(&x, f).coeffs[0b011], 2.0);
    }

    #[test]
    fn exterior_derivative_of_a_gradient_field_vanishes() {
        // A gradient field V = ∇(x₁x₂) = x₂e1 + x₁e2 is curl-free: d² = 0.
        let x = Vga3::basis(1) * 2.0 + Vga3::basis(2) * -1.0;
        let grad_field = |m: &Multivector<Vga3Sig, Dual<f64>>| {
            let mut out = Multivector::<Vga3Sig, Dual<f64>>::zero();
            out.coeffs[1] = m.coeffs[2]; // ∂φ/∂x₁ = x₂
            out.coeffs[2] = m.coeffs[1]; // ∂φ/∂x₂ = x₁
            out
        };
        assert_eq!(exterior_derivative(&x, grad_field), Vga3::zero());
    }
}
