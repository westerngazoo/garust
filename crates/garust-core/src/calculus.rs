//! Geometric calculus: the **multivector derivative** of a scalar-valued
//! function, by forward-mode AD.
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
//! with the [`Dual`] AD scalar — one forward pass per coefficient, seeding
//! `ε` on `x_J` and reading the result's `deriv`. No symbolic work, no finite
//! differences.
//!
//! Two functions, layered:
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
//! **Cost:** `DIM` evaluations of `f` (one per blade). **Degenerate metrics:**
//! a *null* blade has `e_J · e_J = 0`, so its reciprocal does not exist;
//! [`multivector_derivative`] drops those directions (sets them to zero). In a
//! non-degenerate algebra (e.g. spacetime `G(3,1,0)`) every blade squares to
//! `±1`, so the full derivative is recovered.

use crate::algebra::Algebra;
use crate::autodiff::Dual;
use crate::multivector::Multivector;
use crate::scalar::Real;
use crate::signature::blade_product;

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

#[cfg(test)]
mod tests {
    use super::{multivector_derivative, partials};
    use crate::autodiff::Dual;
    use crate::{Multivector, Vga2Sig, Vga3Sig};

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
}
