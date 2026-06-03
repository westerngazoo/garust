//! `#[derive(Algebra)]` — the proc-macro path to a Clifford signature.
//!
//! This crate is the optional, feature-gated companion to
//! [garust-core](https://crates.io/crates/garust-core). The *default* way
//! to mint a signature stays the zero-dependency declarative macro
//! [`define_algebra!`](https://docs.rs/garust-core/latest/garust_core/macro.define_algebra.html),
//! which both declares the marker `struct` and implements
//! [`Algebra`](https://docs.rs/garust-core/latest/garust_core/algebra/trait.Algebra.html)
//! for it. The derive offered here is for when you'd rather write the
//! marker type *yourself* — so it can carry your own derives, attributes,
//! visibility, and doc comments — and have only the `Algebra` impl
//! generated:
//!
//! ```ignore
//! use garust::{Algebra, Multivector};
//!
//! /// 3D Projective GA, `Cl(3, 0, 1)`.
//! #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Algebra)]
//! #[algebra(p = 3, q = 0, r = 1)]
//! pub struct Pga3Sig;
//!
//! let e1 = Multivector::<Pga3Sig, f64>::basis(1);
//! assert_eq!((e1 * e1).scalar_part(), 1.0);
//! ```
//!
//! Enable it through the umbrella crate's feature: `garust = { version =
//! "…", features = ["derive"] }` (or `garust-core` with the same feature).
//! Don't depend on `garust-derive` directly — it's re-exported as
//! `garust::Algebra` / `garust_core::Algebra` so the generated code can find
//! the trait no matter which crate you import.
//!
//! # The `#[algebra(…)]` attribute
//!
//! The signature is given by up to three keys mapping to the metric counts
//! of `Algebra`: `p` generators square to `+1`, `q` to `−1`, and `r` to
//! `0` (the degenerate / null directions). `p` is required; `q` and `r`
//! default to `0`, so a Euclidean algebra is just `#[algebra(p = 3)]` and a
//! projective one `#[algebra(p = 3, r = 1)]`.

use proc_macro::TokenStream;
use proc_macro2::Span;
use proc_macro_crate::{crate_name, FoundCrate};
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Ident, LitInt};

/// Derive [`Algebra`](https://docs.rs/garust-core/latest/garust_core/algebra/trait.Algebra.html)
/// for a signature marker type from a `#[algebra(p = …, q = …, r = …)]`
/// attribute. See the [crate-level docs](crate) for the full story.
#[proc_macro_derive(Algebra, attributes(algebra))]
pub fn derive_algebra(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    expand(ast)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// The fallible core of the derive, kept separate so errors can be turned
/// into a `compile_error!` by the entry point above.
fn expand(ast: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = &ast.ident;

    // A signature is a parameterless marker; generics would have nowhere to
    // go in the generated `impl … for #name` header.
    if !ast.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &ast.generics,
            "#[derive(Algebra)] is for parameterless marker types; remove the generic parameters",
        ));
    }

    let (p, q, r) = parse_signature(name, &ast.attrs)?;
    let krate = garust_path();

    Ok(quote! {
        impl #krate::Algebra for #name {
            const P: usize = #p;
            const Q: usize = #q;
            const R: usize = #r;
            type Blades<T: #krate::Scalar> = [T; 1 << (#p + #q + #r)];
        }
    })
}

/// Read `p` / `q` / `r` out of the `#[algebra(…)]` attribute. `p` is
/// required; `q` and `r` default to `0`. Unknown or duplicate keys, and a
/// missing attribute, are reported as spanned errors.
fn parse_signature(name: &Ident, attrs: &[syn::Attribute]) -> syn::Result<(usize, usize, usize)> {
    let (mut p, mut q, mut r) = (None::<usize>, None::<usize>, None::<usize>);
    let mut seen = false;

    for attr in attrs {
        if !attr.path().is_ident("algebra") {
            continue;
        }
        seen = true;
        attr.parse_nested_meta(|meta| {
            let slot = if meta.path.is_ident("p") {
                &mut p
            } else if meta.path.is_ident("q") {
                &mut q
            } else if meta.path.is_ident("r") {
                &mut r
            } else {
                return Err(meta.error("unknown key; expected `p`, `q`, or `r`"));
            };
            if slot.is_some() {
                return Err(meta.error("duplicate key in #[algebra(…)]"));
            }
            let lit: LitInt = meta.value()?.parse()?;
            *slot = Some(lit.base10_parse()?);
            Ok(())
        })?;
    }

    if !seen {
        return Err(syn::Error::new_spanned(
            name,
            "#[derive(Algebra)] needs a #[algebra(p = …, q = …, r = …)] attribute",
        ));
    }
    let p = p.ok_or_else(|| {
        syn::Error::new_spanned(name, "#[algebra(…)] is missing the required `p` count")
    })?;
    Ok((p, q.unwrap_or(0), r.unwrap_or(0)))
}

/// The path to the crate that re-exports the `Algebra` trait and `Scalar`,
/// resolved against the *caller's* dependencies so the generated code works
/// whether they import the `garust` facade or `garust-core` directly.
///
/// Both crates re-export `Algebra` and `Scalar` at their root, so the same
/// `#krate::Algebra` / `#krate::Scalar` paths apply to either resolution.
fn garust_path() -> proc_macro2::TokenStream {
    // Prefer the facade (the common dependency); fall back to the kernel,
    // then to a literal for the unusual case where neither is in the prelude.
    for dep in ["garust", "garust-core"] {
        match crate_name(dep) {
            Ok(FoundCrate::Itself) => return quote!(crate),
            Ok(FoundCrate::Name(found)) => {
                let id = Ident::new(&found, Span::call_site());
                return quote!(::#id);
            }
            Err(_) => continue,
        }
    }
    quote!(::garust_core)
}
