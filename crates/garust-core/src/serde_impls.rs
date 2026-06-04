//! `serde` support for [`Multivector`], behind the `serde` feature.
//!
//! A multivector *is* its `2^N` blade coefficients, so it (de)serializes as
//! a flat sequence of scalars in blade-index order — e.g. a `Vga3` becomes
//! the JSON array `[c0, c1, …, c7]`. The signature marker `A` is
//! zero-sized, carries no data, and pins the expected length at
//! deserialize time; it never appears on the wire.
//!
//! These impls are hand-written rather than derived for two reasons: the
//! backing store is the generic associated type `A::Blades<T>` (which a
//! derive cannot bound cleanly), and serializing as a sequence works for
//! **every** signature — including `Cga3`'s 32 blades and larger custom
//! algebras — whereas serde's array impls stop at length 32.

use core::fmt;
use core::marker::PhantomData;

use serde::de::{self, Deserialize, Deserializer, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeSeq, Serializer};

use crate::algebra::{Algebra, BladeStore};
use crate::multivector::Multivector;
use crate::scalar::Scalar;

impl<A: Algebra, T: Scalar + Serialize> Serialize for Multivector<A, T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let coeffs = self.coeffs.as_slice();
        let mut seq = serializer.serialize_seq(Some(coeffs.len()))?;
        for c in coeffs {
            seq.serialize_element(c)?;
        }
        seq.end()
    }
}

impl<'de, A: Algebra, T: Scalar + Deserialize<'de>> Deserialize<'de> for Multivector<A, T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct CoeffsVisitor<A, T> {
            marker: PhantomData<(A, T)>,
        }

        impl<'de, A: Algebra, T: Scalar + Deserialize<'de>> Visitor<'de> for CoeffsVisitor<A, T> {
            type Value = Multivector<A, T>;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "a sequence of {} blade coefficients", A::DIM)
            }

            fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
            where
                S: SeqAccess<'de>,
            {
                let mut mv = Multivector::<A, T>::zero();
                {
                    let slots = mv.coeffs.as_mut_slice();
                    for (i, slot) in slots.iter_mut().enumerate() {
                        *slot = seq
                            .next_element()?
                            .ok_or_else(|| de::Error::invalid_length(i, &self))?;
                    }
                }
                // A trailing element means the wrong signature was assumed.
                if seq.next_element::<T>()?.is_some() {
                    return Err(de::Error::invalid_length(A::DIM + 1, &self));
                }
                Ok(mv)
            }
        }

        deserializer.deserialize_seq(CoeffsVisitor::<A, T> {
            marker: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{Cga3, Pga3, Vga3};

    #[test]
    fn vga3_round_trips_as_a_flat_array() {
        let m = Vga3 {
            coeffs: [1.0, -2.0, 3.5, 0.0, 4.0, -5.0, 6.0, 7.25],
        };
        let json = serde_json::to_string(&m).unwrap();
        // Transparent, signature-free wire form: a bare array of coefficients.
        assert!(json.starts_with('[') && json.ends_with(']'));
        let back: Vga3 = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn pga_and_cga_constructors_round_trip() {
        // The degenerate (Pga3, 16) and largest standard (Cga3, 32) signatures.
        let point = Pga3::point(1.0, 2.0, 3.0);
        let back: Pga3 = serde_json::from_str(&serde_json::to_string(&point).unwrap()).unwrap();
        assert_eq!(back, point);

        let sphere = Cga3::sphere(1.0, 2.0, 3.0, 4.0);
        let back: Cga3 = serde_json::from_str(&serde_json::to_string(&sphere).unwrap()).unwrap();
        assert_eq!(back, sphere);
    }

    #[test]
    fn wrong_coefficient_count_is_rejected() {
        // Vga3 needs exactly 8 coefficients.
        assert!(serde_json::from_str::<Vga3>("[1.0, 2.0, 3.0]").is_err());
        assert!(serde_json::from_str::<Vga3>("[0,0,0,0,0,0,0,0,0]").is_err());
    }
}
