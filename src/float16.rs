//! Architecture-independent rounding for binary16 representations.

/// Round binary64 directly to binary16, retaining every discarded bit.
///
/// Available to model conversion and buffer conversion regardless of native
/// runtime features. In particular, scalar WebIDL doubles must not pass
/// through binary32 or a software converter that truncates sticky bits first.
pub(crate) fn f64_to_f16_bits(value: f64) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 48) & 0x8000) as u16;
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    if exponent == 0x7ff {
        return sign | if fraction == 0 { 0x7c00 } else { 0x7e00 };
    }
    // Binary64 subnormals are all below half's smallest rounding midpoint.
    if exponent == 0 {
        return sign;
    }
    let exponent = exponent - 1023;
    if exponent > 15 {
        return sign | 0x7c00;
    }
    let significand = fraction | (1u64 << 52);
    if exponent >= -14 {
        let rounded = round_shift_even(significand, 42);
        // A significand carry advances the exponent, including finite overflow.
        return sign | ((((exponent + 15) as u16) << 10) + (rounded as u16 - 1024));
    }
    if exponent < -25 {
        return sign;
    }
    sign | round_shift_even(significand, (28 - exponent) as u32) as u16
}

fn round_shift_even(value: u64, shift: u32) -> u64 {
    let retained = value >> shift;
    let discarded = value & ((1u64 << shift) - 1);
    let midpoint = 1u64 << (shift - 1);
    retained + u64::from(discarded > midpoint || (discarded == midpoint && retained & 1 != 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn positive_half_value(bits: u16) -> f64 {
        let exponent = bits >> 10;
        let fraction = bits & 1023;
        if exponent == 0 {
            f64::from(fraction) * 2f64.powi(-24)
        } else {
            f64::from(1024 + fraction) * 2f64.powi(i32::from(exponent) - 25)
        }
    }

    #[test]
    fn every_finite_midpoint_and_neighbor_rounds_to_even() {
        // Expectations come from adjacent binary16 encodings, not another
        // narrowing implementation. Both source widths represent each tie
        // exactly; their immediate neighbors must round to opposite endpoints.
        let mut cases = 0;
        for lower in 0u16..0x7bff {
            let midpoint = (positive_half_value(lower) + positive_half_value(lower + 1)) / 2.;
            let tie = lower + (lower & 1);
            for (offset, expected) in [(-1i64, lower), (0, tie), (1, lower + 1)] {
                let single = f32::from_bits(((midpoint as f32).to_bits() as i64 + offset) as u32);
                let double = f64::from_bits((midpoint.to_bits() as i64 + offset) as u64);
                for sign in [false, true] {
                    let expected = expected | if sign { 0x8000 } else { 0 };
                    assert_eq!(
                        f64_to_f16_bits(if sign { -double } else { double }),
                        expected,
                        "binary64 neighbor, lower={lower:#06x}, offset={offset}, sign={sign}"
                    );
                    assert_eq!(
                        f64_to_f16_bits(f64::from(if sign { -single } else { single })),
                        expected,
                        "binary32 neighbor, lower={lower:#06x}, offset={offset}, sign={sign}"
                    );
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 190_458);
    }

    #[test]
    fn every_finite_half_encoding_round_trips_including_signed_zero() {
        for bits in 0u16..=0x7bff {
            let value = positive_half_value(bits);
            assert_eq!(f64_to_f16_bits(value), bits);
            assert_eq!(f64_to_f16_bits(-value), bits | 0x8000);
        }
    }

    #[test]
    fn overflow_underflow_and_nonfinite_classes_keep_their_sign() {
        for (value, expected) in [
            (f64::from_bits(1), 0),
            (f64::MIN_POSITIVE, 0),
            (f64::from_bits(65520f64.to_bits() - 1), 0x7bff),
            (65520., 0x7c00),
            (f64::from_bits(65520f64.to_bits() + 1), 0x7c00),
            (f64::MAX, 0x7c00),
            (f64::INFINITY, 0x7c00),
        ] {
            assert_eq!(f64_to_f16_bits(value), expected, "value={value:?}");
            assert_eq!(f64_to_f16_bits(-value), expected | 0x8000);
        }
        // Canonicalize narrowed NaNs while retaining sign, including payloads
        // present only in the low significand bits discarded by the dependency.
        for payload in [1, 1 << 31, 1 << 51, (1 << 52) - 1] {
            for sign in [0, 1 << 63] {
                let value = f64::from_bits(sign | 0x7ff0_0000_0000_0000 | payload);
                assert_eq!(f64_to_f16_bits(value), 0x7e00 | ((sign >> 48) as u16));
            }
        }
    }
}
