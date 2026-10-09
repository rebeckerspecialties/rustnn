//! Integer-only IEEE binary16/binary32 conversion for exact typed Cast stages.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Widen,
    Narrow,
}

pub fn widen(bits: u16) -> u32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 31;
    let mut fraction = u32::from(bits & 1023);
    match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut exponent = 113u32;
            while fraction & 1024 == 0 {
                fraction <<= 1;
                exponent -= 1;
            }
            sign | (exponent << 23) | ((fraction & 1023) << 13)
        }
        31 => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (fraction << 13),
    }
}

fn rounded_shift(value: u32, shift: u32) -> u32 {
    let retained = value >> shift;
    let discarded = value & ((1u32 << shift) - 1);
    let midpoint = 1u32 << (shift - 1);
    retained + u32::from(discarded > midpoint || (discarded == midpoint && retained & 1 != 0))
}

pub fn narrow(bits: u32) -> u16 {
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 255) as i32;
    let fraction = bits & 0x007f_ffff;
    if exponent == 255 {
        return sign
            | 0x7c00
            | if fraction == 0 {
                0
            } else {
                // Preserve a useful payload and quiet NaNs; their payload is
                // not part of the WebNN numeric Cast classification contract.
                ((fraction >> 13) as u16) | 0x0200
            };
    }
    let half_exponent = exponent - 112;
    if half_exponent >= 31 {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        return sign | rounded_shift(fraction | 0x0080_0000, (14 - half_exponent) as u32) as u16;
    }
    let significand = rounded_shift(fraction, 13);
    // A significand carry advances the exponent, including finite overflow.
    sign | (((half_exponent as u32) << 10) + significand) as u16
}

pub fn cast(bytes: &[u8], direction: Direction) -> Result<Vec<u8>, &'static str> {
    let (source_width, target_width) = match direction {
        Direction::Widen => (2, 4),
        Direction::Narrow => (4, 2),
    };
    if !bytes.len().is_multiple_of(source_width) {
        return Err("Cast input is not an integral number of represented values");
    }
    let size = (bytes.len() / source_width)
        .checked_mul(target_width)
        .filter(|&size| size <= isize::MAX as usize)
        .ok_or("Cast output exceeds addressable storage")?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(size)
        .map_err(|_| "Cast output allocation failed")?;
    match direction {
        Direction::Widen => {
            for bytes in bytes.as_chunks::<2>().0 {
                output.extend_from_slice(&widen(u16::from_ne_bytes(*bytes)).to_ne_bytes());
            }
        }
        Direction::Narrow => {
            for bytes in bytes.as_chunks::<4>().0 {
                output.extend_from_slice(&narrow(u32::from_ne_bytes(*bytes)).to_ne_bytes());
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent numeric oracle: exact binary16 values are representable in
    // f64. Binary-search neighboring values, then compare exact distances.
    fn half_value(bits: u16) -> f64 {
        let sign = if bits & 0x8000 == 0 { 1. } else { -1. };
        let exponent = (bits >> 10) & 31;
        let fraction = bits & 1023;
        match exponent {
            0 => sign * f64::from(fraction) * 2f64.powi(-24),
            31 if fraction == 0 => sign * f64::INFINITY,
            31 => f64::NAN,
            _ => sign * (1. + f64::from(fraction) / 1024.) * 2f64.powi(i32::from(exponent) - 15),
        }
    }

    fn oracle_narrow(bits: u32) -> u16 {
        let value = f32::from_bits(bits);
        let sign = if bits >> 31 == 0 { 0 } else { 0x8000 };
        if value.is_nan() {
            return sign | 0x7e00;
        }
        let magnitude = f64::from(value.abs());
        if magnitude >= 65520. {
            return sign | 0x7c00;
        }
        let mut low = 0u16;
        let mut high = 0x7bffu16;
        while low < high {
            let midpoint = low + (high - low).div_ceil(2);
            if half_value(midpoint) > magnitude {
                high = midpoint - 1;
            } else {
                low = midpoint;
            }
        }
        if low == 0x7bff {
            return sign | low;
        }
        let lower = magnitude - half_value(low);
        let upper = half_value(low + 1) - magnitude;
        sign | if lower < upper || (lower == upper && low & 1 == 0) {
            low
        } else {
            low + 1
        }
    }

    #[test]
    fn every_half_encoding_widens_and_round_trips() {
        for bits in 0..=u16::MAX {
            let actual = f32::from_bits(widen(bits));
            let expected = half_value(bits);
            if expected.is_nan() {
                assert!(actual.is_nan(), "{bits:04x}");
                assert_eq!(actual.is_sign_negative(), bits & 0x8000 != 0);
                assert_eq!(narrow(actual.to_bits()) & 0x7c00, 0x7c00);
                assert_ne!(narrow(actual.to_bits()) & 1023, 0);
            } else {
                assert_eq!(
                    f64::from(actual).to_bits(),
                    expected.to_bits(),
                    "{bits:04x}"
                );
                assert_eq!(narrow(actual.to_bits()), bits, "{bits:04x}");
            }
        }
    }

    #[test]
    fn every_finite_half_midpoint_and_both_neighbors_round_even() {
        for lower in 0..0x7bffu16 {
            let midpoint = ((half_value(lower) + half_value(lower + 1)) / 2.) as f32;
            for magnitude in [
                midpoint.to_bits() - 1,
                midpoint.to_bits(),
                midpoint.to_bits() + 1,
            ] {
                for sign in [0, 0x8000_0000] {
                    let bits = magnitude | sign;
                    assert_eq!(narrow(bits), oracle_narrow(bits), "{bits:08x}");
                }
            }
        }
    }

    #[test]
    fn overflow_zero_subnormal_and_nan_classes() {
        for value in [
            0.,
            -0.,
            2f32.powi(-25),
            -2f32.powi(-25),
            65519.,
            -65519.,
            65520.,
            -65520.,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            assert_eq!(narrow(value.to_bits()), oracle_narrow(value.to_bits()));
        }
        for bits in [0x7f80_0001, 0xff80_0001, 0x7fc0_0123, 0xfff0_abcd] {
            let actual = narrow(bits);
            assert_eq!(actual & 0x8000, ((bits >> 16) & 0x8000) as u16);
            assert_eq!(actual & 0x7c00, 0x7c00);
            assert_ne!(actual & 1023, 0);
        }
    }

    #[test]
    fn randomized_float32_narrowing_matches_independent_distance_oracle() {
        let mut state = 0x36a9_821du32;
        for _ in 0..100_000 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let actual = narrow(state);
            let expected = oracle_narrow(state);
            if f32::from_bits(state).is_nan() {
                assert_eq!(actual & 0xfc00, expected & 0xfc00);
                assert_ne!(actual & 1023, 0);
            } else {
                assert_eq!(actual, expected, "{state:08x}");
            }
        }
    }

    #[test]
    fn unaligned_bytes_and_incomplete_values() {
        let expected = [0x8000u16, 1, 0x83ff, 0x7bff, 0xfc00]
            .into_iter()
            .flat_map(u16::to_ne_bytes)
            .collect::<Vec<_>>();
        let mut unaligned = vec![123];
        unaligned.extend_from_slice(&expected);
        let wide = cast(&unaligned[1..], Direction::Widen).unwrap();
        assert_eq!(wide.len(), 20);
        assert_eq!(cast(&wide, Direction::Narrow).unwrap(), expected);
        assert!(cast(&[0], Direction::Widen).is_err());
        assert!(cast(&[0, 0, 0], Direction::Narrow).is_err());
        assert!(cast(&[], Direction::Widen).unwrap().is_empty());
    }
}
