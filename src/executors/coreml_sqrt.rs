//! Integer-only binary32 square root, independent of host flush-to-zero modes.

/// Round sqrt(M * 2^e) directly to binary32, without a floating-point input
/// conversion. If p=floor(log2(x)/2), its scaled significand is sqrt(Y), where
/// Y=M * 2^(e+46-2p) fits in 48 bits. Compare 4Y with (2*floor(sqrt(Y))+1)^2
/// at the exact rounding midpoint. All positive finite binary32 inputs have
/// normal binary32 roots; only a significand carry can change the exponent.
fn sqrt_bits(bits: u32) -> u32 {
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x007f_ffff;
    if exponent == 0xff && fraction != 0 {
        return bits | 0x0040_0000;
    }
    if exponent == 0 && fraction == 0 {
        return bits;
    }
    if bits >> 31 != 0 {
        return 0x7fc0_0000;
    }
    if exponent == 0xff {
        return bits;
    }
    let (significand, power) = if exponent == 0 {
        (u64::from(fraction), -149)
    } else {
        (u64::from(0x0080_0000 | fraction), exponent as i32 - 150)
    };
    let width = 64 - significand.leading_zeros() as i32;
    let mut result_power = (width - 1 + power).div_euclid(2);
    let shift = power + 46 - 2 * result_power;
    debug_assert!((23..=47).contains(&shift));
    let radicand = significand << shift;
    debug_assert!(radicand < 1 << 48);
    let floor = radicand.isqrt();
    let midpoint = (2 * floor + 1).pow(2);
    let mut rounded = floor
        + u64::from(radicand << 2 > midpoint || (radicand << 2 == midpoint && floor & 1 != 0));
    if rounded == 1 << 24 {
        rounded >>= 1;
        result_power += 1;
    }
    debug_assert!((1 << 23..1 << 24).contains(&rounded));
    ((result_power + 127) as u32) << 23 | (rounded as u32 & 0x007f_ffff)
}

pub(super) fn evaluate(bytes: &[u8]) -> Result<Vec<u8>, &'static str> {
    let (values, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err("typed Float32 Sqrt input has an incomplete element");
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(bytes.len())
        .map_err(|_| "typed Float32 Sqrt allocation failed")?;
    for value in values {
        output.extend_from_slice(&sqrt_bits(u32::from_ne_bytes(*value)).to_ne_bytes());
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqrt_special_values_and_subnormal_inputs_use_independent_bits() {
        // Exact dyadic square/midpoint comparisons, not native Float32 sqrt.
        for (input, expected) in [
            (0, 0),
            (0x8000_0000, 0x8000_0000),
            (1, 0x1a35_04f3),
            (2, 0x1a80_0000),
            (0x007f_ffff, 0x1fff_ffff),
            (0x0080_0000, 0x2000_0000),
            (0x3f80_0000, 0x3f80_0000),
            (0x4080_0000, 0x4000_0000),
            (0x7f80_0000, 0x7f80_0000),
            (0xbf80_0000, 0x7fc0_0000),
            (0xff80_0000, 0x7fc0_0000),
            (0x7f80_0001, 0x7fc0_0001),
            (0xff81_2345, 0xffc1_2345),
        ] {
            assert_eq!(sqrt_bits(input), expected, "input={input:08x}");
        }
        assert!(evaluate(&[1, 2, 3]).is_err());
        let unaligned = [255, 1, 0, 0, 0];
        assert_eq!(
            evaluate(&unaligned[1..]).unwrap(),
            0x1a35_04f3_u32.to_ne_bytes()
        );
    }

    #[test]
    fn sqrt_covers_every_exponent_and_stratified_significands() {
        // Construct f64 from integer fields, so the reference never widens a
        // Float32 subnormal through a potentially flushing instruction. f64
        // sqrt is independent of the integer midpoint implementation above.
        for exponent in 0..255_u32 {
            for fraction in (0..=0x007f_ffff).step_by(257).chain([0x007f_ffff]) {
                let bits = exponent << 23 | fraction;
                let reference = if exponent == 0 {
                    f64::from(fraction) * 2_f64.powi(-149)
                } else {
                    f64::from(fraction | 0x0080_0000) * 2_f64.powi(exponent as i32 - 150)
                };
                assert_eq!(
                    sqrt_bits(bits),
                    (reference.sqrt() as f32).to_bits(),
                    "input={bits:08x}"
                );
            }
        }
    }
}
