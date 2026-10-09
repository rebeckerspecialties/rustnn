//! Exact bit interchange between IEEE binary32 and binary64.

fn rounded_shift(value: u64, shift: u32) -> u64 {
    let kept = value >> shift;
    let remainder = value & ((1_u64 << shift) - 1);
    let midpoint = 1_u64 << (shift - 1);
    kept + u64::from(remainder > midpoint || (remainder == midpoint && kept & 1 != 0))
}

// Unlike a hardware f64->f32 cast, this rounds subnormals even with FPCR.FZ set.
// The existing Cast stage narrows f32->f16, with a different exponent/width.
pub(super) fn narrow(value: f64) -> u32 {
    let bits = value.to_bits();
    let sign = ((bits >> 32) as u32) & 0x8000_0000;
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & 0x000f_ffff_ffff_ffff;
    if exponent == 0x7ff {
        return sign | 0x7f80_0000 | if fraction == 0 { 0 } else { 0x0040_0000 };
    }
    let exponent = exponent - 896;
    if exponent >= 255 {
        return sign | 0x7f80_0000;
    }
    if exponent <= 0 {
        if exponent < -23 {
            return sign;
        }
        return sign | rounded_shift(fraction | (1 << 52), (30 - exponent) as u32) as u32;
    }
    sign | (((exponent as u32) << 23) + rounded_shift(fraction, 29) as u32)
}

/// Assemble the exact binary64 encoding without a Float32 instruction, which
/// could flush a subnormal input under an ambient FPCR.FZ setting.
pub(super) fn widen(bits: u32) -> f64 {
    let sign = u64::from(bits & 0x8000_0000) << 32;
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x007f_ffff;
    let magnitude = match exponent {
        0 if fraction == 0 => 0,
        0 => {
            let width = 32 - fraction.leading_zeros();
            (u64::from(873 + width) << 52)
                | ((u64::from(fraction) << (53 - width)) & 0x000f_ffff_ffff_ffff)
        }
        255 => 0x7ff0_0000_0000_0000 | (u64::from(fraction) << 29),
        _ => (u64::from(exponent + 896) << 52) | (u64::from(fraction) << 29),
    };
    f64::from_bits(sign | magnitude)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary32_interchange_preserves_finite_encodings_and_rounding_boundaries() {
        for sign in [0, 0x8000_0000] {
            for exponent in 0..255 {
                for fraction in [0, 1, 2, 0x003f_ffff, 0x0040_0000, 0x007f_fffe, 0x007f_ffff] {
                    let bits = sign | exponent << 23 | fraction;
                    assert_eq!(narrow(widen(bits)), bits, "{bits:08x}");
                }
            }
        }
        assert_eq!(widen(1).to_bits(), 0x36a0_0000_0000_0000);
        assert_eq!(widen(0x007f_ffff).to_bits(), 0x380f_ffff_c000_0000);
        assert_eq!(narrow(f64::from_bits(0x3690_0000_0000_0000)), 0);
        assert_eq!(narrow(f64::from_bits(0x3690_0000_0000_0001)), 1);
        assert_eq!(narrow(f64::from_bits(0x36a8_0000_0000_0000)), 2);
    }
}
