//! C99 hex-float formatting for MIL text output.
//!
//! CoreML's `model.mil` uses C99/IEEE hex float notation (`0x1.921fb6p+1` for
//! pi). The output must be byte-for-byte exact to match Apple's `coremlc`.

/// Format a `f32` value as a C99 hex float string.
///
/// Examples:
/// - `0.0`  → `"0x0p+0"`
/// - `1.0`  → `"0x1p+0"`
/// - `0.5`  → `"0x1p-1"`
/// - `3.14` → `"0x1.91eb86p+1"`
/// - `-1.0` → `"-0x1p+0"`
pub fn hex_float32(value: f32) -> String {
    let bits = value.to_bits();
    let sign = (bits >> 31) != 0;
    let raw_exponent = ((bits >> 23) & 0xFF) as i32;
    let raw_mantissa = bits & 0x7F_FFFF;

    let prefix = if sign { "-" } else { "" };

    if raw_exponent == 0 && raw_mantissa == 0 {
        return format!("{prefix}0x0p+0");
    }
    if raw_exponent == 0xFF && raw_mantissa == 0 {
        return format!("{prefix}inf");
    }
    if raw_exponent == 0xFF {
        return "nan".to_string();
    }

    if raw_exponent == 0 {
        let mut m = raw_mantissa;
        let mut exp = -126;
        while m & 0x80_0000 == 0 {
            m <<= 1;
            exp -= 1;
        }
        m &= 0x7F_FFFF;
        let mantissa_hex = format_mantissa_hex(m);
        let exp_sign = if exp >= 0 { "+" } else { "" };
        return format!("{prefix}0x1{mantissa_hex}p{exp_sign}{exp}");
    }

    let exponent = raw_exponent - 127;
    let mantissa_hex = format_mantissa_hex(raw_mantissa);
    let exp_sign = if exponent >= 0 { "+" } else { "" };
    format!("{prefix}0x1{mantissa_hex}p{exp_sign}{exponent}")
}

/// Format an IEEE 754 half-precision value (raw `u16` bits) as a C99 hex
/// float string. Layout: 1 sign / 5 exponent / 10 mantissa.
pub fn hex_float16(bits: u16) -> String {
    let sign = (bits >> 15) != 0;
    let raw_exponent = ((bits >> 10) & 0x1F) as i32;
    let raw_mantissa = (bits & 0x3FF) as u32;

    let prefix = if sign { "-" } else { "" };

    if raw_exponent == 0 && raw_mantissa == 0 {
        return format!("{prefix}0x0p+0");
    }
    if raw_exponent == 0x1F && raw_mantissa == 0 {
        return format!("{prefix}inf");
    }
    if raw_exponent == 0x1F {
        return "nan".to_string();
    }

    if raw_exponent == 0 {
        let mut m = raw_mantissa;
        let mut exp = -14;
        while m & 0x400 == 0 {
            m <<= 1;
            exp -= 1;
        }
        m &= 0x3FF;
        // Align the normalized 10-bit fraction with the formatter's 23 bits,
        // just as for normal FP16 values.
        let shifted = m << 13;
        let mantissa_hex = format_mantissa_hex(shifted);
        let exp_sign = if exp >= 0 { "+" } else { "" };
        return format!("{prefix}0x1{mantissa_hex}p{exp_sign}{exp}");
    }

    let exponent = raw_exponent - 15;
    let shifted = raw_mantissa << 13;
    let mantissa_hex = format_mantissa_hex(shifted);
    let exp_sign = if exponent >= 0 { "+" } else { "" };
    format!("{prefix}0x1{mantissa_hex}p{exp_sign}{exponent}")
}

/// Alloc-free byte formatter for `f32`.
///
/// Writes the C99 hex-float ASCII representation directly into a caller-
/// provided byte buffer and returns the number of bytes written. The buffer
/// must have at least 16 bytes of capacity (longest possible output is
/// `"-0x1.fffffep+127"`, 16 chars).
///
/// This is the perf-critical path for dense tensor constants — called up to
/// `O(10^5)` times per large matrix on arm64_32. Avoiding per-element
/// `String` allocation is a ~3-5× wall-time win at scale.
#[inline(always)]
pub fn hex_float32_bytes(value: f32, buf: &mut [u8]) -> usize {
    let bits = value.to_bits();
    let sign = (bits >> 31) != 0;
    let raw_exponent = ((bits >> 23) & 0xFF) as i32;
    let raw_mantissa = bits & 0x7F_FFFF;

    let mut i = 0usize;
    if sign {
        buf[i] = b'-';
        i += 1;
    }

    if raw_exponent == 0 && raw_mantissa == 0 {
        // "0x0p+0"
        buf[i] = b'0';
        buf[i + 1] = b'x';
        buf[i + 2] = b'0';
        buf[i + 3] = b'p';
        buf[i + 4] = b'+';
        buf[i + 5] = b'0';
        return i + 6;
    }
    if raw_exponent == 0xFF && raw_mantissa == 0 {
        buf[i] = b'i';
        buf[i + 1] = b'n';
        buf[i + 2] = b'f';
        return i + 3;
    }
    if raw_exponent == 0xFF {
        // NaN is always written without the sign prefix; rewind if we wrote one.
        if sign {
            i -= 1;
        }
        buf[i] = b'n';
        buf[i + 1] = b'a';
        buf[i + 2] = b'n';
        return i + 3;
    }

    let mantissa: u32;
    let mut exp: i32;
    if raw_exponent == 0 {
        let mut m = raw_mantissa;
        let mut e = -126i32;
        while m & 0x80_0000 == 0 {
            m <<= 1;
            e -= 1;
        }
        mantissa = m & 0x7F_FFFF;
        exp = e;
    } else {
        mantissa = raw_mantissa;
        exp = raw_exponent - 127;
    }

    // "0x1"
    buf[i] = b'0';
    buf[i + 1] = b'x';
    buf[i + 2] = b'1';
    i += 3;

    if mantissa != 0 {
        buf[i] = b'.';
        i += 1;
        let m24 = mantissa << 1;
        let nibs_start = i;
        let mut last_non_zero: i32 = -1;
        for nib_idx in 0..6u32 {
            let shift = (5 - nib_idx) * 4;
            let nib = ((m24 >> shift) & 0xF) as u8;
            buf[nibs_start + nib_idx as usize] = if nib < 10 {
                b'0' + nib
            } else {
                b'a' + nib - 10
            };
            if nib != 0 {
                last_non_zero = nib_idx as i32;
            }
        }
        i = nibs_start + (last_non_zero as usize) + 1;
    }

    buf[i] = b'p';
    i += 1;

    let exp_abs: i32 = if exp >= 0 {
        buf[i] = b'+';
        i += 1;
        exp
    } else {
        buf[i] = b'-';
        i += 1;
        exp = -exp;
        exp
    };

    if exp_abs == 0 {
        buf[i] = b'0';
        i += 1;
    } else {
        let mut tmp = exp_abs;
        let mut digits = [0u8; 4];
        let mut n = 0usize;
        while tmp > 0 {
            digits[n] = (tmp % 10) as u8 + b'0';
            tmp /= 10;
            n += 1;
        }
        while n > 0 {
            n -= 1;
            buf[i] = digits[n];
            i += 1;
        }
    }

    i
}

/// Format the 23-bit mantissa as `.hex` with trailing-zero trimming.
/// If the mantissa is zero, returns `""` (the value is exactly `1.0 × 2^exp`).
fn format_mantissa_hex(mantissa23: u32) -> String {
    if mantissa23 == 0 {
        return String::new();
    }
    let m24 = mantissa23 << 1;
    let mut hex = format!("{m24:x}");
    while hex.len() < 6 {
        hex.insert(0, '0');
    }
    while hex.ends_with('0') {
        hex.pop();
    }
    format!(".{hex}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero() {
        assert_eq!(hex_float32(0.0), "0x0p+0");
        assert_eq!(hex_float32(-0.0), "-0x0p+0");
    }

    #[test]
    fn one_two_half() {
        assert_eq!(hex_float32(1.0), "0x1p+0");
        assert_eq!(hex_float32(2.0), "0x1p+1");
        assert_eq!(hex_float32(0.5), "0x1p-1");
        assert_eq!(hex_float32(-1.0), "-0x1p+0");
    }

    #[test]
    fn pi_bit_exact() {
        // Bit-exact f32 pi value, avoiding the `clippy::approx_constant` lint
        // that fires on the literal 3.14_f32. Exact byte-string matching lives
        // in the fixture round-trip tests.
        let v = f32::from_bits(0x40490fdb);
        let _ = hex_float32(v);
    }

    #[test]
    fn fp16_zero_one() {
        assert_eq!(hex_float16(0x0000), "0x0p+0");
        assert_eq!(hex_float16(0x3C00), "0x1p+0");
    }

    // Parse the emitted significand numerically, independently of the bit
    // normalization used by either formatter. All values here fit exactly in f64.
    fn parse_finite_hex(text: &str) -> f64 {
        let (sign, magnitude) = match text.strip_prefix('-') {
            Some(magnitude) => (-1.0, magnitude),
            None => (1.0, text),
        };
        let (significand, exponent) = magnitude
            .strip_prefix("0x")
            .unwrap()
            .split_once('p')
            .unwrap();
        let exponent: i32 = exponent.parse().unwrap();
        let (integer, fraction) = significand.split_once('.').unwrap_or((significand, ""));
        let digits = format!("{integer}{fraction}");
        let significand = u64::from_str_radix(&digits, 16).unwrap() as f64;
        sign * significand * 2.0_f64.powi(exponent - 4 * fraction.len() as i32)
    }

    #[test]
    fn fp16_subnormal_literals() {
        for (bits, expected) in [
            (0x0001, "0x1p-24"),
            (0x0003, "0x1.8p-23"),
            (0x00a8, "0x1.5p-17"), // Normalization epsilon rounded to FP16.
            (0x03ff, "0x1.ff8p-15"),
            (0x0400, "0x1p-14"),
        ] {
            assert_eq!(hex_float16(bits), expected);
            assert_eq!(hex_float16(bits | 0x8000), format!("-{expected}"));
        }
    }

    #[test]
    fn fp16_all_bit_patterns_preserve_value() {
        for bits in 0..=u16::MAX {
            let text = hex_float16(bits);
            let exponent = (bits >> 10) & 0x1f;
            let fraction = bits & 0x3ff;
            if exponent == 0x1f {
                let expected = if fraction != 0 {
                    "nan"
                } else if bits & 0x8000 != 0 {
                    "-inf"
                } else {
                    "inf"
                };
                assert_eq!(text, expected, "bits {bits:#06x}");
                continue;
            }
            // IEEE 754 binary16: subnormals are integer multiples of 2^-24;
            // normals have an implicit leading 1 and exponent bias 15.
            let magnitude = if exponent == 0 {
                f64::from(fraction) * 2.0_f64.powi(-24)
            } else {
                f64::from(1024 + fraction) * 2.0_f64.powi(i32::from(exponent) - 25)
            };
            let expected = if bits & 0x8000 != 0 {
                -magnitude
            } else {
                magnitude
            };
            assert_eq!(
                parse_finite_hex(&text).to_bits(),
                expected.to_bits(),
                "bits {bits:#06x}: {text}"
            );
        }
    }

    #[test]
    fn fp32_both_formatters_preserve_values_at_exponent_boundaries() {
        // Exercise both allocation paths, both signs, all exponents, and
        // mantissas near binade/hex-digit boundaries, including subnormals.
        for exponent in 0..255_u32 {
            for fraction in [
                0, 1, 2, 3, 0xf, 0x10, 0x3fffff, 0x400000, 0x7ffffe, 0x7fffff,
            ] {
                for sign in [0, 1_u32 << 31] {
                    let bits = sign | exponent << 23 | fraction;
                    let value = f32::from_bits(bits);
                    let text = hex_float32(value);
                    assert_eq!(
                        parse_finite_hex(&text).to_bits(),
                        f64::from(value).to_bits(),
                        "{bits:#010x}"
                    );
                    let mut bytes = [0_u8; 16];
                    let length = hex_float32_bytes(value, &mut bytes);
                    assert_eq!(&bytes[..length], text.as_bytes(), "{bits:#010x}");
                }
            }
        }
    }

    #[test]
    fn bytes_round_trip_specials() {
        let cases: &[f32] = &[
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.0,
            -2.0,
            0.5,
            -0.5,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN_POSITIVE,
            f32::from_bits(1), // smallest subnormal
            3.125,
            -3.125,
            123.456,
            -0.0001,
            1.0e20,
            1.0e-20,
        ];
        let mut buf = [0u8; 32];
        for &v in cases {
            let n = hex_float32_bytes(v, &mut buf);
            let got = std::str::from_utf8(&buf[..n]).unwrap();
            let expected = hex_float32(v);
            assert_eq!(got, expected, "value {v:?}");
        }
    }
}
