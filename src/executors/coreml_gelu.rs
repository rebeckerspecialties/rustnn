//! Cancellation-free Float32 exact GELU, independent of the caller's FP32 FTZ mode.

use super::binary32::narrow;

unsafe extern "C" {
    fn erfc(x: f64) -> f64;
}

fn gelu(bits: u32) -> u32 {
    let sign = bits & 0x8000_0000;
    let magnitude = bits & 0x7fff_ffff;
    // Follow the written x*(1+erf(x/sqrt(2)))/2 expression at -infinity,
    // not its finite-input limit. Preserve signed zero; NaN payloads are not
    // part of the arithmetic contract.
    if magnitude > 0x7f80_0000 || bits == 0xff80_0000 {
        return 0x7fc0_0000;
    }
    if magnitude == 0 || bits == 0x7f80_0000 {
        return bits;
    }
    if magnitude < 0x0100_0000 {
        // x/2 can be a midpoint between binary32 subnormals. The strictly
        // positive x^2/sqrt(2*pi) correction decides the tie for either sign,
        // although it is too small to survive even a binary64 evaluation.
        return sign | ((magnitude + u32::from(sign == 0)) >> 1);
    }
    // The remaining input is normal: assemble its exact binary64 encoding
    // without an FP32 instruction that could flush the input to zero.
    let x = f64::from_bits(
        (u64::from(sign) << 32)
            | (u64::from((magnitude >> 23) + 896) << 52)
            | (u64::from(magnitude & 0x007f_ffff) << 29),
    );
    // Mills' bound gives |GELU(-16)| < exp(-128)/sqrt(2*pi) < 2^-150.
    // The corresponding positive correction is below a binary32 half-ULP.
    if x >= 16.0 {
        return bits;
    }
    if x <= -16.0 {
        return 0x8000_0000;
    }
    // SAFETY: Darwin libm erfc accepts every finite binary64 argument. Keeping
    // the complementary error function avoids cancellation in 1+erf(x).
    narrow((0.5 * x) * unsafe { erfc(-x / std::f64::consts::SQRT_2) })
}

pub(super) fn evaluate(bytes: &[u8]) -> Result<Vec<u8>, &'static str> {
    let (words, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err("typed GELU input length is not a multiple of Float32 width");
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(bytes.len())
        .map_err(|_| "typed GELU allocation failed")?;
    for word in words {
        output.extend_from_slice(&gelu(u32::from_ne_bytes(*word)).to_ne_bytes());
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independently evaluated with Decimal at 140 and 180 digits and checked
    // against mpmath erfc. These are local raw-bit regressions; nonfinite and
    // signed-zero policy is not an additional upstream WPT requirement.
    const CASES: &[(u32, u32)] = &[
        (0xc1200000, 0x9ab83c9b), // -10: native cancellation returned -0.
        (0xc0a00000, 0xb5c05e5d), // -5: native error exceeds 18 ULP.
        (0x00000001, 0x00000001),
        (0x80000001, 0x80000000),
        (0x007fffff, 0x00400000),
        (0x807fffff, 0x803fffff),
        (0x00000000, 0x00000000),
        (0x80000000, 0x80000000),
        (0x41800000, 0x41800000),
        (0xc1800000, 0x80000000),
        (0x7f7fffff, 0x7f7fffff),
        (0xff7fffff, 0x80000000),
        (0x7f800000, 0x7f800000),
        (0xff800000, 0x7fc00000),
        (0x7f800001, 0x7fc00000),
        (0xffc00001, 0x7fc00000),
    ];

    #[test]
    fn gelu_tail_and_rounding_use_independent_raw_bit_references() {
        for &(input, expected) in CASES {
            assert_eq!(gelu(input), expected, "input {input:08x}");
        }
        assert!(evaluate(&[0; 3]).is_err());
        assert!(evaluate(&[]).unwrap().is_empty());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn gelu_is_independent_of_ambient_flush_mode() {
        struct RestoreFpcr(u64);
        impl Drop for RestoreFpcr {
            fn drop(&mut self) {
                // SAFETY: Restore only this thread's original register, also
                // on assertion unwind. Production never changes FPCR.
                unsafe { std::arch::asm!("msr fpcr, {}", in(reg) self.0) };
            }
        }
        let saved: u64;
        // SAFETY: These test-only instructions operate on the current thread.
        unsafe { std::arch::asm!("mrs {}, fpcr", out(reg) saved) };
        let _restore = RestoreFpcr(saved);
        for mode in [
            saved & !((1 << 24) | (1 << 19)),
            saved | (1 << 24) | (1 << 19),
        ] {
            unsafe { std::arch::asm!("msr fpcr, {}", in(reg) mode) };
            let observed: u64;
            unsafe { std::arch::asm!("mrs {}, fpcr", out(reg) observed) };
            assert_eq!(observed, mode);
            for &(input, expected) in CASES {
                assert_eq!(
                    gelu(std::hint::black_box(input)),
                    expected,
                    "input {input:08x}"
                );
            }
        }
    }
}
