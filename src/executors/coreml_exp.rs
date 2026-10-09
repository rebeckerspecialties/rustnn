//! Float32 Exp with bit-preserving interchange and binary64 evaluation.

use super::binary32::{narrow, widen};

unsafe extern "C" {
    fn exp(value: f64) -> f64;
}

fn exp_bits(bits: u32) -> u32 {
    match bits & 0x7fff_ffff {
        0 => 0x3f80_0000,
        0x7f80_0000 => {
            if bits >> 31 == 0 {
                0x7f80_0000
            } else {
                0
            }
        }
        magnitude if magnitude > 0x7f80_0000 => bits | 0x0040_0000,
        _ => {
            // SAFETY: Darwin libm exp accepts every finite binary64 argument.
            // All values that could yield a nonzero finite Float32 result
            // are normal in binary64, so ambient FTZ cannot erase that tail.
            narrow(unsafe { exp(widen(bits)) })
        }
    }
}

pub(super) fn evaluate(bytes: &[u8]) -> Result<Vec<u8>, &'static str> {
    let (values, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err("typed Float32 Exp input has an incomplete element");
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(bytes.len())
        .map_err(|_| "typed Float32 Exp allocation failed")?;
    for value in values {
        output.extend_from_slice(&exp_bits(u32::from_ne_bytes(*value)).to_ne_bytes());
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Decimal140/180 evaluated at exact Float32 inputs; subnormal outputs rounded
    // by exact distance in units of 2^-149. These raw-bit/special-value checks
    // are local fidelity regressions, not a new strict-rounding WPT budget.
    const CASES: &[(u32, u32)] = &[
        (0xc2d0_0000, 0),
        (0xc2ce_0000, 1),
        (0xc2c8_0000, 27),
        (0xc2be_0000, 3940),
        (0xc2b4_0000, 584744),
        (0xc2b0_0000, 4320708),
        (0xbf80_0000, 0x3ebc_5ab2),
        (0x3f80_0000, 0x402d_f854),
        (0, 0x3f80_0000),
        (0x8000_0000, 0x3f80_0000),
        (1, 0x3f80_0000),
        (0x8000_0001, 0x3f80_0000),
        (0x007f_ffff, 0x3f80_0000),
        (0x807f_ffff, 0x3f80_0000),
        (0x7f7f_ffff, 0x7f80_0000),
        (0xff7f_ffff, 0),
        (0x7f80_0000, 0x7f80_0000),
        (0xff80_0000, 0),
        (0x7f80_0001, 0x7fc0_0001),
        (0xff81_2345, 0xffc1_2345),
    ];

    #[test]
    fn exp_special_values_and_representable_tails_use_independent_bits() {
        for &(input, expected) in CASES {
            assert_eq!(exp_bits(input), expected, "input={input:08x}");
        }
        assert!(evaluate(&[1, 2, 3]).is_err());
        assert!(evaluate(&[]).unwrap().is_empty());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn exp_is_independent_of_ambient_flush_mode() {
        struct RestoreFpcr(u64);
        impl Drop for RestoreFpcr {
            fn drop(&mut self) {
                // SAFETY: Restore this test thread's register, including on unwind.
                unsafe { std::arch::asm!("msr fpcr, {}", in(reg) self.0) };
            }
        }
        let saved: u64;
        // SAFETY: Test-only current-thread register access. Production never
        // changes the floating-point environment.
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
                    exp_bits(std::hint::black_box(input)),
                    expected,
                    "input={input:08x}"
                );
            }
        }
    }
}
