//! Optional certified F64 interval fast path, kept separate from the safe exact
//! kernel while integration is being evaluated. No native float-width casts.

use super::{Error, Options, TensorView, exact_dot_validated, layout};
#[cfg(test)]
use super::{Tensor, exact_dot};

const SCRATCH_LIMIT: usize = 65536;

/// Per-executor reusable scratch; dynamic operands are never duplicated.
pub struct Scratch {
    wide: Vec<f64>,
    byte_limit: usize,
}

impl Default for Scratch {
    fn default() -> Self {
        Self::with_byte_limit(SCRATCH_LIMIT)
    }
}

impl Scratch {
    pub fn with_byte_limit(limit: usize) -> Self {
        Self {
            wide: Vec::new(),
            byte_limit: limit.min(SCRATCH_LIMIT),
        }
    }
    pub fn allocated_bytes(&self) -> usize {
        self.wide.capacity() * 8
    }
}

#[derive(Clone, Debug)]
pub struct Report {
    pub shape: Vec<usize>,
    pub integer_fallbacks: usize,
    pub checked_outputs: usize,
    pub scratch_bytes: usize,
}

/// Generic rank>=2, broadcasting, strided and transposed operands. A row is
/// widened once; RHS columns are read directly with validated strides. This
/// neither transposes the whole RHS nor makes a second learned-weight copy.
/// The caller owns output storage and may reuse both it and Scratch per call.
pub fn matmul_into(
    a: &TensorView<'_>,
    b: &TensorView<'_>,
    options: Options,
    output: &mut [f32],
    scratch: &mut Scratch,
) -> Result<Report, Error> {
    let layout = layout(a, b, options)?;
    if output.len() != layout.elements {
        return Err(Error::OutputStorageMismatch {
            expected: layout.elements,
            actual: output.len(),
        });
    }
    let eligible = layout.k <= scratch.byte_limit / 8
        && nearest_even_environment()
        && scratch
            .wide
            .try_reserve_exact(layout.k.saturating_sub(scratch.wide.len()))
            .is_ok();
    let mut fallbacks = 0;
    let mut at = 0;
    for batch in 0..layout.batches {
        let (a_base, b_base) = layout.offsets(a, b, batch);
        for row in 0..layout.m {
            let a_start = a_base + row * a.strides[layout.am];
            scratch.wide.clear();
            let mut finite_a = eligible;
            if eligible {
                for inner in 0..layout.k {
                    let raw = a.data[a_start + inner * a.strides[layout.ak]].to_bits();
                    if (raw >> 23) & 255 == 255 {
                        finite_a = false;
                        break;
                    }
                    scratch.wide.push(widen(raw));
                }
            }
            for column in 0..layout.n {
                let b_start = b_base + column * b.strides[layout.bn];
                let mut certified = None;
                if finite_a {
                    let mut sum = 0.0_f64;
                    let mut absolute = 0.0_f64;
                    let mut finite_b = true;
                    for inner in 0..layout.k {
                        let raw = b.data[b_start + inner * b.strides[layout.bk]].to_bits();
                        if (raw >> 23) & 255 == 255 {
                            finite_b = false;
                            break;
                        }
                        let product = scratch.wide[inner] * widen(raw);
                        sum += product;
                        absolute += product.abs();
                    }
                    if finite_b && absolute == 0.0 {
                        certified = Some(0.0);
                    } else if finite_b {
                        let bound = (4.0 * layout.k as f64 * f64::EPSILON) * absolute;
                        let low = narrow_rne((sum - bound).next_down());
                        let high = narrow_rne((sum + bound).next_up());
                        if low == high {
                            certified = Some(f32::from_bits(low));
                        }
                    }
                }
                output[at] = if let Some(value) = certified {
                    value
                } else {
                    fallbacks += 1;
                    exact_dot_validated((0..layout.k).map(|inner| {
                        (
                            a.data[a_start + inner * a.strides[layout.ak]],
                            b.data[b_start + inner * b.strides[layout.bk]],
                        )
                    }))
                };
                at += 1;
            }
        }
    }
    Ok(Report {
        shape: layout.shape,
        integer_fallbacks: fallbacks,
        checked_outputs: output.len(),
        scratch_bytes: scratch.allocated_bytes(),
    })
}

/// Allocating convenience wrapper; hot paths should prefer matmul_into.
#[cfg(test)]
pub fn matmul(
    a: &TensorView<'_>,
    b: &TensorView<'_>,
    options: Options,
) -> Result<(Tensor, Report), Error> {
    let layout = layout(a, b, options)?;
    let mut data = Vec::new();
    data.try_reserve_exact(layout.elements)
        .map_err(|_| Error::AllocationFailed)?;
    data.resize(layout.elements, 0.0);
    let mut scratch = Scratch::default();
    let report = matmul_into(a, b, options, &mut data, &mut scratch)?;
    Ok((
        Tensor {
            shape: layout.shape,
            data,
        },
        report,
    ))
}

pub fn nearest_even_environment() -> bool {
    let one = std::hint::black_box(f64::from_bits(0x3ff0000000000000));
    let half = std::hint::black_box(f64::from_bits(0x3ca0000000000000));
    let three_halves = std::hint::black_box(f64::from_bits(0x3cb8000000000000));
    (one + half).to_bits() == 0x3ff0000000000000
        && (-one - half).to_bits() == 0xbff0000000000000
        && (one + three_halves).to_bits() == 0x3ff0000000000002
}

fn widen(raw: u32) -> f64 {
    let sign = u64::from(raw >> 31) << 63;
    let exponent = (raw >> 23) & 255;
    let fraction = raw & 0x7fffff;
    let output = if exponent != 0 {
        sign | (u64::from(exponent + 896) << 52) | (u64::from(fraction) << 29)
    } else if fraction == 0 {
        sign
    } else {
        let leading = 31 - fraction.leading_zeros();
        sign | (u64::from(leading + 874) << 52)
            | (u64::from(fraction - (1 << leading)) << (52 - leading))
    };
    f64::from_bits(output)
}

fn narrow_rne(input: f64) -> u32 {
    let raw = input.to_bits();
    let sign = ((raw >> 63) as u32) << 31;
    let biased = ((raw >> 52) & 2047) as u32;
    let fraction = raw & 0xfffffffffffff;
    if biased == 2047 {
        return sign
            | if fraction == 0 {
                0x7f800000
            } else {
                0x7fc00000
            };
    }
    if biased == 0 {
        return sign;
    }
    let mut exponent = biased as i32 - 1023;
    if exponent < -150 {
        return sign;
    }
    let mantissa = fraction | (1 << 52);
    let shift = if exponent < -126 { -exponent - 97 } else { 29 };
    let mut kept = (mantissa >> shift) as u32;
    let discarded = mantissa & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    if discarded > half || (discarded == half && kept & 1 != 0) {
        kept += 1;
    }
    if exponent < -126 {
        return sign | kept;
    }
    if kept >= 1 << 24 {
        kept >>= 1;
        exponent += 1;
    }
    if exponent > 127 {
        return sign | 0x7f800000;
    }
    sign | (((exponent + 127) as u32) << 23) | (kept & 0x7fffff)
}

/// Full-original-width row-major weights, unchanged F32 bits. Dynamic input
/// row scratch is at most64 KiB. The result is certified exact F32 RNE, with
/// integer-bin fallback when an interval cannot establish the unique rounding.
#[cfg(test)]
pub fn projection(
    hidden: &[f32],
    weights: &[f32],
    rows: usize,
) -> Result<(Vec<f32>, usize), Error> {
    let inner = hidden.len();
    if inner == 0 {
        return Err(Error::ZeroDimension { axis: 1 });
    }
    if rows == 0 {
        return Err(Error::ZeroDimension { axis: 0 });
    }
    if inner > i32::MAX as usize {
        return Err(Error::InnerTooLarge);
    }
    let expected = inner.checked_mul(rows).ok_or(Error::SizeOverflow)?;
    if weights.len() != expected {
        return Err(Error::ContiguousStorageMismatch {
            expected,
            actual: weights.len(),
        });
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(rows)
        .map_err(|_| Error::AllocationFailed)?;
    let mut wide = Vec::new();
    let eligible = inner <= SCRATCH_LIMIT / 8
        && nearest_even_environment()
        && hidden.iter().all(|x| x.is_finite())
        && wide.try_reserve_exact(inner).is_ok();
    if eligible {
        wide.extend(hidden.iter().map(|x| widen(x.to_bits())));
    }
    let mut fallbacks = 0;
    for row in weights.chunks_exact(inner) {
        if eligible {
            let mut sum = 0.0_f64;
            let mut absolute = 0.0_f64;
            let mut finite = true;
            for (&a, &b) in wide.iter().zip(row) {
                let raw = b.to_bits();
                if (raw >> 23) & 255 == 255 {
                    finite = false;
                    break;
                }
                let product = a * widen(raw);
                sum += product;
                absolute += product.abs();
            }
            if finite && absolute == 0.0 {
                output.push(0.0);
                continue;
            }
            if finite {
                let bound = (4.0 * inner as f64 * f64::EPSILON) * absolute;
                let low = narrow_rne((sum - bound).next_down());
                let high = narrow_rne((sum + bound).next_up());
                if low == high {
                    output.push(f32::from_bits(low));
                    continue;
                }
            }
        }
        output.push(exact_dot(hidden.iter().copied().zip(row.iter().copied()))?);
        fallbacks += 1;
    }
    Ok((output, fallbacks))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn software_conversions_keep_subnormals_and_ties() {
        assert_eq!(widen(1).to_bits(), (874_u64) << 52);
        assert_eq!(narrow_rne(f64::from_bits((874_u64) << 52)), 1);
        assert_eq!(narrow_rne(1.0 + 2_f64.powi(-24)), 1.0_f32.to_bits());
        assert_eq!(
            narrow_rne(1.0 + 2_f64.powi(-24) + 2_f64.powi(-50)),
            1.0_f32.to_bits() + 1
        );
    }

    #[test]
    fn cancellation_is_not_mistaken_for_certificate() {
        let (out, fallbacks) = projection(
            &[1.0; 5],
            &[2_f32.powi(60), 1.0, 2_f32.powi(-24), -2_f32.powi(60), -1.0],
            1,
        )
        .unwrap();
        assert_eq!(out, [2_f32.powi(-24)]);
        assert_eq!(fallbacks, 1);
    }

    #[test]
    fn exact_products_with_normal_sum_are_certified() {
        let (out, fallbacks) = projection(&[1.0, 2.0], &[3.0, 4.0], 1).unwrap();
        assert_eq!(out, [11.0]);
        assert_eq!(fallbacks, 0);
    }

    #[test]
    fn generic_into_matches_bins_with_transpose_and_batch_broadcast() {
        let av = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bv = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let a = TensorView::contiguous(&av, &[2, 3, 1]).unwrap();
        let b = TensorView::contiguous(&bv, &[1, 2, 3]).unwrap();
        let options = Options {
            transpose_a: true,
            transpose_b: true,
        };
        let expected = super::super::matmul(&a, &b, options).unwrap();
        assert_eq!(
            super::super::matmul_output_shape(&a, &b, options).unwrap(),
            expected.shape
        );
        let mut output = [99.0; 4];
        let mut scratch = Scratch::default();
        let report = matmul_into(&a, &b, options, &mut output, &mut scratch).unwrap();
        assert_eq!(output, expected.data.as_slice());
        assert_eq!(report.shape, expected.shape);
        assert!(report.scratch_bytes <= SCRATCH_LIMIT);
        let allocation = scratch.allocated_bytes();
        matmul_into(&a, &b, options, &mut output, &mut scratch).unwrap();
        assert_eq!(scratch.allocated_bytes(), allocation);
    }

    #[test]
    fn strided_rhs_is_not_materialized_or_reordered() {
        let av = [1.0, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0];
        let bv = [1.0, 99.0, 0.0, 99.0, 99.0, 0.0, 99.0, 1.0];
        let a = TensorView::strided(&av, &[2, 2], &[4, 2]).unwrap();
        let b = TensorView::strided(&bv, &[2, 2], &[5, 2]).unwrap();
        let expected = super::super::matmul(&a, &b, Options::default()).unwrap();
        let (actual, report) = matmul(&a, &b, Options::default()).unwrap();
        assert_eq!(actual.data, expected.data);
        assert_eq!(report.checked_outputs, 4);
    }

    #[test]
    fn low_scratch_budget_uses_bins_without_touching_inputs() {
        let av = [1.0; 5];
        let bv = [2_f32.powi(60), 1.0, 2_f32.powi(-24), -2_f32.powi(60), -1.0];
        let a = TensorView::contiguous(&av, &[1, 5]).unwrap();
        let b = TensorView::contiguous(&bv, &[5, 1]).unwrap();
        let mut scratch = Scratch::with_byte_limit(0);
        let mut output = [0.0];
        let report = matmul_into(&a, &b, Options::default(), &mut output, &mut scratch).unwrap();
        assert_eq!(output, [2_f32.powi(-24)]);
        assert_eq!(report.integer_fallbacks, 1);
        assert_eq!(scratch.allocated_bytes(), 0);
        assert_eq!(av, [1.0; 5]);
    }

    #[test]
    fn wrong_output_length_fails_before_writing() {
        let a = TensorView::contiguous(&[1.0; 4], &[2, 2]).unwrap();
        let b = TensorView::contiguous(&[1.0; 4], &[2, 2]).unwrap();
        let mut output = [99.0; 3];
        let error = matmul_into(
            &a,
            &b,
            Options::default(),
            &mut output,
            &mut Scratch::default(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            Error::OutputStorageMismatch {
                expected: 4,
                actual: 3
            }
        );
        assert_eq!(output, [99.0; 3]);
    }
}
