#![forbid(unsafe_code)]

#[path = "certified.rs"]
pub mod certified;

use std::fmt;

const BINS: usize = 40;
const BASE: i64 = 1 << 16;

/// Exact dyadic products/sum, then one direct binary32 round-to-nearest-even.
/// The finite accumulator uses 320 bytes and handles all binary32 exponents.
/// Exact mathematical zero is canonical +0; negative underflow preserves -0.
/// Nonfinite products follow usual Inf/NaN rules, with canonical quiet NaN.
#[cfg(test)]
pub fn exact_dot<I>(pairs: I) -> Result<f32, Error>
where
    I: IntoIterator<Item = (f32, f32)>,
    I::IntoIter: ExactSizeIterator,
{
    let pairs = pairs.into_iter();
    if pairs.len() > i32::MAX as usize {
        return Err(Error::InnerTooLarge);
    }
    Ok(exact_dot_validated(pairs))
}

fn exact_dot_validated(pairs: impl IntoIterator<Item = (f32, f32)>) -> f32 {
    let mut accumulator = [0_i64; BINS];
    let mut positive_inf = false;
    let mut negative_inf = false;
    for (a, b) in pairs {
        let aa = a.to_bits();
        let bb = b.to_bits();
        let ae = (aa >> 23) & 255;
        let be = (bb >> 23) & 255;
        let mut am = aa & 0x7fffff;
        let mut bm = bb & 0x7fffff;
        if (ae == 255 && am != 0) || (be == 255 && bm != 0) {
            return f32::from_bits(0x7fc00000);
        }
        let negative = (aa ^ bb) >> 31 != 0;
        if ae == 255 || be == 255 {
            if (ae == 0 && am == 0) || (be == 0 && bm == 0) {
                return f32::from_bits(0x7fc00000);
            }
            if negative {
                negative_inf = true;
            } else {
                positive_inf = true;
            }
            continue;
        }
        if ae != 0 {
            am |= 1 << 23;
        }
        if be != 0 {
            bm |= 1 << 23;
        }
        if am == 0 || bm == 0 {
            continue;
        }
        let ax = if ae != 0 { ae as i32 - 150 } else { -149 };
        let bx = if be != 0 { be as i32 - 150 } else { -149 };
        let shift = (ax + bx + 298) as usize;
        let product = u64::from(am) * u64::from(bm);
        let mut fragment = product << (shift % 16);
        let mut bin = shift / 16;
        while fragment != 0 {
            let part = (fragment & 65535) as i64;
            accumulator[bin] += if negative { -part } else { part };
            bin += 1;
            fragment >>= 16;
        }
    }
    if positive_inf && negative_inf {
        return f32::from_bits(0x7fc00000);
    }
    if positive_inf {
        return f32::INFINITY;
    }
    if negative_inf {
        return f32::NEG_INFINITY;
    }
    rounded(&mut accumulator)
}

fn normalize(limbs: &mut [i64; BINS]) {
    for i in 0..BINS - 1 {
        let q = limbs[i].div_euclid(BASE);
        let r = limbs[i].rem_euclid(BASE);
        limbs[i] = r;
        limbs[i + 1] += q;
    }
}

fn bit_at(limbs: &[i64; BINS], bit: usize) -> u32 {
    if bit >= BINS * 16 {
        return 0;
    }
    ((limbs[bit / 16] as u32) >> (bit % 16)) & 1
}

fn any_below(limbs: &[i64; BINS], count: usize) -> bool {
    let whole = count / 16;
    limbs[..whole].iter().any(|&x| x != 0)
        || (!count.is_multiple_of(16) && (limbs[whole] as u32 & ((1 << (count % 16)) - 1)) != 0)
}

fn rounded(limbs: &mut [i64; BINS]) -> f32 {
    normalize(limbs);
    let negative = limbs[BINS - 1] < 0;
    if negative {
        for limb in &mut limbs[..BINS - 1] {
            *limb = BASE - 1 - *limb;
        }
        limbs[BINS - 1] = -1 - limbs[BINS - 1];
        limbs[0] += 1;
        normalize(limbs);
    }
    let Some(highest) = limbs.iter().enumerate().rev().find_map(|(i, &value)| {
        (value != 0).then(|| i * 16 + (63 - value.leading_zeros()) as usize)
    }) else {
        return 0.0;
    };
    let mut exponent = highest as i32 - 298;
    let shift = if exponent < -126 { 149 } else { highest - 23 };
    let mut significand = 0_u32;
    for i in 0..24 {
        significand |= bit_at(limbs, shift + i) << i;
    }
    let guard = bit_at(limbs, shift - 1) != 0;
    let sticky = any_below(limbs, shift - 1);
    if guard && (sticky || significand & 1 != 0) {
        significand += 1;
    }
    let sign = u32::from(negative) << 31;
    if exponent < -126 {
        return f32::from_bits(sign | significand);
    }
    if significand >= 1 << 24 {
        significand >>= 1;
        exponent += 1;
    }
    if exponent > 127 {
        return f32::from_bits(sign | 0x7f800000);
    }
    f32::from_bits(sign | (((exponent + 127) as u32) << 23) | (significand & 0x7fffff))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    RankBelowTwo,
    RankStrideMismatch,
    ZeroDimension {
        axis: usize,
    },
    ZeroStride {
        axis: usize,
    },
    SizeOverflow,
    StorageTooShort {
        needed: usize,
        actual: usize,
    },
    #[cfg(test)]
    ContiguousStorageMismatch {
        expected: usize,
        actual: usize,
    },
    #[cfg(test)]
    ByteLengthMismatch {
        expected: usize,
        actual: usize,
    },
    InnerMismatch {
        a: usize,
        b: usize,
    },
    BroadcastMismatch {
        axis: usize,
        a: usize,
        b: usize,
    },
    InnerTooLarge,
    #[cfg(test)]
    AllocationFailed,
    OutputStorageMismatch {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "exact matmul validation failed: {self:?}")
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Debug)]
pub struct TensorView<'a> {
    data: &'a [f32],
    shape: Vec<usize>,
    strides: Vec<usize>,
}

/// Validate operands/options and return the effective broadcast output shape.
/// This permits callers to reject a mismatched backing before touching it.
#[cfg(test)]
pub fn matmul_output_shape(
    a: &TensorView<'_>,
    b: &TensorView<'_>,
    options: Options,
) -> Result<Vec<usize>, Error> {
    Ok(layout(a, b, options)?.shape)
}

fn checked_product(shape: &[usize]) -> Result<usize, Error> {
    shape
        .iter()
        .enumerate()
        .try_fold(1_usize, |n, (axis, &dim)| {
            if dim == 0 {
                return Err(Error::ZeroDimension { axis });
            }
            n.checked_mul(dim).ok_or(Error::SizeOverflow)
        })
}

impl<'a> TensorView<'a> {
    /// Row-major contiguous storage, with exact shape/element-count validation.
    #[cfg(test)]
    pub fn contiguous(data: &'a [f32], shape: &[usize]) -> Result<Self, Error> {
        let expected = checked_product(shape)?;
        if data.len() != expected {
            return Err(Error::ContiguousStorageMismatch {
                expected,
                actual: data.len(),
            });
        }
        let mut stride = 1_usize;
        let mut strides = vec![0; shape.len()];
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride = stride.checked_mul(shape[i]).ok_or(Error::SizeOverflow)?;
        }
        Self::strided(data, shape, &strides)
    }

    /// Nonzero read-only strides; storage span checked before any arithmetic.
    /// Unit dimensions may have zero strides. General overlapping read views
    /// are allowed, but zero-stride repeated dimensions must be explicit shape-1
    /// batch broadcasting rather than an accidentally missing layout stride.
    pub fn strided(data: &'a [f32], shape: &[usize], strides: &[usize]) -> Result<Self, Error> {
        if shape.len() < 2 {
            return Err(Error::RankBelowTwo);
        }
        if shape.len() != strides.len() {
            return Err(Error::RankStrideMismatch);
        }
        checked_product(shape)?;
        let mut largest = 0_usize;
        for (axis, (&dim, &stride)) in shape.iter().zip(strides).enumerate() {
            if stride == 0 && dim > 1 {
                return Err(Error::ZeroStride { axis });
            }
            largest = largest
                .checked_add((dim - 1).checked_mul(stride).ok_or(Error::SizeOverflow)?)
                .ok_or(Error::SizeOverflow)?;
        }
        let needed = largest.checked_add(1).ok_or(Error::SizeOverflow)?;
        if data.len() < needed {
            return Err(Error::StorageTooShort {
                needed,
                actual: data.len(),
            });
        }
        Ok(Self {
            data,
            shape: shape.to_vec(),
            strides: strides.to_vec(),
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    pub transpose_a: bool,
    pub transpose_b: bool,
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Strict little-endian F32 byte payload validation; no alignment casts/unsafe.
#[cfg(test)]
pub fn decode_f32_le(bytes: &[u8], shape: &[usize]) -> Result<Vec<f32>, Error> {
    let expected = checked_product(shape)?
        .checked_mul(4)
        .ok_or(Error::SizeOverflow)?;
    if bytes.len() != expected {
        return Err(Error::ByteLengthMismatch {
            expected,
            actual: bytes.len(),
        });
    }
    let mut data = Vec::new();
    data.try_reserve_exact(expected / 4)
        .map_err(|_| Error::AllocationFailed)?;
    for chunk in bytes.as_chunks::<4>().0 {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(data)
}

struct Layout {
    shape: Vec<usize>,
    batches: usize,
    elements: usize,
    m: usize,
    k: usize,
    n: usize,
    am: usize,
    ak: usize,
    bk: usize,
    bn: usize,
}

fn layout(a: &TensorView<'_>, b: &TensorView<'_>, options: Options) -> Result<Layout, Error> {
    let ar = a.shape.len();
    let br = b.shape.len();
    let (am, ak) = if options.transpose_a {
        (ar - 1, ar - 2)
    } else {
        (ar - 2, ar - 1)
    };
    let (bk, bn) = if options.transpose_b {
        (br - 1, br - 2)
    } else {
        (br - 2, br - 1)
    };
    let m = a.shape[am];
    let k = a.shape[ak];
    let n = b.shape[bn];
    if k != b.shape[bk] {
        return Err(Error::InnerMismatch {
            a: k,
            b: b.shape[bk],
        });
    }
    if k > i32::MAX as usize {
        return Err(Error::InnerTooLarge);
    }
    let batch_rank = (ar - 2).max(br - 2);
    let mut shape = Vec::with_capacity(batch_rank + 2);
    for axis in 0..batch_rank {
        let adim = if axis >= batch_rank - (ar - 2) {
            a.shape[axis - (batch_rank - (ar - 2))]
        } else {
            1
        };
        let bdim = if axis >= batch_rank - (br - 2) {
            b.shape[axis - (batch_rank - (br - 2))]
        } else {
            1
        };
        if adim != bdim && adim != 1 && bdim != 1 {
            return Err(Error::BroadcastMismatch {
                axis,
                a: adim,
                b: bdim,
            });
        }
        shape.push(adim.max(bdim));
    }
    let batches = checked_product(&shape)?;
    shape.extend([m, n]);
    let elements = checked_product(&shape)?;
    Ok(Layout {
        shape,
        batches,
        elements,
        m,
        k,
        n,
        am,
        ak,
        bk,
        bn,
    })
}

impl Layout {
    fn offsets(&self, a: &TensorView<'_>, b: &TensorView<'_>, batch: usize) -> (usize, usize) {
        let ar = a.shape.len();
        let br = b.shape.len();
        let batch_rank = self.shape.len() - 2;
        let mut remainder = batch;
        let mut a_base = 0;
        let mut b_base = 0;
        for axis in (0..batch_rank).rev() {
            let coordinate = remainder % self.shape[axis];
            remainder /= self.shape[axis];
            if axis >= batch_rank - (ar - 2) {
                let ai = axis - (batch_rank - (ar - 2));
                if a.shape[ai] != 1 {
                    a_base += coordinate * a.strides[ai];
                }
            }
            if axis >= batch_rank - (br - 2) {
                let bi = axis - (batch_rank - (br - 2));
                if b.shape[bi] != 1 {
                    b_base += coordinate * b.strides[bi];
                }
            }
        }
        (a_base, b_base)
    }
}

/// Exact mathematical F32 matmul for rank>=2, with batch broadcasting and
/// transpose options. K is bounded at i32::MAX for proven accumulator bounds.
#[cfg(test)]
pub fn matmul(a: &TensorView<'_>, b: &TensorView<'_>, options: Options) -> Result<Tensor, Error> {
    let layout = layout(a, b, options)?;
    let mut data = Vec::new();
    data.try_reserve_exact(layout.elements)
        .map_err(|_| Error::AllocationFailed)?;
    for batch in 0..layout.batches {
        let (a_base, b_base) = layout.offsets(a, b, batch);
        for row in 0..layout.m {
            for column in 0..layout.n {
                let a_start = a_base + row * a.strides[layout.am];
                let b_start = b_base + column * b.strides[layout.bn];
                let value = exact_dot_validated((0..layout.k).map(|inner| {
                    (
                        a.data[a_start + inner * a.strides[layout.ak]],
                        b.data[b_start + inner * b.strides[layout.bk]],
                    )
                }));
                data.push(value);
            }
        }
    }
    Ok(Tensor {
        shape: layout.shape,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        exact_dot(a.iter().copied().zip(b.iter().copied())).unwrap()
    }

    #[test]
    fn cancellation_keeps_tiny_residue() {
        assert_eq!(
            dot(
                &[1.0; 5],
                &[2_f32.powi(60), 1.0, 2_f32.powi(-24), -2_f32.powi(60), -1.0]
            ),
            2_f32.powi(-24)
        );
    }

    #[test]
    fn products_keep_rounding_tail() {
        assert_eq!(
            dot(
                &[1.0 + 2_f32.powi(-23), 1.0],
                &[1.0 - 2_f32.powi(-23), -1.0]
            ),
            -2_f32.powi(-46)
        );
    }

    #[test]
    fn overflowing_products_may_cancel_to_finite_sum() {
        assert_eq!(
            dot(&[f32::MAX, f32::MAX, 1.0], &[f32::MAX, -f32::MAX, 1.0]),
            1.0
        );
    }

    #[test]
    fn rounding_and_signed_zero() {
        assert_eq!(dot(&[1.0, 2_f32.powi(-24)], &[1.0; 2]), 1.0);
        assert_eq!(
            dot(&[1.0, 2_f32.powi(-24), 2_f32.powi(-50)], &[1.0; 3]).to_bits(),
            1.0_f32.to_bits() + 1
        );
        assert_eq!(
            dot(&[-2_f32.powi(-74)], &[2_f32.powi(-76)]).to_bits(),
            0x80000000
        );
        assert_eq!(dot(&[-0.0], &[1.0]).to_bits(), 0);
        assert_eq!(dot(&[f32::from_bits(1)], &[1.0]).to_bits(), 1);
    }

    #[test]
    fn special_products_follow_defined_policy() {
        assert!(dot(&[f32::INFINITY], &[0.0]).is_nan());
        assert!(dot(&[f32::NAN], &[1.0]).is_nan());
        assert!(dot(&[f32::INFINITY, f32::NEG_INFINITY], &[1.0; 2]).is_nan());
        assert_eq!(dot(&[f32::INFINITY], &[-1.0]), f32::NEG_INFINITY);
    }

    #[test]
    fn batch_broadcast_and_transpose_preserve_mapping() {
        let adata = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bdata = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let a = TensorView::contiguous(&adata, &[2, 1, 3]).unwrap();
        let b = TensorView::contiguous(&bdata, &[1, 3, 2]).unwrap();
        let result = matmul(&a, &b, Options::default()).unwrap();
        assert_eq!(result.shape, [2, 1, 2]);
        assert_eq!(result.data, [4.0, 5.0, 10.0, 11.0]);
        let a = TensorView::contiguous(&adata, &[2, 3, 1]).unwrap();
        let b_transposed = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let b = TensorView::contiguous(&b_transposed, &[1, 2, 3]).unwrap();
        assert_eq!(
            matmul(
                &a,
                &b,
                Options {
                    transpose_a: true,
                    transpose_b: true
                }
            )
            .unwrap()
            .data,
            result.data
        );
    }

    #[test]
    fn noncontiguous_read_strides_work() {
        let a =
            TensorView::strided(&[1.0, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0], &[2, 2], &[4, 2]).unwrap();
        let b = TensorView::contiguous(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        assert_eq!(
            matmul(&a, &b, Options::default()).unwrap().data,
            [1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn invalid_layouts_fail_before_execution() {
        assert!(matches!(
            TensorView::strided(&[1.0], &[2, 2], &[2, 1]),
            Err(Error::StorageTooShort { .. })
        ));
        assert!(matches!(
            TensorView::strided(&[1.0], &[1, 2], &[0, 0]),
            Err(Error::ZeroStride { axis: 1 })
        ));
        assert_eq!(
            TensorView::contiguous(&[], &[0, 2]).unwrap_err(),
            Error::ZeroDimension { axis: 0 }
        );
        assert_eq!(
            TensorView::strided(&[1.0], &[2, 2], &[usize::MAX, 1]).unwrap_err(),
            Error::SizeOverflow
        );
        assert!(matches!(
            decode_f32_le(&[0, 0, 0], &[1, 1]),
            Err(Error::ByteLengthMismatch { .. })
        ));
        assert!(matches!(
            TensorView::contiguous(&[1.0], &[1, 2]),
            Err(Error::ContiguousStorageMismatch { .. })
        ));
    }

    #[test]
    fn incompatible_inner_and_batch_shapes_fail() {
        let a = TensorView::contiguous(&[1.0; 6], &[2, 1, 3]).unwrap();
        let bad_inner = TensorView::contiguous(&[1.0; 4], &[2, 2]).unwrap();
        assert_eq!(
            matmul(&a, &bad_inner, Options::default()).unwrap_err(),
            Error::InnerMismatch { a: 3, b: 2 }
        );
        let bad_batch = TensorView::contiguous(&[1.0; 9], &[3, 3, 1]).unwrap();
        assert_eq!(
            matmul(&a, &bad_batch, Options::default()).unwrap_err(),
            Error::BroadcastMismatch {
                axis: 0,
                a: 2,
                b: 3
            }
        );
    }

    #[test]
    fn broadcasts_both_batch_axes_without_reordering_data() {
        let a = TensorView::contiguous(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 1, 2]).unwrap();
        let b = TensorView::contiguous(&[1.0, 10.0, 2.0, 20.0, 3.0, 30.0], &[1, 3, 2, 1]).unwrap();
        let result = matmul(&a, &b, Options::default()).unwrap();
        assert_eq!(result.shape, [2, 3, 1, 1]);
        assert_eq!(result.data, [21.0, 42.0, 63.0, 43.0, 86.0, 129.0]);
    }

    #[test]
    fn public_dot_rejects_unbounded_inner_size_without_reading() {
        let pairs = (0..(i32::MAX as usize + 1)).map(|_| (1.0, 1.0));
        assert_eq!(exact_dot(pairs).unwrap_err(), Error::InnerTooLarge);
    }
}
