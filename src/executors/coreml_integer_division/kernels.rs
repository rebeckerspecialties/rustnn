//! Exact, checked Int32 division over represented tensor storage.

fn element_count(shape: &[usize]) -> Result<usize, &'static str> {
    shape.iter().try_fold(1usize, |count, &extent| {
        if extent == 0 {
            return Err("Int32 division requires positive tensor extents");
        }
        count
            .checked_mul(extent)
            .ok_or("Int32 division tensor size overflow")
    })
}

fn strides(shape: &[usize], rank: usize) -> Result<Vec<usize>, &'static str> {
    let mut strides = vec![0; rank];
    let mut stride = 1usize;
    for (axis, &extent) in shape.iter().enumerate().rev() {
        if extent != 1 {
            strides[rank - shape.len() + axis] = stride;
        }
        stride = stride
            .checked_mul(extent)
            .ok_or("Int32 division tensor size overflow")?;
    }
    Ok(strides)
}

pub(in super::super) fn evaluate(
    left: &[u8],
    left_shape: &[usize],
    right: &[u8],
    right_shape: &[usize],
) -> Result<(Vec<usize>, Vec<u8>), &'static str> {
    for (bytes, shape) in [(left, left_shape), (right, right_shape)] {
        if element_count(shape)?.checked_mul(4) != Some(bytes.len()) {
            return Err("Int32 division storage does not match its declared shape");
        }
    }
    let rank = left_shape.len().max(right_shape.len());
    let mut shape = vec![1; rank];
    for source in [left_shape, right_shape] {
        for (axis, &extent) in source.iter().enumerate() {
            let result = &mut shape[rank - source.len() + axis];
            if *result != 1 && extent != 1 && *result != extent {
                return Err("Int32 division input shapes do not broadcast");
            }
            *result = (*result).max(extent);
        }
    }
    let count = element_count(&shape)?;
    let length = count
        .checked_mul(4)
        .ok_or("Int32 division output byte length overflow")?;
    let left_strides = strides(left_shape, rank)?;
    let right_strides = strides(right_shape, rank)?;
    // This private result is published only after every quotient succeeds.
    // Division by zero and an unrepresentable quotient have no assigned
    // result here; neither may panic, wrap or partially overwrite an output.
    let mut result = Vec::new();
    result
        .try_reserve_exact(length)
        .map_err(|_| "Int32 division output allocation failed")?;
    for index in 0..count {
        let mut remaining = index;
        let mut a_index = 0;
        let mut b_index = 0;
        for axis in (0..rank).rev() {
            let coordinate = remaining % shape[axis];
            remaining /= shape[axis];
            a_index += coordinate * left_strides[axis];
            b_index += coordinate * right_strides[axis];
        }
        let a = i32::from_le_bytes(left[a_index * 4..a_index * 4 + 4].try_into().unwrap());
        let b = i32::from_le_bytes(right[b_index * 4..b_index * 4 + 4].try_into().unwrap());
        if b == 0 {
            return Err("Int32 division by zero has no supported result");
        }
        let quotient = a
            .checked_div(b)
            .ok_or("Int32 division quotient is outside the represented range")?;
        result.extend_from_slice(&quotient.to_le_bytes());
    }
    Ok((shape, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(values: &[i32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn values(bytes: &[u8]) -> Vec<i32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|value| i32::from_le_bytes(*value))
            .collect()
    }

    #[test]
    fn signed_boundary_grid_matches_widened_reference() {
        let cases = [
            i32::MIN,
            i32::MIN + 1,
            -16_777_217,
            -7,
            -3,
            -1,
            0,
            1,
            2,
            3,
            7,
            16_777_217,
            i32::MAX - 1,
            i32::MAX,
        ];
        for a in cases {
            for b in cases {
                let result = evaluate(&bytes(&[a]), &[], &bytes(&[b]), &[]);
                if b == 0 || (a == i32::MIN && b == -1) {
                    assert!(result.is_err(), "{a} / {b}");
                } else {
                    let (shape, result) = result.unwrap();
                    assert!(shape.is_empty());
                    assert_eq!(values(&result), [(i64::from(a) / i64::from(b)) as i32]);
                }
            }
        }
    }

    #[test]
    fn floor_remainder_correction_matches_truncation_for_representable_quotients() {
        let mut state = 0x8320_59a3_u32;
        for _ in 0..100_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let a = i64::from(state as i32);
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let b = i64::from(state as i32);
            if b == 0 || (a == i64::from(i32::MIN) && b == -1) {
                continue;
            }
            let truncation = a / b;
            let remainder = a % b;
            let floor = truncation - i64::from(remainder != 0 && (a < 0) != (b < 0));
            let corrected = floor + i64::from(floor < 0 && remainder != 0);
            assert_eq!(corrected, truncation);
            let (_, actual) =
                evaluate(&bytes(&[a as i32]), &[1], &bytes(&[b as i32]), &[]).unwrap();
            assert_eq!(values(&actual), [truncation as i32]);
        }
    }

    #[test]
    fn multidirectional_broadcast_and_singletons_preserve_integer_bits() {
        let (shape, output) = evaluate(
            &bytes(&[16_777_217, -16_777_217]),
            &[2, 1],
            &bytes(&[1, -1, 3]),
            &[1, 3],
        )
        .unwrap();
        assert_eq!(shape, [2, 3]);
        assert_eq!(
            values(&output),
            [
                16_777_217,
                -16_777_217,
                5_592_405,
                -16_777_217,
                16_777_217,
                -5_592_405
            ]
        );
    }

    #[test]
    fn invalid_storage_shapes_and_late_undefined_quotients_are_errors() {
        let valid = bytes(&[1, 2]);
        for (left, left_shape, right, right_shape) in [
            (valid.as_slice(), vec![3], valid.as_slice(), vec![2]),
            (valid.as_slice(), vec![0], valid.as_slice(), vec![2]),
        ] {
            assert!(evaluate(left, &left_shape, right, &right_shape).is_err());
        }
        assert!(evaluate(&valid, &[2], &bytes(&[1, 2, 3]), &[3]).is_err());
        assert!(evaluate(&valid, &[2], &bytes(&[1, 0]), &[2]).is_err());
        assert!(evaluate(&bytes(&[1, i32::MIN]), &[2], &bytes(&[1, -1]), &[2]).is_err());
        assert!(element_count(&[usize::MAX, 2]).is_err());
    }
}
