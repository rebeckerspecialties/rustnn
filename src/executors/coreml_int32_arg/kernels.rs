//! Exact comparisons over represented Int32 values and checked axis geometry.

pub(in super::super) fn evaluate(
    input: &[u8],
    shape: &[usize],
    axis: usize,
    keep_dimensions: bool,
    maximum: bool,
) -> Result<(Vec<usize>, Vec<u8>), &'static str> {
    if axis >= shape.len() || shape.contains(&0) {
        return Err("Int32 argument reduction requires a valid axis and positive extents");
    }
    let count = shape
        .iter()
        .try_fold(1usize, |count, &size| count.checked_mul(size))
        .ok_or("Int32 argument reduction size overflow")?;
    if count.checked_mul(4) != Some(input.len()) || shape[axis] > i32::MAX as usize {
        return Err("Int32 argument reduction storage or index range is invalid");
    }
    let width = shape[axis];
    let inner = shape[axis + 1..].iter().product::<usize>();
    let cells = count / width;
    let mut result = Vec::new();
    result
        .try_reserve_exact(cells * 4)
        .map_err(|_| "Int32 argument reduction allocation failed")?;
    let value =
        |index: usize| i32::from_le_bytes(input[index * 4..index * 4 + 4].try_into().unwrap());
    for cell in 0..cells {
        let start = cell / inner * width * inner + cell % inner;
        let mut selected_index = 0;
        let mut selected = value(start);
        for index in 1..width {
            let current = value(start + index * inner);
            if if maximum {
                current > selected
            } else {
                current < selected
            } {
                selected = current;
                selected_index = index;
            }
        }
        // The first true tie is this implementation's choice, not a WebNN requirement.
        result.extend_from_slice(&(selected_index as i32).to_le_bytes());
    }
    let mut output_shape = shape.to_vec();
    if keep_dimensions {
        output_shape[axis] = 1;
    } else {
        output_shape.remove(axis);
    }
    Ok((output_shape, result))
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

    #[test]
    fn adjacent_extrema_and_axis_layout_are_exact() {
        let input = [
            16_777_216,
            16_777_217,
            i32::MIN,
            i32::MIN + 1,
            i32::MAX,
            i32::MAX - 1,
            -16_777_216,
            -16_777_217,
        ];
        for (axis, maximum, expected) in [
            (0, false, vec![0, 0, 0, 0]),
            (0, true, vec![1, 1, 1, 1]),
            (1, false, vec![1, 1, 1, 1]),
            (1, true, vec![0, 0, 0, 0]),
            (2, false, vec![0, 0, 1, 1]),
            (2, true, vec![1, 1, 0, 0]),
        ] {
            for keep in [false, true] {
                let (shape, result) =
                    evaluate(&bytes(&input), &[2, 2, 2], axis, keep, maximum).unwrap();
                assert_eq!(result, bytes(&expected));
                assert_eq!(
                    shape,
                    if keep {
                        let mut shape = vec![2; 3];
                        shape[axis] = 1;
                        shape
                    } else {
                        vec![2; 2]
                    }
                );
            }
        }
        let (shape, result) = evaluate(&bytes(&[5, 5]), &[2], 0, false, true).unwrap();
        assert!(shape.is_empty());
        assert_eq!(result, bytes(&[0])); // deterministic local tie choice
    }

    #[test]
    fn malformed_geometry_and_storage_fail_without_output() {
        for (shape, axis) in [
            (vec![], 0),
            (vec![0], 0),
            (vec![2], 1),
            (vec![usize::MAX, 2], 0),
        ] {
            assert!(evaluate(&[], &shape, axis, false, false).is_err());
        }
        assert!(evaluate(&[0; 3], &[1], 0, false, false).is_err());
        assert!(evaluate(&bytes(&[1, 2]), &[1], 0, false, false).is_err());
    }
}
