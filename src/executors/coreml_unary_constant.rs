//! Source-proven constant inputs for typed floating-point unary children.

use std::borrow::Cow;

use super::*;
use crate::error::GraphError;
use crate::protos::coreml::specification::{FeatureDescription, FeatureType};

#[derive(Debug, Clone, PartialEq)]
enum Storage {
    Immediate(Vec<u8>),
    Blob {
        offset: u64,
        width: usize,
        length: usize,
    },
    Resolved(std::ops::Range<usize>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstantUnary {
    pub kind: Kind,
    storage: Storage,
    pub shape: Vec<i64>,
    transforms: Vec<Transform>,
}

#[derive(Debug, Clone, PartialEq)]
enum Transform {
    Cast(Direction),
    Transpose {
        shape: Vec<usize>,
        permutation: Vec<usize>,
        width: usize,
    },
}

impl ConstantUnary {
    pub fn resolve(&mut self, weights: Option<&[u8]>) -> Result<(), GraphError> {
        let Storage::Blob {
            offset,
            width,
            length,
        } = &self.storage
        else {
            return Ok(());
        };
        let invalid = || GraphError::CoremlRuntimeFailed {
            reason: "typed unary constant does not match its original weight record".into(),
        };
        let weights = weights.ok_or_else(invalid)?;
        let ranges = crate::converters::weight_ranges(weights)?;
        let range = ranges.get(offset).ok_or_else(invalid)?;
        let header = usize::try_from(*offset).map_err(|_| invalid())?;
        let record_type = u32::from_le_bytes(weights[header + 4..header + 8].try_into().unwrap());
        if range.len() != *length || record_type != if *width == 2 { 1 } else { 2 } {
            return Err(invalid());
        }
        self.storage = Storage::Resolved(range.clone());
        Ok(())
    }

    pub fn bytes<'a>(&'a self, weights: Option<&'a [u8]>) -> Result<Cow<'a, [u8]>, GraphError> {
        let invalid = || GraphError::CoremlRuntimeFailed {
            reason: "typed unary constant does not match its original weight record".into(),
        };
        let bytes = match &self.storage {
            Storage::Immediate(bytes) => bytes.as_slice(),
            Storage::Resolved(range) => weights
                .and_then(|bytes| bytes.get(range.clone()))
                .ok_or_else(invalid)?,
            Storage::Blob { .. } => return Err(invalid()),
        };
        let mut bytes = Cow::Borrowed(bytes);
        for transform in &self.transforms {
            bytes = Cow::Owned(match transform {
                Transform::Cast(direction) => {
                    cast(&bytes, *direction).map_err(|reason| GraphError::CoremlRuntimeFailed {
                        reason: reason.into(),
                    })?
                }
                Transform::Transpose {
                    shape,
                    permutation,
                    width,
                } => {
                    let mut output = Vec::new();
                    output
                        .try_reserve_exact(bytes.len())
                        .map_err(|_| invalid())?;
                    output.resize(bytes.len(), 0);
                    let mut strides = vec![1; shape.len()];
                    for axis in (0..shape.len().saturating_sub(1)).rev() {
                        strides[axis] = strides[axis + 1] * shape[axis + 1];
                    }
                    for index in 0..bytes.len() / width {
                        let mut cursor = index;
                        let mut source = 0;
                        for &axis in permutation.iter().rev() {
                            source += (cursor % shape[axis]) * strides[axis];
                            cursor /= shape[axis];
                        }
                        output[index * width..(index + 1) * width]
                            .copy_from_slice(&bytes[source * width..(source + 1) * width]);
                    }
                    output
                }
            });
        }
        Ok(bytes)
    }
}

fn shape(ty: &mil::TensorType) -> Option<Vec<i64>> {
    shape_with_storage_limit(ty, usize::MAX as u64)
}

fn shape_with_storage_limit(ty: &mil::TensorType, max_extent: u64) -> Option<Vec<i64>> {
    if ty.rank < 0 || ty.rank as usize != ty.dimensions.len() || !ty.attributes.is_empty() {
        return None;
    }
    ty.dimensions
        .iter()
        .map(|dimension| match &dimension.dimension {
            Some(dimension::Dimension::Constant(value))
                if value.size > 0 && value.size <= max_extent =>
            {
                i64::try_from(value.size).ok()
            }
            _ => None,
        })
        .collect()
}

fn feature(value: &mil::NamedValueType) -> Option<FeatureDescription> {
    let ty = tensor(value.r#type.as_ref()?)?;
    let mut shape = shape(ty)?;
    if shape.is_empty() {
        shape.push(1);
    }
    Some(FeatureDescription {
        name: value.name.clone(),
        r#type: Some(FeatureType {
            r#type: Some(feature_type::Type::MultiArrayType(ArrayFeatureType {
                shape,
                data_type: match ty.data_type {
                    10 => 65552,
                    11 => 65568,
                    _ => return None,
                },
                ..Default::default()
            })),
            ..Default::default()
        }),
        ..Default::default()
    })
}

// Test each operation through the same typed-unary provenance/feature checks.
// The original bytes must first pass the known-wire guard, so rebuilding this
// small classification view cannot erase unknown source semantics.
fn unary_view(
    model: &Model,
    input: &mil::NamedValueType,
    operation: &mil::Operation,
) -> Option<Kind> {
    let model::Type::MlProgram(original) = model.r#type.as_ref()? else {
        return None;
    };
    let opset = original.functions.get("main")?.opset.clone();
    // Do not clone the constant's original immediate payload just to classify
    // its consumer. The original wire and prefix have already been checked.
    let view = Model {
        specification_version: model.specification_version,
        description: Some(crate::protos::coreml::specification::ModelDescription {
            input: vec![feature(input)?],
            output: vec![feature(operation.outputs.first()?)?],
            ..Default::default()
        }),
        r#type: Some(model::Type::MlProgram(mil::Program {
            version: 1,
            functions: [(
                "main".into(),
                mil::Function {
                    inputs: vec![input.clone()],
                    opset: opset.clone(),
                    block_specializations: [(
                        opset,
                        mil::Block {
                            operations: vec![operation.clone()],
                            outputs: vec![operation.outputs.first()?.name.clone()],
                            ..Default::default()
                        },
                    )]
                    .into(),
                    ..Default::default()
                },
            )]
            .into(),
            ..Default::default()
        })),
        ..Default::default()
    };
    classify(&view, &view.encode_to_vec())
}

pub fn classify_constant(model: &Model, source: &[u8]) -> Option<ConstantUnary> {
    if model.encoded_len() != source.len() || known_wire(source, Schema::Model, 0).is_none() {
        return None;
    }
    let model::Type::MlProgram(program) = model.r#type.as_ref()? else {
        return None;
    };
    if program.version != 1 || program.functions.len() != 1 || !program.attributes.is_empty() {
        return None;
    }
    let function = program.functions.get("main")?;
    if !function.inputs.is_empty()
        || function.block_specializations.len() != 1
        || !function.attributes.is_empty()
    {
        return None;
    }
    let block = function.block_specializations.get(&function.opset)?;
    if block.operations.len() < 2
        || !block.inputs.is_empty()
        || !block.attributes.is_empty()
        || block.outputs.len() != 1
    {
        return None;
    }
    let mut names = std::collections::HashSet::new();
    for operation in &block.operations {
        if operation
            .outputs
            .iter()
            .any(|value| value.name.is_empty() || !names.insert(&value.name))
        {
            return None;
        }
    }
    let constant = &block.operations[0];
    if constant.r#type != "const"
        || !constant.inputs.is_empty()
        || !constant.blocks.is_empty()
        || constant.outputs.len() != 1
        || constant.attributes.len() != 1
    {
        return None;
    }
    let value = constant.attributes.get("val")?;
    let input = &constant.outputs[0];
    if input.name.is_empty() || value.r#type != input.r#type {
        return None;
    }
    let ty = tensor(input.r#type.as_ref()?)?;
    let dimensions = shape(ty)?;
    let width: usize = match ty.data_type {
        10 => 2,
        11 => 4,
        _ => return None,
    };
    let length = dimensions
        .iter()
        .try_fold(width, |size, &extent| size.checked_mul(extent as usize))?;
    let storage = match value.value.as_ref()? {
        value::Value::BlobFileValue(blob)
            if blob.file_name == "@model_path/weights/weights.bin" =>
        {
            Storage::Blob {
                offset: blob.offset,
                width,
                length,
            }
        }
        value::Value::ImmediateValue(value) => {
            let value::immediate_value::Value::Tensor(value) = value.value.as_ref()? else {
                return None;
            };
            let bytes = match value.value.as_ref()? {
                tensor_value::Value::Floats(value) if width == 4 => value
                    .values
                    .iter()
                    .flat_map(|value| value.to_bits().to_ne_bytes())
                    .collect::<Vec<_>>(),
                tensor_value::Value::Bytes(value) if width == 2 => value.values.to_vec(),
                _ => return None,
            };
            if bytes.len() != length {
                return None;
            }
            Storage::Immediate(bytes)
        }
        _ => return None,
    };
    let mut input = input;
    let mut transforms = Vec::new();
    for operation in &block.operations[1..block.operations.len() - 1] {
        if operation.outputs.len() != 1
            || !operation.blocks.is_empty()
            || !operation.attributes.is_empty()
            || named_input(operation, "x")? != input.name
        {
            return None;
        }
        let source = tensor(input.r#type.as_ref()?)?;
        let output = &operation.outputs[0];
        let target = tensor(output.r#type.as_ref()?)?;
        let source_shape = shape(source)?;
        let target_shape = shape(target)?;
        let elements = |shape: &[i64]| {
            shape
                .iter()
                .try_fold(1_usize, |n, &size| n.checked_mul(size as usize))
        };
        if elements(&source_shape)? != elements(&target_shape)? {
            return None;
        }
        match operation.r#type.as_str() {
            "cast" => {
                let Some(Kind::FloatCast(direction)) = unary_view(model, input, operation) else {
                    return None;
                };
                transforms.push(Transform::Cast(direction));
            }
            "identity" if operation.inputs.len() == 1 && source == target => {}
            "reshape" if operation.inputs.len() == 2 && source.data_type == target.data_type => {
                if integers(operation, "shape")?
                    .iter()
                    .map(|&n| i64::from(n))
                    .collect::<Vec<_>>()
                    != target_shape
                {
                    return None;
                }
            }
            "transpose" if operation.inputs.len() == 2 && source.data_type == target.data_type => {
                let permutation: Vec<_> = integers(operation, "perm")?
                    .iter()
                    .map(|&n| usize::try_from(n).ok())
                    .collect::<Option<_>>()?;
                if permutation.len() != source_shape.len()
                    || target_shape.len() != source_shape.len()
                {
                    return None;
                }
                let mut seen = vec![false; permutation.len()];
                for (axis, &source_axis) in permutation.iter().enumerate() {
                    if source_axis >= seen.len()
                        || seen[source_axis]
                        || target_shape[axis] != source_shape[source_axis]
                    {
                        return None;
                    }
                    seen[source_axis] = true;
                }
                transforms.push(Transform::Transpose {
                    shape: source_shape.iter().map(|&n| n as usize).collect(),
                    permutation,
                    width: if source.data_type == 10 { 2 } else { 4 },
                });
            }
            "mul" | "real_div"
                if operation.inputs.len() == 2
                    && source == target
                    && source.data_type == 11
                    && unit(operation, "y") => {}
            _ => return None,
        }
        input = output;
    }
    let operation = block.operations.last()?;
    let kind = unary_view(model, input, operation)?;
    if !matches!(kind, Kind::Gelu | Kind::Sqrt) {
        return None;
    }
    let description = model.description.as_ref()?;
    let output = &operation.outputs[0];
    if !description.input.is_empty()
        || description.output.len() != 1
        || block.outputs[0] != output.name
        || description.output[0].name != output.name
        || description.output[0].r#type.as_ref()?.is_optional
    {
        return None;
    }
    let output_type = tensor(output.r#type.as_ref()?)?;
    let array = array(&description.output[0])?;
    if array.data_type != 65568 || !compatible_shape(output_type, array) {
        return None;
    }
    let dimensions = shape(tensor(input.r#type.as_ref()?)?)?;
    let shape = if dimensions.is_empty() {
        vec![1]
    } else {
        dimensions
    };
    Some(ConstantUnary {
        kind,
        storage,
        shape,
        transforms,
    })
}

fn named_input<'a>(operation: &'a mil::Operation, name: &str) -> Option<&'a str> {
    let arguments = &operation.inputs.get(name)?.arguments;
    let [binding] = arguments.as_slice() else {
        return None;
    };
    match binding.binding.as_ref()? {
        argument::binding::Binding::Name(name) => Some(name),
        _ => None,
    }
}

fn literal<'a>(
    operation: &'a mil::Operation,
    name: &str,
) -> Option<(&'a mil::TensorType, &'a mil::TensorValue)> {
    let [binding] = operation.inputs.get(name)?.arguments.as_slice() else {
        return None;
    };
    let argument::binding::Binding::Value(value) = binding.binding.as_ref()? else {
        return None;
    };
    let ty = tensor(value.r#type.as_ref()?)?;
    let value::Value::ImmediateValue(value) = value.value.as_ref()? else {
        return None;
    };
    let value::immediate_value::Value::Tensor(value) = value.value.as_ref()? else {
        return None;
    };
    Some((ty, value))
}

fn integers<'a>(operation: &'a mil::Operation, name: &str) -> Option<&'a [i32]> {
    let (ty, value) = literal(operation, name)?;
    let tensor_value::Value::Ints(value) = value.value.as_ref()? else {
        return None;
    };
    if ty.data_type != 23 || shape(ty)? != [value.values.len() as i64] {
        return None;
    }
    Some(&value.values)
}

fn unit(operation: &mil::Operation, name: &str) -> bool {
    literal(operation, name).is_some_and(|(ty, value)| ty.data_type == 11 && shape(ty).is_some_and(|shape| shape.is_empty() || shape == [1]) && matches!(&value.value, Some(tensor_value::Value::Floats(value)) if value.values.as_slice() == [1.0]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converters::WeightFileBuilder;

    #[test]
    fn constant_shapes_do_not_truncate_at_the_arm64_32_storage_boundary() {
        // Exercise the 32-bit limit even when this test runs on a 64-bit Mac.
        let limit = u64::from(u32::MAX);
        for (size, expected) in [
            (0, None),
            (1, Some(vec![1])),
            (limit, Some(vec![limit as i64])),
            (limit + 1, None),
            (limit + 2, None),
            (u64::MAX, None),
        ] {
            let ty = mil::TensorType {
                data_type: 11,
                rank: 1,
                dimensions: vec![mil::Dimension {
                    dimension: Some(dimension::Dimension::Constant(
                        dimension::ConstantDimension { size },
                    )),
                }],
                ..Default::default()
            };
            assert_eq!(
                shape_with_storage_limit(&ty, limit),
                expected,
                "size {size}"
            );
        }
    }

    #[test]
    fn gelu_constant_ranges_validate_record_type_length_and_original_bits_once() {
        let payload: Vec<_> = [0xc1200000_u32, 0x00000001, 0x80000000, 0x7fc01234]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let mut builder = WeightFileBuilder::new();
        let offset = builder.add_weight(0, 2, &payload).unwrap();
        let weights = builder.finalize();
        let source = ConstantUnary {
            kind: Kind::Gelu,
            storage: Storage::Blob {
                offset,
                width: 4,
                length: payload.len(),
            },
            shape: vec![4],
            transforms: vec![],
        };
        assert!(
            source.bytes(Some(&weights)).is_err(),
            "resolve before exposing a span"
        );
        let mut resolved = source.clone();
        resolved.resolve(Some(&weights)).unwrap();
        for _ in 0..3 {
            let bytes = resolved.bytes(Some(&weights)).unwrap();
            assert!(matches!(bytes, Cow::Borrowed(_)));
            assert_eq!(bytes.as_ptr(), weights[128..].as_ptr());
            assert_eq!(bytes.as_ref(), payload);
        }
        assert!(source.clone().resolve(None).is_err());
        for mutation in 0..4 {
            let mut changed = weights.clone();
            match mutation {
                0 => changed[68..72].copy_from_slice(&1_u32.to_le_bytes()),
                1 => changed[72..80].copy_from_slice(&8_u64.to_le_bytes()),
                2 => changed[80..88].copy_from_slice(&192_u64.to_le_bytes()),
                3 => {
                    changed.truncate(135);
                }
                _ => unreachable!(),
            }
            assert!(
                source.clone().resolve(Some(&changed)).is_err(),
                "mutation {mutation}"
            );
        }
        let mut wrong = source;
        wrong.storage = Storage::Blob {
            offset: 128,
            width: 4,
            length: payload.len(),
        };
        assert!(wrong.resolve(Some(&weights)).is_err());
    }
}
