//! Classify only complete, source-proven pure binary16/binary32 Cast programs.

use prost::Message;

use crate::protos::coreml::mil_spec::{
    self as mil, argument, dimension, tensor_value, value, value_type,
};
use crate::protos::coreml::specification::{
    ArrayFeatureType, Model, array_feature_type, feature_type, model,
};

#[path = "coreml_float_cast/kernels.rs"]
mod kernels;
pub(super) use kernels::{Direction, cast};

#[derive(Clone, Copy)]
enum Schema {
    Model,
    Description,
    Metadata,
    StringsEntry,
    Feature,
    FeatureType,
    Array,
    Shapes,
    Shape,
    Ranges,
    Range,
    Program,
    FunctionEntry,
    Function,
    BlockEntry,
    Block,
    Operation,
    ArgumentEntry,
    Argument,
    Binding,
    NamedType,
    ValueType,
    TensorType,
    Dimension,
    ConstantDimension,
    UnknownDimension,
    Value,
    Immediate,
    TensorValue,
    Strings,
}

enum WireType {
    Varint,
    Bytes,
    PackedVarints,
    Message(Schema),
}

fn field_type(schema: Schema, field: u64) -> Option<WireType> {
    use Schema as S;
    use WireType::{Bytes as B, Message as M, PackedVarints as P, Varint as V};
    Some(match (schema, field) {
        (S::Model, 1) => V,
        (S::Model, 2) => M(S::Description),
        (S::Model, 502) => M(S::Program),
        (S::Description, 1 | 10) => M(S::Feature),
        (S::Description, 100) => M(S::Metadata),
        (S::Metadata, 1..=4) => B,
        (S::Metadata, 100) => M(S::StringsEntry),
        (S::StringsEntry, 1 | 2) => B,
        (S::Feature, 1 | 2) => B,
        (S::Feature, 3) => M(S::FeatureType),
        (S::FeatureType, 5) => M(S::Array),
        (S::FeatureType, 1000) => V,
        (S::Array, 1) | (S::Shape, 1) => P,
        (S::Array, 2) => V,
        (S::Array, 21) => M(S::Shapes),
        (S::Array, 31) => M(S::Ranges),
        (S::Shapes, 1) => M(S::Shape),
        (S::Ranges, 1) => M(S::Range),
        (S::Range, 1 | 2) => V,
        (S::Program, 1) => V,
        (S::Program, 2) => M(S::FunctionEntry),
        (S::Program, 3) => B,
        (S::FunctionEntry, 1) | (S::BlockEntry, 1) | (S::ArgumentEntry, 1) => B,
        (S::FunctionEntry, 2) => M(S::Function),
        (S::BlockEntry, 2) => M(S::Block),
        (S::ArgumentEntry, 2) => M(S::Argument),
        (S::Function, 1) => M(S::NamedType),
        (S::Function, 2) => B,
        (S::Function, 3) => M(S::BlockEntry),
        (S::Block, 2) => B,
        (S::Block, 3) => M(S::Operation),
        (S::Operation, 1) => B,
        (S::Operation, 2) => M(S::ArgumentEntry),
        (S::Operation, 3) => M(S::NamedType),
        (S::Argument, 1) => M(S::Binding),
        (S::Binding, 1) => B,
        (S::Binding, 2) => M(S::Value),
        (S::NamedType, 1) => B,
        (S::NamedType, 2) | (S::Value, 2) => M(S::ValueType),
        (S::ValueType, 1) => M(S::TensorType),
        (S::TensorType, 1 | 2) => V,
        (S::TensorType, 3) => M(S::Dimension),
        (S::Dimension, 1) => M(S::ConstantDimension),
        (S::Dimension, 2) => M(S::UnknownDimension),
        (S::ConstantDimension, 1) | (S::UnknownDimension, 1) => V,
        (S::Value, 1) => B,
        (S::Value, 3) => M(S::Immediate),
        (S::Immediate, 1) => M(S::TensorValue),
        (S::TensorValue, 4) => M(S::Strings),
        (S::Strings, 1) => B,
        _ => return None,
    })
}

// Unknown fields at any nested level stay native, even if their wire length
// happens to equal a known message's re-encoding. This whitelist describes the
// complete straight-line Cast subset, not a rewriting of its source bytes.
fn known_wire(mut bytes: &[u8], schema: Schema, depth: u8) -> Option<()> {
    fn varint(bytes: &mut &[u8]) -> Option<u64> {
        let mut result = 0_u64;
        for shift in (0..70).step_by(7) {
            let (&byte, remaining) = bytes.split_first()?;
            *bytes = remaining;
            if shift == 63 && byte > 1 {
                return None;
            }
            result |= u64::from(byte & 127) << shift;
            if byte & 128 == 0 {
                return Some(result);
            }
        }
        None
    }
    if depth > 32 {
        return None;
    }
    while !bytes.is_empty() {
        let tag = varint(&mut bytes)?;
        match (field_type(schema, tag >> 3)?, tag & 7) {
            (WireType::Varint | WireType::PackedVarints, 0) => {
                varint(&mut bytes)?;
            }
            (kind @ (WireType::Bytes | WireType::PackedVarints | WireType::Message(_)), 2) => {
                let length = usize::try_from(varint(&mut bytes)?).ok()?;
                let (value, remaining) = bytes.split_at_checked(length)?;
                if let WireType::Message(schema) = kind {
                    known_wire(value, schema, depth + 1)?;
                }
                bytes = remaining;
            }
            _ => return None,
        }
    }
    Some(())
}

fn tensor(ty: &mil::ValueType) -> Option<&mil::TensorType> {
    match ty.r#type.as_ref()? {
        value_type::Type::TensorType(tensor) => Some(tensor),
        _ => None,
    }
}

fn array(
    feature: &crate::protos::coreml::specification::FeatureDescription,
) -> Option<&ArrayFeatureType> {
    match feature.r#type.as_ref()?.r#type.as_ref()? {
        feature_type::Type::MultiArrayType(array) => Some(array),
        _ => None,
    }
}

fn compatible_shape(tensor: &mil::TensorType, array: &ArrayFeatureType) -> bool {
    if !tensor.attributes.is_empty()
        || tensor.rank < 0
        || tensor.rank as usize != tensor.dimensions.len()
    {
        return false;
    }
    if tensor.rank == 0 {
        return array.shape == [1] && array.shape_flexibility.is_none();
    }
    if tensor.dimensions.len() != array.shape.len() {
        return false;
    }
    tensor
        .dimensions
        .iter()
        .zip(&array.shape)
        .enumerate()
        .all(|(axis, (dimension, &size))| match &dimension.dimension {
            Some(dimension::Dimension::Constant(constant)) => {
                u64::try_from(size).ok() == Some(constant.size)
            }
            Some(dimension::Dimension::Unknown(unknown)) if !unknown.variadic => {
                match &array.shape_flexibility {
                    Some(array_feature_type::ShapeFlexibility::ShapeRange(bounds)) => {
                        bounds.size_ranges.len() == array.shape.len()
                            && bounds.size_ranges[axis].lower_bound > 0
                            && bounds.size_ranges[axis].upper_bound >= size
                    }
                    Some(array_feature_type::ShapeFlexibility::EnumeratedShapes(bounds)) => {
                        !bounds.shapes.is_empty()
                            && bounds.shapes.iter().all(|shape| {
                                shape.shape.len() == array.shape.len()
                                    && shape.shape.iter().all(|&size| size > 0)
                            })
                    }
                    _ => false,
                }
            }
            _ => false,
        })
}

/// Conservatively keep programs with extra operations, attributes, unresolved
/// types/shapes or unknown source fields native. No feature name/proxy dtype
/// or hardware-generation heuristic establishes logical Cast provenance.
pub(super) fn classify(model: &Model, source: &[u8]) -> Option<Direction> {
    // Ordinary converter wire is canonical in encoded length. Extra unknown
    // fields or redundant wire encodings deliberately do not get a host path.
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
    if function.inputs.len() != 1
        || function.block_specializations.len() != 1
        || !function.attributes.is_empty()
    {
        return None;
    }
    let block = function.block_specializations.get(&function.opset)?;
    if block.operations.len() != 1
        || block.outputs.len() != 1
        || !block.inputs.is_empty()
        || !block.attributes.is_empty()
    {
        return None;
    }
    let operation = &block.operations[0];
    if operation.r#type != "cast"
        || operation.outputs.len() != 1
        || operation.inputs.len() != 2
        || !operation.blocks.is_empty()
        || !operation.attributes.is_empty()
    {
        return None;
    }
    let binding = operation.inputs.get("x")?;
    if binding.arguments.len() != 1 {
        return None;
    }
    let argument::binding::Binding::Name(name) = binding.arguments[0].binding.as_ref()? else {
        return None;
    };
    let input = &function.inputs[0];
    let output = &operation.outputs[0];
    if *name != input.name || block.outputs[0] != output.name {
        return None;
    }
    let source_type = tensor(input.r#type.as_ref()?)?;
    let target_type = tensor(output.r#type.as_ref()?)?;
    let (direction, source_code, target_code, target_name) =
        match (source_type.data_type, target_type.data_type) {
            (10, 11) => (Direction::Widen, 65552, 65568, "fp32"),
            (11, 10) => (Direction::Narrow, 65568, 65552, "fp16"),
            _ => return None,
        };
    let mut target_shape = target_type.clone();
    target_shape.data_type = source_type.data_type;
    if target_shape != *source_type {
        return None;
    }
    let dtype = operation.inputs.get("dtype")?;
    if dtype.arguments.len() != 1 {
        return None;
    }
    let argument::binding::Binding::Value(dtype) = dtype.arguments[0].binding.as_ref()? else {
        return None;
    };
    let dtype_type = tensor(dtype.r#type.as_ref()?)?;
    if dtype_type.data_type != 2
        || dtype_type.rank != 0
        || !dtype_type.dimensions.is_empty()
        || !dtype_type.attributes.is_empty()
    {
        return None;
    }
    let value::Value::ImmediateValue(dtype) = dtype.value.as_ref()? else {
        return None;
    };
    let value::immediate_value::Value::Tensor(dtype) = dtype.value.as_ref()? else {
        return None;
    };
    let tensor_value::Value::Strings(dtype) = dtype.value.as_ref()? else {
        return None;
    };
    if dtype.values.len() != 1 || dtype.values[0] != target_name {
        return None;
    }
    let description = model.description.as_ref()?;
    if description.input.len() != 1
        || description.output.len() != 1
        || description.input[0].name != input.name
        || description.output[0].name != output.name
    {
        return None;
    }
    if description.input[0].r#type.as_ref()?.is_optional
        || description.output[0].r#type.as_ref()?.is_optional
    {
        return None;
    }
    let source_feature = array(&description.input[0])?;
    let target_feature = array(&description.output[0])?;
    if source_feature.data_type != source_code
        || target_feature.data_type != target_code
        || !compatible_shape(source_type, source_feature)
        || !compatible_shape(target_type, target_feature)
    {
        return None;
    }
    let mut target_shape = target_feature.clone();
    target_shape.data_type = source_code;
    (target_shape == *source_feature).then_some(direction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::coreml::specification::{FeatureDescription, FeatureType, ModelDescription};

    fn tensor_type(dtype: i32) -> mil::ValueType {
        mil::ValueType {
            r#type: Some(value_type::Type::TensorType(mil::TensorType {
                data_type: dtype,
                rank: 1,
                dimensions: vec![mil::Dimension {
                    dimension: Some(dimension::Dimension::Constant(
                        mil::dimension::ConstantDimension { size: 4 },
                    )),
                }],
                ..Default::default()
            })),
        }
    }

    fn model() -> Model {
        let feature = |name: &str, code| FeatureDescription {
            name: name.into(),
            r#type: Some(FeatureType {
                r#type: Some(feature_type::Type::MultiArrayType(ArrayFeatureType {
                    shape: vec![4],
                    data_type: code,
                    ..Default::default()
                })),
                ..Default::default()
            }),
            ..Default::default()
        };
        let dtype = mil::Value {
            r#type: Some(mil::ValueType {
                r#type: Some(value_type::Type::TensorType(mil::TensorType {
                    data_type: 2,
                    ..Default::default()
                })),
            }),
            value: Some(value::Value::ImmediateValue(mil::value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(mil::TensorValue {
                    value: Some(tensor_value::Value::Strings(
                        mil::tensor_value::RepeatedStrings {
                            values: vec!["fp16".into()],
                        },
                    )),
                })),
            })),
            ..Default::default()
        };
        let operation = mil::Operation {
            r#type: "cast".into(),
            inputs: [
                (
                    "x".into(),
                    mil::Argument {
                        arguments: vec![argument::Binding {
                            binding: Some(argument::binding::Binding::Name("source".into())),
                        }],
                    },
                ),
                (
                    "dtype".into(),
                    mil::Argument {
                        arguments: vec![argument::Binding {
                            binding: Some(argument::binding::Binding::Value(dtype)),
                        }],
                    },
                ),
            ]
            .into(),
            outputs: vec![mil::NamedValueType {
                name: "result".into(),
                r#type: Some(tensor_type(10)),
            }],
            ..Default::default()
        };
        let function = mil::Function {
            inputs: vec![mil::NamedValueType {
                name: "source".into(),
                r#type: Some(tensor_type(11)),
            }],
            opset: "CoreML7".into(),
            block_specializations: [(
                "CoreML7".into(),
                mil::Block {
                    operations: vec![operation],
                    outputs: vec!["result".into()],
                    ..Default::default()
                },
            )]
            .into(),
            ..Default::default()
        };
        Model {
            specification_version: 8,
            description: Some(ModelDescription {
                input: vec![feature("source", 65568)],
                output: vec![feature("result", 65552)],
                ..Default::default()
            }),
            r#type: Some(model::Type::MlProgram(mil::Program {
                version: 1,
                functions: [("main".into(), function)].into(),
                ..Default::default()
            })),
            ..Default::default()
        }
    }

    #[test]
    fn complete_source_cast_is_classified_without_name_or_proxy_heuristics() {
        let model = model();
        assert_eq!(
            classify(&model, &model.encode_to_vec()),
            Some(Direction::Narrow)
        );
        for mutation in 0..5 {
            let mut model = model.clone();
            match mutation {
                0 => {
                    model.description.as_mut().unwrap().input[0]
                        .r#type
                        .as_mut()
                        .unwrap()
                        .is_optional = true
                }
                1 => {
                    let description = model.description.as_mut().unwrap();
                    description.state.push(description.input[0].clone());
                }
                2 => {
                    model.description.as_mut().unwrap().input[0]
                        .r#type
                        .as_mut()
                        .unwrap()
                        .r#type = Some(feature_type::Type::MultiArrayType(ArrayFeatureType {
                        shape: vec![4],
                        data_type: 131104,
                        ..Default::default()
                    }))
                }
                3 => {
                    let model::Type::MlProgram(program) = model.r#type.as_mut().unwrap() else {
                        unreachable!()
                    };
                    program
                        .functions
                        .get_mut("main")
                        .unwrap()
                        .block_specializations
                        .get_mut("CoreML7")
                        .unwrap()
                        .operations[0]
                        .r#type = "identity".into();
                }
                4 => model.description.as_mut().unwrap().output[0].name = "unresolved".into(),
                _ => unreachable!(),
            }
            assert_eq!(
                classify(&model, &model.encode_to_vec()),
                None,
                "mutation {mutation}"
            );
        }
    }

    #[test]
    fn unknown_nested_wire_fields_never_establish_a_host_stage() {
        for schema in [
            Schema::Model,
            Schema::Program,
            Schema::Operation,
            Schema::TensorType,
            Schema::Array,
            Schema::Binding,
            Schema::Metadata,
        ] {
            assert!(known_wire(&[0xf8, 0x3f, 7], schema, 0).is_none());
        }
        let model = model();
        let mut source = model.encode_to_vec();
        source.extend([0xf8, 0x3f, 7]);
        assert_eq!(
            classify(&Model::decode(source.as_slice()).unwrap(), &source),
            None
        );
    }
}
