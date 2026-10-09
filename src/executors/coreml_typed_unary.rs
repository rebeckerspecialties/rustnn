//! Classify complete, source-proven typed unary programs for exact host stages.

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

#[path = "coreml_binary32.rs"]
mod binary32;

#[path = "coreml_exp.rs"]
mod exp;

#[path = "coreml_gelu.rs"]
mod gelu;

#[path = "coreml_sqrt.rs"]
mod sqrt;

#[path = "coreml_unary_constant.rs"]
mod constant;
pub(super) use constant::{ConstantUnary, classify_constant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    FloatCast(Direction),
    Gelu,
    Sqrt,
    Exp,
}

pub(super) fn evaluate(bytes: &[u8], kind: Kind) -> Result<Vec<u8>, &'static str> {
    match kind {
        Kind::FloatCast(direction) => cast(bytes, direction),
        Kind::Gelu => gelu::evaluate(bytes),
        Kind::Sqrt => sqrt::evaluate(bytes),
        Kind::Exp => exp::evaluate(bytes),
    }
}

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
    ValueEntry,
    Floats,
    Ints,
    Bytes,
    Blob,
}

enum WireType {
    Varint,
    Bytes,
    PackedVarints,
    PackedFloats,
    Message(Schema),
}

fn field_type(schema: Schema, field: u64) -> Option<WireType> {
    use Schema as S;
    use WireType::{Bytes as B, Message as M, PackedFloats as F, PackedVarints as P, Varint as V};
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
        (S::Operation, 5) => M(S::ValueEntry),
        (S::ValueEntry, 1) => B,
        (S::ValueEntry, 2) => M(S::Value),
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
        (S::Value, 5) => M(S::Blob),
        (S::Immediate, 1) => M(S::TensorValue),
        (S::TensorValue, 1) => M(S::Floats),
        (S::TensorValue, 2) => M(S::Ints),
        (S::TensorValue, 7) => M(S::Bytes),
        (S::TensorValue, 4) => M(S::Strings),
        (S::Strings, 1) => B,
        (S::Floats, 1) => F,
        (S::Ints, 1) => P,
        (S::Bytes, 1) | (S::Blob, 1) => B,
        (S::Blob, 2) => V,
        _ => return None,
    })
}

// Unknown fields at any nested level stay native, even if their wire length
// happens to equal a known message's re-encoding. This whitelist describes the
// complete straight-line typed unary subset, not a rewriting of its source bytes.
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
            (WireType::PackedFloats, 5) => {
                bytes = bytes.get(4..)?;
            }
            (
                kind @ (WireType::Bytes
                | WireType::PackedVarints
                | WireType::PackedFloats
                | WireType::Message(_)),
                2,
            ) => {
                let length = usize::try_from(varint(&mut bytes)?).ok()?;
                let (value, remaining) = bytes.split_at_checked(length)?;
                if let WireType::Message(schema) = kind {
                    known_wire(value, schema, depth + 1)?;
                } else if matches!(kind, WireType::PackedFloats) && value.len() % 4 != 0 {
                    return None;
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
    if tensor.dimensions.len() != array.shape.len() || array.shape.iter().any(|&size| size <= 0) {
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
                    && match &array.shape_flexibility {
                        None => true,
                        Some(array_feature_type::ShapeFlexibility::ShapeRange(bounds)) => {
                            bounds.size_ranges.len() == array.shape.len()
                                && bounds.size_ranges[axis].lower_bound == constant.size
                                && bounds.size_ranges[axis].upper_bound == size
                        }
                        Some(array_feature_type::ShapeFlexibility::EnumeratedShapes(bounds)) => {
                            !bounds.shapes.is_empty()
                                && bounds.shapes.iter().all(|shape| {
                                    shape.shape.len() == array.shape.len()
                                        && shape.shape[axis] == size
                                })
                        }
                    }
            }
            Some(dimension::Dimension::Unknown(unknown)) if !unknown.variadic => {
                match &array.shape_flexibility {
                    Some(array_feature_type::ShapeFlexibility::ShapeRange(bounds)) => {
                        bounds.size_ranges.len() == array.shape.len()
                            && size > 0
                            && bounds.size_ranges[axis].lower_bound <= size as u64
                            && bounds.size_ranges[axis].upper_bound >= size
                    }
                    Some(array_feature_type::ShapeFlexibility::EnumeratedShapes(bounds)) => {
                        !bounds.shapes.is_empty()
                            && bounds.shapes.iter().any(|shape| shape.shape == array.shape)
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
pub(super) fn classify(model: &Model, source: &[u8]) -> Option<Kind> {
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
    if matches!(operation.r#type.as_str(), "gelu" | "sqrt" | "exp") && function.opset != "CoreML7" {
        return None;
    }
    if !matches!(operation.r#type.as_str(), "cast" | "gelu" | "sqrt" | "exp")
        || operation.outputs.len() != 1
        || operation.inputs.len()
            != if matches!(operation.r#type.as_str(), "sqrt" | "exp") {
                1
            } else {
                2
            }
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
    if *name != input.name || block.outputs[0] != output.name || input.name == output.name {
        return None;
    }
    let source_type = tensor(input.r#type.as_ref()?)?;
    let target_type = tensor(output.r#type.as_ref()?)?;
    let (kind, source_code, target_code, argument) = match (
        operation.r#type.as_str(),
        source_type.data_type,
        target_type.data_type,
    ) {
        ("cast", 10, 11) => (
            Kind::FloatCast(Direction::Widen),
            65552,
            65568,
            Some(("dtype", "fp32")),
        ),
        ("cast", 11, 10) => (
            Kind::FloatCast(Direction::Narrow),
            65568,
            65552,
            Some(("dtype", "fp16")),
        ),
        ("gelu", 11, 11) => (Kind::Gelu, 65568, 65568, Some(("mode", "EXACT"))),
        ("sqrt", 11, 11) => (Kind::Sqrt, 65568, 65568, None),
        ("exp", 11, 11) => (Kind::Exp, 65568, 65568, None),
        _ => return None,
    };
    let canonical_shape = |ty: &mil::TensorType| {
        let mut ty = ty.clone();
        ty.data_type = 0;
        if ty.rank == 0 && ty.dimensions.is_empty() {
            ty.rank = 1;
            ty.dimensions = vec![mil::Dimension {
                dimension: Some(dimension::Dimension::Constant(
                    mil::dimension::ConstantDimension { size: 1 },
                )),
            }];
        }
        ty
    };
    // CoreML features represent public scalar tensors as [1]. This is only
    // that one-element adapter, not a general reshape or broadcast rule.
    if canonical_shape(target_type) != canonical_shape(source_type) {
        return None;
    }
    if let Some((argument, target_name)) = argument {
        let dtype = operation.inputs.get(argument)?;
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
    (target_shape == *source_feature).then_some(kind)
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

    fn gelu_model() -> Model {
        let mut model = model();
        let model::Type::MlProgram(program) = model.r#type.as_mut().unwrap() else {
            unreachable!()
        };
        let operation = &mut program
            .functions
            .get_mut("main")
            .unwrap()
            .block_specializations
            .get_mut("CoreML7")
            .unwrap()
            .operations[0];
        operation.r#type = "gelu".into();
        operation.outputs[0].r#type = Some(tensor_type(11));
        let mut mode = operation.inputs.remove("dtype").unwrap();
        let argument::binding::Binding::Value(value) = mode.arguments[0].binding.as_mut().unwrap()
        else {
            unreachable!()
        };
        let value::Value::ImmediateValue(value) = value.value.as_mut().unwrap() else {
            unreachable!()
        };
        let value::immediate_value::Value::Tensor(value) = value.value.as_mut().unwrap() else {
            unreachable!()
        };
        let tensor_value::Value::Strings(value) = value.value.as_mut().unwrap() else {
            unreachable!()
        };
        value.values[0] = "EXACT".into();
        operation.inputs.insert("mode".into(), mode);
        let feature_type::Type::MultiArrayType(output) = model.description.as_mut().unwrap().output
            [0]
        .r#type
        .as_mut()
        .unwrap()
        .r#type
        .as_mut()
        .unwrap() else {
            unreachable!()
        };
        output.data_type = 65568;
        model
    }

    fn sqrt_model() -> Model {
        let mut model = gelu_model();
        let model::Type::MlProgram(program) = model.r#type.as_mut().unwrap() else {
            unreachable!()
        };
        let operation = &mut program
            .functions
            .get_mut("main")
            .unwrap()
            .block_specializations
            .get_mut("CoreML7")
            .unwrap()
            .operations[0];
        operation.r#type = "sqrt".into();
        operation.inputs.remove("mode");
        model
    }

    #[test]
    fn sqrt_and_exp_require_complete_float32_source_provenance() {
        for kind in [Kind::Sqrt, Kind::Exp] {
            let mut model = sqrt_model();
            if kind == Kind::Exp {
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
                    .r#type = "exp".into();
            }
            assert_eq!(classify(&model, &model.encode_to_vec()), Some(kind));
            for mutation in 0..5 {
                let mut changed = model.clone();
                let model::Type::MlProgram(program) = changed.r#type.as_mut().unwrap() else {
                    unreachable!()
                };
                let function = program.functions.get_mut("main").unwrap();
                let block = function.block_specializations.get_mut("CoreML7").unwrap();
                match mutation {
                    0 => {
                        block.operations[0]
                            .inputs
                            .insert("extra".into(), Default::default());
                    }
                    1 => {
                        block.operations[0].outputs[0].r#type = Some(tensor_type(10));
                    }
                    2 => {
                        block.operations.push(block.operations[0].clone());
                    }
                    3 => {
                        function.inputs[0].r#type = Some(tensor_type(10));
                    }
                    4 => {
                        changed.description.as_mut().unwrap().input[0]
                            .r#type
                            .as_mut()
                            .unwrap()
                            .is_optional = true;
                    }
                    _ => unreachable!(),
                }
                assert_eq!(
                    classify(&changed, &changed.encode_to_vec()),
                    None,
                    "mutation {mutation}"
                );
            }
            let mut unknown = model.encode_to_vec();
            unknown.extend([0xf8, 0x7f, 0x01]);
            assert_eq!(
                classify(&Model::decode(unknown.as_slice()).unwrap(), &unknown),
                None
            );
        }
    }

    #[test]
    fn exp_proved_float32_program_has_an_accuracy_preserving_stage() {
        let mut model = sqrt_model();
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
            .r#type = "exp".into();
        assert_eq!(classify(&model, &model.encode_to_vec()), Some(Kind::Exp));
    }

    #[test]
    fn exact_gelu_requires_complete_mode_types_and_feature_provenance() {
        let model = gelu_model();
        assert_eq!(classify(&model, &model.encode_to_vec()), Some(Kind::Gelu));
        for mutation in 0..6 {
            let mut changed = model.clone();
            let model::Type::MlProgram(program) = changed.r#type.as_mut().unwrap() else {
                unreachable!()
            };
            let block = program
                .functions
                .get_mut("main")
                .unwrap()
                .block_specializations
                .get_mut("CoreML7")
                .unwrap();
            let operation = &mut block.operations[0];
            match mutation {
                0 => {
                    operation.inputs.remove("mode");
                }
                1 => {
                    let argument::binding::Binding::Value(value) =
                        operation.inputs.get_mut("mode").unwrap().arguments[0]
                            .binding
                            .as_mut()
                            .unwrap()
                    else {
                        unreachable!()
                    };
                    let value::Value::ImmediateValue(value) = value.value.as_mut().unwrap() else {
                        unreachable!()
                    };
                    let value::immediate_value::Value::Tensor(value) =
                        value.value.as_mut().unwrap()
                    else {
                        unreachable!()
                    };
                    let tensor_value::Value::Strings(value) = value.value.as_mut().unwrap() else {
                        unreachable!()
                    };
                    value.values[0] = "TANH_APPROXIMATION".into();
                }
                2 => {
                    operation.outputs[0].r#type = Some(tensor_type(10));
                }
                3 => {
                    operation.outputs[0].name = "source".into();
                }
                4 => {
                    block.operations.push(block.operations[0].clone());
                }
                5 => {
                    changed.description.as_mut().unwrap().input[0]
                        .r#type
                        .as_mut()
                        .unwrap()
                        .is_optional = true;
                }
                _ => unreachable!(),
            }
            assert_eq!(
                classify(&changed, &changed.encode_to_vec()),
                None,
                "mutation {mutation}"
            );
        }
        let mut unknown = model.encode_to_vec();
        unknown.extend([0xf8, 0x7f, 0x01]);
        let decoded = Model::decode(unknown.as_slice()).unwrap();
        assert_eq!(classify(&decoded, &unknown), None);
    }

    #[test]
    fn feature_flexibility_cannot_relax_a_static_mil_axis() {
        use crate::protos::coreml::specification::SizeRange;
        use array_feature_type::{EnumeratedShapes, Shape, ShapeFlexibility, ShapeRange};
        for original in [model(), gelu_model(), sqrt_model()] {
            for mixed in [false, true] {
                let mut changed = original.clone();
                let model::Type::MlProgram(program) = changed.r#type.as_mut().unwrap() else {
                    unreachable!()
                };
                let function = program.functions.get_mut("main").unwrap();
                let output = &mut function
                    .block_specializations
                    .get_mut("CoreML7")
                    .unwrap()
                    .operations[0]
                    .outputs[0];
                if mixed {
                    for value in [&mut function.inputs[0], output] {
                        let value_type::Type::TensorType(tensor) =
                            value.r#type.as_mut().unwrap().r#type.as_mut().unwrap()
                        else {
                            unreachable!()
                        };
                        tensor.rank = 2;
                        tensor.dimensions.insert(
                            0,
                            mil::Dimension {
                                dimension: Some(dimension::Dimension::Unknown(Default::default())),
                            },
                        );
                    }
                }
                let description = changed.description.as_mut().unwrap();
                for feature in [&mut description.input[0], &mut description.output[0]] {
                    let feature_type::Type::MultiArrayType(array) =
                        feature.r#type.as_mut().unwrap().r#type.as_mut().unwrap()
                    else {
                        unreachable!()
                    };
                    array.shape_flexibility = Some(if mixed {
                        array.shape.insert(0, 1);
                        ShapeFlexibility::EnumeratedShapes(EnumeratedShapes {
                            shapes: vec![Shape { shape: vec![1, 4] }, Shape { shape: vec![2, 3] }],
                        })
                    } else {
                        ShapeFlexibility::ShapeRange(ShapeRange {
                            size_ranges: vec![SizeRange {
                                lower_bound: 1,
                                upper_bound: 8,
                            }],
                        })
                    });
                }
                assert_eq!(
                    classify(&changed, &changed.encode_to_vec()),
                    None,
                    "mixed={mixed}"
                );
                let description = changed.description.as_mut().unwrap();
                for feature in [&mut description.input[0], &mut description.output[0]] {
                    let feature_type::Type::MultiArrayType(array) =
                        feature.r#type.as_mut().unwrap().r#type.as_mut().unwrap()
                    else {
                        unreachable!()
                    };
                    match array.shape_flexibility.as_mut().unwrap() {
                        ShapeFlexibility::EnumeratedShapes(bounds) => bounds.shapes[1].shape[1] = 4,
                        ShapeFlexibility::ShapeRange(bounds) => {
                            bounds.size_ranges[0].lower_bound = 4;
                            bounds.size_ranges[0].upper_bound = 4;
                        }
                    }
                }
                assert!(classify(&changed, &changed.encode_to_vec()).is_some());
            }
        }
    }

    #[test]
    fn complete_source_cast_is_classified_without_name_or_proxy_heuristics() {
        let model = model();
        assert_eq!(
            classify(&model, &model.encode_to_vec()),
            Some(Kind::FloatCast(Direction::Narrow))
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
