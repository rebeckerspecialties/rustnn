//! Exact transport for a complete Int32 Identity child, including produced values.

use super::typed_unary;
use crate::protos::coreml::mil_spec::{self as mil, argument};
use crate::protos::coreml::specification::{Model, array_feature_type, model};

pub(super) fn classify(model: &Model, source: &[u8]) -> bool {
    checked(model, source).is_some()
}

fn checked(model: &Model, source: &[u8]) -> Option<()> {
    if !typed_unary::known_source(model, source) {
        return None;
    }
    let model::Type::MlProgram(program) = model.r#type.as_ref()? else {
        return None;
    };
    if program.version != 1 || program.functions.len() != 1 || !program.attributes.is_empty() {
        return None;
    }
    let function = program.functions.get("main")?;
    if function.opset != "CoreML7"
        || function.inputs.len() != 1
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
    if operation.r#type != "identity"
        || operation.inputs.len() != 1
        || operation.outputs.len() != 1
        || !operation.attributes.is_empty()
        || !operation.blocks.is_empty()
    {
        return None;
    }
    let bindings = &operation.inputs.get("x")?.arguments;
    if bindings.len() != 1 {
        return None;
    }
    let argument::binding::Binding::Name(name) = bindings[0].binding.as_ref()? else {
        return None;
    };
    let input = &function.inputs[0];
    let output = &operation.outputs[0];
    if *name != input.name || input.name == output.name || block.outputs[0] != output.name {
        return None;
    }
    let input_type = typed_unary::tensor(input.r#type.as_ref()?)?;
    let output_type = typed_unary::tensor(output.r#type.as_ref()?)?;
    if input_type.data_type != mil::DataType::Int32 as i32 || input_type != output_type {
        return None;
    }
    let description = model.description.as_ref()?;
    if description.input.len() != 1
        || description.output.len() != 1
        || description.input[0].name != input.name
        || description.output[0].name != output.name
        || description.input[0].r#type.as_ref()?.is_optional
        || description.output[0].r#type.as_ref()?.is_optional
    {
        return None;
    }
    let input_feature = typed_unary::array(&description.input[0])?;
    let output_feature = typed_unary::array(&description.output[0])?;
    if input_feature.data_type != array_feature_type::ArrayDataType::Int32 as i32
        || input_feature != output_feature
        || !typed_unary::compatible_shape(input_type, input_feature)
        || !typed_unary::compatible_shape(output_type, output_feature)
    {
        return None;
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converters::{CoremlMlProgramConverter, GraphConverter};
    use crate::graph::{
        DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, to_dimension_vector,
    };
    use crate::operators::Operation;
    use prost::Message;

    fn fixture() -> Model {
        let operand = |name: &str, kind| Operand {
            name: Some(name.into()),
            kind,
            descriptor: OperandDescriptor {
                data_type: DataType::Int32,
                shape: to_dimension_vector(&[4]),
                pending_permutation: vec![],
            },
        };
        let graph = GraphInfo {
            operands: vec![
                operand("source", OperandKind::Input),
                operand("result", OperandKind::Output),
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![Operation::Identity {
                input: 0,
                outputs: vec![1],
                options: None,
            }],
            ..Default::default()
        };
        Model::decode(
            CoremlMlProgramConverter
                .convert(&graph)
                .unwrap()
                .data
                .as_slice(),
        )
        .unwrap()
    }

    #[test]
    fn int32_identity_requires_the_complete_typed_program_not_a_copy_name() {
        let original = fixture();
        assert!(classify(&original, &original.encode_to_vec()));
        let mut unknown = original.encode_to_vec();
        unknown.extend([0xa0, 0x06, 1]);
        assert!(!classify(
            &Model::decode(unknown.as_slice()).unwrap(),
            &unknown
        ));
        for mutation in 0..12 {
            let mut model = original.clone();
            let model::Type::MlProgram(program) = model.r#type.as_mut().unwrap() else {
                panic!()
            };
            let function = program.functions.get_mut("main").unwrap();
            let block = function.block_specializations.get_mut("CoreML7").unwrap();
            let operation = &mut block.operations[0];
            match mutation {
                0 => operation.r#type = "cast".into(),
                1 => operation.r#type = "transpose".into(),
                2 => {
                    operation.inputs.get_mut("x").unwrap().arguments[0].binding =
                        Some(argument::binding::Binding::Name("unproved".into()))
                }
                3 => operation.outputs[0].r#type = None,
                4 => {
                    operation
                        .attributes
                        .insert("unknown".into(), Default::default());
                }
                5 => operation.blocks.push(Default::default()),
                6 => {
                    let copy = operation.clone();
                    block.operations.push(copy);
                }
                7 => block.outputs[0] = "unproved".into(),
                8 => function.opset = "CoreML8".into(),
                9 => {
                    model.description.as_mut().unwrap().input[0]
                        .r#type
                        .as_mut()
                        .unwrap()
                        .is_optional = true
                }
                10 => model.description.as_mut().unwrap().output[0].name = "unproved".into(),
                11 => {
                    let mil::value_type::Type::TensorType(ty) = operation.outputs[0]
                        .r#type
                        .as_mut()
                        .unwrap()
                        .r#type
                        .as_mut()
                        .unwrap()
                    else {
                        panic!()
                    };
                    ty.data_type = mil::DataType::Float32 as i32;
                }
                _ => unreachable!(),
            }
            assert!(
                !classify(&model, &model.encode_to_vec()),
                "mutation {mutation}"
            );
        }
    }

    #[test]
    fn int32_identity_requires_bounded_matching_actual_shape_contracts() {
        use crate::protos::coreml::specification::feature_type;
        let mut model = fixture();
        let description = model.description.as_mut().unwrap();
        for feature in description.input.iter_mut().chain(&mut description.output) {
            let feature_type::Type::MultiArrayType(array) =
                feature.r#type.as_mut().unwrap().r#type.as_mut().unwrap()
            else {
                panic!()
            };
            array.shape_flexibility = Some(array_feature_type::ShapeFlexibility::ShapeRange(
                array_feature_type::ShapeRange {
                    size_ranges: vec![crate::protos::coreml::specification::SizeRange {
                        lower_bound: 1,
                        upper_bound: 4,
                    }],
                },
            ));
        }
        assert!(
            !classify(&model, &model.encode_to_vec()),
            "a static MIL axis must match every allowed shape"
        );
        let model::Type::MlProgram(program) = model.r#type.as_mut().unwrap() else {
            panic!()
        };
        let function = program.functions.get_mut("main").unwrap();
        let output = &mut function
            .block_specializations
            .get_mut("CoreML7")
            .unwrap()
            .operations[0]
            .outputs[0];
        for value in [&mut function.inputs[0], output] {
            let mil::value_type::Type::TensorType(ty) =
                value.r#type.as_mut().unwrap().r#type.as_mut().unwrap()
            else {
                panic!()
            };
            ty.dimensions[0].dimension = Some(mil::dimension::Dimension::Unknown(
                mil::dimension::UnknownDimension { variadic: false },
            ));
        }
        assert!(classify(&model, &model.encode_to_vec()));
        let mut changed = model.clone();
        let feature = &mut changed.description.as_mut().unwrap().output[0];
        let feature_type::Type::MultiArrayType(array) =
            feature.r#type.as_mut().unwrap().r#type.as_mut().unwrap()
        else {
            panic!()
        };
        array.shape = vec![3];
        assert!(
            !classify(&changed, &changed.encode_to_vec()),
            "input/output contracts must match"
        );
        let mut changed = model;
        let feature = &mut changed.description.as_mut().unwrap().input[0];
        let feature_type::Type::MultiArrayType(array) =
            feature.r#type.as_mut().unwrap().r#type.as_mut().unwrap()
        else {
            panic!()
        };
        array.shape_flexibility = None;
        assert!(
            !classify(&changed, &changed.encode_to_vec()),
            "unknown MIL axes need actual feature bounds"
        );
    }
}
