use super::*;
use crate::converters::{CoremlMlProgramConverter, GraphConverter};
use crate::graph::{
    ConstantData, DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, to_dimension_vector,
};
use crate::operator_enums::MLOperandDataType;
use crate::operator_options::MLArgMinMaxOptions;
use crate::operators::Operation;
use prost::Message;

fn fixture(constant: bool) -> Model {
    let operand = |name: &str, kind, shape: &[u32]| Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: DataType::Int32,
            shape: to_dimension_vector(shape),
            pending_permutation: vec![],
        },
    };
    let graph = GraphInfo {
        operands: vec![
            operand(
                "source",
                if constant {
                    OperandKind::Constant
                } else {
                    OperandKind::Input
                },
                &[2, 3],
            ),
            operand("result", OperandKind::Output, &[2]),
        ],
        input_operands: if constant { vec![] } else { vec![0] },
        output_operands: vec![1],
        constant_operand_ids_to_handles: if constant {
            [(
                0,
                ConstantData {
                    data: [16_777_216_i32, 16_777_217, 1, -16_777_216, -16_777_217, -1]
                        .into_iter()
                        .flat_map(i32::to_le_bytes)
                        .collect(),
                    label: None,
                },
            )]
            .into_iter()
            .collect()
        } else {
            Default::default()
        },
        operations: vec![Operation::ArgMax {
            input: 0,
            axis: 1,
            outputs: vec![1],
            options: Some(MLArgMinMaxOptions {
                output_data_type: MLOperandDataType::Int32,
                ..Default::default()
            }),
        }],
        ..Default::default()
    };
    let bytes = CoremlMlProgramConverter.convert(&graph).unwrap().data;
    fn child(model: Model) -> Option<Model> {
        match model.r#type.as_ref()? {
            model::Type::Pipeline(pipeline) => pipeline.models.iter().cloned().find_map(child),
            model::Type::MlProgram(program)
                if program.functions.values().any(|function| {
                    function.block_specializations.values().any(|block| {
                        block
                            .operations
                            .iter()
                            .any(|operation| operation.r#type == "reduce_argmax")
                    })
                }) =>
            {
                Some(model)
            }
            _ => None,
        }
    }
    child(Model::decode(bytes.as_slice()).unwrap()).unwrap()
}

fn function(model: &mut Model) -> &mut mil::Function {
    let model::Type::MlProgram(program) = model.r#type.as_mut().unwrap() else {
        panic!()
    };
    program.functions.get_mut("main").unwrap()
}
fn operation(model: &mut Model) -> &mut mil::Operation {
    function(model)
        .block_specializations
        .get_mut("CoreML7")
        .unwrap()
        .operations
        .last_mut()
        .unwrap()
}
fn parameter<'a>(model: &'a mut Model, key: &str) -> &'a mut mil::Value {
    let mil::argument::binding::Binding::Value(value) =
        operation(model).inputs.get_mut(key).unwrap().arguments[0]
            .binding
            .as_mut()
            .unwrap()
    else {
        panic!()
    };
    value
}
fn tensor_mut(value: &mut mil::ValueType) -> &mut mil::TensorType {
    let mil::value_type::Type::TensorType(tensor) = value.r#type.as_mut().unwrap() else {
        panic!()
    };
    tensor
}
fn accepted(model: &Model) -> bool {
    classify(model, &model.encode_to_vec()).is_some()
}

#[test]
fn int32_arg_classifier_checks_complete_source_parameters_and_outputs() {
    for constant in [false, true] {
        assert!(accepted(&fixture(constant)));
    }
    for mutation in 0..11 {
        let mut model = fixture(false);
        match mutation {
            0 => {
                operation(&mut model)
                    .inputs
                    .insert("unknown".into(), Default::default());
            }
            1 => {
                let axis = operation(&mut model).inputs["axis"].clone();
                operation(&mut model)
                    .inputs
                    .insert("keep_dims".into(), axis);
            }
            2 => {
                let ty = tensor_mut(parameter(&mut model, "axis").r#type.as_mut().unwrap());
                ty.rank = 1;
                ty.dimensions = vec![mil::Dimension {
                    dimension: Some(dimension::Dimension::Constant(
                        dimension::ConstantDimension { size: 1 },
                    )),
                }];
            }
            3 => {
                let value::Value::ImmediateValue(value) =
                    parameter(&mut model, "axis").value.as_mut().unwrap()
                else {
                    panic!()
                };
                let value::immediate_value::Value::Tensor(value) = value.value.as_mut().unwrap()
                else {
                    panic!()
                };
                let tensor_value::Value::Ints(values) = value.value.as_mut().unwrap() else {
                    panic!()
                };
                values.values[0] = -1;
            }
            4 => {
                let value::Value::ImmediateValue(value) =
                    parameter(&mut model, "keep_dims").value.as_mut().unwrap()
                else {
                    panic!()
                };
                let value::immediate_value::Value::Tensor(value) = value.value.as_mut().unwrap()
                else {
                    panic!()
                };
                let tensor_value::Value::Bools(values) = value.value.as_mut().unwrap() else {
                    panic!()
                };
                values.values.push(false);
            }
            5 => {
                tensor_mut(function(&mut model).inputs[0].r#type.as_mut().unwrap()).data_type =
                    mil::DataType::Float32 as i32;
            }
            6 => {
                model.description.as_mut().unwrap().input[0]
                    .r#type
                    .as_mut()
                    .unwrap()
                    .is_optional = true;
            }
            7 => {
                tensor_mut(operation(&mut model).outputs[0].r#type.as_mut().unwrap()).dimensions
                    [0] = mil::Dimension {
                    dimension: Some(dimension::Dimension::Constant(
                        dimension::ConstantDimension { size: 1 },
                    )),
                };
            }
            8 => {
                let extra = operation(&mut model).clone();
                function(&mut model)
                    .block_specializations
                    .get_mut("CoreML7")
                    .unwrap()
                    .operations
                    .push(extra);
            }
            9 => {
                let ty = tensor_mut(parameter(&mut model, "keep_dims").r#type.as_mut().unwrap());
                ty.rank = 1;
                ty.dimensions = vec![mil::Dimension {
                    dimension: Some(dimension::Dimension::Constant(
                        dimension::ConstantDimension { size: 1 },
                    )),
                }];
            }
            _ => {
                let value::Value::ImmediateValue(value) =
                    parameter(&mut model, "axis").value.as_mut().unwrap()
                else {
                    panic!()
                };
                let value::immediate_value::Value::Tensor(value) = value.value.as_mut().unwrap()
                else {
                    panic!()
                };
                let tensor_value::Value::Ints(values) = value.value.as_mut().unwrap() else {
                    panic!()
                };
                values.values[0] = 2;
            }
        }
        assert!(!accepted(&model), "mutation={mutation}");
    }
    let model = fixture(false);
    let mut wire = model.encode_to_vec();
    wire.extend_from_slice(&[0xa0, 0x06, 0x01]);
    assert!(classify(&model, &wire).is_none());
}

#[test]
fn int32_arg_classifier_preserves_dynamic_axis_bounds_without_output_guessing() {
    let mut model = fixture(false);
    tensor_mut(function(&mut model).inputs[0].r#type.as_mut().unwrap()).dimensions[1] =
        mil::Dimension {
            dimension: Some(dimension::Dimension::Unknown(dimension::UnknownDimension {
                variadic: false,
            })),
        };
    let crate::protos::coreml::specification::feature_type::Type::MultiArrayType(input) =
        model.description.as_mut().unwrap().input[0]
            .r#type
            .as_mut()
            .unwrap()
            .r#type
            .as_mut()
            .unwrap()
    else {
        panic!()
    };
    input.shape_flexibility = Some(array_feature_type::ShapeFlexibility::ShapeRange(
        array_feature_type::ShapeRange {
            size_ranges: vec![
                crate::protos::coreml::specification::SizeRange {
                    lower_bound: 2,
                    upper_bound: 2,
                },
                crate::protos::coreml::specification::SizeRange {
                    lower_bound: 0,
                    upper_bound: 8,
                },
            ],
        },
    ));
    assert!(accepted(&model));
    let mut changed = model.clone();
    let crate::protos::coreml::specification::feature_type::Type::MultiArrayType(input) =
        changed.description.as_mut().unwrap().input[0]
            .r#type
            .as_mut()
            .unwrap()
            .r#type
            .as_mut()
            .unwrap()
    else {
        panic!()
    };
    let array_feature_type::ShapeFlexibility::ShapeRange(range) =
        input.shape_flexibility.as_mut().unwrap()
    else {
        panic!()
    };
    range.size_ranges[0].upper_bound = 3;
    assert!(!accepted(&changed));
    let mut changed = model.clone();
    let crate::protos::coreml::specification::feature_type::Type::MultiArrayType(input) =
        changed.description.as_mut().unwrap().input[0]
            .r#type
            .as_mut()
            .unwrap()
            .r#type
            .as_mut()
            .unwrap()
    else {
        panic!()
    };
    let array_feature_type::ShapeFlexibility::ShapeRange(range) =
        input.shape_flexibility.as_mut().unwrap()
    else {
        panic!()
    };
    range.size_ranges[1].upper_bound = i64::from(i32::MAX) + 1;
    assert!(!accepted(&changed));
    let crate::protos::coreml::specification::feature_type::Type::MultiArrayType(output) =
        model.description.as_mut().unwrap().output[0]
            .r#type
            .as_mut()
            .unwrap()
            .r#type
            .as_mut()
            .unwrap()
    else {
        panic!()
    };
    output.shape[0] = 3;
    assert!(!accepted(&model));
}
