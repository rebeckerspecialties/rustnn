//! Quantization parameters that already broadcast must not freeze active extents.

use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
#[cfg(feature = "dynamic-inputs")]
use rustnn::graph::DynamicDimension;
use rustnn::graph::{DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind};
use rustnn::operators::Operation;
use rustnn::protos::coreml::{mil_spec, specification};
use serde_json::json;

#[cfg(feature = "dynamic-inputs")]
fn dynamic(name: &str, max_size: u32) -> Dimension {
    Dimension::Dynamic(DynamicDimension {
        name: name.into(),
        max_size,
    })
}

fn graph(
    name: &str,
    float_type: DataType,
    shape: Vec<Dimension>,
    parameters: Vec<Dimension>,
) -> GraphInfo {
    let quantize = name == "quantizeLinear";
    let operand = |name: &str, kind, data_type, shape| Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type,
            shape,
            pending_permutation: vec![],
        },
    };
    GraphInfo {
        operands: vec![
            operand(
                "input",
                OperandKind::Input,
                if quantize {
                    float_type
                } else {
                    DataType::Int32
                },
                shape.clone(),
            ),
            operand("scale", OperandKind::Input, float_type, parameters.clone()),
            operand(
                "zero_point",
                OperandKind::Input,
                DataType::Int32,
                parameters,
            ),
            operand(
                "result",
                OperandKind::Output,
                if quantize {
                    DataType::Int32
                } else {
                    float_type
                },
                shape,
            ),
        ],
        operations: vec![
            Operation::from_json_attributes(name, &[0, 1, 2], &[3], &json!({})).unwrap(),
        ],
        input_operands: vec![0, 1, 2],
        output_operands: vec![3],
        ..Default::default()
    }
}

fn operations(model: &specification::Model) -> Vec<&mil_spec::Operation> {
    let programs = match model.r#type.as_ref().unwrap() {
        specification::model::Type::MlProgram(program) => vec![program],
        specification::model::Type::Pipeline(pipeline) => pipeline
            .models
            .iter()
            .map(|model| {
                let specification::model::Type::MlProgram(program) = model.r#type.as_ref().unwrap()
                else {
                    panic!("MLProgram child")
                };
                program
            })
            .collect(),
        _ => panic!("MLProgram or Pipeline"),
    };
    programs
        .into_iter()
        .flat_map(|program| &program.functions)
        .flat_map(|(_, function)| &function.block_specializations)
        .flat_map(|(_, block)| &block.operations)
        .collect()
}

#[test]
fn scalar_and_static_broadcast_parameters_need_no_reshape() {
    for shape in [vec![], vec![Dimension::Static(3), Dimension::Static(4)]] {
        let parameters = vec![Dimension::Static(1); shape.len()];
        for name in ["quantizeLinear", "dequantizeLinear"] {
            let converted = CoremlMlProgramConverter
                .convert(&graph(
                    name,
                    DataType::Float32,
                    shape.clone(),
                    parameters.clone(),
                ))
                .unwrap();
            let model = specification::Model::decode(converted.data.as_slice()).unwrap();
            assert!(
                !operations(&model)
                    .iter()
                    .any(|operation| operation.r#type == "reshape"),
                "{name}:{shape:?}"
            );
        }
    }
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn broadcast_quantization_keeps_symbolic_data_dimensions() {
    for float_type in [DataType::Float16, DataType::Float32] {
        for parameters in [
            vec![Dimension::Static(1), Dimension::Static(1)],
            vec![Dimension::Static(1), Dimension::Static(4)],
        ] {
            for name in ["quantizeLinear", "dequantizeLinear"] {
                let original = graph(
                    name,
                    float_type,
                    vec![dynamic("batch", 3), Dimension::Static(4)],
                    parameters.clone(),
                );
                let converted = CoremlMlProgramConverter.convert(&original).unwrap();
                let model = specification::Model::decode(converted.data.as_slice()).unwrap();
                let operations = operations(&model);
                assert!(
                    !operations
                        .iter()
                        .any(|operation| operation.r#type == "reshape"),
                    "{name}:{float_type:?}"
                );
                for operation in operations {
                    for output in &operation.outputs {
                        if output.name.contains("_q_") || output.name.contains("_dq_") {
                            let mil_spec::value_type::Type::TensorType(tensor) =
                                output.r#type.as_ref().unwrap().r#type.as_ref().unwrap()
                            else {
                                panic!("tensor")
                            };
                            if output.name.contains("scale") || output.name.contains("zp") {
                                continue;
                            }
                            assert!(
                                matches!(
                                    tensor.dimensions[0].dimension,
                                    Some(mil_spec::dimension::Dimension::Unknown(_))
                                ),
                                "active batch must not be replaced with its max: {}",
                                output.name
                            );
                        }
                    }
                }
                assert_eq!(
                    original.operands[0].descriptor.shape[0],
                    dynamic("batch", 3)
                );
            }
        }
    }
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn unproven_dynamic_block_ratio_is_rejected_not_frozen_at_maximum() {
    for name in ["quantizeLinear", "dequantizeLinear"] {
        let original = graph(
            name,
            DataType::Float32,
            vec![dynamic("columns", 6)],
            vec![Dimension::Static(2)],
        );
        let error = match CoremlMlProgramConverter.convert(&original) {
            Ok(_) => panic!("{name} accepted an unproven dynamic block ratio"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("dynamic blockwise"), "{error}");
    }
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn native_constant_parameters_cannot_cover_a_dynamic_axis_by_its_maximum() {
    for name in ["quantizeLinear", "dequantizeLinear"] {
        let mut original = graph(
            name,
            DataType::Float32,
            vec![dynamic("batch", 3), Dimension::Static(4)],
            vec![Dimension::Static(3), Dimension::Static(1)],
        );
        original.operands[if name == "quantizeLinear" { 3 } else { 0 }]
            .descriptor
            .data_type = DataType::Int8;
        original.operands[2].descriptor.data_type = DataType::Int8;
        for id in [1, 2] {
            original.operands[id].kind = OperandKind::Constant;
        }
        original.input_operands = vec![0];
        original.constant_operand_ids_to_handles = [
            (
                1,
                rustnn::graph::ConstantData {
                    data: [1.0f32; 3].into_iter().flat_map(f32::to_le_bytes).collect(),
                    label: None,
                },
            ),
            (
                2,
                rustnn::graph::ConstantData {
                    data: vec![0; 3],
                    label: None,
                },
            ),
        ]
        .into_iter()
        .collect();
        let error = match CoremlMlProgramConverter.convert(&original) {
            Ok(_) => panic!("{name} used the maximum as a fixed per-channel extent"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("dynamic blockwise"), "{error}");
    }
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn static_channel_blocks_expand_only_parameters_with_dynamic_batch() {
    for name in ["quantizeLinear", "dequantizeLinear"] {
        let original = graph(
            name,
            DataType::Float32,
            vec![dynamic("batch", 3), Dimension::Static(4)],
            vec![Dimension::Static(1), Dimension::Static(2)],
        );
        let converted = CoremlMlProgramConverter.convert(&original).unwrap();
        let model = specification::Model::decode(converted.data.as_slice()).unwrap();
        let operations = operations(&model);
        assert!(
            operations
                .iter()
                .any(|operation| operation.r#type == "tile")
        );
        for operation in operations
            .iter()
            .filter(|operation| operation.r#type == "reshape")
        {
            let mil_spec::argument::binding::Binding::Name(input) =
                operation.inputs["x"].arguments[0].binding.as_ref().unwrap()
            else {
                panic!("named parameter")
            };
            assert!(
                input.contains("scale") || input.contains("zp"),
                "data must not be reshaped: {input}"
            );
        }
    }
}

#[cfg(all(
    target_os = "macos",
    feature = "coreml-runtime",
    feature = "dynamic-inputs"
))]
mod runtime {
    use super::*;
    use rustnn::backend_selection::{BackendDevice, DeviceType};
    use rustnn::mlcontext::{
        MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    };
    use rustnn::operator_enums::MLOperandDataType;

    #[test]
    fn reused_quantization_graphs_grow_and_shrink_active_batch() {
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            for name in ["quantizeLinear", "dequantizeLinear"] {
                for (columns, scales, zero_points) in [
                    (1, vec![2.0_f32], vec![1_i32]),
                    (2, vec![2.0_f32, 4.0], vec![1_i32, -1]),
                ] {
                    let mut context = MLContext::create(
                        &MLContextOptions::new(
                            MLPowerPreference::Default,
                            policy != DeviceType::Cpu,
                        )
                        .with_rustnn_device_hint(BackendDevice::Coreml {
                            device_type: policy,
                        }),
                    )
                    .unwrap();
                    let original = graph(
                        name,
                        DataType::Float32,
                        vec![dynamic("batch", 3), Dimension::Static(4)],
                        vec![Dimension::Static(1), Dimension::Static(columns)],
                    );
                    let mut compiled = context.rustnn_build_graph(original).unwrap();
                    let scale = context
                        .create_tensor(
                            &MLTensorDescriptor::new(
                                MLOperandDataType::Float32,
                                vec![1, columns as u64],
                            )
                            .to_writable(),
                        )
                        .unwrap();
                    let zero = context
                        .create_tensor(
                            &MLTensorDescriptor::new(
                                MLOperandDataType::Int32,
                                vec![1, columns as u64],
                            )
                            .to_writable(),
                        )
                        .unwrap();
                    context.write_tensor(&scale, &scales).unwrap();
                    context.write_tensor(&zero, &zero_points).unwrap();
                    for batch in [1, 3, 1] {
                        let quantize = name == "quantizeLinear";
                        let input_type = if quantize {
                            MLOperandDataType::Float32
                        } else {
                            MLOperandDataType::Int32
                        };
                        let output_type = if quantize {
                            MLOperandDataType::Int32
                        } else {
                            MLOperandDataType::Float32
                        };
                        let input = context
                            .create_tensor(
                                &MLTensorDescriptor::new(input_type, vec![batch, 4]).to_writable(),
                            )
                            .unwrap();
                        let output = context
                            .create_tensor(
                                &MLTensorDescriptor::new(output_type, vec![batch, 4]).to_readable(),
                            )
                            .unwrap();
                        let values: Vec<i32> =
                            (0..batch as usize).flat_map(|_| [-3, -1, 2, 5]).collect();
                        if quantize {
                            context
                                .write_tensor(
                                    &input,
                                    &values.iter().map(|&value| value as f32).collect::<Vec<_>>(),
                                )
                                .unwrap();
                        } else {
                            context.write_tensor(&input, &values).unwrap();
                        }
                        context
                            .dispatch(
                                &mut compiled,
                                &MLNamedTensors::from_iter([
                                    ("input", &input),
                                    ("scale", &scale),
                                    ("zero_point", &zero),
                                ]),
                                &MLNamedTensors::from_iter([("result", &output)]),
                            )
                            .unwrap();
                        if quantize {
                            let mut actual = vec![i32::MIN; values.len()];
                            context.read_tensor(&output, &mut actual).unwrap();
                            let expected: Vec<i32> = values
                                .iter()
                                .enumerate()
                                .map(|(index, &value)| {
                                    let parameter = (index % 4) / (4 / columns as usize);
                                    (value as f32 / scales[parameter]).round_ties_even() as i32
                                        + zero_points[parameter]
                                })
                                .collect();
                            assert_eq!(actual, expected, "{name}:{policy:?}:batch={batch}");
                        } else {
                            let mut actual = vec![f32::NAN; values.len()];
                            context.read_tensor(&output, &mut actual).unwrap();
                            let expected: Vec<f32> = values
                                .iter()
                                .enumerate()
                                .map(|(index, &value)| {
                                    let parameter = (index % 4) / (4 / columns as usize);
                                    (value - zero_points[parameter]) as f32 * scales[parameter]
                                })
                                .collect();
                            assert_eq!(actual, expected, "{name}:{policy:?}:batch={batch}");
                        }
                    }
                }
            }
        }
    }
}
