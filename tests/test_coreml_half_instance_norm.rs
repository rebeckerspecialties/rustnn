//! Native instance normalization keeps its complete computation in Float32.

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
mod runtime {
    use super::*;
    use half::f16;
    use rustnn::backend_selection::{BackendDevice, DeviceType};
    use rustnn::mlcontext::{
        MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    };
    use rustnn::operator_enums::MLOperandDataType;

    #[test]
    fn float32_instance_norm_honors_zero_and_tiny_epsilon_for_both_layouts() {
        let unit = 2f32.powi(-24);
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            for layout in ["nchw", "nhwc"] {
                for epsilon in [0., f64::from(unit), 1e-6] {
                    for affine in [false, true] {
                        let source = float32_graph(layout, epsilon, affine);
                        let mut context = MLContext::create(
                            &MLContextOptions::new(
                                MLPowerPreference::Default,
                                policy != DeviceType::Cpu,
                            )
                            .with_rustnn_device_hint(
                                BackendDevice::Coreml {
                                    device_type: policy,
                                },
                            ),
                        )
                        .unwrap();
                        let mut graph = context.rustnn_build_graph(source).unwrap();
                        let descriptor =
                            MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, 2, 2, 2]);
                        let input = context
                            .create_tensor(&descriptor.clone().to_writable())
                            .unwrap();
                        let output = context.create_tensor(&descriptor.to_readable()).unwrap();
                        let values: Vec<_> = (0..8)
                            .map(|i| {
                                let position = if layout == "nchw" { i % 4 } else { i / 2 };
                                if position % 2 == 0 { unit } else { -unit }
                            })
                            .collect();
                        let expected: Vec<_> = values
                            .iter()
                            .enumerate()
                            .map(|(i, &value)| {
                                let channel = if layout == "nchw" { i / 4 } else { i % 2 };
                                let normalized = f64::from(value)
                                    / (f64::from(unit).powi(2) + f64::from(epsilon as f32)).sqrt();
                                if affine {
                                    (normalized * [0.75, -0.5][channel]
                                        + f64::from([unit, -unit][channel]))
                                        as f32
                                } else {
                                    normalized as f32
                                }
                            })
                            .collect();
                        context.write_tensor(&input, &values).unwrap();
                        context
                            .dispatch(
                                &mut graph,
                                &MLNamedTensors::from([("input", &input)]),
                                &MLNamedTensors::from([("result", &output)]),
                            )
                            .unwrap();
                        let mut actual = [0f32; 8];
                        context.read_tensor(&output, &mut actual).unwrap();
                        for (i, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                            assert!(
                                (i64::from(actual.to_bits()) - i64::from(expected.to_bits())).abs()
                                    <= 2,
                                "{layout}/{policy:?}/epsilon={epsilon}/affine={affine}/index={i}: \
                                 {actual} != {expected}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn minimum_half_epsilon_is_not_replaced_by_the_native_default() {
        let epsilon = 2f64.powi(-25) + 2f64.powi(-55);
        let values = [1u16, 0x8001, 1, 0x8001];
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            for layout in ["nchw", "nhwc"] {
                let mut source = graph(layout, Some(epsilon));
                let shape = if layout == "nchw" {
                    vec![1, 1, 2, 2]
                } else {
                    vec![1, 2, 2, 1]
                };
                for id in [0, 3] {
                    source.operands[id].descriptor.shape =
                        shape.iter().copied().map(Dimension::Static).collect();
                }
                for id in [1, 2] {
                    source.operands[id].descriptor.shape = vec![Dimension::Static(1)];
                }
                source
                    .constant_operand_ids_to_handles
                    .get_mut(&1)
                    .unwrap()
                    .data = 0x3c00u16.to_le_bytes().to_vec();
                source
                    .constant_operand_ids_to_handles
                    .get_mut(&2)
                    .unwrap()
                    .data = 0u16.to_le_bytes().to_vec();
                let mut context = MLContext::create(
                    &MLContextOptions::new(MLPowerPreference::Default, policy != DeviceType::Cpu)
                        .with_rustnn_device_hint(BackendDevice::Coreml {
                            device_type: policy,
                        }),
                )
                .unwrap();
                let mut graph = context.rustnn_build_graph(source).unwrap();
                let descriptor = MLTensorDescriptor::new(
                    MLOperandDataType::Float16,
                    shape.iter().map(|&n| u64::from(n)).collect(),
                );
                let input = context
                    .create_tensor(&descriptor.clone().to_writable())
                    .unwrap();
                let output = context.create_tensor(&descriptor.to_readable()).unwrap();
                context.write_tensor(&input, &values).unwrap();
                context
                    .dispatch(
                        &mut graph,
                        &MLNamedTensors::from([("input", &input)]),
                        &MLNamedTensors::from([("result", &output)]),
                    )
                    .unwrap();
                let mut actual = [0u16; 4];
                context.read_tensor(&output, &mut actual).unwrap();
                assert_eq!(
                    actual,
                    [0x0c00, 0x8c00, 0x0c00, 0x8c00],
                    "{layout}/{policy:?}"
                );
            }
        }
    }

    #[test]
    fn low_variance_half_instance_norm_matches_independent_centered_formula() {
        let unit = 2f32.powi(-24);
        let channels = [
            [
                1. - 2f32.powi(-11),
                1.,
                1. + 2f32.powi(-10),
                1. + 2f32.powi(-9),
            ],
            [unit, -3. * unit, 0., 7. * unit],
        ];
        let scale = [0.75, -0.5];
        let bias = [unit, -unit];
        let epsilon = 2f64.powi(-14);
        let mut reference = [[0u16; 4]; 2];
        for channel in 0..2 {
            let mean = channels[channel]
                .iter()
                .map(|value| f64::from(*value))
                .sum::<f64>()
                / 4.;
            let variance = channels[channel]
                .iter()
                .map(|value| (f64::from(*value) - mean).powi(2))
                .sum::<f64>()
                / 4.;
            for position in 0..4 {
                reference[channel][position] = f16::from_f64(
                    (f64::from(channels[channel][position]) - mean) / (variance + epsilon).sqrt()
                        * scale[channel]
                        + f64::from(bias[channel]),
                )
                .to_bits();
            }
        }
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            for layout in ["nchw", "nhwc"] {
                let mut context = MLContext::create(
                    &MLContextOptions::new(MLPowerPreference::Default, policy != DeviceType::Cpu)
                        .with_rustnn_device_hint(BackendDevice::Coreml {
                            device_type: policy,
                        }),
                )
                .unwrap();
                let mut graph = context
                    .rustnn_build_graph(graph(layout, Some(epsilon)))
                    .unwrap();
                let descriptor =
                    MLTensorDescriptor::new(MLOperandDataType::Float16, vec![1, 2, 2, 2]);
                let input = context
                    .create_tensor(&descriptor.clone().to_writable())
                    .unwrap();
                let output = context.create_tensor(&descriptor.to_readable()).unwrap();
                let coordinates: Vec<_> = if layout == "nchw" {
                    (0..2)
                        .flat_map(|channel| (0..4).map(move |position| (channel, position)))
                        .collect()
                } else {
                    (0..4)
                        .flat_map(|position| (0..2).map(move |channel| (channel, position)))
                        .collect()
                };
                let values: Vec<_> = coordinates
                    .iter()
                    .map(|&(channel, position)| {
                        f16::from_f32(channels[channel][position]).to_bits()
                    })
                    .collect();
                let expected: Vec<_> = coordinates
                    .iter()
                    .map(|&(channel, position)| reference[channel][position])
                    .collect();
                context.write_tensor(&input, &values).unwrap();
                context
                    .dispatch(
                        &mut graph,
                        &MLNamedTensors::from([("input", &input)]),
                        &MLNamedTensors::from([("result", &output)]),
                    )
                    .unwrap();
                let mut actual = vec![0u16; 8];
                context.read_tensor(&output, &mut actual).unwrap();
                assert_eq!(actual, expected, "{policy:?}, {layout}");
            }
        }
    }
}

use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{
    ConstantData, DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
};
use rustnn::operators::Operation;
use rustnn::protos::coreml::{mil_spec, specification};
use serde_json::json;

fn operand(name: &str, shape: &[u32], kind: OperandKind) -> Operand {
    Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: DataType::Float16,
            shape: shape.iter().copied().map(Dimension::Static).collect(),
            pending_permutation: vec![],
        },
    }
}

fn graph(layout: &str, epsilon: Option<f64>) -> GraphInfo {
    let mut attributes = json!({"layout":layout,"scale":1,"bias":2});
    if let Some(epsilon) = epsilon {
        attributes["epsilon"] = json!(epsilon);
    }
    GraphInfo {
        operands: vec![
            operand("input", &[1, 2, 2, 2], OperandKind::Input),
            operand("scale", &[2], OperandKind::Constant),
            operand("bias", &[2], OperandKind::Constant),
            operand("result", &[1, 2, 2, 2], OperandKind::Output),
        ],
        input_operands: vec![0],
        output_operands: vec![3],
        operations: vec![
            Operation::from_json_attributes("instanceNormalization", &[0], &[3], &attributes)
                .unwrap(),
        ],
        constant_operand_ids_to_handles: [
            (1, [0.75f32, -0.5]),
            (2, [2.0f32.powi(-24), -2.0f32.powi(-24)]),
        ]
        .into_iter()
        .map(|(id, values)| {
            (
                id,
                ConstantData {
                    data: values
                        .into_iter()
                        .flat_map(|v| half::f16::from_f32(v).to_bits().to_le_bytes())
                        .collect(),
                    label: None,
                },
            )
        })
        .collect(),
        ..Default::default()
    }
}

fn operations(model: specification::Model) -> Vec<mil_spec::Operation> {
    match model.r#type.unwrap() {
        specification::model::Type::MlProgram(program) => {
            let function = &program.functions["main"];
            function.block_specializations[&function.opset]
                .operations
                .clone()
        }
        specification::model::Type::Pipeline(pipeline) => {
            pipeline.models.into_iter().flat_map(operations).collect()
        }
        _ => panic!("MLProgram or Pipeline"),
    }
}

fn float32_graph(layout: &str, epsilon: f64, affine: bool) -> GraphInfo {
    let mut graph = graph(layout, Some(epsilon));
    for operand in &mut graph.operands {
        operand.descriptor.data_type = DataType::Float32;
    }
    for constant in graph.constant_operand_ids_to_handles.values_mut() {
        constant.data = constant
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|bytes| {
                half::f16::from_bits(u16::from_le_bytes(*bytes))
                    .to_f32()
                    .to_le_bytes()
            })
            .collect();
    }
    if !affine {
        graph.operands = vec![graph.operands[0].clone(), graph.operands[3].clone()];
        graph.output_operands = vec![1];
        graph.operations = vec![
            Operation::from_json_attributes(
                "instanceNormalization",
                &[0],
                &[1],
                &json!({"layout":layout,"epsilon":epsilon}),
            )
            .unwrap(),
        ];
        graph.constant_operand_ids_to_handles.clear();
    }
    graph
}

#[test]
fn float32_instance_norm_uses_explicit_spatial_axes_and_preserves_epsilon() {
    for layout in ["nchw", "nhwc"] {
        for affine in [false, true] {
            for epsilon in [0., 2f64.powi(-24), 1e-6] {
                let graph = float32_graph(layout, epsilon, affine);
                let original = serde_json::to_vec(&graph).unwrap();
                let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
                assert_eq!(serde_json::to_vec(&graph).unwrap(), original);
                let model = specification::Model::decode(converted.data.as_slice()).unwrap();
                if affine {
                    assert!(
                        matches!(model.r#type, Some(specification::model::Type::Pipeline(_))),
                        "source Float32 spatial normalization must materialize before affine work"
                    );
                }
                let operations = operations(model);
                assert!(!operations.iter().any(|op| op.r#type == "instance_norm"));
                let kernel = operations
                    .iter()
                    .find(|op| op.r#type == "layer_norm")
                    .unwrap();
                assert!(!kernel.inputs.contains_key("gamma"));
                assert!(!kernel.inputs.contains_key("beta"));
                let immediate = |name: &str| {
                    let Some(mil_spec::argument::binding::Binding::Value(value)) =
                        &kernel.inputs[name].arguments[0].binding
                    else {
                        panic!("immediate {name}")
                    };
                    let Some(mil_spec::value::Value::ImmediateValue(value)) = &value.value else {
                        panic!("immediate {name}")
                    };
                    let Some(mil_spec::value::immediate_value::Value::Tensor(tensor)) =
                        &value.value
                    else {
                        panic!("tensor {name}")
                    };
                    tensor.value.as_ref().unwrap()
                };
                let mil_spec::tensor_value::Value::Ints(axes) = immediate("axes") else {
                    panic!("axes")
                };
                assert_eq!(
                    axes.values,
                    if layout == "nchw" {
                        vec![2, 3]
                    } else {
                        vec![1, 2]
                    }
                );
                let mil_spec::tensor_value::Value::Floats(values) = immediate("epsilon") else {
                    panic!("epsilon")
                };
                assert_eq!(values.values, [epsilon as f32]);
            }
        }
    }
}

#[test]
fn half_instance_norm_uses_spatial_layer_norm_even_without_affine_parameters() {
    for layout in ["nchw", "nhwc"] {
        for affine in [true, false] {
            let mut graph = graph(layout, Some(2f64.powi(-25) + 2f64.powi(-55)));
            if !affine {
                graph.operands = vec![graph.operands[0].clone(), graph.operands[3].clone()];
                graph.output_operands = vec![1];
                graph.operations = vec![
                    Operation::from_json_attributes(
                        "instanceNormalization",
                        &[0],
                        &[1],
                        &json!({"layout":layout,"epsilon":2f64.powi(-25) + 2f64.powi(-55)}),
                    )
                    .unwrap(),
                ];
                graph.constant_operand_ids_to_handles.clear();
            }
            let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
            let operations =
                operations(specification::Model::decode(converted.data.as_slice()).unwrap());
            assert!(!operations.iter().any(|op| op.r#type == "instance_norm"));
            let kernel = operations
                .iter()
                .find(|op| op.r#type == "layer_norm")
                .unwrap();
            let mil_spec::argument::binding::Binding::Value(value) =
                kernel.inputs["axes"].arguments[0].binding.as_ref().unwrap()
            else {
                panic!("axes")
            };
            let Some(mil_spec::value::Value::ImmediateValue(value)) = &value.value else {
                panic!("immediate")
            };
            let Some(mil_spec::value::immediate_value::Value::Tensor(tensor)) = &value.value else {
                panic!("tensor")
            };
            let Some(mil_spec::tensor_value::Value::Ints(axes)) = &tensor.value else {
                panic!("ints")
            };
            assert_eq!(
                axes.values,
                if layout == "nchw" {
                    vec![2, 3]
                } else {
                    vec![1, 2]
                }
            );
        }
    }
}

fn check(graph: &GraphInfo, expected_epsilon: f32) {
    use mil_spec::{argument::binding::Binding, tensor_value, value, value_type};
    let original = serde_json::to_vec(graph).unwrap();
    let converted = CoremlMlProgramConverter.convert(graph).unwrap();
    assert_eq!(
        serde_json::to_vec(graph).unwrap(),
        original,
        "conversion must preserve declared GraphInfo"
    );
    let operations = operations(specification::Model::decode(converted.data.as_slice()).unwrap());
    let kernels: Vec<_> = operations
        .iter()
        .filter(|op| matches!(op.r#type.as_str(), "instance_norm" | "layer_norm"))
        .collect();
    assert_eq!(kernels.len(), 1);
    assert!(!kernels[0].inputs.contains_key("gamma"));
    assert!(!kernels[0].inputs.contains_key("beta"));
    for output in &kernels[0].outputs {
        let Some(value_type::Type::TensorType(tensor)) = &output.r#type.as_ref().unwrap().r#type
        else {
            panic!("tensor")
        };
        assert_eq!(
            tensor.data_type,
            mil_spec::DataType::Float32 as i32,
            "native instance_norm must not center values in Half"
        );
    }
    let Binding::Value(epsilon) = kernels[0].inputs["epsilon"].arguments[0]
        .binding
        .as_ref()
        .unwrap()
    else {
        panic!("epsilon")
    };
    let value::Value::ImmediateValue(immediate) = epsilon.value.as_ref().unwrap() else {
        panic!("immediate")
    };
    let value::immediate_value::Value::Tensor(tensor) = immediate.value.as_ref().unwrap() else {
        panic!("tensor")
    };
    let tensor_value::Value::Floats(values) = tensor.value.as_ref().unwrap() else {
        panic!("widened epsilon")
    };
    assert_eq!(values.values, [expected_epsilon]);
    let constants: Vec<_> = operations
        .iter()
        .filter(|op| op.r#type == "const")
        .flat_map(|op| op.attributes.values())
        .filter(|v| matches!(v.value, Some(value::Value::BlobFileValue(_))))
        .collect();
    if graph.operations[0]
        .all_input_operands()
        .iter()
        .any(|&id| graph.operands[id as usize].kind == OperandKind::Constant)
    {
        assert!(!constants.is_empty());
    }
    for constant in constants {
        let Some(value_type::Type::TensorType(tensor)) = &constant.r#type.as_ref().unwrap().r#type
        else {
            panic!("tensor")
        };
        assert_eq!(
            tensor.data_type,
            mil_spec::DataType::Float16 as i32,
            "stored affine parameters remain Half"
        );
    }
}

#[test]
fn half_instance_norm_keeps_centering_and_affine_wide_for_each_layout() {
    for layout in ["nchw", "nhwc"] {
        check(
            &graph(layout, Some(0.0001)),
            half::f16::from_f32(0.0001).to_f32(),
        );
    }
}

#[test]
fn absent_instance_norm_options_round_the_default_epsilon_to_source_half() {
    let mut graph = graph("nchw", None);
    let Operation::InstanceNormalization { options, .. } = &mut graph.operations[0] else {
        unreachable!()
    };
    *options = None;
    check(&graph, half::f16::from_f32(1e-5).to_f32());
}

#[test]
fn half_instance_norm_rounds_epsilon_directly_from_binary64() {
    for layout in ["nchw", "nhwc"] {
        for epsilon in [
            2f64.powi(-25) + 2f64.powi(-55),
            1.00048828125 - 2f64.powi(-40),
            1.00048828125 + 2f64.powi(-40),
        ] {
            check(
                &graph(layout, Some(epsilon)),
                half::f16::from_f64(epsilon).to_f32(),
            );
        }
    }
}

#[test]
#[cfg(feature = "dynamic-inputs")]
fn half_instance_norm_keeps_dynamic_batch_bounds_for_each_layout() {
    for layout in ["nchw", "nhwc"] {
        let mut graph = graph(layout, Some(0.0001));
        for id in [0, 3] {
            graph.operands[id].descriptor.shape[0] =
                Dimension::Dynamic(rustnn::graph::DynamicDimension {
                    name: "batch".into(),
                    max_size: 3,
                });
        }
        check(&graph, half::f16::from_f32(0.0001).to_f32());
    }
}
