//! Evidenced Half arithmetic kernels compute wide without changing GraphInfo.

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
mod runtime {
    use super::*;
    use half::f16;
    use rustnn::backend_selection::{BackendDevice, DeviceType};
    use rustnn::mlcontext::{
        MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    };
    use rustnn::operator_enums::MLOperandDataType;

    fn predict(graph: GraphInfo, shape: &[u64], values: &[f32], policy: DeviceType) -> Vec<u16> {
        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, policy != DeviceType::Cpu)
                .with_rustnn_device_hint(BackendDevice::Coreml {
                    device_type: policy,
                }),
        )
        .unwrap();
        let output_shape: Vec<_> = graph.operands[graph.output_operands[0] as usize]
            .descriptor
            .static_or_max_shape()
            .into_iter()
            .map(u64::from)
            .collect();
        let mut graph = context.rustnn_build_graph(graph).unwrap();
        let input = context
            .create_tensor(
                &MLTensorDescriptor::new(MLOperandDataType::Float16, shape.to_vec()).to_writable(),
            )
            .unwrap();
        let output = context
            .create_tensor(
                &MLTensorDescriptor::new(MLOperandDataType::Float16, output_shape.clone())
                    .to_readable(),
            )
            .unwrap();
        let values: Vec<_> = values
            .iter()
            .map(|value| f16::from_f32(*value).to_bits())
            .collect();
        context.write_tensor(&input, &values).unwrap();
        context
            .dispatch(
                &mut graph,
                &MLNamedTensors::from([("input", &input)]),
                &MLNamedTensors::from([("result", &output)]),
            )
            .unwrap();
        let count: usize = output_shape
            .into_iter()
            .map(|value| value as usize)
            .product();
        let mut actual = vec![0u16; count];
        context.read_tensor(&output, &mut actual).unwrap();
        actual
    }

    #[test]
    fn half_reductions_preserve_exact_represented_terms() {
        let unit = 2f32.powi(-24);
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            for exclusive in [false, true] {
                for reversed in [false, true] {
                    let values = [unit, -3. * unit, 0., 7. * unit];
                    let mut expected = vec![0u16; 4];
                    let mut sum = 0f64;
                    let indices: Vec<usize> = if reversed {
                        (0..4).rev().collect()
                    } else {
                        (0..4).collect()
                    };
                    for index in indices {
                        if !exclusive {
                            sum += f64::from(values[index]);
                        }
                        expected[index] = f16::from_f64(sum).to_bits();
                        if exclusive {
                            sum += f64::from(values[index]);
                        }
                    }
                    let graph = simple(
                        "cumulativeSum",
                        &[4],
                        &[4],
                        json!({"axis":0,"exclusive":exclusive,"reversed":reversed}),
                    );
                    assert_eq!(
                        predict(graph, &[4], &values, policy),
                        expected,
                        "{policy:?}, exclusive={exclusive}, reversed={reversed}"
                    );
                }
            }
            assert_eq!(
                predict(
                    simple(
                        "reduceMin",
                        &[4],
                        &[1],
                        json!({"axes":[0],"keepDimensions":true})
                    ),
                    &[4],
                    &[unit, -3. * unit, 0., 7. * unit],
                    policy
                ),
                [f16::from_f32(-3. * unit).to_bits()],
                "{policy:?}"
            );
            let values = [0.03125f32, 0.015625];
            let expected = values
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>();
            assert_eq!(
                predict(
                    simple(
                        "reduceSumSquare",
                        &[2],
                        &[1],
                        json!({"axes":[0],"keepDimensions":true})
                    ),
                    &[2],
                    &values,
                    policy
                ),
                [f16::from_f64(expected).to_bits()],
                "{policy:?}"
            );
        }
    }

    #[test]
    fn half_convolution_remains_finite_after_cancellation() {
        let graph = GraphInfo {
            operands: vec![
                operand("input", &[1, 2, 1, 1], OperandKind::Input),
                operand("weight", &[1, 2, 1, 1], OperandKind::Constant),
                operand("result", &[1, 1, 1, 1], OperandKind::Output),
            ],
            input_operands: vec![0],
            output_operands: vec![2],
            operations: vec![
                Operation::from_json_attributes("conv2d", &[0, 1], &[2], &json!({})).unwrap(),
            ],
            constant_operand_ids_to_handles: [(1, constant(&[2., -2.]))].into_iter().collect(),
            ..Default::default()
        };
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            let actual = predict(graph.clone(), &[1, 2, 1, 1], &[65504., 65504.], policy);
            assert_eq!(
                actual,
                [0],
                "{policy:?}: finite cancellation must not become infinity"
            );
        }
    }

    #[test]
    fn half_pool_and_constant_batch_norm_do_not_round_internal_stages() {
        let unit = 2f32.powi(-24);
        let values = [unit, -3. * unit, 0., 7. * unit];
        let epsilon = 2f64.powi(-14);
        let graph = GraphInfo {
            operands: vec![
                operand("input", &[1, 1, 2, 2], OperandKind::Input),
                operand("mean", &[1], OperandKind::Constant),
                operand("variance", &[1], OperandKind::Constant),
                operand("scale", &[1], OperandKind::Constant),
                operand("bias", &[1], OperandKind::Constant),
                operand("result", &[1, 1, 2, 2], OperandKind::Output),
            ],
            input_operands: vec![0],
            output_operands: vec![5],
            operations: vec![
                Operation::from_json_attributes(
                    "batchNormalization",
                    &[0, 1, 2],
                    &[5],
                    &json!({"epsilon":epsilon,"scale":3,"bias":4}),
                )
                .unwrap(),
            ],
            constant_operand_ids_to_handles: [
                (1, constant(&[0.])),
                (2, constant(&[0.])),
                (3, constant(&[0.75])),
                (4, constant(&[unit])),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        let expected: Vec<_> = values
            .iter()
            .map(|value| {
                f16::from_f64(f64::from(*value) / epsilon.sqrt() * 0.75 + f64::from(unit)).to_bits()
            })
            .collect();
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            assert_eq!(
                predict(graph.clone(), &[1, 1, 2, 2], &values, policy),
                expected,
                "{policy:?}"
            );
            let average =
                f16::from_f64(values.iter().map(|value| f64::from(*value)).sum::<f64>() / 4.)
                    .to_bits();
            assert_eq!(
                predict(
                    simple(
                        "averagePool2d",
                        &[1, 1, 2, 2],
                        &[1, 1, 1, 1],
                        json!({"windowDimensions":[2,2]})
                    ),
                    &[1, 1, 2, 2],
                    &values,
                    policy
                ),
                [average],
                "{policy:?}"
            );
        }
    }
    #[test]
    fn half_batch_norm_retains_the_epsilon_sticky_bit_before_sqrt_and_affine() {
        let midpoint = 2f64.powi(-25);
        let epsilon = f64::from_bits(midpoint.to_bits() + 1);
        // Source-Half epsilon is 2^-24, hence sqrt(epsilon) is exactly 2^-12.
        // Narrowing epsilon to zero instead would make the division infinite.
        let graph = GraphInfo {
            operands: vec![
                operand("input", &[1, 1, 1, 2], OperandKind::Input),
                operand("mean", &[1], OperandKind::Constant),
                operand("variance", &[1], OperandKind::Constant),
                operand("scale", &[1], OperandKind::Constant),
                operand("bias", &[1], OperandKind::Constant),
                operand("result", &[1, 1, 1, 2], OperandKind::Output),
            ],
            input_operands: vec![0],
            output_operands: vec![5],
            operations: vec![
                Operation::from_json_attributes(
                    "batchNormalization",
                    &[0, 1, 2],
                    &[5],
                    &json!({"epsilon":epsilon,"scale":3,"bias":4}),
                )
                .unwrap(),
            ],
            constant_operand_ids_to_handles: [
                (1, constant(&[0.])),
                (2, constant(&[0.])),
                (3, constant(&[0.5])),
                (4, constant(&[0.25])),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            assert_eq!(
                predict(
                    graph.clone(),
                    &[1, 1, 1, 2],
                    &[2f32.powi(-12), -2f32.powi(-12)],
                    policy
                ),
                [0x3a00, 0xb400],
                "{policy:?}: represented epsilon must precede wide normalization and affine work"
            );
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

fn constant(values: &[f32]) -> ConstantData {
    ConstantData {
        data: values
            .iter()
            .flat_map(|value| half::f16::from_f32(*value).to_bits().to_le_bytes())
            .collect(),
        label: None,
    }
}

fn simple(operation: &str, input: &[u32], output: &[u32], options: serde_json::Value) -> GraphInfo {
    GraphInfo {
        operands: vec![
            operand("input", input, OperandKind::Input),
            operand("result", output, OperandKind::Output),
        ],
        input_operands: vec![0],
        output_operands: vec![1],
        operations: vec![Operation::from_json_attributes(operation, &[0], &[1], &options).unwrap()],
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
        _ => panic!("expected MLProgram or Pipeline"),
    }
}

fn tensor_dtype(value: &mil_spec::NamedValueType) -> i32 {
    let Some(mil_spec::value_type::Type::TensorType(tensor)) =
        &value.r#type.as_ref().unwrap().r#type
    else {
        panic!("tensor")
    };
    tensor.data_type
}

fn check_kernel(graph: GraphInfo, kernel: &str) -> Vec<mil_spec::Operation> {
    let original = serde_json::to_vec(&graph).unwrap();
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    assert_eq!(
        serde_json::to_vec(&graph).unwrap(),
        original,
        "conversion cannot retag the WebNN graph"
    );
    let operations = operations(specification::Model::decode(converted.data.as_slice()).unwrap());
    let kernels: Vec<_> = operations
        .iter()
        .filter(|operation| operation.r#type == kernel)
        .collect();
    assert!(!kernels.is_empty(), "{kernel}");
    for operation in kernels {
        assert!(
            operation
                .outputs
                .iter()
                .all(|value| tensor_dtype(value) == mil_spec::DataType::Float32 as i32),
            "{kernel} accumulates/selects in Float32"
        );
    }
    assert!(
        operations
            .iter()
            .flat_map(|op| &op.outputs)
            .any(|value| value.name == "result"
                && tensor_dtype(value) == mil_spec::DataType::Float16 as i32),
        "public result keeps its declared Half dtype"
    );
    operations
}

fn check_batch_norm(graph: GraphInfo, expected_epsilon: u16) {
    use mil_spec::{argument::binding::Binding, tensor_value, value};
    let operations = check_kernel(graph, "real_div");
    assert!(operations.iter().all(|op| op.r#type != "batch_norm"));
    assert!(
        operations
            .iter()
            .filter(|op| op.r#type == "add"
                && op
                    .outputs
                    .iter()
                    .any(|output| output.name.ends_with("_bn_veps")))
            .any(|op| {
                let Some(argument) = op.inputs.get("y") else {
                    return false;
                };
                let Some(Binding::Value(epsilon_value)) = argument
                    .arguments
                    .first()
                    .and_then(|binding| binding.binding.as_ref())
                else {
                    return false;
                };
                let Some(value::Value::ImmediateValue(immediate)) = epsilon_value.value.as_ref()
                else {
                    return false;
                };
                let Some(value::immediate_value::Value::Tensor(tensor)) = immediate.value.as_ref()
                else {
                    return false;
                };
                let Some(tensor_value::Value::Floats(values)) = tensor.value.as_ref() else {
                    return false;
                };
                values.values.len() == 1
                    && values.values[0].to_bits()
                        == half::f16::from_bits(expected_epsilon).to_f32().to_bits()
            }),
        "the complete formula uses source-Half-rounded epsilon"
    );
    for operation in operations.iter().filter(|op| {
        matches!(
            op.r#type.as_str(),
            "sub" | "add" | "mul" | "sqrt" | "real_div"
        )
    }) {
        assert!(
            operation
                .outputs
                .iter()
                .all(|value| tensor_dtype(value) == mil_spec::DataType::Float32 as i32)
        );
    }
}

#[test]
fn half_batch_norm_rounds_epsilon_directly_from_binary64() {
    for (epsilon, expected) in [
        (2f64.powi(-25) + 2f64.powi(-55), 0x0001),
        (1.00048828125 - 2f64.powi(-40), 0x3c00),
        (1.00048828125 + 2f64.powi(-40), 0x3c01),
        (f64::from_bits(2f64.powi(-25).to_bits() + 1), 0x0001),
        (f64::from_bits(1.00048828125f64.to_bits() - 1), 0x3c00),
        (f64::from_bits(1.00048828125f64.to_bits() + 1), 0x3c01),
        (f64::from_bits(1.00146484375f64.to_bits() - 1), 0x3c01),
        (f64::from_bits(1.00146484375f64.to_bits() + 1), 0x3c02),
    ] {
        let graph = GraphInfo {
            operands: vec![
                operand("input", &[1, 1, 1, 2], OperandKind::Input),
                operand("mean", &[1], OperandKind::Constant),
                operand("variance", &[1], OperandKind::Constant),
                operand("result", &[1, 1, 1, 2], OperandKind::Output),
            ],
            input_operands: vec![0],
            output_operands: vec![3],
            operations: vec![
                Operation::from_json_attributes(
                    "batchNormalization",
                    &[0, 1, 2],
                    &[3],
                    &json!({"epsilon":epsilon}),
                )
                .unwrap(),
            ],
            constant_operand_ids_to_handles: [(1, constant(&[0.])), (2, constant(&[0.]))]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        check_batch_norm(graph, expected);
    }
}

#[test]
fn half_cumulative_sum_keeps_all_direction_and_exclusion_controls() {
    for exclusive in [false, true] {
        for reversed in [false, true] {
            check_kernel(
                simple(
                    "cumulativeSum",
                    &[2, 4],
                    &[2, 4],
                    json!({"axis":1,"exclusive":exclusive,"reversed":reversed}),
                ),
                "cumsum",
            );
        }
    }
}

#[test]
fn half_min_and_sum_square_preserve_finite_subnormal_results() {
    check_kernel(
        simple("reduceMin", &[2, 4], &[2], json!({"axes":[1]})),
        "reduce_min",
    );
    check_kernel(
        simple(
            "reduceSumSquare",
            &[2, 4],
            &[2, 1],
            json!({"axes":[1],"keepDimensions":true}),
        ),
        "reduce_sum_square",
    );
}

#[test]
fn half_average_pool_computes_the_complete_window_wide() {
    check_kernel(
        simple(
            "averagePool2d",
            &[1, 1, 2, 2],
            &[1, 1, 1, 1],
            json!({"windowDimensions":[2,2]}),
        ),
        "avg_pool",
    );
}

#[test]
fn half_convolution_keeps_original_half_weight_and_bias_bytes() {
    let graph = GraphInfo {
        operands: vec![
            operand("input", &[1, 2, 1, 1], OperandKind::Input),
            operand("weight", &[1, 2, 1, 1], OperandKind::Constant),
            operand("bias", &[1], OperandKind::Constant),
            operand("result", &[1, 1, 1, 1], OperandKind::Output),
        ],
        input_operands: vec![0],
        output_operands: vec![3],
        operations: vec![
            Operation::from_json_attributes("conv2d", &[0, 1], &[3], &json!({"bias":2})).unwrap(),
        ],
        constant_operand_ids_to_handles: [(1, constant(&[2., -2.])), (2, constant(&[0.]))]
            .into_iter()
            .collect(),
        ..Default::default()
    };
    let operations = check_kernel(graph, "conv");
    use mil_spec::{value, value_type};
    let blobs: Vec<_> = operations
        .iter()
        .filter(|operation| operation.r#type == "const")
        .flat_map(|operation| operation.attributes.values())
        .filter(|value| matches!(&value.value, Some(value::Value::BlobFileValue(_))))
        .collect();
    assert!(!blobs.is_empty());
    for value in blobs {
        let Some(value_type::Type::TensorType(tensor)) = &value.r#type.as_ref().unwrap().r#type
        else {
            panic!("constant tensor")
        };
        assert_eq!(
            tensor.data_type,
            mil_spec::DataType::Float16 as i32,
            "stored constant was not expanded to Float32"
        );
    }
}

#[test]
fn half_batch_norm_widens_its_already_input_rounded_epsilon() {
    for input_shape in [vec![1, 1, 2, 2], vec![1]] {
        let graph = GraphInfo {
            operands: vec![
                operand("input", &input_shape, OperandKind::Input),
                operand("mean", &[1], OperandKind::Constant),
                operand("variance", &[1], OperandKind::Constant),
                operand("result", &input_shape, OperandKind::Output),
            ],
            input_operands: vec![0],
            output_operands: vec![3],
            operations: vec![
                Operation::from_json_attributes(
                    "batchNormalization",
                    &[0, 1, 2],
                    &[3],
                    &json!({"axis":if input_shape.len()==1 {0} else {1},"epsilon":0.0001}),
                )
                .unwrap(),
            ],
            constant_operand_ids_to_handles: [(1, constant(&[0.])), (2, constant(&[0.]))]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        check_batch_norm(graph, 0x068e);
    }
}

#[test]
#[cfg(feature = "dynamic-inputs")]
fn protected_dynamic_batch_norm_widening_keeps_actual_source_bounds() {
    let mut graph = GraphInfo {
        operands: vec![
            operand("input", &[3, 1, 2, 2], OperandKind::Input),
            operand("mean", &[1], OperandKind::Constant),
            operand("variance", &[1], OperandKind::Constant),
            operand("result", &[3, 1, 2, 2], OperandKind::Output),
        ],
        input_operands: vec![0],
        output_operands: vec![3],
        operations: vec![
            Operation::from_json_attributes(
                "batchNormalization",
                &[0, 1, 2],
                &[3],
                &json!({"epsilon":0.0001}),
            )
            .unwrap(),
        ],
        constant_operand_ids_to_handles: [(1, constant(&[0.])), (2, constant(&[0.]))]
            .into_iter()
            .collect(),
        ..Default::default()
    };
    for id in [0, 3] {
        graph.operands[id].descriptor.shape[0] =
            Dimension::Dynamic(rustnn::graph::DynamicDimension {
                name: "batch".into(),
                max_size: 3,
            });
    }
    check_batch_norm(graph, 0x068e);
}

#[test]
fn runtime_half_batch_norm_keeps_one_complete_wide_compute_region() {
    for dynamic in [false, true] {
        if dynamic && !cfg!(feature = "dynamic-inputs") {
            continue;
        }
        let mut graph = GraphInfo {
            operands: vec![
                operand("input", &[3, 2, 2, 2], OperandKind::Input),
                operand("mean", &[2], OperandKind::Input),
                operand("variance", &[2], OperandKind::Input),
                operand("scale", &[2], OperandKind::Input),
                operand("bias", &[2], OperandKind::Input),
                operand("result", &[3, 2, 2, 2], OperandKind::Output),
            ],
            input_operands: vec![0, 1, 2, 3, 4],
            output_operands: vec![5],
            operations: vec![
                Operation::from_json_attributes(
                    "batchNormalization",
                    &[0, 1, 2],
                    &[5],
                    &json!({"epsilon":0.0001,"scale":3,"bias":4}),
                )
                .unwrap(),
            ],
            ..Default::default()
        };
        if dynamic {
            for id in [0, 5] {
                graph.operands[id].descriptor.shape[0] =
                    Dimension::Dynamic(rustnn::graph::DynamicDimension {
                        name: "batch".into(),
                        max_size: 3,
                    });
            }
        }
        let operations = check_kernel(graph, "real_div");
        for operation in operations.iter().filter(|op| {
            matches!(
                op.r#type.as_str(),
                "sub" | "add" | "mul" | "sqrt" | "real_div"
            )
        }) {
            assert!(
                operation
                    .outputs
                    .iter()
                    .all(|value| tensor_dtype(value) == mil_spec::DataType::Float32 as i32),
                "{} rounds an internal BatchNorm value to Half",
                operation.r#type
            );
        }
    }
}

#[test]
fn absent_batch_norm_options_still_round_the_default_epsilon_to_source_half() {
    let mut graph = GraphInfo {
        operands: vec![
            operand("input", &[1, 1, 2, 2], OperandKind::Input),
            operand("mean", &[1], OperandKind::Constant),
            operand("variance", &[1], OperandKind::Constant),
            operand("result", &[1, 1, 2, 2], OperandKind::Output),
        ],
        input_operands: vec![0],
        output_operands: vec![3],
        operations: vec![
            Operation::from_json_attributes("batchNormalization", &[0, 1, 2], &[3], &json!({}))
                .unwrap(),
        ],
        constant_operand_ids_to_handles: [(1, constant(&[0.])), (2, constant(&[0.]))]
            .into_iter()
            .collect(),
        ..Default::default()
    };
    let Operation::BatchNormalization { options, .. } = &mut graph.operations[0] else {
        unreachable!()
    };
    *options = None;
    check_batch_norm(graph, 0x00a8);
}

#[test]
fn widened_constant_batch_norm_parameters_do_not_require_native_const_features() {
    let graph = GraphInfo {
        operands: vec![
            operand("input", &[1, 2, 2, 2], OperandKind::Input),
            operand("mean", &[2], OperandKind::Constant),
            operand("variance", &[2], OperandKind::Constant),
            operand("scale", &[2], OperandKind::Constant),
            operand("bias", &[2], OperandKind::Constant),
            operand("result", &[1, 2, 2, 2], OperandKind::Output),
        ],
        input_operands: vec![0],
        output_operands: vec![5],
        operations: vec![
            Operation::from_json_attributes(
                "batchNormalization",
                &[0, 1, 2],
                &[5],
                &json!({"scale":3,"bias":4,"epsilon":0.0001}),
            )
            .unwrap(),
        ],
        constant_operand_ids_to_handles: [
            (1, constant(&[0., 0.])),
            (2, constant(&[0., 0.])),
            (3, constant(&[0.75, -0.5])),
            (4, constant(&[0., 0.])),
        ]
        .into_iter()
        .collect(),
        ..Default::default()
    };
    let operations = check_kernel(graph, "real_div");
    assert!(
        operations
            .iter()
            .all(|operation| operation.r#type != "batch_norm"),
        "widened gamma/beta are live values; native BatchNorm rejects them as non-const"
    );
}
