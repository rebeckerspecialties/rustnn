//! WebNN double options round directly to the operation's input dtype.

use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{
    ConstantData, DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
};
use rustnn::operators::Operation;
use rustnn::protos::coreml::{mil_spec, specification};
use serde_json::json;

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
mod runtime {
    use super::*;
    use rustnn::backend_selection::{BackendDevice, DeviceType};
    use rustnn::mlcontext::{
        MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    };
    use rustnn::operator_enums::MLOperandDataType;

    #[test]
    fn represented_half_options_preserve_small_products_and_exact_clip_values() {
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            let mut context = MLContext::create(
                &MLContextOptions::new(MLPowerPreference::Default, policy != DeviceType::Cpu)
                    .with_rustnn_device_hint(BackendDevice::Coreml {
                        device_type: policy,
                    }),
            )
            .unwrap();
            for (operation, options, values, expected) in [
                (
                    "linear",
                    json!({"alpha":2f64.powi(-25)+2f64.powi(-55),"beta":0}),
                    vec![0x3c00u16, 0xbc00, 0x6400, 0xe400],
                    vec![0x0001u16, 0x8001, 0x0400, 0x8400],
                ),
                (
                    "clamp",
                    json!({"minValue":-2,"maxValue":1.00146484375-2f64.powi(-40)}),
                    vec![0x0000, 0x3c00, 0x4000],
                    vec![0x0000, 0x3c00, 0x3c01],
                ),
            ] {
                let mut description = unary(operation, options);
                for operand in &mut description.operands {
                    operand.descriptor.shape = vec![Dimension::Static(values.len() as u32)];
                }
                let mut graph = context.rustnn_build_graph(description).unwrap();
                let shape = vec![values.len() as u64];
                let input = context
                    .create_tensor(
                        &MLTensorDescriptor::new(MLOperandDataType::Float16, shape.clone())
                            .to_writable(),
                    )
                    .unwrap();
                let output = context
                    .create_tensor(
                        &MLTensorDescriptor::new(MLOperandDataType::Float16, shape).to_readable(),
                    )
                    .unwrap();
                context
                    .write_tensor(
                        &input,
                        &values
                            .iter()
                            .flat_map(|bits| bits.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                for _ in 0..2 {
                    context
                        .dispatch(
                            &mut graph,
                            &MLNamedTensors::from_iter([("input", &input)]),
                            &MLNamedTensors::from_iter([("result", &output)]),
                        )
                        .unwrap();
                    let mut actual = vec![0; values.len() * 2];
                    context.read_tensor(&output, &mut actual).unwrap();
                    assert_eq!(
                        actual,
                        expected
                            .iter()
                            .flat_map(|bits| bits.to_le_bytes())
                            .collect::<Vec<_>>(),
                        "{operation}:{policy:?}"
                    );
                }
            }
        }
    }
}

fn output_dtype(operation: &mil_spec::Operation) -> i32 {
    let mil_spec::value_type::Type::TensorType(tensor) = operation.outputs[0]
        .r#type
        .as_ref()
        .unwrap()
        .r#type
        .as_ref()
        .unwrap()
    else {
        panic!("tensor output")
    };
    tensor.data_type
}

#[test]
fn half_linear_and_clamp_compute_in_float32_without_half_intermediates() {
    for (name, kernels, options) in [
        (
            "linear",
            vec!["mul", "add"],
            json!({"alpha":2f64.powi(-25)+2f64.powi(-55),"beta":0}),
        ),
        (
            "clamp",
            vec!["clip"],
            json!({"minValue":-2,"maxValue":1.00146484375-2f64.powi(-40)}),
        ),
    ] {
        let graph = unary(name, options);
        let operations = operations(&graph);
        for kernel in kernels {
            let found: Vec<_> = operations
                .iter()
                .filter(|operation| operation.r#type == kernel)
                .collect();
            assert!(!found.is_empty(), "{name}:{kernel}");
            for operation in found {
                assert_eq!(
                    output_dtype(operation),
                    mil_spec::DataType::Float32 as i32,
                    "{name} must retain a complete Float32 computation region"
                );
            }
        }
        assert_eq!(graph.operands[0].descriptor.data_type, DataType::Float16);
        assert_eq!(graph.operands[1].descriptor.data_type, DataType::Float16);
    }
}

#[test]
fn legal_nonfinite_mlnumber_values_keep_half_classes() {
    for (value, bits) in [("Infinity", 0x7c00), ("-Infinity", 0xfc00), ("NaN", 0x7e00)] {
        let mut graph = unary("linear", json!({}));
        graph.operands[1].descriptor.shape = [1, 1, 2, 4].map(Dimension::Static).into();
        graph.operations[0] = Operation::from_json_attributes(
            "pad",
            &[0],
            &[1],
            &json!({"beginningPadding":[0,0,0,1],"endingPadding":[0,0,0,1],"value":value}),
        )
        .unwrap();
        let operation = operations(&graph)
            .into_iter()
            .find(|op| op.r#type == "pad")
            .unwrap();
        let actual = immediate_half(&operation.inputs["constant_val"]).unwrap();
        if value == "NaN" {
            assert!(half::f16::from_bits(actual).is_nan());
        } else {
            assert_eq!(actual, bits);
        }
    }
}

fn operand(name: &str, shape: &[u32], kind: OperandKind, dtype: DataType) -> Operand {
    Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: dtype,
            shape: shape.iter().copied().map(Dimension::Static).collect(),
            pending_permutation: vec![],
        },
    }
}

fn unary(name: &str, options: serde_json::Value) -> GraphInfo {
    GraphInfo {
        operands: vec![
            operand(
                "input",
                &[1, 1, 2, 2],
                OperandKind::Input,
                DataType::Float16,
            ),
            operand(
                "result",
                &[1, 1, 2, 2],
                OperandKind::Output,
                DataType::Float16,
            ),
        ],
        input_operands: vec![0],
        output_operands: vec![1],
        operations: vec![Operation::from_json_attributes(name, &[0], &[1], &options).unwrap()],
        ..Default::default()
    }
}

fn operations(graph: &GraphInfo) -> Vec<mil_spec::Operation> {
    let model = CoremlMlProgramConverter.convert(graph).unwrap();
    let model = specification::Model::decode(model.data.as_slice()).unwrap();
    let children = match model.r#type.unwrap() {
        specification::model::Type::MlProgram(program) => vec![program],
        specification::model::Type::Pipeline(pipeline) => pipeline
            .models
            .into_iter()
            .map(|model| match model.r#type.unwrap() {
                specification::model::Type::MlProgram(program) => program,
                _ => panic!("MLProgram child"),
            })
            .collect(),
        _ => panic!("MLProgram or Pipeline"),
    };
    children
        .into_iter()
        .flat_map(|program| {
            let function = &program.functions["main"];
            function.block_specializations[&function.opset]
                .operations
                .clone()
        })
        .collect()
}

fn immediate_half(argument: &mil_spec::Argument) -> Option<u16> {
    use mil_spec::{argument::binding::Binding, tensor_value, value, value_type};
    let Binding::Value(value) = argument.arguments.first()?.binding.as_ref()? else {
        return None;
    };
    let Some(value_type::Type::TensorType(dtype)) = &value.r#type.as_ref()?.r#type else {
        return None;
    };
    let value::Value::ImmediateValue(value) = value.value.as_ref()? else {
        return None;
    };
    let value::immediate_value::Value::Tensor(tensor) = value.value.as_ref()? else {
        return None;
    };
    match (dtype.data_type, tensor.value.as_ref()?) {
        (kind, tensor_value::Value::Bytes(bytes)) if kind == mil_spec::DataType::Float16 as i32 => {
            Some(u16::from_le_bytes(bytes.values.as_ref().try_into().ok()?))
        }
        (kind, tensor_value::Value::Floats(values))
            if kind == mil_spec::DataType::Float32 as i32 =>
        {
            let value = *values.values.first()?;
            let represented = half::f16::from_f32(value);
            assert_eq!(value.to_bits(), represented.to_f32().to_bits());
            Some(represented.to_bits())
        }
        _ => None,
    }
}

fn expect_scalar(graph: &GraphInfo, kernel: &str, key: &str, expected: u16) {
    let operations = operations(graph);
    let op = operations
        .iter()
        .find(|operation| operation.r#type == kernel)
        .unwrap();
    assert_eq!(
        immediate_half(&op.inputs[key]),
        Some(expected),
        "{kernel}.{key} must round the original double directly to Half"
    );
}

#[test]
fn linear_options_round_half_midpoints_without_a_float32_intermediate() {
    for (value, bits) in [
        (1.00048828125 + 2f64.powi(-40), 0x3c01),
        (1.00146484375 - 2f64.powi(-40), 0x3c01),
        (2f64.powi(-25) + 2f64.powi(-55), 0x0001),
        (1023.5 * 2f64.powi(-24) - 2f64.powi(-65), 0x03ff),
        (65520. - 2f64.powi(-30), 0x7bff),
    ] {
        expect_scalar(
            &unary("linear", json!({"alpha":value,"beta":0})),
            "mul",
            "y",
            bits,
        );
        expect_scalar(
            &unary("linear", json!({"alpha":1,"beta":-value})),
            "add",
            "y",
            bits | 0x8000,
        );
    }
}

#[test]
fn activations_and_clamp_use_source_half_scalar_options() {
    let value = 1.00048828125 + 2f64.powi(-40);
    for (name, kernel) in [
        ("elu", "elu"),
        ("leakyRelu", "leaky_relu"),
        ("hardSigmoid", "sigmoid_hard"),
    ] {
        expect_scalar(
            &unary(name, json!({"alpha":value,"beta":0})),
            kernel,
            "alpha",
            0x3c01,
        );
    }
    expect_scalar(
        &unary("clamp", json!({"minValue":value,"maxValue":2})),
        "clip",
        "alpha",
        0x3c01,
    );
}

#[test]
fn normalization_epsilon_rounds_the_original_double_to_half() {
    let epsilon = 1.00048828125 + 2f64.powi(-40);
    for (name, kernel) in [
        ("layerNormalization", "layer_norm"),
        ("instanceNormalization", "instance_norm"),
    ] {
        expect_scalar(
            &unary(name, json!({"epsilon":epsilon})),
            kernel,
            "epsilon",
            0x3c01,
        );
    }
    let mut graph = unary("linear", json!({}));
    graph.operands.push(operand(
        "mean",
        &[1],
        OperandKind::Constant,
        DataType::Float16,
    ));
    graph.operands.push(operand(
        "variance",
        &[1],
        OperandKind::Constant,
        DataType::Float16,
    ));
    graph.constant_operand_ids_to_handles = [
        (
            2,
            ConstantData {
                data: vec![0, 0],
                label: None,
            },
        ),
        (
            3,
            ConstantData {
                data: vec![0, 0],
                label: None,
            },
        ),
    ]
    .into_iter()
    .collect();
    graph.operations[0] = Operation::from_json_attributes(
        "batchNormalization",
        &[0, 2, 3],
        &[1],
        &json!({"epsilon":epsilon}),
    )
    .unwrap();
    expect_scalar(&graph, "batch_norm", "epsilon", 0x3c01);
}

#[test]
fn absent_normalization_options_still_cast_default_epsilon_to_half() {
    for (name, kernel) in [
        ("layerNormalization", "layer_norm"),
        ("instanceNormalization", "instance_norm"),
    ] {
        let mut graph = unary(name, json!({}));
        match &mut graph.operations[0] {
            Operation::LayerNormalization { options, .. } => *options = None,
            Operation::InstanceNormalization { options, .. } => *options = None,
            _ => unreachable!(),
        }
        expect_scalar(&graph, kernel, "epsilon", 0x00a8);
    }
}

#[test]
fn pad_value_and_gemm_coefficients_round_original_double_directly() {
    let value = 1.00048828125 + 2f64.powi(-40);
    let mut pad = unary("linear", json!({}));
    pad.operands[1].descriptor.shape = [1, 1, 2, 4].map(Dimension::Static).into();
    pad.operations[0] = Operation::from_json_attributes(
        "pad",
        &[0],
        &[1],
        &json!({"beginningPadding":[0,0,0,1],"endingPadding":[0,0,0,1],"value":value}),
    )
    .unwrap();
    expect_scalar(&pad, "pad", "constant_val", 0x3c01);
    let gemm = GraphInfo {
        operands: vec![
            operand("a", &[1, 1], OperandKind::Input, DataType::Float16),
            operand("b", &[1, 1], OperandKind::Input, DataType::Float16),
            operand("result", &[1, 1], OperandKind::Output, DataType::Float16),
        ],
        input_operands: vec![0, 1],
        output_operands: vec![2],
        operations: vec![
            Operation::from_json_attributes("gemm", &[0, 1], &[2], &json!({"alpha":value}))
                .unwrap(),
        ],
        ..Default::default()
    };
    expect_scalar(&gemm, "mul", "y", 0x3c01);
}

#[test]
fn gemm_does_not_discard_a_represented_one_ulp_coefficient() {
    let graph = GraphInfo {
        operands: vec![
            operand("a", &[1, 1], OperandKind::Input, DataType::Float32),
            operand("b", &[1, 1], OperandKind::Input, DataType::Float32),
            operand("c", &[1, 1], OperandKind::Input, DataType::Float32),
            operand("result", &[1, 1], OperandKind::Output, DataType::Float32),
        ],
        input_operands: vec![0, 1, 2],
        output_operands: vec![3],
        operations: vec![
            Operation::from_json_attributes(
                "gemm",
                &[0, 1],
                &[3],
                &json!({"alpha":1.0+2f64.powi(-23),"beta":1.0+2f64.powi(-23),"c":2}),
            )
            .unwrap(),
        ],
        ..Default::default()
    };
    let operations = operations(&graph);
    assert_eq!(
        operations.iter().filter(|op| op.r#type == "mul").count(),
        2,
        "only an exactly represented unit coefficient can omit a scaling operation"
    );
}

fn coefficient_boundaries() -> Vec<(f64, u16)> {
    // Adjacent Half encodings independently define each exact midpoint. Use
    // binary64's immediate neighbors so even the lowest sticky bit matters.
    let mut cases = vec![(0., 0), (-0., 0x8000), (65504., 0x7bff)];
    for (midpoint, lower, upper) in [
        (2f64.powi(-25), 0x0000, 0x0001),
        (1023.5 * 2f64.powi(-24), 0x03ff, 0x0400),
        (1.00048828125, 0x3c00, 0x3c01),
        (1.00146484375, 0x3c01, 0x3c02),
        (65520., 0x7bff, 0x7c00),
    ] {
        let tie = if lower & 1 == 0 { lower } else { upper };
        for (offset, expected) in [(-1i64, lower), (0, tie), (1, upper)] {
            let value = f64::from_bits((midpoint.to_bits() as i64 + offset) as u64);
            cases.push((value, expected));
            cases.push((-value, expected | 0x8000));
        }
    }
    cases
}

#[test]
fn scalar_paths_keep_lowest_sticky_bit_at_underflow_normal_and_overflow_boundaries() {
    for (value, expected) in coefficient_boundaries() {
        let linear = unary("linear", json!({"alpha":value,"beta":value}));
        expect_scalar(&linear, "mul", "y", expected);
        expect_scalar(&linear, "add", "y", expected);

        for (name, kernel) in [("elu", "elu"), ("leakyRelu", "leaky_relu")] {
            expect_scalar(
                &unary(name, json!({"alpha":value})),
                kernel,
                "alpha",
                expected,
            );
        }
        let sigmoid = unary("hardSigmoid", json!({"alpha":value,"beta":value}));
        expect_scalar(&sigmoid, "sigmoid_hard", "alpha", expected);
        expect_scalar(&sigmoid, "sigmoid_hard", "beta", expected);

        for (options, key) in [
            (json!({"minValue":value,"maxValue":"Infinity"}), "alpha"),
            (json!({"minValue":"-Infinity","maxValue":value}), "beta"),
        ] {
            expect_scalar(&unary("clamp", options), "clip", key, expected);
        }

        let mut pad = unary("linear", json!({}));
        pad.operands[1].descriptor.shape = [1, 1, 2, 4].map(Dimension::Static).into();
        pad.operations[0] = Operation::from_json_attributes(
            "pad",
            &[0],
            &[1],
            &json!({"beginningPadding":[0,0,0,1],"endingPadding":[0,0,0,1],"value":value}),
        )
        .unwrap();
        expect_scalar(&pad, "pad", "constant_val", expected);
    }
}

#[test]
fn gemm_scale_and_bias_keep_lowest_sticky_bit_and_only_omit_exact_one() {
    for (value, expected) in coefficient_boundaries() {
        let graph = GraphInfo {
            operands: vec![
                operand("a", &[1, 1], OperandKind::Input, DataType::Float16),
                operand("b", &[1, 1], OperandKind::Input, DataType::Float16),
                operand("c", &[1, 1], OperandKind::Input, DataType::Float16),
                operand("result", &[1, 1], OperandKind::Output, DataType::Float16),
            ],
            input_operands: vec![0, 1, 2],
            output_operands: vec![3],
            operations: vec![
                Operation::from_json_attributes(
                    "gemm",
                    &[0, 1],
                    &[3],
                    &json!({"alpha":value,"beta":value,"c":2}),
                )
                .unwrap(),
            ],
            ..Default::default()
        };
        let actual: Vec<_> = operations(&graph)
            .iter()
            .filter(|op| op.r#type == "mul")
            .map(|op| immediate_half(&op.inputs["y"]).unwrap())
            .collect();
        let wanted = if expected == 0x3c00 {
            vec![]
        } else {
            vec![expected; 2]
        };
        assert_eq!(actual, wanted, "GEMM alpha/beta, original double={value:?}");
    }
}

#[test]
fn normalization_epsilon_keeps_lowest_sticky_bit_with_and_without_affine_operands() {
    for (epsilon, expected) in coefficient_boundaries()
        .into_iter()
        .filter(|(v, _)| *v >= 0.)
    {
        for affine in [false, true] {
            for (name, kernel) in [
                ("layerNormalization", "layer_norm"),
                ("instanceNormalization", "instance_norm"),
                ("batchNormalization", "batch_norm"),
            ] {
                let mut graph = unary("linear", json!({}));
                let mut options = json!({"epsilon":epsilon});
                let mut inputs = vec![0];
                if name == "batchNormalization" {
                    for (parameter, bits) in [("mean", 0x0000u16), ("variance", 0x3c00)] {
                        let id = graph.operands.len() as u32;
                        graph.operands.push(operand(
                            parameter,
                            &[1],
                            OperandKind::Constant,
                            DataType::Float16,
                        ));
                        graph.constant_operand_ids_to_handles.insert(
                            id,
                            ConstantData {
                                data: bits.to_le_bytes().to_vec(),
                                label: None,
                            },
                        );
                        inputs.push(id);
                    }
                }
                if affine {
                    if name == "layerNormalization" {
                        options["axes"] = json!([1]);
                    }
                    for (parameter, bits) in [("scale", 0x3c01u16), ("bias", 0x8001)] {
                        let id = graph.operands.len() as u32;
                        graph.operands.push(operand(
                            parameter,
                            &[1],
                            OperandKind::Constant,
                            DataType::Float16,
                        ));
                        graph.constant_operand_ids_to_handles.insert(
                            id,
                            ConstantData {
                                data: bits.to_le_bytes().to_vec(),
                                label: None,
                            },
                        );
                        options[parameter] = json!(id);
                    }
                }
                graph.operations[0] =
                    Operation::from_json_attributes(name, &inputs, &[1], &options).unwrap();
                expect_scalar(&graph, kernel, "epsilon", expected);
                assert!(
                    graph
                        .operands
                        .iter()
                        .all(|operand| operand.descriptor.data_type == DataType::Float16)
                );
            }
        }
    }
}
