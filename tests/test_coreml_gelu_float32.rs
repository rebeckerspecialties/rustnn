//! Cancellation-sensitive GELU through the public context and tensor paths.

#![cfg(all(target_os = "macos", feature = "coreml-runtime"))]

use rustnn::backend_selection::{BackendDevice, DeviceType};
use rustnn::graph::{
    ConstantData, DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
};
use rustnn::mlcontext::{
    MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    RustNNOptions,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operators::Operation;

// Independent Decimal 140/180-digit references, also checked with mpmath erfc.
// These exact-bit local checks are stronger than the finite 18-ULP WPT gate.
const INPUT: [u32; 8] = [
    0xc1200000, 0xc0a00000, 0xbf800000, 0x3f800000, 0x00000001, 0x80000001, 0xc1580000, 0xc1600000,
];
const EXPECTED: [u32; 8] = [
    0x9ab83c9b, 0xb5c05e5d, 0xbe227686, 0x3f57625f, 0x00000001, 0x80000000, 0x8001263e, 0x8000004e,
];

fn context(mode: usize, policy: DeviceType) -> MLContext<'static> {
    let mut options = RustNNOptions::default();
    options.coreml.reuse_tensor_storage = mode != 0;
    options.coreml.output_backings = mode == 2;
    MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, policy != DeviceType::Cpu)
            .with_rustnn_device_hint(BackendDevice::Coreml {
                device_type: policy,
            })
            .with_rustnn_options(options),
    )
    .unwrap()
}

fn graph(shape: Vec<Dimension>) -> GraphInfo {
    let descriptor = OperandDescriptor {
        data_type: DataType::Float32,
        shape,
        pending_permutation: vec![],
    };
    GraphInfo {
        operands: vec![
            Operand {
                name: Some("input".into()),
                kind: OperandKind::Input,
                descriptor: descriptor.clone(),
            },
            Operand {
                name: Some("result".into()),
                kind: OperandKind::Output,
                descriptor,
            },
        ],
        operations: vec![Operation::Gelu {
            input: 0,
            options: None,
            outputs: vec![1],
        }],
        input_operands: vec![0],
        output_operands: vec![1],
        ..Default::default()
    }
}

#[test]
fn gelu_float32_tail_subnormals_and_owned_outputs_in_all_storage_modes() {
    for mode in 0..3 {
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            let mut context = context(mode, policy);
            let mut graph = context
                .rustnn_build_graph(graph(vec![Dimension::Static(8)]))
                .unwrap();
            let descriptor = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![8]);
            let input = context
                .create_tensor(&descriptor.clone().to_writable())
                .unwrap();
            let output = context.create_tensor(&descriptor.to_readable()).unwrap();
            for _ in 0..3 {
                context.write_tensor(&input, &INPUT).unwrap();
                context
                    .dispatch(
                        &mut graph,
                        &MLNamedTensors::from([("input", &input)]),
                        &MLNamedTensors::from([("result", &output)]),
                    )
                    .unwrap();
                context.write_tensor(&input, &[0_u32; 8]).unwrap();
                let mut actual = [0_u32; 8];
                context.read_tensor(&output, &mut actual).unwrap();
                assert_eq!(actual, EXPECTED, "mode={mode}, policy={policy:?}");
            }
        }
    }
}

#[test]
fn gelu_float32_native_neighbors_and_fanout_keep_the_repaired_tail() {
    for mode in 0..3 {
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            let mut context = context(mode, policy);
            let mut source = graph(vec![Dimension::Static(4)]);
            source.operands[1].kind = OperandKind::Intermediate;
            source.operands[1].name = Some("negative".into());
            let mut output = source.operands[1].clone();
            output.kind = OperandKind::Output;
            output.name = Some("gelu".into());
            source.operands.push(output.clone());
            output.name = Some("result".into());
            source.operands.push(output);
            source.operations = vec![
                Operation::Neg {
                    input: 0,
                    options: None,
                    outputs: vec![1],
                },
                Operation::Gelu {
                    input: 1,
                    options: None,
                    outputs: vec![2],
                },
                Operation::Neg {
                    input: 2,
                    options: None,
                    outputs: vec![3],
                },
            ];
            source.output_operands = vec![2, 3];
            let mut graph = context.rustnn_build_graph(source).unwrap();
            let descriptor = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![4]);
            let input = context
                .create_tensor(&descriptor.clone().to_writable())
                .unwrap();
            let gelu = context
                .create_tensor(&descriptor.clone().to_readable())
                .unwrap();
            let result = context.create_tensor(&descriptor.to_readable()).unwrap();
            context
                .write_tensor(&input, &[10.0_f32, 5.0, 13.5, 14.0])
                .unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("input", &input)]),
                    &MLNamedTensors::from([("gelu", &gelu), ("result", &result)]),
                )
                .unwrap();
            let expected = [EXPECTED[0], EXPECTED[1], EXPECTED[6], EXPECTED[7]];
            let mut actual = [0_u32; 4];
            context.read_tensor(&gelu, &mut actual).unwrap();
            assert_eq!(actual, expected, "mode={mode}, policy={policy:?}");
            context.read_tensor(&result, &mut actual).unwrap();
            assert_eq!(
                actual,
                expected.map(|x| x ^ 0x80000000),
                "mode={mode}, policy={policy:?}"
            );
        }
    }
}

#[test]
fn gelu_float32_constant_scalar_and_blob_preserve_original_values() {
    for mode in 0..3 {
        for scalar in [false, true] {
            let mut source = graph(if scalar {
                vec![]
            } else {
                vec![Dimension::Static(8)]
            });
            source.operands[0].kind = OperandKind::Constant;
            source.input_operands.clear();
            let input = if scalar { &INPUT[..1] } else { &INPUT[..] };
            source.constant_operand_ids_to_handles.insert(
                0,
                ConstantData {
                    data: input.iter().flat_map(|x| x.to_le_bytes()).collect(),
                    label: None,
                },
            );
            let mut context = context(mode, DeviceType::Cpu);
            let mut graph = context.rustnn_build_graph(source).unwrap();
            let descriptor = MLTensorDescriptor::new(
                MLOperandDataType::Float32,
                if scalar { vec![] } else { vec![8] },
            )
            .to_readable();
            let output = context.create_tensor(&descriptor).unwrap();
            for _ in 0..2 {
                context
                    .dispatch(
                        &mut graph,
                        &MLNamedTensors::new(),
                        &MLNamedTensors::from([("result", &output)]),
                    )
                    .unwrap();
                let mut actual = vec![0_u32; input.len()];
                context.read_tensor(&output, &mut actual).unwrap();
                assert_eq!(
                    actual,
                    EXPECTED[..input.len()],
                    "mode={mode}, scalar={scalar}"
                );
            }
        }
    }
}

#[test]
fn gelu_float32_constant_view_closure_and_native_consumer_preserve_tail() {
    let input = [
        INPUT[0], INPUT[1], INPUT[2], INPUT[3], INPUT[3], INPUT[2], INPUT[1], INPUT[0],
    ];
    let expected = [
        EXPECTED[0],
        EXPECTED[3],
        EXPECTED[1],
        EXPECTED[2],
        EXPECTED[2],
        EXPECTED[1],
        EXPECTED[3],
        EXPECTED[0],
    ];
    for mode in 0..3 {
        let mut source = graph(vec![Dimension::Static(8)]);
        source.operands[0].kind = OperandKind::Constant;
        source.operands[0].descriptor.shape = vec![Dimension::Static(2), Dimension::Static(4)];
        source.operands[1].kind = OperandKind::Intermediate;
        source.operands[1].name = Some("transposed".into());
        source.operands[1].descriptor.shape = vec![Dimension::Static(4), Dimension::Static(2)];
        for (name, kind) in [
            ("flattened", OperandKind::Intermediate),
            ("gelu", OperandKind::Output),
            ("result", OperandKind::Output),
        ] {
            let mut operand = source.operands[1].clone();
            operand.name = Some(name.into());
            operand.kind = kind;
            operand.descriptor.shape = vec![Dimension::Static(8)];
            source.operands.push(operand);
        }
        source.input_operands.clear();
        source.output_operands = vec![3, 4];
        source.operations = vec![
            Operation::from_json_attributes(
                "transpose",
                &[0],
                &[1],
                &serde_json::json!({"permutation": [1,0]}),
            )
            .unwrap(),
            Operation::from_json_attributes(
                "reshape",
                &[1],
                &[2],
                &serde_json::json!({"newShape": [8]}),
            )
            .unwrap(),
            Operation::Gelu {
                input: 2,
                options: None,
                outputs: vec![3],
            },
            Operation::Neg {
                input: 3,
                options: None,
                outputs: vec![4],
            },
        ];
        source.constant_operand_ids_to_handles.insert(
            0,
            ConstantData {
                data: input.iter().flat_map(|x| x.to_le_bytes()).collect(),
                label: None,
            },
        );
        let mut context = context(mode, DeviceType::Cpu);
        let mut graph = context.rustnn_build_graph(source).unwrap();
        let descriptor = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![8]).to_readable();
        let gelu = context.create_tensor(&descriptor).unwrap();
        let result = context.create_tensor(&descriptor).unwrap();
        context
            .dispatch(
                &mut graph,
                &MLNamedTensors::new(),
                &MLNamedTensors::from([("gelu", &gelu), ("result", &result)]),
            )
            .unwrap();
        let mut actual = [0_u32; 8];
        context.read_tensor(&gelu, &mut actual).unwrap();
        assert_eq!(actual, expected);
        context.read_tensor(&result, &mut actual).unwrap();
        assert_eq!(actual, expected.map(|x| x ^ 0x80000000));
    }
}

#[test]
fn gelu_float32_constant_arithmetic_is_materialized_before_typed_evaluation() {
    for mode in 0..3 {
        for scalar in [false, true] {
            let shape = if scalar {
                vec![]
            } else {
                vec![Dimension::Static(2)]
            };
            let mut source = graph(shape);
            let mut addend = source.operands[0].clone();
            addend.kind = OperandKind::Constant;
            addend.name = Some("addend".into());
            source.operands.insert(1, addend);
            source.operands[0].kind = OperandKind::Constant;
            source.operands[2].kind = OperandKind::Intermediate;
            source.operands[2].name = Some("sum".into());
            for name in ["gelu", "result"] {
                let mut output = source.operands[2].clone();
                output.name = Some(name.into());
                output.kind = OperandKind::Output;
                source.operands.push(output);
            }
            source.operations = vec![
                Operation::from_json_attributes("add", &[0, 1], &[2], &serde_json::Value::Null)
                    .unwrap(),
                Operation::Gelu {
                    input: 2,
                    options: None,
                    outputs: vec![3],
                },
                Operation::Neg {
                    input: 3,
                    options: None,
                    outputs: vec![4],
                },
            ];
            source.input_operands.clear();
            source.output_operands = vec![3, 4];
            let count = if scalar { 1 } else { 2 };
            for (id, input) in [
                (0, &[-11.0_f32, -6.0][..count]),
                (1, &[1.0_f32, 1.0][..count]),
            ] {
                source.constant_operand_ids_to_handles.insert(
                    id,
                    ConstantData {
                        data: input.iter().flat_map(|x| x.to_le_bytes()).collect(),
                        label: None,
                    },
                );
            }
            let mut context = context(mode, DeviceType::Cpu);
            let mut graph = context.rustnn_build_graph(source).unwrap();
            let descriptor = MLTensorDescriptor::new(
                MLOperandDataType::Float32,
                if scalar { vec![] } else { vec![2] },
            )
            .to_readable();
            let gelu = context.create_tensor(&descriptor).unwrap();
            let result = context.create_tensor(&descriptor).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::new(),
                    &MLNamedTensors::from([("gelu", &gelu), ("result", &result)]),
                )
                .unwrap();
            let mut actual = vec![0_u32; count];
            context.read_tensor(&gelu, &mut actual).unwrap();
            assert_eq!(actual, EXPECTED[..count]);
            context.read_tensor(&result, &mut actual).unwrap();
            assert_eq!(
                actual,
                EXPECTED[..count]
                    .iter()
                    .map(|x| x ^ 0x80000000)
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn gelu_float32_dequantized_constant_is_materialized_before_evaluation() {
    for mode in 0..3 {
        let mut source = graph(vec![Dimension::Static(2)]);
        source.operands[0].kind = OperandKind::Constant;
        source.operands[0].descriptor.data_type = DataType::Int8;
        let mut scale = source.operands[0].clone();
        scale.name = Some("scale".into());
        scale.descriptor.data_type = DataType::Float32;
        scale.descriptor.shape.clear();
        source.operands.insert(1, scale);
        source.operands[2].kind = OperandKind::Intermediate;
        source.operands[2].name = Some("dequantized".into());
        for name in ["gelu", "result"] {
            let mut operand = source.operands[2].clone();
            operand.kind = OperandKind::Output;
            operand.name = Some(name.into());
            source.operands.push(operand);
        }
        let mut zero_point = source.operands[0].clone();
        zero_point.name = Some("zero_point".into());
        zero_point.descriptor.shape.clear();
        source.operands.push(zero_point);
        source.input_operands.clear();
        source.output_operands = vec![3, 4];
        source.constant_operand_ids_to_handles.insert(
            0,
            ConstantData {
                data: vec![246, 251],
                label: None,
            },
        );
        source.constant_operand_ids_to_handles.insert(
            1,
            ConstantData {
                data: 1.0_f32.to_le_bytes().to_vec(),
                label: None,
            },
        );
        source.constant_operand_ids_to_handles.insert(
            5,
            ConstantData {
                data: vec![0],
                label: None,
            },
        );
        source.operations = vec![
            Operation::from_json_attributes(
                "dequantizeLinear",
                &[0, 1, 5],
                &[2],
                &serde_json::json!({}),
            )
            .unwrap(),
            Operation::Gelu {
                input: 2,
                options: None,
                outputs: vec![3],
            },
            Operation::Neg {
                input: 3,
                options: None,
                outputs: vec![4],
            },
        ];
        let mut context = context(mode, DeviceType::Cpu);
        let mut graph = context.rustnn_build_graph(source).unwrap();
        let descriptor = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2]).to_readable();
        let gelu = context.create_tensor(&descriptor).unwrap();
        let result = context.create_tensor(&descriptor).unwrap();
        context
            .dispatch(
                &mut graph,
                &MLNamedTensors::new(),
                &MLNamedTensors::from([("gelu", &gelu), ("result", &result)]),
            )
            .unwrap();
        let mut actual = [0_u32; 2];
        context.read_tensor(&gelu, &mut actual).unwrap();
        assert_eq!(actual, EXPECTED[..2]);
        context.read_tensor(&result, &mut actual).unwrap();
        assert_eq!(actual, [EXPECTED[0] ^ 0x80000000, EXPECTED[1] ^ 0x80000000]);
    }
}

#[test]
fn gelu_float16_constant_promotion_and_native_consumer_preserve_rounding() {
    for mode in 0..3 {
        for scalar in [false, true] {
            let mut source = graph(if scalar {
                vec![]
            } else {
                vec![Dimension::Static(2)]
            });
            for operand in &mut source.operands {
                operand.descriptor.data_type = DataType::Float16;
            }
            source.operands[0].kind = OperandKind::Constant;
            source.operands[1].name = Some("gelu".into());
            let mut output = source.operands[1].clone();
            output.name = Some("result".into());
            source.operands.push(output);
            source.operations.push(Operation::Neg {
                input: 1,
                options: None,
                outputs: vec![2],
            });
            source.input_operands.clear();
            source.output_operands = vec![1, 2];
            let input = if scalar {
                &[0xc500_u16][..]
            } else {
                &[0xc500_u16, 0xc900][..]
            };
            source.constant_operand_ids_to_handles.insert(
                0,
                ConstantData {
                    data: input.iter().flat_map(|x| x.to_le_bytes()).collect(),
                    label: None,
                },
            );
            let mut context = context(mode, DeviceType::Cpu);
            let mut graph = context.rustnn_build_graph(source).unwrap();
            let descriptor = MLTensorDescriptor::new(
                MLOperandDataType::Float16,
                if scalar { vec![] } else { vec![2] },
            )
            .to_readable();
            let gelu = context.create_tensor(&descriptor).unwrap();
            let result = context.create_tensor(&descriptor).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::new(),
                    &MLNamedTensors::from([("gelu", &gelu), ("result", &result)]),
                )
                .unwrap();
            let mut actual = vec![0_u16; input.len()];
            context.read_tensor(&gelu, &mut actual).unwrap();
            assert_eq!(actual, [0x8018, 0x8000][..input.len()]);
            context.read_tensor(&result, &mut actual).unwrap();
            assert_eq!(actual, [0x0018, 0x0000][..input.len()]);
        }
    }
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn gelu_float32_actual_shapes_grow_and_shrink_without_padding_values() {
    for mode in 0..3 {
        let mut context = context(mode, DeviceType::Cpu);
        let mut graph = context
            .rustnn_build_graph(graph(vec![Dimension::Dynamic(
                rustnn::graph::DynamicDimension {
                    name: "length".into(),
                    max_size: 8,
                },
            )]))
            .unwrap();
        for length in [1, 8, 3] {
            let descriptor =
                MLTensorDescriptor::new(MLOperandDataType::Float32, vec![length as u64]);
            let input = context
                .create_tensor(&descriptor.clone().to_writable())
                .unwrap();
            let output = context.create_tensor(&descriptor.to_readable()).unwrap();
            context.write_tensor(&input, &INPUT[..length]).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("input", &input)]),
                    &MLNamedTensors::from([("result", &output)]),
                )
                .unwrap();
            let mut actual = vec![0_u32; length];
            context.read_tensor(&output, &mut actual).unwrap();
            assert_eq!(actual, EXPECTED[..length]);
            assert_eq!(output.shape(), [length as u64]);
        }
    }
}
