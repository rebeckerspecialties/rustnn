//! Source Float32 Sqrt boundaries through public tensors, not native proxies.

#![cfg(all(target_os = "macos", feature = "coreml-runtime"))]

use rustnn::backend_selection::{BackendDevice, DeviceType};
use rustnn::executors::coreml::CoremlLoadRoute;
use rustnn::graph::{
    ConstantData, DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
};
use rustnn::mlcontext::{
    LoadDiagnostics, MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference,
    MLTensorDescriptor, RustNNOptions,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operators::Operation;

// Exact dyadic midpoint references. The first three roots are normal even
// though their inputs are subnormal; returning zero exceeds the 1-ULP WPT gate.
const INPUT: [u32; 8] = [
    1,
    2,
    0x007f_ffff,
    0x0080_0000,
    0,
    0x8000_0000,
    0x4080_0000,
    0x7f80_0000,
];
const EXPECTED: [u32; 8] = [
    0x1a35_04f3,
    0x1a80_0000,
    0x1fff_ffff,
    0x2000_0000,
    0,
    0x8000_0000,
    0x4000_0000,
    0x7f80_0000,
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
        operations: vec![Operation::Sqrt {
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
fn sqrt_float32_runtime_constant_scalar_and_owned_outputs() {
    for mode in 0..3 {
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            for constant in [false, true] {
                for scalar in [false, true] {
                    let shape = if scalar { vec![] } else { vec![8] };
                    let count = if scalar { 1 } else { 8 };
                    let mut source =
                        graph(shape.iter().map(|&size| Dimension::Static(size)).collect());
                    if constant {
                        source.operands[0].kind = OperandKind::Constant;
                        source.input_operands.clear();
                        source.constant_operand_ids_to_handles.insert(
                            0,
                            ConstantData {
                                data: INPUT[..count]
                                    .iter()
                                    .flat_map(|x| x.to_le_bytes())
                                    .collect(),
                                label: None,
                            },
                        );
                    }
                    let mut context = context(mode, policy);
                    let mut graph = context.rustnn_build_graph(source).unwrap();
                    let Some(LoadDiagnostics::Coreml(load)) = graph.rustnn_load_diagnostics()
                    else {
                        panic!("missing diagnostics")
                    };
                    assert_eq!(load.route, CoremlLoadRoute::TypedHost);
                    assert_eq!(load.loaded_compute_units, "NOT_APPLICABLE");
                    assert!(load.failures.is_empty());
                    let descriptor = MLTensorDescriptor::new(
                        MLOperandDataType::Float32,
                        shape.into_iter().map(u64::from).collect(),
                    );
                    let input = context
                        .create_tensor(&descriptor.clone().to_writable())
                        .unwrap();
                    let output = context.create_tensor(&descriptor.to_readable()).unwrap();
                    for _ in 0..2 {
                        context.write_tensor(&input, &INPUT[..count]).unwrap();
                        let inputs = if constant {
                            MLNamedTensors::new()
                        } else {
                            MLNamedTensors::from([("input", &input)])
                        };
                        context
                            .dispatch(
                                &mut graph,
                                &inputs,
                                &MLNamedTensors::from([("result", &output)]),
                            )
                            .unwrap();
                        context.write_tensor(&input, &vec![0_u32; count]).unwrap();
                        let mut actual = vec![0_u32; count];
                        context.read_tensor(&output, &mut actual).unwrap();
                        assert_eq!(
                            actual,
                            EXPECTED[..count],
                            "mode={mode}, policy={policy:?}, constant={constant}, scalar={scalar}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn sqrt_float32_constant_transpose_and_native_consumer_keep_root_values() {
    let input = [
        INPUT[0],
        INPUT[1],
        INPUT[2],
        INPUT[3],
        0x3f80_0000,
        0x4080_0000,
        0x4110_0000,
        0x4180_0000,
    ];
    let roots = [
        EXPECTED[0],
        EXPECTED[1],
        EXPECTED[2],
        EXPECTED[3],
        0x3f80_0000,
        0x4000_0000,
        0x4040_0000,
        0x4080_0000,
    ];
    for mode in 0..3 {
        let mut source = graph(vec![Dimension::Static(2), Dimension::Static(4)]);
        source.operands[0].kind = OperandKind::Constant;
        source.operands[1].descriptor.shape = vec![Dimension::Static(4), Dimension::Static(2)];
        source.operands[1].kind = OperandKind::Intermediate;
        source.operands[1].name = Some("transposed".into());
        for name in ["root", "result"] {
            let mut output = source.operands[1].clone();
            output.name = Some(name.into());
            output.kind = OperandKind::Output;
            source.operands.push(output);
        }
        source.input_operands.clear();
        source.output_operands = vec![2, 3];
        source.constant_operand_ids_to_handles.insert(
            0,
            ConstantData {
                data: input.iter().flat_map(|x| x.to_le_bytes()).collect(),
                label: None,
            },
        );
        source.operations = vec![
            Operation::from_json_attributes(
                "transpose",
                &[0],
                &[1],
                &serde_json::json!({"permutation": [1,0]}),
            )
            .unwrap(),
            Operation::Sqrt {
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
        let mut context = context(mode, DeviceType::Cpu);
        let mut graph = context.rustnn_build_graph(source).unwrap();
        let descriptor =
            MLTensorDescriptor::new(MLOperandDataType::Float32, vec![4, 2]).to_readable();
        let root = context.create_tensor(&descriptor).unwrap();
        let result = context.create_tensor(&descriptor).unwrap();
        context
            .dispatch(
                &mut graph,
                &MLNamedTensors::new(),
                &MLNamedTensors::from([("root", &root), ("result", &result)]),
            )
            .unwrap();
        let expected = [0, 4, 1, 5, 2, 6, 3, 7].map(|i| roots[i]);
        let mut actual = [0_u32; 8];
        context.read_tensor(&root, &mut actual).unwrap();
        assert_eq!(actual, expected);
        context.read_tensor(&result, &mut actual).unwrap();
        assert_eq!(actual, expected.map(|x| x ^ 0x8000_0000));
    }
}

#[test]
fn sqrt_float32_native_constant_producer_is_not_reinterpreted_as_a_source_constant() {
    for mode in 0..3 {
        let mut source = graph(vec![Dimension::Static(2)]);
        source.operands[0].kind = OperandKind::Constant;
        source.operands[1].kind = OperandKind::Intermediate;
        source.operands[1].name = Some("absolute".into());
        let mut output = source.operands[1].clone();
        output.name = Some("result".into());
        output.kind = OperandKind::Output;
        source.operands.push(output);
        source.input_operands.clear();
        source.output_operands = vec![2];
        source.operations = vec![
            Operation::Abs {
                input: 0,
                options: None,
                outputs: vec![1],
            },
            Operation::Sqrt {
                input: 1,
                options: None,
                outputs: vec![2],
            },
        ];
        source.constant_operand_ids_to_handles.insert(
            0,
            ConstantData {
                data: [-4_f32, -9.0]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect(),
                label: None,
            },
        );
        let mut context = context(mode, DeviceType::Cpu);
        let mut graph = context.rustnn_build_graph(source).unwrap();
        let result = context
            .create_tensor(
                &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2]).to_readable(),
            )
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &MLNamedTensors::new(),
                &MLNamedTensors::from([("result", &result)]),
            )
            .unwrap();
        let mut actual = [0_u32; 2];
        context.read_tensor(&result, &mut actual).unwrap();
        assert_eq!(actual, [2_f32.to_bits(), 3_f32.to_bits()]);
    }
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn sqrt_float32_bounded_actual_shapes_grow_then_shrink() {
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
        for count in [1, 8, 3] {
            let descriptor =
                MLTensorDescriptor::new(MLOperandDataType::Float32, vec![count as u64]);
            let input = context
                .create_tensor(&descriptor.clone().to_writable())
                .unwrap();
            let output = context.create_tensor(&descriptor.to_readable()).unwrap();
            context.write_tensor(&input, &INPUT[..count]).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("input", &input)]),
                    &MLNamedTensors::from([("result", &output)]),
                )
                .unwrap();
            let mut actual = vec![0_u32; count];
            context.read_tensor(&output, &mut actual).unwrap();
            assert_eq!(actual, EXPECTED[..count]);
            assert_eq!(output.shape(), [count as u64]);
        }
    }
}
